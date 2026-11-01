"""事件研究：信息 CAR 与可交易 CAR。

- 信息 CAR：按收盘到收盘的市场调整异常收益累加，第 0 日（首个可交易日）包含开盘跳空；
  这部分反映“公告含多少信息”，但第 0 日跳空对日频收盘后决策者是拿不到的。
- 可交易 CAR：用 sim.fill 在可成交约束下真实买卖，扣除交易成本，再减去同期市场收益。
- 显著性：先按公告日聚合（同日事件取均值），再对日均值做 t 检验，
  以缓解年报季事件聚集造成的截面相关（Kolari & Pynnönen 2010 指出的问题的简化处理）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..data.panel import Panel
from ..sim.fill import simulate
from ..timeutil.rules import stamp_duty_rate

COMMISSION = 0.00025
SLIPPAGE = 0.001  # 单边


def market_series(p: Panel) -> tuple[np.ndarray, np.ndarray]:
    """等权市场：收盘到收盘收益、开盘到收盘收益（剔除 ST）。"""
    r = p.ret.copy()
    r[p.is_st] = np.nan
    intraday = p.close / p.open - 1.0
    intraday[p.is_st] = np.nan
    with np.errstate(invalid="ignore"):
        return np.nanmean(r, axis=1), np.nanmean(intraday, axis=1)


def eligible(p: Panel, j: int, d: int, min_days: int = 60, min_amount: float = 3e7) -> bool:
    """决策日 d（公告日收盘）可见信息下的股票池过滤。"""
    if d < min_days or p.is_st[d, j]:
        return False
    if np.isnan(p.close[d - min_days : d + 1, j]).sum() > min_days * 0.5:
        return False
    amt = np.nanmean(p.amount_cny[d - 19 : d + 1, j])
    return bool(amt >= min_amount)


def round_trip_cost(p: Panel, exit_idx: int) -> float:
    return 2 * COMMISSION + 2 * SLIPPAGE + float(stamp_duty_rate(p.dates[exit_idx].date(), "sell"))


def _mkt_window(M: np.ndarray, MI: np.ndarray, entry: int, exit_: int) -> float:
    r = 1.0 + MI[entry]
    if exit_ > entry:
        r *= np.prod(1.0 + np.nan_to_num(M[entry + 1 : exit_ + 1]))
    return r - 1.0


def run(p: Panel, events: pd.DataFrame, horizon: int = 20) -> pd.DataFrame:
    """events 需含列：code（带 .SH/.SZ）、notice_date、category。返回逐事件结果。"""
    M, MI = market_series(p)
    code_idx = {c: i for i, c in enumerate(p.codes)}
    T = len(p.dates)
    out = []
    for ev in events.itertuples(index=False):
        j = code_idx.get(ev.code)
        if j is None:
            continue
        nd = pd.Timestamp(ev.notice_date)
        t0 = int(p.dates.searchsorted(nd, side="right"))  # 日期精度：下一交易日可交易
        d = t0 - 1
        if t0 + horizon >= T or d < 0 or not eligible(p, j, d):
            continue
        ar = p.ret[t0 : t0 + horizon + 1, j] - M[t0 : t0 + horizon + 1]
        ar = np.nan_to_num(ar)
        row = dict(
            code=ev.code,
            notice_date=nd,
            t0=p.dates[t0],
            category=ev.category,
            info_car_0_1=ar[0:2].sum(),
            info_car_2_5=ar[2:6].sum(),
            info_car_6_20=ar[6:21].sum(),
            open_limit_up_t0=bool(p.open[t0, j] >= p.high_limit[t0, j] - 1e-6) if not np.isnan(p.open[t0, j]) else True,
        )
        for name, start, hold in (("now", t0, horizon + 1), ("skip2", t0 + 2, horizon - 1)):
            f = simulate(p, j, start, hold)
            if f is None:
                row[f"tr_{name}"] = np.nan
                row[f"filled_{name}"] = False
                continue
            row[f"filled_{name}"] = True
            row[f"delay_{name}"] = f.entry_idx - start
            row[f"tr_{name}"] = f.gross_ret - _mkt_window(M, MI, f.entry_idx, f.exit_idx) - round_trip_cost(p, f.exit_idx)
        for extra in ("magnitude",):
            if hasattr(ev, extra):
                row[extra] = getattr(ev, extra)
        out.append(row)
    return pd.DataFrame(out)


def clustered_t(df: pd.DataFrame, col: str, by: str = "notice_date") -> tuple[float, float, int]:
    """按公告日聚合后的均值与 t 值。返回 (mean, t, n_dates)。"""
    s = df.dropna(subset=[col]).groupby(by)[col].mean()
    n = len(s)
    if n < 3:
        return float("nan"), float("nan"), n
    m = s.mean()
    return float(m), float(m / (s.std(ddof=1) / np.sqrt(n))), n


def summarize(res: pd.DataFrame, group: str = "category") -> pd.DataFrame:
    rows = []
    for g, df in res.groupby(group):
        r = {group: g, "n": len(df), "open_limit_up_rate": df["open_limit_up_t0"].mean()}
        for col in ("info_car_0_1", "info_car_2_5", "info_car_6_20", "tr_now", "tr_skip2"):
            m, t, nd = clustered_t(df, col)
            r[f"{col}_mean"], r[f"{col}_t"] = m, t
        r["fill_rate_now"] = df["filled_now"].mean()
        rows.append(r)
    return pd.DataFrame(rows).sort_values("n", ascending=False)
