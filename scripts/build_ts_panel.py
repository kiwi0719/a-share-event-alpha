"""由 Tushare daily（按交易日下载的 parquet）构建全市场面板：主板 + 创业板 + 科创板，含退市股（不含北交所）。

- 价格不复权，prev_close 为交易所前收盘（除权参考价），close/prev_close 即真实日收益。
- 120 积分拿不到 stk_limit，涨跌停价按 rules.price_limit 由前收盘计算；ST 由 namechange 的简称区间判断。
- 上市日期用数据中首次出现的交易日近似（2014-01-02 已存在的股票视为老股）。
- 拿不到 daily_basic，circ_mv_cny 用 20 日平均成交额代替，仅用于“剔除最小 30%”稳健性检验（改名为流动性分位）。

用法: python scripts/build_ts_panel.py TS_DIR OUT.pkl
"""

import glob
import pickle
import sys
from datetime import date

import numpy as np
import pandas as pd

from event_alpha.data.panel import Panel
from event_alpha.timeutil.rules import Board, board_of, price_limit


def st_mask(nc: pd.DataFrame, dates: pd.DatetimeIndex, codes: pd.Index) -> np.ndarray:
    m = np.zeros((len(dates), len(codes)), dtype=bool)
    ci = {c: i for i, c in enumerate(codes)}
    st = nc[nc["name"].str.contains("ST", na=False)]
    for r in st.itertuples():
        j = ci.get(r.ts_code)
        if j is None:
            continue
        a = dates.searchsorted(pd.Timestamp(r.start_date))
        b = dates.searchsorted(pd.Timestamp(r.end_date), side="right") if isinstance(r.end_date, str) and r.end_date else len(dates)
        m[a:b, j] = True
    return m


def limit_pct_matrix(dates, codes, is_st, nth):
    """返回 (up, down) 比例矩阵；NaN 表示不设涨跌幅。按 (板块, 是否 ST, 制度区间, 新股期) 分组计算，避免逐格调用。"""
    up = np.full(is_st.shape, np.nan)
    dn = np.full(is_st.shape, np.nan)
    boards = np.array([board_of(c) for c in codes])
    # 制度切换日：取每个交易日的代表性规则（同一日同一组合结果相同）
    for bi, b in enumerate([Board.MAIN, Board.CHINEXT, Board.STAR, Board.BSE]):
        cols = np.where(boards == b)[0]
        if len(cols) == 0:
            continue
        for t, d in enumerate(dates):
            dd = d.date()
            for st in (False, True):
                for n_bucket in (1, 3, 10_000):  # 首日 / 新股 2-5 日 / 普通
                    lim = price_limit(b, dd, is_st=st, nth_trading_day_since_listing=n_bucket)
                    sel = cols[(is_st[t, cols] == st) & (np.where(nth[t, cols] == 1, 1, np.where(nth[t, cols] <= 5, 3, 10_000)) == n_bucket)]
                    if len(sel) == 0 or lim.unlimited:
                        continue
                    up[t, sel] = float(lim.up_pct)
                    dn[t, sel] = float(lim.down_pct)
    return up, dn


def round_half_up(x):
    return np.floor(x * 100 + 0.5 + 1e-9) / 100


def main(ts_dir, out):
    files = sorted(glob.glob(f"{ts_dir}/daily/*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df["date"] = pd.to_datetime(df["trade_date"])
    df = df.drop_duplicates(["ts_code", "date"])
    # 不含北交所：需要单独开通权限，且 2024 年代码切换为 92xxxx 后 Tushare 前收盘有异常值
    df = df[df.ts_code.str[:2].isin(["60", "00", "30", "68"])]
    dates = pd.DatetimeIndex(sorted(df.date.unique()))
    codes = pd.Index(sorted(df.ts_code.unique()))

    def wide(col):
        return df.pivot(index="date", columns="ts_code", values=col).reindex(index=dates, columns=codes).to_numpy(float)

    o, c, pc, amt = wide("open"), wide("close"), wide("pre_close"), wide("amount") * 1000.0
    first = df.groupby("ts_code")["date"].min().reindex(codes)
    first_idx = dates.searchsorted(first.values)
    old = first_idx == 0
    tgrid = np.arange(len(dates))[:, None]
    nth = np.where(old[None, :], 10_000, tgrid - first_idx[None, :] + 1).astype(float)
    nth[nth < 1] = np.nan

    nc = pd.read_parquet(f"{ts_dir}/namechange.parquet")
    is_st = st_mask(nc, dates, codes)
    up, dn = limit_pct_matrix(dates, codes, is_st, np.nan_to_num(nth, nan=10_000))
    hl = round_half_up(pc * (1 + up))
    ll = round_half_up(pc * (1 - dn))
    hl[np.isnan(c)] = np.nan
    ll[np.isnan(c)] = np.nan

    # 自检：收盘价不应超出计算出的涨跌停价
    with np.errstate(invalid="ignore"):
        bad_up = np.nansum(c > hl + 0.011) / np.sum(~np.isnan(hl))
        bad_dn = np.nansum(c < ll - 0.011) / np.sum(~np.isnan(ll))
        at_up = np.nansum(np.abs(c - hl) < 0.005) / np.sum(~np.isnan(hl))
    print(f"[check] close>涨停价 {bad_up:.5%}；close<跌停价 {bad_dn:.5%}；收盘涨停占比 {at_up:.3%}")

    liq = pd.DataFrame(amt).rolling(20, min_periods=5).mean().to_numpy()
    p = Panel(dates=dates, codes=codes, open=o, close=c, prev_close=pc, high_limit=hl, low_limit=ll,
              amount_cny=amt, circ_mv_cny=liq, is_st=is_st, list_date=first)
    with open(out, "wb") as f:
        pickle.dump(p, f)
    print("panel", p.close.shape, "codes by board:",
          pd.Series([board_of(x).value for x in codes]).value_counts().to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:3])
