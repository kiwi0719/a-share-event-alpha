"""模拟盘：用冻结的 B 模型（预注册 v4）在最新收盘日选 5 只，并给历史持仓记账。

规则（与回测一致）：
- 决策日 d 收盘后打分；d+1 开盘买入（开盘涨停或停牌则该只作废，不替补）；持有 20 个交易日，d+20 收盘卖出。
- 每 20 个交易日调仓一次；模型权重每年初用此前 3 年数据重训一次（当年内冻结）。
- 成本：佣金 2.5bp×2 + 滑点 10bp×2 + 卖出印花税。基准为中证1000 同期涨跌。

用法:
  python scripts/paper_trade.py pick   PANEL.pkl YJYG GGCG CONTENT.jsonl LEDGER.csv
  python scripts/paper_trade.py settle PANEL.pkl INDEX.parquet LEDGER.csv
"""

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from event_alpha.research.event_study import COMMISSION, SLIPPAGE, market_series
from event_alpha.timeutil.rules import stamp_duty_rate

HOLD = 20


def _abl():
    spec = importlib.util.spec_from_file_location("abl", Path(__file__).parent / "run_factor_ablation.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def cross_section(p, d, trad, ev, first, amt20, with_target):
    abl = _abl.cache
    N = len(p.codes)
    mask = (~p.is_st[d]) & (d - first >= 60) & (amt20[d] >= 3e7) & ~np.isnan(p.close[d])
    if with_target:  # 训练截面与消融脚本保持一致：排名前先剔除 d+1 开盘不可买的股票
        mask &= ~np.isnan(p.open[d + 1]) & (p.open[d + 1] < p.high_limit[d + 1] - 1e-6)
    cols = {k: abl.rank_pct(f(d), mask) for k, f in trad.items()}
    for k, E in ev.items():
        v = E[d].astype(float)
        cols[k] = abl.rank_pct(np.nan_to_num(v, nan=-200.0), mask) if k == "good_mag_60" else np.where(mask, v, np.nan)
    df = pd.DataFrame(cols)
    df["j"] = np.arange(N)
    if with_target:
        e, x = d + 1, d + HOLD
        M, MI = with_target
        gross = (p.close[e] / p.open[e]) * np.prod(1 + np.nan_to_num(p_ret_cache[e + 1 : x + 1]), axis=0) - 1
        mkt = (1 + MI[e]) * np.prod(1 + np.nan_to_num(M[e + 1 : x + 1])) - 1
        df["ex"] = gross - mkt
    df = df[mask].copy()
    for c in abl.A_FEATS + ["good_mag_60"]:
        df[c] = df[c].fillna(0.5) - 0.5
    if with_target:
        df["y"] = df.ex.rank(pct=True) - 0.5
    return df


def train_weights(p, year, trad, ev, first, amt20, feats):
    """与消融脚本一致：用 [year-3, year) 内、持有期在 year 之前结束的调仓截面训练岭回归。"""
    abl = _abl.cache
    M, MI = market_series(p)
    T = len(p.dates)
    start = int(p.dates.searchsorted(pd.Timestamp("2015-01-01")))
    grid = range(max(start, 130), T - HOLD - 1, abl.STEP)
    rows = []
    for d in grid:
        if not (pd.Timestamp(year - 3, 1, 1) <= p.dates[d] and p.dates[d + HOLD] < pd.Timestamp(year, 1, 1)):
            continue
        rows.append(cross_section(p, d, trad, ev, first, amt20, (M, MI)))
    tr = pd.concat(rows)
    X, y = tr[feats].to_numpy(float), tr.y.to_numpy(float)
    return np.linalg.solve(X.T @ X + abl.ALPHA * np.eye(X.shape[1]), X.T @ y)


def pick(panel_path, yjyg, ggcg, content, ledger_path):
    global p_ret_cache
    abl = _abl.cache = _abl()
    with open(panel_path, "rb") as f:
        p = pickle.load(f)
    p_ret_cache = p.ret
    trad = abl.traditional(p)
    ev = abl.load_events(p, yjyg, ggcg, content)
    feats = abl.A_FEATS + list(ev)
    first = p.dates.searchsorted(pd.DatetimeIndex(p.list_date))
    amt20 = pd.DataFrame(p.amount_cny).rolling(20, min_periods=10).mean().to_numpy()
    d = len(p.dates) - 1
    year = p.dates[d].year
    wpath = Path(ledger_path).with_name(f"weights_{year}.json")
    if wpath.exists():
        w = np.array(json.loads(wpath.read_text())["w"])
    else:
        w = train_weights(p, year, trad, ev, first, amt20, feats)
        wpath.write_text(json.dumps(dict(year=year, feats=feats, w=w.tolist()), ensure_ascii=False, indent=1))
    cs = cross_section(p, d, trad, ev, first, amt20, None)
    cs["score"] = cs[feats].to_numpy(float) @ w
    top = cs.nlargest(5, "score")
    out = pd.DataFrame(dict(
        decision_date=p.dates[d].date(), code=p.codes[top.j.to_numpy()], score=top.score.round(4).to_numpy(),
        good_mag_60=top.good_mag_60.round(3).to_numpy(), big_contract_20=top.big_contract_20.to_numpy(),
        entry_date=pd.NaT, entry_open=np.nan, exit_date=pd.NaT, exit_close=np.nan, net_ret=np.nan, bench_ret=np.nan,
        status="pending"))
    lp = Path(ledger_path)
    led = pd.read_csv(lp) if lp.exists() else pd.DataFrame(columns=out.columns)
    if str(p.dates[d].date()) in set(led.decision_date.astype(str)):
        print("该决策日已选过，跳过")
        return
    pd.concat([led, out]).to_csv(lp, index=False)
    print(out[["decision_date", "code", "score"]].to_string(index=False))


def settle(panel_path, index_path, ledger_path):
    with open(panel_path, "rb") as f:
        p = pickle.load(f)
    led = pd.read_csv(ledger_path, dtype={"entry_date": object, "exit_date": object, "status": object})
    idx = pd.read_parquet(index_path)
    csi = idx[idx.code == "sh000852"].set_index("date")["close"]
    csi.index = pd.to_datetime(csi.index)
    ci = {c: i for i, c in enumerate(p.codes)}
    R = p.ret
    for i, r in led.iterrows():
        if r.status in ("closed", "void"):
            continue
        d = int(p.dates.searchsorted(pd.Timestamp(r.decision_date)))
        e, x, j = d + 1, d + HOLD, ci[r.code]
        if e >= len(p.dates):
            continue
        if np.isnan(p.open[e, j]) or p.open[e, j] >= p.high_limit[e, j] - 1e-6:
            led.loc[i, ["status", "entry_date"]] = ["void", str(p.dates[e].date())]
            continue
        led.loc[i, ["entry_date", "entry_open", "status"]] = [str(p.dates[e].date()), p.open[e, j], "open"]
        last = min(x, len(p.dates) - 1)
        gross = (p.close[e, j] / p.open[e, j]) * np.prod(1 + np.nan_to_num(R[e + 1 : last + 1, j])) - 1
        bench = csi.asof(p.dates[last]) / csi.asof(p.dates[d]) - 1
        if x <= len(p.dates) - 1:
            cost = 2 * COMMISSION + 2 * SLIPPAGE + float(stamp_duty_rate(p.dates[x].date(), "sell"))
            led.loc[i, ["exit_date", "exit_close", "net_ret", "bench_ret", "status"]] = [
                str(p.dates[x].date()), p.close[x, j], gross - cost, bench, "closed"]
        else:
            led.loc[i, ["net_ret", "bench_ret"]] = [gross - 2 * COMMISSION - SLIPPAGE, bench]  # 浮动
    led.to_csv(ledger_path, index=False)
    s = led[led.status.isin(["open", "closed"])].groupby("decision_date").agg(
        n=("code", "size"), ret=("net_ret", "mean"), bench=("bench_ret", "first"))
    s["excess"] = s.ret - s.bench
    print(s.round(4).to_string())


if __name__ == "__main__":
    {"pick": pick, "settle": settle}[sys.argv[1]](*sys.argv[2:])
