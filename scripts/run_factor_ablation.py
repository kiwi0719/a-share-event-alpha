"""事件作为因子的消融检验（预注册 addendum_v4）。

A 组：传统量价因子；B 组：A + 事件特征。岭回归，按年滚动（前 3 年训练），2018–2026 样本外。

用法: python scripts/run_factor_ablation.py PANEL.pkl YJYG.parquet GGCG.parquet CONTENT.jsonl INDEX.parquet OUT_DIR
"""

import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from event_alpha.research.event_study import COMMISSION, SLIPPAGE, market_series
from event_alpha.timeutil.rules import stamp_duty_rate

HOLD = 20
STEP = 20
ALPHA = 1.0
GOOD = ["预增", "扭亏", "略增", "续盈"]
BAD = ["首亏", "预减", "略减", "增亏", "续亏"]
A_FEATS = ["mom_20", "mom_120_skip20", "rev_5", "vol_20", "log_amt_20", "max_ret_20"]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parent / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def visible_idx(p, dates):
    """公告日（日期精度）在其后第一个交易日开盘可交易，决策日 = 该日前一交易日收盘。与事件研究一致。"""
    return p.dates.searchsorted(pd.DatetimeIndex(dates), side="right") - 1


def event_matrix(p, df):
    """df: code, notice_date [, value]；返回 (T, N) 矩阵，在可见日处计 1（或 value 最大值）。"""
    ci = {c: i for i, c in enumerate(p.codes)}
    E = np.zeros(p.close.shape, dtype=np.float32)
    df = df[df.code.isin(ci)]
    v = visible_idx(p, df.notice_date)
    j = df.code.map(ci).to_numpy()
    ok = v >= 0
    E[v[ok], j[ok]] = 1.0
    return E


def rolling_any(E, w):
    c = np.cumsum(E, axis=0)
    out = c.copy()
    out[w:] = c[w:] - c[:-w]
    return (out > 0).astype(np.float32)


def load_events(p, yjyg_path, ggcg_path, content_path):
    rs = _load_script("run_event_study")
    yj = rs.load_events(yjyg_path, p.codes)
    gg = pd.read_parquet(ggcg_path)
    suffix = {c.split(".")[0]: c for c in p.codes}
    gg["code"] = gg["代码"].astype(str).str.zfill(6).map(suffix)
    gg["notice_date"] = pd.to_datetime(gg["公告日"])
    gg = gg.dropna(subset=["code", "notice_date"])
    rc = _load_script("run_contract_study")
    ct = rc.load_events(content_path, p.codes)
    ev = {
        "good": yj[yj.category.isin(GOOD)],
        "bad": yj[yj.category.isin(BAD)],
        "buy": gg[gg["持股变动信息-增减"] == "增持"],
        "sell": gg[gg["持股变动信息-增减"] == "减持"],
        "contract": ct,
        "big_contract": ct[ct.ratio >= 0.30],
    }
    feats = {}
    for k, df in ev.items():
        E = event_matrix(p, df)
        feats[f"{k}_20"] = rolling_any(E, 20)
        feats[f"{k}_60"] = rolling_any(E, 60)
    # 预增幅度：60 日内最近一次好消息预告的变动幅度
    g = ev["good"].dropna(subset=["magnitude"])
    ci = {c: i for i, c in enumerate(p.codes)}
    g = g[g.code.isin(ci)].sort_values("notice_date")
    Mg = np.full(p.close.shape, np.nan, dtype=np.float32)
    Mg[visible_idx(p, g.notice_date), g.code.map(ci).to_numpy()] = np.clip(g.magnitude.to_numpy(), -100, 500)
    feats["good_mag_60"] = pd.DataFrame(Mg).ffill(limit=59).to_numpy()
    return feats


def traditional(p):
    Rraw = p.ret
    R = np.nan_to_num(Rraw, nan=0.0)
    L = np.vstack([np.zeros((1, R.shape[1])), np.cumsum(np.log1p(np.clip(R, -0.99, None)), axis=0)])  # L[t+1]=sum R[0..t]

    def cum(a, b):  # 返回函数：在 d 处 sum R[d-a .. d-b]
        return lambda d: np.expm1(L[d - b + 1] - L[d - a])

    amt = pd.DataFrame(p.amount_cny)
    return dict(
        mom_20=cum(19, 0),
        mom_120_skip20=cum(119, 20),
        rev_5=cum(4, 0),
        vol_20=lambda d: np.nanstd(Rraw[d - 19 : d + 1], axis=0),
        log_amt_20=lambda d, A=amt.rolling(20, min_periods=10).mean().to_numpy(): np.log1p(A[d]),
        max_ret_20=lambda d: np.max(np.where(np.isnan(Rraw[d - 19 : d + 1]), -1, Rraw[d - 19 : d + 1]), axis=0),
    )


def rank_pct(x, mask):
    out = np.full(x.shape, np.nan)
    s = pd.Series(x[mask])
    out[mask] = s.rank(pct=True).to_numpy()
    return out


def main(panel_path, yjyg_path, ggcg_path, content_path, index_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(panel_path, "rb") as f:
        p = pickle.load(f)
    T, N = p.close.shape
    M, MI = market_series(p)
    R = p.ret
    trad = traditional(p)
    ev = load_events(p, yjyg_path, ggcg_path, content_path)
    B_FEATS = A_FEATS + list(ev)
    first = p.dates.searchsorted(pd.DatetimeIndex(p.list_date))
    amt20 = pd.DataFrame(p.amount_cny).rolling(20, min_periods=10).mean().to_numpy()

    start = int(p.dates.searchsorted(pd.Timestamp("2015-01-01")))
    rows = []
    for d in range(max(start, 130), T - HOLD - 1, STEP):
        e = d + 1
        x = d + HOLD
        mask = (~p.is_st[d]) & (d - first >= 60) & (amt20[d] >= 3e7) & ~np.isnan(p.close[d])
        mask &= ~np.isnan(p.open[e]) & (p.open[e] < p.high_limit[e] - 1e-6)
        if mask.sum() < 100:
            continue
        gross = (p.close[e] / p.open[e]) * np.prod(1 + np.nan_to_num(R[e + 1 : x + 1]), axis=0) - 1
        mkt = (1 + MI[e]) * np.prod(1 + np.nan_to_num(M[e + 1 : x + 1])) - 1
        cols = {k: rank_pct(f(d), mask) for k, f in trad.items()}
        for k, E in ev.items():
            v = E[d].astype(float)
            if k == "good_mag_60":
                cols[k] = rank_pct(np.nan_to_num(v, nan=-200.0), mask)
            else:
                cols[k] = np.where(mask, v, np.nan)
        df = pd.DataFrame(cols)
        df["j"] = np.arange(N)
        df["gross"] = gross
        df["ex"] = gross - mkt
        df = df[mask].copy()
        df["y"] = df.ex.rank(pct=True) - 0.5
        df["d"] = d
        rows.append(df)
    data = pd.concat(rows, ignore_index=True)
    for c in A_FEATS + ["good_mag_60"]:
        data[c] = data[c].fillna(0.5) - 0.5
    data["date"] = p.dates[data.d.to_numpy()]
    data["exit_date"] = p.dates[data.d.to_numpy() + HOLD]
    print("截面数", data.d.nunique(), "样本", len(data))

    def ridge(Xtr, ytr):
        k = Xtr.shape[1]
        return np.linalg.solve(Xtr.T @ Xtr + ALPHA * np.eye(k), Xtr.T @ ytr)

    preds = []
    coefs = []
    for year in range(2018, 2027):
        test = data[data.date.dt.year == year]
        if test.empty:
            continue
        train = data[(data.date >= pd.Timestamp(year - 3, 1, 1)) & (data.exit_date < pd.Timestamp(year, 1, 1))]
        res = test[["d", "j", "gross", "ex", "y"]].copy()
        for name, feats in (("A", A_FEATS), ("B", B_FEATS)):
            w = ridge(train[feats].to_numpy(float), train.y.to_numpy(float))
            res[f"pred_{name}"] = test[feats].to_numpy(float) @ w
            coefs.append(pd.Series(w, index=feats, name=f"{year}_{name}"))
        preds.append(res)
    pr = pd.concat(preds)
    pd.concat(coefs, axis=1).to_csv(out / "coefs.csv")

    # IC
    ic = pr.groupby("d").apply(lambda g: pd.Series({
        "A": g.pred_A.rank().corr(g.y.rank()), "B": g.pred_B.rank().corr(g.y.rank())}))
    ic["diff"] = ic.B - ic.A

    def tstat(s):
        return s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))

    # 前 5 只 / 前 10%
    def top(g, col, k):
        return g.nlargest(k, col)

    port = []
    idx = pd.read_parquet(index_path)
    csi = idx[idx.code == "sh000852"].set_index("date")["close"]
    csi.index = pd.to_datetime(csi.index)
    for d, g in pr.groupby("d"):
        x = d + HOLD
        cost = 2 * COMMISSION + 2 * SLIPPAGE + float(stamp_duty_rate(p.dates[x].date(), "sell"))
        bench = csi.asof(p.dates[x]) / csi.asof(p.dates[d]) - 1
        row = dict(d=p.dates[d], bench=bench)
        for name in ("A", "B"):
            row[f"top5_{name}"] = top(g, f"pred_{name}", 5).gross.mean() - cost
            k = max(int(len(g) * 0.1), 1)
            row[f"dec_ex_{name}"] = top(g, f"pred_{name}", k).ex.mean()
        port.append(row)
    pt = pd.DataFrame(port).set_index("d")
    years = (pt.index[-1] - pt.index[0]).days / 365.25 + HOLD / 245

    def ann(r):
        return (1 + r).prod() ** (1 / years) - 1

    summary = {
        "IC_A": ic.A.mean(), "IC_A_t": tstat(ic.A), "IC_B": ic.B.mean(), "IC_B_t": tstat(ic.B),
        "IC_diff": ic["diff"].mean(), "IC_diff_t": tstat(ic["diff"]),
        "top5_A_ann": ann(pt.top5_A), "top5_B_ann": ann(pt.top5_B), "csi1000_ann": ann(pt.bench),
        "top5_A_excess": ann(pt.top5_A) - ann(pt.bench), "top5_B_excess": ann(pt.top5_B) - ann(pt.bench),
        "dec_ex_A_per20d": pt.dec_ex_A.mean(), "dec_ex_B_per20d": pt.dec_ex_B.mean(),
    }
    rel_A = ((1 + pt.top5_A).cumprod() / (1 + pt.bench).cumprod())
    rel_B = ((1 + pt.top5_B).cumprod() / (1 + pt.bench).cumprod())
    summary["rel_maxdd_A"] = (rel_A / rel_A.cummax() - 1).min()
    summary["rel_maxdd_B"] = (rel_B / rel_B.cummax() - 1).min()
    summary["pass_gate"] = bool(summary["IC_diff"] > 0 and summary["IC_diff_t"] > 2
                                and summary["top5_B_excess"] - summary["top5_A_excess"] >= 0.01)
    s = pd.Series(summary)
    s.to_csv(out / "ablation_summary.csv")
    ic.to_csv(out / "ic.csv")
    pt.to_csv(out / "portfolio.csv")
    print(s.round(4).to_string())
    by_year = pd.DataFrame({
        "IC_A": ic.A.groupby(p.dates[ic.index].year).mean(),
        "IC_B": ic.B.groupby(p.dates[ic.index].year).mean(),
        "top5_A": pt.top5_A.groupby(pt.index.year).apply(lambda r: (1 + r).prod() - 1),
        "top5_B": pt.top5_B.groupby(pt.index.year).apply(lambda r: (1 + r).prod() - 1),
        "csi1000": pt.bench.groupby(pt.index.year).apply(lambda r: (1 + r).prod() - 1),
    })
    by_year.to_csv(out / "by_year.csv")
    print(by_year.round(4).to_string())
    print(pd.concat(coefs, axis=1).filter(like="_B").mean(axis=1).round(4).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:7])
