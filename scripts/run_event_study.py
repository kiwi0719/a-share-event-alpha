"""业绩预告事件研究（预注册见 research/preregistration.yaml）。

用法: python scripts/run_event_study.py PANEL.pkl YJYG.parquet OUT_DIR
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from event_alpha.research.event_study import clustered_t, run, summarize

GOOD = ["预增", "扭亏", "略增", "续盈"]
BAD = ["首亏", "预减", "增亏", "续亏", "略减"]


def load_events(path: str, codes: pd.Index) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["预测指标"] == "归属于上市公司股东的净利润"].copy()
    suffix = {c.split(".")[0]: c for c in codes}
    df["code"] = df["股票代码"].astype(str).str.zfill(6).map(suffix)
    df = df.dropna(subset=["code", "公告日期"])
    df["notice_date"] = pd.to_datetime(df["公告日期"])
    df["category"] = df["预告类型"]
    df["magnitude"] = pd.to_numeric(df["业绩变动幅度"], errors="coerce")
    df = df.drop_duplicates(["code", "notice_date", "report_period"])
    df = df.sort_values("notice_date").drop_duplicates(["code", "notice_date"])
    return df[["code", "notice_date", "category", "magnitude", "report_period"]]


def bh(pvals: pd.Series, q: float = 0.10) -> pd.Series:
    p = pvals.dropna().sort_values()
    m = len(p)
    thresh = q * np.arange(1, m + 1) / m
    passed = p.values <= thresh
    k = np.max(np.where(passed)[0]) + 1 if passed.any() else 0
    return pvals.index.isin(p.index[:k])


def main(panel_path, events_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(panel_path, "rb") as f:
        p = pickle.load(f)
    ev = load_events(events_path, p.codes)
    ev = ev[(ev.notice_date >= "2015-01-01")]
    print("events", len(ev))
    res = run(p, ev)
    res["period"] = np.where(res["t0"] >= "2024-01-01", "2024+", "2015-2023")
    # 流通市值分位（决策日）
    code_idx = {c: i for i, c in enumerate(p.codes)}
    mv_rank = []
    for r in res.itertuples():
        d = int(p.dates.searchsorted(r.t0)) - 1
        row = p.circ_mv_cny[d]
        v = row[code_idx[r.code]]
        mv_rank.append(np.nanmean(row < v) if not np.isnan(v) else np.nan)
    res["mv_pct"] = mv_rank
    res.to_parquet(out / "event_results.parquet")

    # H3 子组
    h3 = res[(res.category == "预增") & (res.magnitude >= 100)].assign(category="预增≥100%")
    allres = pd.concat([res, h3])
    summ = summarize(allres)
    tests = summ.set_index("category")
    dfree = allres.groupby("category")["notice_date"].nunique()
    tests["p_skip2"] = [2 * stats.t.sf(abs(t), max(dfree[c] - 1, 1)) for c, t in tests["tr_skip2_t"].items()]
    tests["bh_pass_10pct"] = bh(tests["p_skip2"])
    tests.to_csv(out / "summary_all.csv")

    rob = []
    for name, sub in (
        ("2015-2023", allres[allres.period == "2015-2023"]),
        ("2024+", allres[allres.period == "2024+"]),
        ("剔除最小30%市值", allres[allres.mv_pct >= 0.3]),
    ):
        s = summarize(sub).set_index("category")[["n", "tr_skip2_mean", "tr_skip2_t", "tr_now_mean"]]
        s.columns = [f"{name}:{c}" for c in s.columns]
        rob.append(s)
    pd.concat(rob, axis=1).to_csv(out / "robustness.csv")

    by_year = (
        allres[allres.category.isin(GOOD + ["预增≥100%"])]
        .assign(year=lambda d: d.t0.dt.year)
        .groupby(["category", "year"])["tr_skip2"]
        .mean()
        .unstack()
    )
    by_year.to_csv(out / "by_year_skip2.csv")
    pd.set_option("display.width", 250)
    cols = ["n", "open_limit_up_rate", "info_car_0_1_mean", "info_car_2_5_mean", "info_car_6_20_mean",
            "tr_now_mean", "tr_now_t", "tr_skip2_mean", "tr_skip2_t", "p_skip2", "bh_pass_10pct"]
    print(tests[cols].round(4).to_string())
    print(pd.concat(rob, axis=1).round(4).to_string())
    print(by_year.round(4).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:4])
