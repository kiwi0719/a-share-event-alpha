"""股东增减持事件研究（预注册 addendum_v2）。用法: python scripts/run_ggcg_study.py PANEL.pkl GGCG.parquet OUT_DIR"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from event_alpha.research.event_study import run, summarize


def load(path, codes):
    d = pd.read_parquet(path)
    suffix = {c.split(".")[0]: c for c in codes}
    d["code"] = d["代码"].astype(str).str.zfill(6).map(suffix)
    d["notice_date"] = pd.to_datetime(d["公告日"], errors="coerce")
    d["pct"] = pd.to_numeric(d["持股变动信息-占总股本比例"], errors="coerce")
    d = d.dropna(subset=["code", "notice_date"])
    g = d.groupby(["code", "notice_date", "持股变动信息-增减"], as_index=False)["pct"].sum()
    g = g.rename(columns={"持股变动信息-增减": "category", "pct": "magnitude"})
    return g


def main(panel_path, path, out_dir):
    out = Path(out_dir)
    with open(panel_path, "rb") as f:
        p = pickle.load(f)
    ev = load(path, p.codes)
    ev = ev[ev.notice_date >= "2015-01-01"]
    print(ev.groupby([ev.notice_date.dt.year, "category"]).size().unstack())
    res = run(p, ev)
    big = res[(res.category == "增持") & (res.magnitude >= 0.5)].assign(category="增持≥0.5%")
    allres = pd.concat([res, big])
    allres["period"] = np.where(allres.t0 >= "2024-01-01", "2024+", "2015-2023")
    allres.to_parquet(out / "ggcg_results.parquet")
    s = summarize(allres).set_index("category")
    nd = allres.groupby("category")["notice_date"].nunique()
    s["p_skip2"] = [2 * stats.t.sf(abs(t), max(nd[c] - 1, 1)) for c, t in s["tr_skip2_t"].items()]
    cols = ["n", "open_limit_up_rate", "info_car_0_1_mean", "info_car_2_5_mean", "info_car_6_20_mean",
            "tr_now_mean", "tr_now_t", "tr_skip2_mean", "tr_skip2_t", "p_skip2"]
    pd.set_option("display.width", 250)
    print(s[cols].round(4).to_string())
    for per, sub in allres.groupby("period"):
        print(per)
        print(summarize(sub).set_index("category")[["n", "tr_skip2_mean", "tr_skip2_t"]].round(4).to_string())
    s.to_csv(out / "ggcg_summary.csv")


if __name__ == "__main__":
    main(*sys.argv[1:4])
