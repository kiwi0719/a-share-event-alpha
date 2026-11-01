"""重大合同 / 中标事件研究（预注册 addendum_v3）。

用法: python scripts/run_contract_study.py PANEL.pkl CONTENT.jsonl PLACEBO_DIR OUT_DIR
CONTENT.jsonl 每行: art_code, code(6位), date, title, type, text
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from event_alpha.events.contracts import classify_title, extract
from event_alpha.research.event_study import clustered_t, run

BUCKETS = [(0, 0.10, "占营收<10%"), (0.10, 0.30, "占营收10%-30%"), (0.30, np.inf, "占营收≥30%")]


def load_events(path, codes):
    rows = []
    for line in open(path):
        r = json.loads(line)
        if not r.get("text"):
            continue
        if r["type"] != "重大合同" and not classify_title(r["title"]):
            continue
        f = extract(r["title"], r["text"])
        rows.append(dict(code6=str(r["code"]).zfill(6), notice_date=pd.Timestamp(r["date"][:10]), title=r["title"],
                         amount_cny=f.amount_cny, ratio=f.revenue_ratio, cand=f.is_candidate, fx=f.foreign_currency))
    df = pd.DataFrame(rows)
    print("解析公告", len(df), "候选人/预中标", int(df.cand.sum()), "有金额", int(df.amount_cny.notna().sum()),
          "有营收占比", int(df.ratio.notna().sum()))
    df = df[~df.cand & ~(df.fx & df.ratio.isna())]
    suffix = {c.split(".")[0]: c for c in codes}
    df["code"] = df.code6.map(suffix)
    df = df.dropna(subset=["code"])
    df = df.sort_values("ratio", na_position="first").drop_duplicates(["code", "notice_date"], keep="last")
    df["magnitude"] = df.ratio
    df["category"] = "全部合同"
    return df


def main(panel_path, content_path, placebo_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(panel_path, "rb") as f:
        p = pickle.load(f)
    ev = load_events(content_path, p.codes)
    print("事件（去重后）", len(ev), "其中有营收占比", int(ev.ratio.notna().sum()))
    res = run(p, ev[["code", "notice_date", "category", "magnitude"]])
    res["ratio"] = res["magnitude"]
    res["period"] = np.where(res.t0 >= "2024-01-01", "2024+", "2015-2023")
    res.to_parquet(out / "contract_results.parquet")

    groups = {"全部合同": res}
    for lo, hi, name in BUCKETS:
        groups[name] = res[(res.ratio >= lo) & (res.ratio < hi)]
    groups["无营收占比"] = res[res.ratio.isna()]

    pl = pd.read_parquet(Path(placebo_dir) / "placebo_results.parquet")
    rows = []
    for per in ("全样本", "2015-2023", "2024+"):
        plp = pl if per == "全样本" else pl[pl.period == per]
        base = plp.dropna(subset=["tr_skip2"]).groupby("notice_date")["tr_skip2"].mean()
        for name, d in groups.items():
            if per != "全样本":
                d = d[d.period == per]
            s = d.dropna(subset=["tr_skip2"]).groupby("notice_date")["tr_skip2"].mean()
            if len(s) < 10:
                rows.append(dict(period=per, group=name, n=len(d)))
                continue
            t, pv = stats.ttest_ind(s, base, equal_var=False)
            m0, t0_, _ = clustered_t(d, "info_car_0_1")
            rows.append(dict(period=per, group=name, n=len(d), open_limit_up=d.open_limit_up_t0.mean(),
                             info_car_0_1=m0, tr_now=d.tr_now.mean(), tr_skip2=s.mean(), placebo=base.mean(),
                             diff=s.mean() - base.mean(), t=t, p=pv))
    r = pd.DataFrame(rows)
    # BH 覆盖 H7（≥30%）与 H9（全部）
    full = r[(r.period == "全样本") & r.group.isin(["全部合同", "占营收≥30%"])].sort_values("p")
    m = len(full)
    passed = full.p.values <= 0.10 * np.arange(1, m + 1) / m
    k = passed.nonzero()[0].max() + 1 if passed.any() else 0
    r["bh10"] = r.index.isin(full.index[:k])
    r.to_csv(out / "contract_vs_placebo.csv", index=False)
    pd.set_option("display.width", 250)
    for per, d in r.groupby("period", sort=False):
        print("==", per)
        print(d.drop(columns="period").round(4).to_string(index=False))
    # H8 单调性（描述性）
    means = [groups[n].tr_skip2.mean() for _, _, n in BUCKETS]
    print("H8 三档 tr_skip2 均值:", np.round(means, 4), "单调上升" if means[0] < means[1] < means[2] else "非单调")
    by_year = res.assign(year=res.t0.dt.year).groupby("year")["tr_skip2"].agg(["count", "mean"])
    print(by_year.round(4).T.to_string())


if __name__ == "__main__":
    main(*sys.argv[1:5])
