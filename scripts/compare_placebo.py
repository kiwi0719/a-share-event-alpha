"""事件 vs 安慰剂：按公告日聚合后做 Welch t 检验，并做 BH 校正。
用法: python scripts/compare_placebo.py OUT_DIR"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

out = Path(sys.argv[1])
pl = pd.read_parquet(out / "placebo_results.parquet")
yj = pd.read_parquet(out / "event_results.parquet")
yj = pd.concat([yj, yj[(yj.category == "预增") & (yj.magnitude >= 100)].assign(category="预增≥100%")])
gg = pd.read_parquet(out / "ggcg_results.parquet")
ev = pd.concat([yj.assign(src="业绩预告"), gg.assign(src="增减持")])
ev["period"] = np.where(ev.t0 >= "2024-01-01", "2024+", "2015-2023")
rows = []
for per in ("全样本", "2015-2023", "2024+"):
    plp = pl if per == "全样本" else pl[pl.period == per]
    base = plp.dropna(subset=["tr_skip2"]).groupby("notice_date")["tr_skip2"].mean()
    for (src, cat), d in ev.groupby(["src", "category"]):
        if per != "全样本":
            d = d[d.period == per]
        s = d.dropna(subset=["tr_skip2"]).groupby("notice_date")["tr_skip2"].mean()
        if len(s) < 10:
            continue
        t, pv = stats.ttest_ind(s, base, equal_var=False)
        rows.append(dict(period=per, src=src, category=cat, n=len(d), tr_skip2=s.mean(), placebo=base.mean(),
                         diff=s.mean() - base.mean(), t=t, p=pv))
r = pd.DataFrame(rows)


def bh(p, q=0.10):
    o = np.argsort(p)
    m = len(p)
    ok = p[o] <= q * np.arange(1, m + 1) / m
    k = ok.nonzero()[0].max() + 1 if ok.any() else 0
    res = np.zeros(m, bool)
    res[o[:k]] = True
    return res


r["bh10"] = False
full = r.period == "全样本"
r.loc[full, "bh10"] = bh(r.loc[full, "p"].to_numpy())
r.to_csv(out / "vs_placebo.csv", index=False)
pd.set_option("display.width", 200)
for per, d in r.groupby("period", sort=False):
    print("==", per)
    print(d.drop(columns="period").sort_values("diff", ascending=False).round(4).to_string(index=False))
