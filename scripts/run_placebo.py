"""安慰剂对照：在同一股票池过滤下随机抽取 (股票, 日期) 作为伪事件，估计基准偏差。
用法: python scripts/run_placebo.py PANEL.pkl OUT_DIR N"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from event_alpha.research.event_study import run, summarize

panel_path, out_dir, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
with open(panel_path, "rb") as f:
    p = pickle.load(f)
rng = np.random.default_rng(20261101)
lo = int(p.dates.searchsorted(pd.Timestamp("2015-01-01")))
hi = len(p.dates) - 25
rows = []
while len(rows) < n:
    i = rng.integers(lo, hi)
    j = rng.integers(0, len(p.codes))
    if np.isnan(p.close[i, j]):
        continue
    rows.append((p.codes[j], p.dates[i], "placebo", np.nan))
ev = pd.DataFrame(rows, columns=["code", "notice_date", "category", "magnitude"])
res = run(p, ev)
res["period"] = np.where(res.t0 >= "2024-01-01", "2024+", "2015-2023")
res.to_parquet(Path(out_dir) / "placebo_results.parquet")
print(summarize(res).round(4).T.to_string())
for per, sub in res.groupby("period"):
    print(per, len(sub), round(sub.tr_skip2.mean(), 4), round(sub.tr_now.mean(), 4))
