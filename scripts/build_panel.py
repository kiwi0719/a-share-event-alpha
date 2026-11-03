"""从 duckdb 的 px 表构建宽矩阵并缓存。用法: python scripts/build_panel.py DB OUT.pkl"""
import pickle
import sys
import time

from event_alpha.data.panel import load_panel

t = time.time()
p = load_panel(sys.argv[1], start="2014-01-01", table="px")
with open(sys.argv[2], "wb") as f:
    pickle.dump(p, f, protocol=5)
print(len(p.dates), len(p.codes), f"{time.time() - t:.0f}s")
