"""每日规则初筛：哪些转债接近或已经触发“向下修正转股价”条款。

条款因券而异，这里用最常见的默认值做初筛（正股收盘价低于转股价 85%，30 个交易日中至少 15 日），
并标出“接近”（已满 10 日）。真正的条款以募集说明书为准，由后续抽取步骤核对。

转股价由东方财富估值序列反推：转股价 = 100 × 正股收盘价 / 转股价值。

用法: python scripts/cb_reset_screen.py CB_DIR PANEL.pkl OUT.csv
"""

import glob
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RATIO, NEED, WINDOW, NEAR = 0.85, 15, 30, 10


def main(cb_dir, panel_path, out_csv):
    with open(panel_path, "rb") as f:
        p = pickle.load(f)
    lst = pd.read_parquet(f"{cb_dir}/cb_list_em.parquet")
    lst["code"] = lst["债券代码"].astype(str)
    suffix = {c.split(".")[0]: c for c in p.codes}
    ci = {c: i for i, c in enumerate(p.codes)}
    last_day = p.dates[-1]
    rows = []
    for r in lst.itertuples():
        f = Path(cb_dir) / "value" / f"{r.code}.parquet"
        dfile = Path(cb_dir) / "daily" / f"{r.code}.parquet"
        if not f.exists() or not dfile.exists():
            continue
        v = pd.read_parquet(f)
        if v.empty:
            continue
        v["date"] = pd.to_datetime(v["日期"])
        if v.date.max() < last_day - pd.Timedelta(days=5):
            continue  # 已退市或停止交易
        sc = suffix.get(str(r.正股代码))
        if sc is None:
            continue
        j = ci[sc]
        s = pd.Series(p.close[:, j], index=p.dates).dropna()
        v = v.set_index("date").join(s.rename("stock"), how="inner").dropna(subset=["转股价值", "stock"])
        v = v[v["转股价值"] > 0].tail(WINDOW)
        if len(v) < 20:
            continue
        conv = 100 * v.stock / v["转股价值"]
        below = int((v.stock < RATIO * conv).sum())
        d = pd.read_parquet(dfile)
        price = float(d["close"].iloc[-1]) if len(d) else np.nan
        rows.append(dict(bond=r.code, name=r.债券简称, stock=sc, stock_name=r.正股简称, conv_price=round(conv.iloc[-1], 3),
                         stock_close=float(v.stock.iloc[-1]), stock_to_conv=round(v.stock.iloc[-1] / conv.iloc[-1], 3),
                         days_below_85=below, bond_price=price, premium_pct=round(float(v["转股溢价率"].iloc[-1]), 1),
                         rating=r.信用评级, size_yi=r.发行规模))
    df = pd.DataFrame(rows)
    df["status"] = np.where(df.days_below_85 >= NEED, "已满足默认触发条件", np.where(df.days_below_85 >= NEAR, "接近触发", ""))
    out = df[df.status != ""].sort_values("days_below_85", ascending=False)
    out.to_csv(out_csv, index=False)
    print(f"活跃转债 {len(df)}，已满足默认触发 {int((df.status == '已满足默认触发条件').sum())}，接近 {int((df.status == '接近触发').sum())}")
    print(out.head(30).to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:4])
