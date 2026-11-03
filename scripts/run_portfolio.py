"""卫星组合回测（walk-forward 选事件类别）vs 中证 1000。

规则（与蓝图 v0.3 决策记录一致）：
- 每年初，只用此前 3 年、且持有期已结束的事件结果，挑选 tr_skip2 均值 > 0 且 t > 2 的类别；
- 事件 t0 后第 2 个交易日开盘买入（开盘涨停/停牌顺延，最多 3 日），持有 19 个交易日后收盘卖出；
- 最多 5 个仓位，等权（每仓位 = 当时卫星净值的 1/5），候选超额时按业绩变动幅度从高到低；
- 成本：佣金单边 5bp（小资金下 5 元最低收费的近似）、滑点单边 10bp、卖出印花税按日期。

用法: python scripts/run_portfolio.py PANEL.pkl EVENT_RESULTS.parquet INDEX.parquet OUT_DIR
"""

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from event_alpha.research.event_study import clustered_t
from event_alpha.sim.fill import find_entry, find_exit
from event_alpha.timeutil.rules import stamp_duty_rate

SLOTS = 5
HOLD = 19
COMM = 0.0005
SLIP = 0.001


def select_categories(res: pd.DataFrame, year: int) -> list[str]:
    lo, hi = pd.Timestamp(year - 3, 1, 1), pd.Timestamp(year - 1, 11, 30)
    past = res[(res.t0 >= lo) & (res.t0 <= hi)]
    picked = []
    for cat, df in past.groupby("category"):
        m, t, n = clustered_t(df, "tr_skip2")
        if n >= 20 and m > 0 and t > 2:
            picked.append(cat)
    return picked


def main(panel_path, res_path, index_path, out_dir):
    out = Path(out_dir)
    with open(panel_path, "rb") as f:
        p = pickle.load(f)
    res = pd.read_parquet(res_path)
    h3 = res[(res.category == "预增") & (res.magnitude >= 100)].assign(category="预增≥100%")
    res_all = pd.concat([res, h3])
    code_idx = {c: i for i, c in enumerate(p.codes)}
    T = len(p.dates)

    start_year, end_year = 2018, 2026
    fixed = os.environ.get("FIXED_CATEGORIES")  # 探索性：固定类别，不做 walk-forward 选择
    if fixed:
        cats_by_year = {y: fixed.split(",") for y in range(start_year, end_year + 1)}
    else:
        cats_by_year = {y: select_categories(res_all, y) for y in range(start_year, end_year + 1)}
    for y, c in cats_by_year.items():
        print(y, c)

    # 候选：按计划买入日（t0 后第 2 个交易日）分组
    cand = {}
    for y, cats in cats_by_year.items():
        sub = res_all[(res_all.t0.dt.year == y) & res_all.category.isin(cats)]
        for r in sub.itertuples():
            t0 = int(p.dates.searchsorted(r.t0))
            cand.setdefault(t0 + 2, []).append((r.magnitude if not np.isnan(r.magnitude) else 0.0, r.code))

    first = int(p.dates.searchsorted(pd.Timestamp(start_year, 1, 1)))
    nav = 1.0
    cash = 1.0
    positions = []  # dict(j, value, exit_target, pending_entry)
    navs, trades = [], []
    pending = []  # (deadline_idx, j, planned_start)
    for i in range(first, T):
        # 1) 更新持仓市值（收盘到收盘）
        for pos in positions:
            if pos["entry"] == i:
                continue
            r = p.close[i, pos["j"]] / p.prev_close[i, pos["j"]]
            if not np.isnan(r):
                pos["value"] *= r
        # 2) 新候选加入待买队列
        for mag, code in sorted(cand.get(i, []), reverse=True):
            j = code_idx[code]
            if any(q["j"] == j for q in positions) or any(q[1] == j for q in pending):
                continue
            pending.append((i + 3, j, i, mag))
        # 3) 开盘买入
        free = SLOTS - len(positions)
        still = []
        for deadline, j, st, mag in sorted(pending, key=lambda x: -x[3]):
            if i > deadline:
                continue
            if free <= 0:
                still.append((deadline, j, st, mag))
                continue
            e = find_entry(p, j, i, max_defer=0)
            if e is None:
                still.append((deadline, j, st, mag))
                continue
            alloc = min(nav / SLOTS, cash)
            if alloc <= 0:
                break
            cost = alloc * (COMM + SLIP)
            cash -= alloc
            v = (alloc - cost) * p.close[i, j] / p.open[i, j]
            positions.append(dict(j=j, value=v, entry=i, exit_target=i + HOLD - 1, cost_basis=alloc))
            free -= 1
        pending = still
        # 4) 收盘卖出
        keep = []
        for pos in positions:
            if i >= pos["exit_target"] and find_exit(p, pos["j"], i, max_defer=0) == i:
                fee = pos["value"] * (COMM + SLIP + float(stamp_duty_rate(p.dates[i].date(), "sell")))
                proceeds = pos["value"] - fee
                cash += proceeds
                trades.append(dict(code=p.codes[pos["j"]], entry=p.dates[pos["entry"]], exit=p.dates[i],
                                   ret=proceeds / pos["cost_basis"] - 1))
            else:
                keep.append(pos)
        positions = keep
        nav = cash + sum(q["value"] for q in positions)
        navs.append((p.dates[i], nav, len(positions)))

    nv = pd.DataFrame(navs, columns=["date", "nav", "n_pos"]).set_index("date")
    idx = pd.read_parquet(index_path)
    for code, name in (("sh000852", "csi1000"), ("sh000300", "csi300")):
        s = idx[idx.code == code].set_index("date")["close"].reindex(nv.index).ffill()
        nv[name] = s / s.iloc[0]
    nv["combo_30_70"] = 1 + 0.3 * (nv.nav - 1) + 0.7 * (nv.csi1000 - 1)  # 期初配置、不再平衡的近似
    rel = nv.nav / nv.csi1000
    nv["rel"] = rel
    nv.to_csv(out / "portfolio_nav.csv")
    td = pd.DataFrame(trades)
    td.to_csv(out / "trades.csv", index=False)

    years = (nv.index[-1] - nv.index[0]).days / 365.25
    def ann(x):
        return x.iloc[-1] ** (1 / years) - 1
    rel_dd = (rel / rel.cummax() - 1).min()
    abs_dd = (nv.nav / nv.nav.cummax() - 1).min()
    yearly = nv[["nav", "csi1000", "csi300"]].resample("YE").last()
    yearly = yearly / yearly.shift(1).fillna(1.0) - 1
    yearly["excess_vs_csi1000"] = yearly.nav - yearly.csi1000
    print(f"区间 {nv.index[0].date()} ~ {nv.index[-1].date()}，{years:.1f} 年，交易 {len(td)} 笔，胜率 {(td.ret > 0).mean():.1%}")
    print(f"卫星年化 {ann(nv.nav):.2%}；中证1000(价格指数) 年化 {ann(nv.csi1000):.2%}；年化超额 {ann(rel):.2%}")
    print(f"相对中证1000最大回撤 {rel_dd:.2%}；绝对最大回撤 {abs_dd:.2%}；平均持仓数 {nv.n_pos.mean():.2f}")
    print(yearly.round(4).to_string())
    (out / "portfolio_summary.txt").write_text(
        f"years={years:.2f} trades={len(td)} win={(td.ret > 0).mean():.4f}\n"
        f"sat_ann={ann(nv.nav):.4f} csi1000_ann={ann(nv.csi1000):.4f} excess_ann={ann(rel):.4f}\n"
        f"rel_maxdd={rel_dd:.4f} abs_maxdd={abs_dd:.4f} avg_pos={nv.n_pos.mean():.2f}\n"
        + yearly.round(4).to_string()
    )


if __name__ == "__main__":
    main(*sys.argv[1:5])
