"""可转债检验（预注册 addendum_v5）：H10 双低轮动，H11 董事会提议下修事件。

用法: python scripts/run_cb_study.py CB_DIR NOTICES_DIR INDEX.parquet OUT_DIR
CB_DIR 下：cb_list_em.parquet、daily/{code}.parquet（新浪 OHLC）、value/{code}.parquet（东财估值序列）
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

STEP = 20
COST = 2 * 0.0001 + 2 * 0.001  # 佣金 1bp×2 + 滑点 10bp×2，无印花税
TOP = 10


def load_panel(cb_dir):
    lst = pd.read_parquet(f"{cb_dir}/cb_list_em.parquet")
    lst["code"] = lst["债券代码"].astype(str)
    frames = []
    for f in glob.glob(f"{cb_dir}/daily/*.parquet"):
        d = pd.read_parquet(f)
        if d.empty:
            continue
        code = Path(f).stem
        d["code"] = code
        v = pd.read_parquet(f"{cb_dir}/value/{code}.parquet")
        if not v.empty:
            v = v.rename(columns={"日期": "date", "转股溢价率": "premium", "转股价值": "conv_value", "纯债价值": "bond_value"})
            v["date"] = pd.to_datetime(v["date"])
            d["date"] = pd.to_datetime(d["date"])
            d = d.merge(v[["date", "premium", "conv_value", "bond_value"]], on="date", how="left")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["code", "date"]).sort_values(["code", "date"])
    df["amount"] = df.volume * df.close  # 新浪成交量单位为张，面值 100 元/张，价格按每百元面值报价
    term = lst.set_index("code")["申购日期"]
    df["maturity"] = pd.to_datetime(df.code.map(term), errors="coerce") + pd.DateOffset(years=6)
    dates = pd.DatetimeIndex(sorted(df.date.unique()))
    codes = pd.Index(sorted(df.code.unique()))

    def wide(c):
        return df.pivot(index="date", columns="code", values=c).reindex(index=dates, columns=codes).to_numpy(float)

    P = dict(open=wide("open"), close=wide("close"), premium=wide("premium"), amount=wide("amount"))
    P["maturity"] = df.groupby("code")["maturity"].first().reindex(codes)
    P["first"] = dates.searchsorted(df.groupby("code")["date"].min().reindex(codes))
    P["stock"] = lst.set_index("code")["正股代码"].astype(str).reindex(codes)
    return dates, codes, P


def hold_return(P, j, e, x):
    """e 开盘买入、x 收盘卖出；中途停止交易（强赎/到期）则按最后收盘价计。"""
    o = P["open"][e, j]
    c = P["close"][e : x + 1, j]
    c = c[~np.isnan(c)]
    if np.isnan(o) or len(c) == 0:
        return np.nan
    return c[-1] / o - 1


def eligible(dates, P, d):
    amt20 = np.nanmean(P["amount"][max(0, d - 19) : d + 1], axis=0)
    days_to_mat = np.asarray((pd.DatetimeIndex(P["maturity"]) - dates[d]).days, dtype=float)
    return ((d - P["first"]) >= 5) & ~(days_to_mat <= 90) & ~np.isnan(P["close"][d]) & (amt20 >= 1e6)


def h10(dates, codes, P, start="2018-01-01", top=TOP, max_price=None):
    rows = []
    s = int(dates.searchsorted(pd.Timestamp(start)))
    for d in range(s, len(dates) - 2, STEP):
        e, x = d + 1, min(d + STEP, len(dates) - 1)
        m = eligible(dates, P, d) & ~np.isnan(P["premium"][d]) & ~np.isnan(P["open"][e])
        if max_price:
            m &= P["close"][d] <= max_price
        if m.sum() < 30:
            continue
        idx = np.where(m)[0]
        score = P["close"][d, idx] + P["premium"][d, idx]
        pick = idx[np.argsort(score)[:top]]
        r_pick = np.nanmean([hold_return(P, j, e, x) for j in pick]) - COST
        r_all = np.nanmean([hold_return(P, j, e, x) for j in idx]) - COST
        rows.append(dict(d=dates[d], exit=dates[x], n_pool=len(idx), pick=r_pick, pool=r_all,
                         avg_price=P["close"][d, pick].mean(), avg_prem=P["premium"][d, pick].mean()))
    return pd.DataFrame(rows).set_index("d")


def stats(r, col, base):
    ex = r[col] - r[base]
    years = (r.exit.iloc[-1] - r.index[0]).days / 365.25
    ann = (1 + r[col]).prod() ** (1 / years) - 1
    ann_b = (1 + r[base]).prod() ** (1 / years) - 1
    nav = (1 + r[col]).cumprod()
    return dict(n=len(r), ann=ann, ann_base=ann_b, ann_excess=ann - ann_b,
                t=ex.mean() / (ex.std(ddof=1) / np.sqrt(len(ex))), win=(ex > 0).mean(),
                maxdd=(nav / nav.cummax() - 1).min())


def h11(dates, codes, P, notices_dir):
    fs = sorted(glob.glob(f"{notices_dir}/20*.parquet"))
    ns = pd.concat([pd.read_parquet(f) for f in fs if pd.read_parquet(f, columns=None).shape[0] > 0], ignore_index=True)
    t = ns["公告标题"].astype(str)
    sel = t.str.contains("向下修正") & t.str.contains("提议|董事会") & ~t.str.contains("不向下修正|不提议|暂不|结果|完成|实施|进展|股东大会")
    ev = ns[sel].copy()
    ev["notice_date"] = pd.to_datetime(ev["公告日期"])
    ev["stock"] = ev["代码"].astype(str).str.zfill(6)
    by_stock = P["stock"].rename_axis("code").reset_index().groupby("正股代码")["code"].apply(list)
    ci = {c: i for i, c in enumerate(codes)}
    rows = []
    for r in ev.drop_duplicates(["stock", "notice_date"]).itertuples():
        for code in by_stock.get(r.stock, []):
            j = ci[code]
            t0 = int(dates.searchsorted(r.notice_date, side="right"))
            d = t0 - 1
            if t0 + 19 >= len(dates) or d < 0 or not eligible(dates, P, d)[j]:
                continue
            x = t0 + 19
            m = eligible(dates, P, d) & ~np.isnan(P["open"][t0])
            pool = np.nanmean([hold_return(P, k, t0, x) for k in np.where(m)[0]])
            rr = hold_return(P, j, t0, x)
            if np.isnan(rr):
                continue
            pre = P["close"][d, j] / P["close"][max(d - 1, 0), j] - 1
            rows.append(dict(code=code, notice_date=r.notice_date, title=r.公告标题, ret=rr - COST, pool=pool - COST,
                             ex=rr - pool, day0=pre, price=P["close"][d, j], premium=P["premium"][d, j]))
    return pd.DataFrame(rows)


def main(cb_dir, notices_dir, index_path, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dates, codes, P = load_panel(cb_dir)
    print("转债", len(codes), "交易日", len(dates), dates[0].date(), dates[-1].date())
    idx = pd.read_parquet(index_path)
    csi = idx[idx.code == "sh000852"].set_index("date")["close"]
    csi.index = pd.to_datetime(csi.index)

    res = {}
    for name, kw in (("双低10", {}), ("双低20", dict(top=20)), ("双低10_价格≤130", dict(max_price=130))):
        r = h10(dates, codes, P, **kw)
        r["csi1000"] = [csi.asof(x) / csi.asof(d) - 1 for d, x in zip(r.index, r.exit)]
        r.to_csv(out / f"h10_{name}.csv")
        res[name] = r
    rows = []
    for name, r in res.items():
        for per, rr in (("全样本", r), ("2018-2023", r[r.index < "2024-01-01"]), ("2024+", r[r.index >= "2024-01-01"])):
            s = stats(rr, "pick", "pool")
            s2 = stats(rr, "pick", "csi1000")
            rows.append(dict(strategy=name, period=per, **s, excess_vs_csi1000=s2["ann_excess"], t_vs_csi1000=s2["t"]))
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "h10_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summ.round(4).to_string(index=False))
    r = res["双低10"]
    rel = (1 + r.pick).cumprod() / (1 + r.csi1000).cumprod()
    print("双低10 相对中证1000最大回撤 %.4f；平均价格 %.1f 平均溢价率 %.1f%%" % ((rel / rel.cummax() - 1).min(), r.avg_price.mean(), r.avg_prem.mean()))
    yearly = r.groupby(r.index.year)[["pick", "pool", "csi1000"]].apply(lambda g: (1 + g).prod() - 1)
    print(yearly.round(4).to_string())

    ev = h11(dates, codes, P, notices_dir)
    ev.to_csv(out / "h11_events.csv", index=False)
    if len(ev):
        g = ev.groupby("notice_date")["ex"].mean()
        print("H11 提议下修事件 n=%d 日期=%d  20日超额均值 %.4f  t %.2f  公告日当天涨跌 %.4f" % (
            len(ev), len(g), g.mean(), g.mean() / (g.std(ddof=1) / np.sqrt(len(g))), ev.day0.mean()))
        ev["period"] = np.where(ev.notice_date >= "2024-01-01", "2024+", "2018-2023")
        print(ev.groupby("period")["ex"].agg(["count", "mean"]).round(4).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:5])
