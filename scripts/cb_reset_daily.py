"""转债下修事件每日扫描（H11 模拟盘 + 提醒清单）。

每个交易日收盘后运行：
1. 抓当日全部公告（东方财富，公开接口），按标题分类：提议下修 / 不下修 / 预计触发 / 股东会下修结果。
2. 提议下修 → 生成信号：下一交易日开盘买入对应转债，持有 20 个交易日后收盘卖出（与回测一致）。
3. 生成今日清单（Markdown）：明天要买、到期要卖、观察名单。原始公告列表存档并记录抓取时间。

用法: python scripts/cb_reset_daily.py YYYYMMDD CB_LIST.parquet TRADE_CAL.parquet OUT_DIR
"""

import datetime as dt
import sys
from pathlib import Path

import akshare as ak
import pandas as pd

HOLD = 20


def classify(t: str) -> str:
    if "向下修正" not in t and "下修" not in t:
        return ""
    if "不向下修正" in t or "不提议" in t or "暂不" in t:
        return "不下修"
    if "提议向下修正" in t and "更新后" not in t:
        return "提议下修"
    if "预计" in t and "触发" in t:
        return "预计触发"
    if "向下修正" in t and ("结果" in t or "暨转股停复牌" in t or "完成" in t):
        return "下修结果"
    return "其他下修相关"


def active_bonds(cb_dir):
    """仍在交易的转债：日线最近 10 天内有数据。"""
    if _ACTIVE:
        return _ACTIVE
    for f in Path(cb_dir, "daily").glob("*.parquet"):
        d = pd.read_parquet(f)
        if "date" in d and len(d) and pd.to_datetime(d["date"]).max() >= pd.Timestamp.now() - pd.Timedelta(days=14):
            _ACTIVE.add(f.stem)
    return _ACTIVE


_ACTIVE: set = set()


def main(day, cb_list, cal_path, out_dir):
    """day 为本次扫描的截止日；从上次扫描日次日起逐个自然日抓公告（周末公告也要抓）。"""
    out = Path(out_dir)
    state = out / "last_scanned.txt"
    # 上次扫描日也重抓一遍：傍晚运行时当天还有晚间公告没发布
    start = pd.Timestamp(state.read_text().strip()) if state.exists() else pd.Timestamp(day)
    for d in pd.date_range(start, pd.Timestamp(day)):
        scan(d.strftime("%Y%m%d"), cb_list, cal_path, out, refresh=True)
        state.write_text(d.strftime("%Y%m%d"))
    report(day, cal_path, out)


def scan(day, cb_list, cal_path, out, refresh=False):
    (out / "raw").mkdir(parents=True, exist_ok=True)
    cal = pd.read_parquet(cal_path)
    days = sorted(pd.to_datetime(cal[cal.is_open == 1].cal_date))
    d = pd.Timestamp(day)
    nxt = [x for x in days if x > d]
    raw = out / "raw" / f"notices_{day}.parquet"
    if raw.exists() and not refresh:
        ns = pd.read_parquet(raw)
    else:
        try:
            ns = ak.stock_notice_report(symbol="全部", date=day)
        except KeyError:  # 当日无公告
            ns = pd.DataFrame(columns=["代码", "名称", "公告标题", "公告类型", "公告日期", "网址"])
        ns["fetched_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        ns.to_parquet(raw)
    ns["kind"] = ns["公告标题"].astype(str).map(classify)
    hit = ns[ns.kind != ""].copy()
    lst = pd.read_parquet(cb_list)
    lst["正股代码"] = lst["正股代码"].astype(str)
    act = active_bonds(Path(cb_list).parent)
    lst = lst[lst["债券代码"].astype(str).isin(act)]
    hit = hit.merge(lst[["正股代码", "债券代码", "债券简称"]], left_on="代码", right_on="正股代码", how="left")
    # 同一正股有多只存续转债时，优先取标题里点名的那只
    named = hit.apply(lambda r: isinstance(r["债券简称"], str) and r["债券简称"] in r["公告标题"], axis=1)
    multi = hit.groupby(["代码", "公告标题"])["债券代码"].transform("count") > 1
    any_named = named.groupby([hit["代码"], hit["公告标题"]]).transform("any")
    hit = hit[~multi | ~any_named | named]

    sig_path = out / "signals.csv"
    sig = pd.read_csv(sig_path, dtype=str) if sig_path.exists() else pd.DataFrame(
        columns=["notice_date", "stock", "stock_name", "bond", "bond_name", "title", "buy_date", "sell_date"])
    new = hit[(hit.kind == "提议下修") & hit["债券代码"].notna()]
    rows = []
    for r in new.itertuples():
        if len(nxt) <= HOLD:
            break
        rows.append(dict(notice_date=day, stock=r.代码, stock_name=r.名称, bond=r.债券代码, bond_name=r.债券简称,
                         title=r.公告标题, buy_date=nxt[0].strftime("%Y%m%d"), sell_date=nxt[HOLD - 1].strftime("%Y%m%d")))
    sig = pd.concat([sig, pd.DataFrame(rows)]).drop_duplicates(["notice_date", "bond"])
    sig.to_csv(sig_path, index=False)
    hit.to_csv(out / "raw" / f"hits_{day}.csv", index=False)


def report(day, cal_path, out):
    cal = pd.read_parquet(cal_path)
    days = sorted(pd.to_datetime(cal[cal.is_open == 1].cal_date))
    nxt = [x for x in days if x > pd.Timestamp(day)]
    sig = pd.read_csv(out / "signals.csv", dtype=str)
    hit = pd.concat([pd.read_csv(f, dtype=str) for f in sorted((out / "raw").glob("hits_*.csv"))[-3:]], ignore_index=True)
    tomorrow = nxt[0].strftime("%Y%m%d") if nxt else ""
    buy = sig[sig.buy_date == tomorrow]
    sell = sig[sig.sell_date == tomorrow]
    hold = sig[(sig.buy_date <= day) & (sig.sell_date >= tomorrow)]
    lines = [f"# 转债下修清单 {day}（下一交易日 {tomorrow}）", ""]
    lines += ["## 下一交易日开盘买入（今日公告董事会提议下修）"] + (
        [f"- {r.bond_name}（{r.bond}）← {r.stock_name}：{r.title}" for r in buy.itertuples()] or ["- 无"])
    lines += ["", "## 下一交易日收盘卖出（持有满 20 个交易日）"] + (
        [f"- {r.bond_name}（{r.bond}），{r.buy_date} 买入" for r in sell.itertuples()] or ["- 无"])
    lines += ["", f"## 持有中（{len(hold)} 只）"] + [f"- {r.bond_name}（{r.bond}），{r.buy_date} 买入，{r.sell_date} 卖出" for r in hold.itertuples()]
    for k in ("预计触发", "不下修", "下修结果"):
        h = hit[hit.kind == k]
        lines += ["", f"## 观察：{k}（{len(h)}）"] + [f"- {r.名称}：{r.公告标题}" for r in h.itertuples()]
    lines += ["", "规则：董事会提议下修公告后次日开盘买入，持有 20 个交易日。回测（2018–2026，504 次）相对等权转债池平均超额 +2.1%，t=3.84。仅供研究，不构成投资建议。"]
    md = "\n".join(lines)
    (out / f"todo_{day}.md").write_text(md)
    (out / "todo_latest.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main(*sys.argv[1:5])
