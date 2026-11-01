"""把长表日线转成 日期 × 股票 的宽矩阵，供向量化回测使用。

数据约定（ellendan/a-share-prices，源自 Tushare）：
- 价格为不复权价；prev_close 是交易所公布的前收盘（除权除息日为除权参考价），
  因此 close / prev_close - 1 即为含分红送转调整的真实日收益，不需要复权因子。
- volume 单位为手，turnover 单位为千元（见开源踩坑复查 #1，统一转成元）。
- 停牌日没有记录，宽矩阵中为 NaN。
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    codes: pd.Index
    open: np.ndarray
    close: np.ndarray
    prev_close: np.ndarray
    high_limit: np.ndarray
    low_limit: np.ndarray
    amount_cny: np.ndarray
    circ_mv_cny: np.ndarray
    is_st: np.ndarray  # bool，按当日简称判断
    list_date: pd.Series  # code -> Timestamp

    @property
    def ret(self) -> np.ndarray:
        """收盘到收盘日收益（停牌日 NaN）。"""
        return self.close / self.prev_close - 1.0

    def date_index(self, d) -> int:
        return int(self.dates.searchsorted(pd.Timestamp(d)))


def _st_from_limits(prev_close: np.ndarray, high_limit: np.ndarray) -> np.ndarray:
    """由当日涨停幅度反推 ST（主板 ST 为 5%）。

    数据集的 name 字段在 2023 年前多为回填的当前简称，按简称判断 ST 会错：
    2016、2020 年“名称含 ST”的样本涨停幅度中位数是 10%。涨跌停价是交易所当日公布值，天然 PIT。
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = high_limit / prev_close - 1.0
    return np.abs(pct - 0.05) < 0.006


def load_panel(csv_or_db: str, start: str = "2014-01-01", table: str | None = None) -> Panel:
    con = duckdb.connect()
    src = f"read_csv_auto('{csv_or_db}', sample_size=200000)" if table is None else table
    if table is not None:
        con = duckdb.connect(csv_or_db, read_only=True)
    df = con.execute(
        f"""
        select code, date, open, close, prev_close, high_limit, low_limit,
               turnover * 1000.0 as amount_cny, circ_mv * 10000.0 as circ_mv_cny,
               coalesce(name like '%ST%', false) as is_st, list_date
        from {src}
        where date >= '{start}'
        """
    ).fetchdf()
    df["date"] = pd.to_datetime(df["date"])
    # 源数据增量更新时出现过整行重复（如 2025-12-18 全市场重复一次），去重保留一条
    n0 = len(df)
    df = df.drop_duplicates(["code", "date"], keep="last")
    if len(df) < n0:
        print(f"[panel] 去除重复行 {n0 - len(df)} 条")
    dates = pd.DatetimeIndex(sorted(df["date"].unique()))
    codes = pd.Index(sorted(df["code"].unique()))

    def wide(col, dtype=float):
        w = df.pivot(index="date", columns="code", values=col).reindex(index=dates, columns=codes)
        if dtype is bool:
            return w.fillna(False).to_numpy(dtype=bool)
        return w.to_numpy(dtype=float)

    list_date = pd.to_datetime(df.groupby("code")["list_date"].first()).reindex(codes)
    return Panel(
        dates=dates,
        codes=codes,
        open=wide("open"),
        close=wide("close"),
        prev_close=wide("prev_close"),
        high_limit=wide("high_limit"),
        low_limit=wide("low_limit"),
        amount_cny=wide("amount_cny"),
        circ_mv_cny=wide("circ_mv_cny"),
        is_st=_st_from_limits(wide("prev_close"), wide("high_limit")),
        list_date=list_date,
    )
