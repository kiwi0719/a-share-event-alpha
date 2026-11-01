"""公告的最早可交易时点。

规则（与蓝图 v0.3 一致）：
1. 只知道日期（time_precision='day'）的公告一律视为盘后发布，下一交易日开盘可交易；
2. 精确到分钟：交易日 09:15 集合竞价开始前发布 -> 当日开盘；其余（盘中、盘后）-> 下一交易日开盘；
3. 非交易日发布 -> 下一交易日开盘；
4. 若目标交易日停牌，顺延到第一个非停牌交易日。
"""

from __future__ import annotations

import bisect
from collections.abc import Callable, Sequence
from datetime import date, datetime, time
from typing import Literal
from zoneinfo import ZoneInfo

CN_TZ = ZoneInfo("Asia/Shanghai")
OPEN_AUCTION = time(9, 15)
MARKET_OPEN = time(9, 30)

TimePrecision = Literal["day", "minute"]


class TradingCalendar:
    def __init__(self, trading_days: Sequence[date]):
        days = sorted(set(trading_days))
        if not days:
            raise ValueError("交易日历为空")
        self._days = days
        self._set = set(days)

    def is_trading_day(self, d: date) -> bool:
        return d in self._set

    def next_trading_day(self, d: date, *, inclusive: bool = False) -> date:
        i = bisect.bisect_left(self._days, d) if inclusive else bisect.bisect_right(self._days, d)
        if i >= len(self._days):
            raise LookupError(f"{d} 之后没有交易日，日历需要延长")
        return self._days[i]


def _to_cn(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        return ts.replace(tzinfo=CN_TZ)
    return ts.astimezone(CN_TZ)


def tradable_at(
    published_at: datetime | date,
    time_precision: TimePrecision,
    calendar: TradingCalendar,
    is_suspended: Callable[[date], bool] | None = None,
) -> datetime:
    """返回最早可交易时点（某交易日 09:30，北京时间）。"""
    if time_precision == "day":
        d = published_at.date() if isinstance(published_at, datetime) else published_at
        target = calendar.next_trading_day(d)
    elif time_precision == "minute":
        if not isinstance(published_at, datetime):
            raise TypeError("minute 精度需要 datetime")
        ts = _to_cn(published_at)
        d = ts.date()
        if calendar.is_trading_day(d) and ts.time() < OPEN_AUCTION:
            target = d
        else:
            target = calendar.next_trading_day(d)
    else:
        raise ValueError(f"未知 time_precision: {time_precision}")

    if is_suspended is not None:
        guard = 0
        while is_suspended(target):
            target = calendar.next_trading_day(target)
            guard += 1
            if guard > 2000:
                raise RuntimeError("停牌顺延超过 2000 个交易日，请检查停牌数据")

    return datetime.combine(target, MARKET_OPEN, tzinfo=CN_TZ)
