"""可成交性约束下的单笔交易模拟。

买入：从目标交易日起，以开盘价买入；当日停牌或开盘价已在涨停价（一字板/开盘封板）
视为买不到，顺延至下一交易日，最多顺延 max_defer 天，仍买不到则放弃。
卖出：目标卖出日收盘价卖出；停牌或收盘在跌停价视为卖不出，顺延。
收益用 close/prev_close 链式计算，自动包含分红送转（prev_close 为除权参考价）。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data.panel import Panel

EPS = 1e-6


@dataclass
class Fill:
    entry_idx: int
    exit_idx: int
    gross_ret: float  # 买入开盘到卖出收盘的总收益


def find_entry(p: Panel, j: int, t: int, max_defer: int = 3) -> int | None:
    T = len(p.dates)
    for k in range(max_defer + 1):
        i = t + k
        if i >= T:
            return None
        o, hl = p.open[i, j], p.high_limit[i, j]
        if np.isnan(o):
            continue
        if not np.isnan(hl) and o >= hl - EPS:
            continue
        return i
    return None


def find_exit(p: Panel, j: int, target: int, max_defer: int = 60) -> int | None:
    T = len(p.dates)
    for k in range(max_defer + 1):
        i = target + k
        if i >= T:
            return None
        c, ll = p.close[i, j], p.low_limit[i, j]
        if np.isnan(c):
            continue
        if not np.isnan(ll) and c <= ll + EPS:
            continue
        return i
    return None


def holding_return(p: Panel, j: int, entry: int, exit_: int) -> float:
    r = p.close[entry, j] / p.open[entry, j]
    if exit_ > entry:
        seg = p.close[entry + 1 : exit_ + 1, j] / p.prev_close[entry + 1 : exit_ + 1, j]
        r *= np.nanprod(seg)  # 停牌日 NaN 视为收益 0
    return float(r - 1.0)


def simulate(p: Panel, j: int, t: int, hold: int, max_defer: int = 3) -> Fill | None:
    """在第 t 个交易日（或顺延后）开盘买入，持有 hold 个交易日后收盘卖出。"""
    e = find_entry(p, j, t, max_defer)
    if e is None:
        return None
    x = find_exit(p, j, e + hold - 1)
    if x is None:
        return None
    return Fill(e, x, holding_return(p, j, e, x))
