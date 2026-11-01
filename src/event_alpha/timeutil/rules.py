"""按日期 × 板块版本化的 A 股交易规则。

回测引擎不得把涨跌幅、税费写成常量，一律通过本模块查询。
标注 VERIFY 的条目来自整理，使用前请对照交易所原文核对。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum


class Board(str, Enum):
    MAIN = "main"  # 沪深主板
    CHINEXT = "chinext"  # 创业板
    STAR = "star"  # 科创板
    BSE = "bse"  # 北交所（含原新三板精选层）


def board_of(symbol: str) -> Board:
    """由 6 位代码推断板块。symbol 形如 '600000' 或 '600000.SH'。"""
    code = symbol.split(".")[0]
    if code.startswith("688") or code.startswith("689"):
        return Board.STAR
    if code.startswith("300") or code.startswith("301") or code.startswith("302"):
        return Board.CHINEXT
    if code.startswith(("8", "4", "92")):
        return Board.BSE
    if code.startswith(("60", "00")):
        return Board.MAIN
    raise ValueError(f"无法识别板块: {symbol}")


# 关键制度日期（VERIFY）
CHINEXT_REFORM = date(2020, 8, 24)  # 创业板注册制，涨跌幅 10% -> 20%
MAIN_REGISTRATION = date(2023, 4, 10)  # 主板全面注册制首批上市，新股前 5 日不设涨跌幅
STAMP_DUTY_CUT = date(2023, 8, 28)  # 印花税卖方 0.1% -> 0.05%


@dataclass(frozen=True)
class PriceLimit:
    """None 表示当日不设涨跌幅限制。"""

    up_pct: Decimal | None
    down_pct: Decimal | None

    @property
    def unlimited(self) -> bool:
        return self.up_pct is None


def _pct(x: str) -> Decimal:
    return Decimal(x)


def price_limit(
    board: Board,
    trade_date: date,
    *,
    is_st: bool = False,
    nth_trading_day_since_listing: int | None = None,
) -> PriceLimit:
    """返回某板块某日的涨跌幅比例。

    nth_trading_day_since_listing: 上市第几个交易日（上市首日为 1）。未知时传 None，
    按普通交易日处理；调用方应尽量提供，以免新股期被误判为可涨停。
    """
    n = nth_trading_day_since_listing

    if board is Board.STAR:
        if n is not None and n <= 5:
            return PriceLimit(None, None)
        return PriceLimit(_pct("0.20"), _pct("0.20"))

    if board is Board.CHINEXT:
        if trade_date >= CHINEXT_REFORM:
            if n is not None and n <= 5:
                return PriceLimit(None, None)
            return PriceLimit(_pct("0.20"), _pct("0.20"))
        return _main_board_legacy(is_st, n)

    if board is Board.BSE:
        # VERIFY: 北交所/精选层上市首日不设涨跌幅，其后 30%
        if n is not None and n <= 1:
            return PriceLimit(None, None)
        return PriceLimit(_pct("0.30"), _pct("0.30"))

    # 主板
    if trade_date >= MAIN_REGISTRATION and n is not None and n <= 5:
        # VERIFY: 注册制前核准制上市的主板股票在 2023-04-10 后不适用此规则，
        # 但它们的 n 早已远大于 5，因此不受影响。
        return PriceLimit(None, None)
    return _main_board_legacy(is_st, n)


def _main_board_legacy(is_st: bool, n: int | None) -> PriceLimit:
    if n == 1:
        # VERIFY: 核准制新股首日涨幅上限 44%、跌幅下限 36%（相对发行价）
        return PriceLimit(_pct("0.44"), _pct("0.36"))
    if is_st:
        # VERIFY: 主板风险警示股涨跌幅曾为 5%，需按交易所最新规则核对是否已调整
        return PriceLimit(_pct("0.05"), _pct("0.05"))
    return PriceLimit(_pct("0.10"), _pct("0.10"))


def _round_price(x: Decimal) -> Decimal:
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def limit_prices(pre_close: float | Decimal, limit: PriceLimit) -> tuple[Decimal, Decimal] | None:
    """由前收盘价（不复权）计算涨停价、跌停价；不设限返回 None。"""
    if limit.unlimited:
        return None
    pc = Decimal(str(pre_close))
    up = _round_price(pc * (1 + limit.up_pct))
    down = _round_price(pc * (1 - limit.down_pct))
    return up, down


def stamp_duty_rate(trade_date: date, side: str) -> Decimal:
    """印花税率，仅卖方征收。side: 'buy' | 'sell'。"""
    if side not in ("buy", "sell"):
        raise ValueError(side)
    if side == "buy":
        return Decimal("0")
    return Decimal("0.0005") if trade_date >= STAMP_DUTY_CUT else Decimal("0.001")


def commission(amount_cny: float | Decimal, rate: Decimal = Decimal("0.00025"), minimum: Decimal = Decimal("5")) -> Decimal:
    """券商佣金，含单笔最低收费。开通“免五”时传 minimum=Decimal('0')。"""
    amt = Decimal(str(amount_cny))
    return max(amt * rate, minimum)
