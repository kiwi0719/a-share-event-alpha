from datetime import date
from decimal import Decimal

import pytest

from event_alpha.timeutil.rules import (
    Board, board_of, commission, limit_prices, price_limit, stamp_duty_rate,
)


@pytest.mark.parametrize("sym,board", [
    ("600000.SH", Board.MAIN), ("000001", Board.MAIN), ("300750.SZ", Board.CHINEXT),
    ("301000", Board.CHINEXT), ("688981.SH", Board.STAR), ("830799.BJ", Board.BSE),
    ("920001", Board.BSE),
])
def test_board_of(sym, board):
    assert board_of(sym) is board


def test_chinext_limit_changes_on_reform_date():
    assert price_limit(Board.CHINEXT, date(2020, 8, 21)).up_pct == Decimal("0.10")
    assert price_limit(Board.CHINEXT, date(2020, 8, 24)).up_pct == Decimal("0.20")


def test_registration_new_stock_unlimited_first_five_days():
    assert price_limit(Board.STAR, date(2024, 1, 2), nth_trading_day_since_listing=5).unlimited
    assert not price_limit(Board.STAR, date(2024, 1, 2), nth_trading_day_since_listing=6).unlimited
    assert price_limit(Board.MAIN, date(2023, 4, 10), nth_trading_day_since_listing=1).unlimited


def test_main_board_legacy_ipo_first_day():
    lim = price_limit(Board.MAIN, date(2019, 6, 3), nth_trading_day_since_listing=1)
    assert (lim.up_pct, lim.down_pct) == (Decimal("0.44"), Decimal("0.36"))


def test_st_main_board():
    assert price_limit(Board.MAIN, date(2022, 1, 4), is_st=True).up_pct == Decimal("0.05")


def test_limit_prices_round_half_up():
    up, down = limit_prices(10.05, price_limit(Board.MAIN, date(2024, 1, 2)))
    assert up == Decimal("11.06")  # 11.055 -> 11.06
    assert down == Decimal("9.05")  # 9.045 -> 9.05


def test_limit_prices_none_when_unlimited():
    assert limit_prices(10, price_limit(Board.STAR, date(2024, 1, 2), nth_trading_day_since_listing=1)) is None


def test_stamp_duty():
    assert stamp_duty_rate(date(2023, 8, 25), "sell") == Decimal("0.001")
    assert stamp_duty_rate(date(2023, 8, 28), "sell") == Decimal("0.0005")
    assert stamp_duty_rate(date(2023, 8, 28), "buy") == 0


def test_commission_minimum():
    assert commission(10_000) == Decimal("5")  # 2.5 元 < 最低 5 元
    assert commission(100_000) == Decimal("25.00000")
    assert commission(10_000, minimum=Decimal("0")) == Decimal("2.50000")
