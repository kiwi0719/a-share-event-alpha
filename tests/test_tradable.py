from datetime import date, datetime

import pytest

from event_alpha.timeutil.tradable import CN_TZ, TradingCalendar, tradable_at

# 2024-09-27 周五，2024-09-30 周一，国庆 10-01 至 10-07 休市，10-08 周二开市
CAL = TradingCalendar([date(2024, 9, 26), date(2024, 9, 27), date(2024, 9, 30), date(2024, 10, 8), date(2024, 10, 9)])


def at(d):
    return datetime(d.year, d.month, d.day, 9, 30, tzinfo=CN_TZ)


def test_day_precision_is_treated_as_after_close():
    assert tradable_at(date(2024, 9, 27), "day", CAL) == at(date(2024, 9, 30))


def test_day_precision_ignores_time_component():
    assert tradable_at(datetime(2024, 9, 27, 7, 0), "day", CAL) == at(date(2024, 9, 30))


def test_minute_before_auction_same_day():
    assert tradable_at(datetime(2024, 9, 27, 8, 30), "minute", CAL) == at(date(2024, 9, 27))


def test_minute_at_auction_start_goes_next_day():
    assert tradable_at(datetime(2024, 9, 27, 9, 15), "minute", CAL) == at(date(2024, 9, 30))


def test_minute_intraday_conservative_next_day():
    assert tradable_at(datetime(2024, 9, 27, 11, 0), "minute", CAL) == at(date(2024, 9, 30))


def test_minute_after_close_next_day():
    assert tradable_at(datetime(2024, 9, 27, 18, 0), "minute", CAL) == at(date(2024, 9, 30))


def test_holiday_morning_goes_to_next_trading_day():
    assert tradable_at(datetime(2024, 10, 3, 8, 0), "minute", CAL) == at(date(2024, 10, 8))


def test_utc_input_is_converted():
    # 2024-09-27 00:30 UTC = 08:30 北京时间，开盘前
    from datetime import timezone
    ts = datetime(2024, 9, 27, 0, 30, tzinfo=timezone.utc)
    assert tradable_at(ts, "minute", CAL) == at(date(2024, 9, 27))


def test_suspension_rolls_forward():
    suspended = {date(2024, 9, 30), date(2024, 10, 8)}
    got = tradable_at(date(2024, 9, 27), "day", CAL, is_suspended=lambda d: d in suspended)
    assert got == at(date(2024, 10, 9))


def test_calendar_end_raises():
    with pytest.raises(LookupError):
        tradable_at(date(2024, 10, 9), "day", CAL)


def test_minute_requires_datetime():
    with pytest.raises(TypeError):
        tradable_at(date(2024, 9, 27), "minute", CAL)
