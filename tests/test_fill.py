import numpy as np
import pandas as pd

from event_alpha.data.panel import Panel
from event_alpha.sim.fill import find_entry, find_exit, simulate


def make_panel():
    n = 6
    dates = pd.bdate_range("2024-01-01", periods=n)
    pc = np.array([[10.0], [11.0], [12.1], [12.1], [12.0], [12.5]])
    close = np.array([[11.0], [12.1], [12.1], [12.0], [12.5], [12.5]])
    opn = np.array([[10.5], [12.1], [np.nan], [12.0], [12.2], [12.4]])  # 第1日开盘涨停，第2日停牌
    close[2, 0] = np.nan
    hl = np.round(pc * 1.1, 2)
    ll = np.round(pc * 0.9, 2)
    ll[4, 0] = 12.5  # 第4日收盘跌停（构造）
    one = np.ones_like(pc)
    return Panel(dates, pd.Index(["600000.SH"]), opn, close, pc, hl, ll, one, one,
                 np.zeros_like(pc, bool), pd.Series([pd.Timestamp("2000-01-01")]))


def test_entry_skips_limit_up_open_and_suspension():
    p = make_panel()
    assert find_entry(p, 0, 1) == 3
    assert find_entry(p, 0, 1, max_defer=1) is None


def test_exit_defers_on_limit_down_close():
    p = make_panel()
    assert find_exit(p, 0, 4) == 5


def test_holding_return_chains_prev_close():
    p = make_panel()
    f = simulate(p, 0, 3, hold=1)
    assert f.entry_idx == 3 and f.exit_idx == 3
    assert abs(f.gross_ret - (12.0 / 12.0 - 1)) < 1e-12
    f = simulate(p, 0, 3, hold=3)  # 目标第5日
    assert f.exit_idx == 5
    expected = (12.0 / 12.0) * (12.5 / 12.0) * (12.5 / 12.5) - 1
    assert abs(f.gross_ret - expected) < 1e-12
