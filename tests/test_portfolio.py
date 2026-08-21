"""Unit tests for portfolio accounting, sizing, and mark-to-market."""

from __future__ import annotations

from collections import deque
from datetime import datetime

import pytest

from backtester.events import (
    Direction,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
    SignalType,
)
from backtester.portfolio import Portfolio

TS = datetime(2020, 1, 1)


def _pf(stub_data, **kw):
    return Portfolio(stub_data, ["AAPL"], initial_capital=100_000.0, **kw)


def test_signal_sizes_to_target_position(stub_data):
    pf = _pf(stub_data, risk_per_position=0.2)
    events: deque = deque()
    pf.on_signal(SignalEvent("AAPL", TS, SignalType.LONG), events)
    order: OrderEvent = events.pop()
    # 100k * 0.2 / 100 = 200 shares
    assert order.direction is Direction.BUY
    assert order.quantity == 200


def test_short_signal_targets_negative(stub_data):
    pf = _pf(stub_data, risk_per_position=0.2)
    events: deque = deque()
    pf.on_signal(SignalEvent("AAPL", TS, SignalType.SHORT), events)
    order: OrderEvent = events.pop()
    assert order.direction is Direction.SELL
    assert order.quantity == 200


def test_exit_flattens_existing_position(stub_data):
    pf = _pf(stub_data, risk_per_position=0.2)
    pf.positions["AAPL"] = 200
    events: deque = deque()
    pf.on_signal(SignalEvent("AAPL", TS, SignalType.EXIT), events)
    order: OrderEvent = events.pop()
    assert order.direction is Direction.SELL
    assert order.quantity == 200


def test_no_order_when_already_at_target(stub_data):
    pf = _pf(stub_data, risk_per_position=0.2)
    pf.positions["AAPL"] = 200
    events: deque = deque()
    pf.on_signal(SignalEvent("AAPL", TS, SignalType.LONG), events)
    assert len(events) == 0


def test_fill_updates_cash_and_positions(stub_data):
    pf = _pf(stub_data)
    pf.on_fill(FillEvent("AAPL", TS, Direction.BUY, 200, 100.0, 5.0, 2.0))
    assert pf.positions["AAPL"] == 200
    assert pf.cash == pytest.approx(100_000 - 200 * 100.0 - 5.0)
    assert pf.num_trades == 1


def test_buy_then_sell_cash_conservation(stub_data):
    pf = _pf(stub_data)
    pf.on_fill(FillEvent("AAPL", TS, Direction.BUY, 100, 100.0, 1.0, 0.0))
    pf.on_fill(FillEvent("AAPL", TS, Direction.SELL, 100, 101.0, 1.0, 0.0))
    # bought 100@100 (-10000-1), sold 100@101 (+10100-1) => net +98
    assert pf.positions["AAPL"] == 0
    assert pf.cash == pytest.approx(100_000 + (10_100 - 10_000) - 2.0)


def test_mark_to_market_equity_invariant(stub_data):
    pf = _pf(stub_data)
    pf.on_fill(FillEvent("AAPL", TS, Direction.BUY, 200, 100.0, 0.0, 0.0))
    stub_data.set_price("AAPL", 110.0)
    pf.update_timeindex(MarketEvent(TS))
    # equity must equal cash + position * current price
    assert pf.current_equity == pytest.approx(pf.cash + 200 * 110.0)
    assert pf.equity_curve[-1][1] == pytest.approx(pf.current_equity)


def test_invalid_risk_rejected(stub_data):
    with pytest.raises(ValueError):
        Portfolio(stub_data, ["AAPL"], risk_per_position=0.0)
    with pytest.raises(ValueError):
        Portfolio(stub_data, ["AAPL"], risk_per_position=1.5)
