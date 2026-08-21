"""Unit tests for the simulated execution handler (costs + slippage)."""

from __future__ import annotations

from collections import deque
from datetime import datetime

import pytest

from backtester.events import Direction, FillEvent, OrderEvent, OrderType
from backtester.execution import SimulatedExecutionHandler

TS = datetime(2020, 1, 1)


def _order(direction, qty=10):
    return OrderEvent("AAPL", TS, OrderType.MARKET, qty, direction)


def test_slippage_pushes_buy_price_up(stub_data):
    ex = SimulatedExecutionHandler(
        stub_data, commission_bps=0.0, min_commission=0.0, slippage_bps=10.0
    )
    events: deque = deque()
    ex.execute_order(_order(Direction.BUY), events)
    fill: FillEvent = events.pop()
    # base 100, 10bps => +0.10
    assert fill.fill_price == pytest.approx(100.10)
    assert fill.slippage == pytest.approx(0.10 * 10)


def test_slippage_pushes_sell_price_down(stub_data):
    ex = SimulatedExecutionHandler(
        stub_data, commission_bps=0.0, min_commission=0.0, slippage_bps=10.0
    )
    events: deque = deque()
    ex.execute_order(_order(Direction.SELL), events)
    fill: FillEvent = events.pop()
    assert fill.fill_price == pytest.approx(99.90)


def test_commission_floor_applies(stub_data):
    ex = SimulatedExecutionHandler(
        stub_data, commission_bps=1.0, min_commission=5.0, slippage_bps=0.0
    )
    events: deque = deque()
    ex.execute_order(_order(Direction.BUY, qty=10), events)  # notional ~1000
    fill: FillEvent = events.pop()
    # 1000 * 1bps = 0.10 -> floored to 5.0
    assert fill.commission == pytest.approx(5.0)


def test_commission_scales_with_notional(stub_data):
    ex = SimulatedExecutionHandler(
        stub_data, commission_bps=10.0, min_commission=0.0, slippage_bps=0.0
    )
    events: deque = deque()
    ex.execute_order(_order(Direction.BUY, qty=1000), events)  # notional 100_000
    fill: FillEvent = events.pop()
    assert fill.commission == pytest.approx(100_000 * 10 / 10_000)


def test_zero_quantity_produces_no_fill(stub_data):
    ex = SimulatedExecutionHandler(stub_data)
    events: deque = deque()
    ex.execute_order(_order(Direction.BUY, qty=0), events)
    assert len(events) == 0


def test_negative_cost_params_rejected(stub_data):
    with pytest.raises(ValueError):
        SimulatedExecutionHandler(stub_data, slippage_bps=-1.0)
