"""Unit tests for the event dataclasses."""

from __future__ import annotations

from datetime import datetime

import pytest

from backtester.events import (
    Direction,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderType,
    SignalEvent,
    SignalType,
)

TS = datetime(2020, 1, 1)


def test_market_event_type_tag():
    ev = MarketEvent(TS)
    assert ev.type is EventType.MARKET
    assert ev.timestamp == TS


def test_signal_event_defaults_and_coercion():
    ev = SignalEvent("AAPL", TS, SignalType.LONG, strength=1)
    assert ev.type is EventType.SIGNAL
    assert ev.signal_type is SignalType.LONG
    assert isinstance(ev.strength, float)


def test_order_event_rejects_negative_quantity():
    with pytest.raises(ValueError):
        OrderEvent("AAPL", TS, OrderType.MARKET, -5, Direction.BUY)


def test_events_are_immutable():
    ev = MarketEvent(TS)
    with pytest.raises(Exception):
        ev.timestamp = datetime(2021, 1, 1)  # type: ignore[misc]


def test_fill_signed_quantity():
    buy = FillEvent("AAPL", TS, Direction.BUY, 10, 100.0, 1.0, 0.5)
    sell = FillEvent("AAPL", TS, Direction.SELL, 7, 100.0, 1.0, 0.5)
    assert buy.signed_quantity == 10
    assert sell.signed_quantity == -7
