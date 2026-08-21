"""Shared test fixtures and helpers."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta

import pytest

from backtester.data import Bar, InMemoryDataHandler
from backtester.events import Event


class StubData:
    """Minimal DataHandler-compatible stub returning fixed prices per symbol."""

    def __init__(self, prices: dict[str, float]):
        self._prices = dict(prices)
        self.symbols = list(prices)
        self.continue_backtest = True

    def set_price(self, symbol: str, price: float) -> None:
        self._prices[symbol] = price

    def get_latest_bars(self, symbol: str, n: int = 1):  # pragma: no cover - trivial
        price = self._prices[symbol]
        bar = Bar(datetime(2020, 1, 1), price, price, price, price, 1000.0)
        return [bar] * n

    def get_latest_bar_value(
        self, symbol: str, field: str = "close", n_back: int = 0
    ) -> float:
        return self._prices[symbol]

    def update_bars(self, events: deque[Event]) -> None:  # pragma: no cover
        self.continue_backtest = False


@pytest.fixture
def stub_data():
    return StubData({"AAPL": 100.0})


def make_bars(closes, start=datetime(2020, 1, 1)):
    """Build simple bars from a list of close prices."""
    return [
        Bar(start + i * timedelta(days=1), c, c + 1, c - 1, c, 1000.0)
        for i, c in enumerate(closes)
    ]


@pytest.fixture
def two_symbol_handler():
    """A tiny 5-bar, 2-symbol in-memory handler."""
    bars = {
        "A": make_bars([10, 11, 12, 13, 14]),
        "B": make_bars([20, 19, 18, 17, 16]),
    }
    return InMemoryDataHandler(["A", "B"], bars)
