"""Unit tests for strategies, including a structural no-look-ahead guarantee."""

from __future__ import annotations

from collections import deque

from backtester.data import InMemoryDataHandler, generate_cointegrated_pair
from backtester.events import Event, MarketEvent, SignalEvent, SignalType
from backtester.strategy import (
    MeanReversionPairsStrategy,
    MomentumStrategy,
    Strategy,
)

from .conftest import make_bars


def _drain_signals(handler: InMemoryDataHandler, strategy: Strategy) -> list[SignalEvent]:
    """Run a handler to exhaustion, collecting every signal a strategy emits."""
    signals: list[SignalEvent] = []
    events: deque[Event] = deque()
    while handler.continue_backtest:
        handler.update_bars(events)
        while events:
            ev = events.popleft()
            if isinstance(ev, MarketEvent):
                strategy.calculate_signals(ev, events)
            elif isinstance(ev, SignalEvent):
                signals.append(ev)
    return signals


def test_momentum_goes_long_on_uptrend():
    handler = InMemoryDataHandler(["A"], {"A": make_bars(list(range(1, 21)))})
    strat = MomentumStrategy(handler, ["A"], fast_window=2, slow_window=5)
    signals = _drain_signals(handler, strat)
    assert signals, "expected at least one signal on a clean uptrend"
    assert signals[0].signal_type is SignalType.LONG


def test_momentum_emits_only_on_state_change():
    handler = InMemoryDataHandler(["A"], {"A": make_bars(list(range(1, 31)))})
    strat = MomentumStrategy(handler, ["A"], fast_window=2, slow_window=5)
    signals = _drain_signals(handler, strat)
    # A pure uptrend crosses to LONG once and never flips back.
    assert sum(s.signal_type is SignalType.LONG for s in signals) == 1


def test_momentum_rejects_bad_windows():
    handler = InMemoryDataHandler(["A"], {"A": make_bars([1, 2, 3])})
    try:
        MomentumStrategy(handler, ["A"], fast_window=10, slow_window=5)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for fast >= slow")


def test_pairs_enters_and_exits_in_pairs():
    x, y = generate_cointegrated_pair(500, seed=11)
    handler = InMemoryDataHandler(["PAIR_X", "PAIR_Y"], {"PAIR_X": x, "PAIR_Y": y})
    strat = MeanReversionPairsStrategy(
        handler, "PAIR_X", "PAIR_Y", lookback=40, entry_z=1.5, exit_z=0.5
    )
    signals = _drain_signals(handler, strat)
    assert signals, "cointegrated pair should generate trades"
    # Signals must always arrive two-at-a-time (one per leg).
    assert len(signals) % 2 == 0
    exits = [s for s in signals if s.signal_type is SignalType.EXIT]
    assert exits, "expected at least one flatten (mean reversion)"


class _SpyStrategy(Strategy):
    """Records how many bars are visible each time it is called."""

    def __init__(self, data, symbol):
        self.data = data
        self.symbol = symbol
        self.visible_counts: list[int] = []

    def calculate_signals(self, event: MarketEvent, events: deque[Event]) -> None:
        self.visible_counts.append(len(self.data.get_latest_bars(self.symbol, 10_000)))


def test_strategy_never_sees_future_bars():
    handler = InMemoryDataHandler(["A"], {"A": make_bars(list(range(1, 11)))})
    spy = _SpyStrategy(handler, "A")
    _drain_signals(handler, spy)
    # On call k the strategy must see exactly k bars -- never any future data.
    assert spy.visible_counts == list(range(1, 11))
