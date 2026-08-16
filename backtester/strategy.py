"""Trading strategies.

Every strategy implements the same :class:`Strategy` interface: react to a
:class:`MarketEvent` by reading *only* the bars available up to the current
timestamp and emitting :class:`SignalEvent` s. Because the data handler has not
yet loaded future bars, look-ahead bias is structurally impossible.

Two production-quality strategies are provided:

* :class:`MomentumStrategy` — dual moving-average trend following on single
  names.
* :class:`MeanReversionPairsStrategy` — z-score reversion on a cointegrated
  pair with a rolling hedge ratio.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from .data import DataHandler
from .events import Event, MarketEvent, SignalEvent, SignalType

logger = logging.getLogger(__name__)


class Strategy(ABC):
    """Consumes market data and emits trading signals."""

    @abstractmethod
    def calculate_signals(self, event: MarketEvent, events: deque[Event]) -> None:
        """Inspect the latest data and push any :class:`SignalEvent` s."""


class MomentumStrategy(Strategy):
    """Dual moving-average crossover trend follower.

    Goes long when the fast MA is above the slow MA. Optionally shorts on the
    reverse cross; otherwise flattens. Signals are only emitted on state
    changes to keep the event stream clean.

    Args:
        data: The data handler.
        symbols: Symbols to trade independently.
        fast_window: Fast moving-average lookback (bars).
        slow_window: Slow moving-average lookback (bars); must exceed fast.
        allow_short: If True, emit SHORT on a bearish cross instead of EXIT.
    """

    def __init__(
        self,
        data: DataHandler,
        symbols: list[str],
        *,
        fast_window: int = 20,
        slow_window: int = 60,
        allow_short: bool = False,
    ) -> None:
        if fast_window < 1 or slow_window < 1:
            raise ValueError("windows must be positive")
        if fast_window >= slow_window:
            raise ValueError("fast_window must be smaller than slow_window")
        self.data = data
        self.symbols = list(symbols)
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.allow_short = allow_short
        self._state: dict[str, SignalType | None] = {s: None for s in self.symbols}

    def calculate_signals(self, event: MarketEvent, events: deque[Event]) -> None:
        for symbol in self.symbols:
            bars = self.data.get_latest_bars(symbol, self.slow_window)
            if len(bars) < self.slow_window:
                continue
            closes = np.array([b.close for b in bars], dtype=float)
            fast_ma = closes[-self.fast_window :].mean()
            slow_ma = closes.mean()

            if fast_ma > slow_ma:
                desired = SignalType.LONG
            elif self.allow_short:
                desired = SignalType.SHORT
            else:
                desired = SignalType.EXIT

            if desired is not self._state[symbol]:
                self._state[symbol] = desired
                events.append(SignalEvent(symbol, event.timestamp, desired, strength=1.0))


class MeanReversionPairsStrategy(Strategy):
    """Z-score mean-reversion on a cointegrated pair.

    The spread is ``log(y) - beta * log(x)`` where ``beta`` is re-estimated by
    OLS over the rolling lookback window. When the spread's z-score exceeds
    ``entry_z`` the spread is shorted (short ``y``, long ``x``); below
    ``-entry_z`` it is bought (long ``y``, short ``x``). Positions are flattened
    when ``|z|`` falls below ``exit_z``.

    Args:
        data: The data handler.
        symbol_x: The independent leg (hedge instrument).
        symbol_y: The dependent leg.
        lookback: Rolling window for beta, mean and std of the spread.
        entry_z: Absolute z-score to open a position.
        exit_z: Absolute z-score to close a position.
    """

    def __init__(
        self,
        data: DataHandler,
        symbol_x: str,
        symbol_y: str,
        *,
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ) -> None:
        if lookback < 3:
            raise ValueError("lookback must be >= 3 for a stable regression")
        if entry_z <= exit_z:
            raise ValueError("entry_z must exceed exit_z")
        self.data = data
        self.symbol_x = symbol_x
        self.symbol_y = symbol_y
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self._in_market = False  # currently holding a spread position?
        self._long_spread = False  # direction of the held spread

    def _hedge_weights(self, beta: float) -> tuple[float, float]:
        """Return ``(weight_y, weight_x)`` giving a beta-weighted dollar hedge.

        For the spread ``log_y - beta*log_x`` to be common-factor neutral we
        need ``notional_x = beta * notional_y``. We express that as per-leg
        sizing strengths in ``[0, 1]`` by normalising by the larger leg. A
        non-positive/degenerate beta falls back to equal notionals.
        """
        if not np.isfinite(beta) or beta <= 0:
            return 1.0, 1.0
        return (1.0, beta) if beta <= 1.0 else (1.0 / beta, 1.0)

    def _emit(
        self,
        ts,
        sig_y: SignalType,
        sig_x: SignalType,
        weights: tuple[float, float],
        events: deque[Event],
    ) -> None:
        w_y, w_x = weights
        events.append(SignalEvent(self.symbol_y, ts, sig_y, strength=w_y))
        events.append(SignalEvent(self.symbol_x, ts, sig_x, strength=w_x))

    def calculate_signals(self, event: MarketEvent, events: deque[Event]) -> None:
        bars_x = self.data.get_latest_bars(self.symbol_x, self.lookback)
        bars_y = self.data.get_latest_bars(self.symbol_y, self.lookback)
        if len(bars_x) < self.lookback or len(bars_y) < self.lookback:
            return

        log_x = np.log(np.array([b.close for b in bars_x], dtype=float))
        log_y = np.log(np.array([b.close for b in bars_y], dtype=float))

        # Rolling OLS hedge ratio: log_y = alpha + beta * log_x.
        beta, alpha = np.polyfit(log_x, log_y, 1)
        spread = log_y - (beta * log_x + alpha)
        std = spread.std(ddof=1)
        if std == 0 or not np.isfinite(std):
            return
        z = (spread[-1] - spread.mean()) / std
        ts = event.timestamp
        weights = self._hedge_weights(float(beta))
        flat = (1.0, 1.0)  # strength is irrelevant when flattening to zero

        if not self._in_market:
            if z >= self.entry_z:
                # spread rich -> short spread: short y, long x
                self._emit(ts, SignalType.SHORT, SignalType.LONG, weights, events)
                self._in_market, self._long_spread = True, False
            elif z <= -self.entry_z:
                # spread cheap -> long spread: long y, short x
                self._emit(ts, SignalType.LONG, SignalType.SHORT, weights, events)
                self._in_market, self._long_spread = True, True
        else:
            reverted = abs(z) <= self.exit_z
            flipped = (self._long_spread and z >= self.entry_z) or (
                not self._long_spread and z <= -self.entry_z
            )
            if reverted or flipped:
                self._emit(ts, SignalType.EXIT, SignalType.EXIT, flat, events)
                self._in_market, self._long_spread = False, False
