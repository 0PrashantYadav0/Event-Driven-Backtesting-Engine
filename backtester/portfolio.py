"""Portfolio: position sizing, order generation, and P&L accounting.

The portfolio sits between strategies and execution. It:

1. Converts :class:`SignalEvent` s into :class:`OrderEvent` s using a
   target-position sizing scheme (fraction of equity per position). Sizing off
   a *target* position (rather than blindly adding) makes rebalancing and
   flattening fall out naturally and prevents accidental position stacking.
2. Enforces a **book-level gross-leverage cap**: the combined notional of all
   positions (existing + those requested on the current bar) is scaled down to
   respect ``max_gross_leverage * equity``. This is a true portfolio
   constraint, not a per-name one.
3. Consumes :class:`FillEvent` s to update cash and positions, charging
   commission (slippage is already baked into the fill price).
4. Marks the book to market on every :class:`MarketEvent`, appending to the
   equity curve used for performance analytics.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from datetime import datetime

import numpy as np

from .data import DataHandler
from .events import (
    Direction,
    Event,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderType,
    SignalEvent,
    SignalType,
)

logger = logging.getLogger(__name__)


class Portfolio:
    """Tracks cash, positions and equity; turns signals into sized orders.

    Args:
        data: Data handler used to read current prices.
        symbols: Universe of tradable symbols.
        initial_capital: Starting cash.
        risk_per_position: Fraction of current equity to allocate per unit of
            signal strength (e.g. ``0.20`` => full-strength signal targets 20%
            of equity in that name). Must be in ``(0, 1]``.
        max_gross_leverage: Cap on *aggregate* gross exposure / equity. Orders
            on a bar are scaled so total gross notional never exceeds this.
    """

    def __init__(
        self,
        data: DataHandler,
        symbols: list[str],
        *,
        initial_capital: float = 100_000.0,
        risk_per_position: float = 0.20,
        max_gross_leverage: float = 1.0,
    ) -> None:
        if not 0 < risk_per_position <= 1:
            raise ValueError("risk_per_position must be in (0, 1]")
        if max_gross_leverage <= 0:
            raise ValueError("max_gross_leverage must be positive")

        self.data = data
        self.symbols = list(symbols)
        self.initial_capital = float(initial_capital)
        self.risk_per_position = risk_per_position
        self.max_gross_leverage = max_gross_leverage

        self.cash = float(initial_capital)
        self.positions: dict[str, int] = defaultdict(int)
        self.current_equity = float(initial_capital)
        self.num_trades = 0

        self.equity_curve: list[tuple[datetime, float]] = []
        # Cumulative trade count snapshotted at each bar *before* that bar's
        # fills. Lets walk-forward attribute trades to the out-of-sample window.
        self.trade_count_curve: list[int] = []

        # Target positions requested on the current bar (symbol -> target qty),
        # reset each bar. Used to enforce a book-level leverage cap even when
        # several legs are sized within the same bar (e.g. pairs trades).
        self._bar_targets: dict[str, int] = {}
        self._bar_stamp: datetime | None = None

    # ---- marking to market ------------------------------------------------ #
    def _holdings_value(self) -> float:
        total = 0.0
        for symbol, qty in self.positions.items():
            if qty == 0:
                continue
            price = self.data.get_latest_bar_value(symbol)
            total += qty * price
        return total

    def update_timeindex(self, event: MarketEvent) -> None:
        """Mark the book to market at the current bar and record equity."""
        self.current_equity = self.cash + self._holdings_value()
        self.equity_curve.append((event.timestamp, self.current_equity))
        self.trade_count_curve.append(self.num_trades)
        # New bar -> reset per-bar target planning.
        self._bar_targets = {}
        self._bar_stamp = event.timestamp

    # ---- signal -> order -------------------------------------------------- #
    def _planned_gross_excluding(self, symbol: str) -> float:
        """Gross notional already committed on this bar, excluding ``symbol``.

        Combines positions we're *not* re-sizing this bar with the pending
        targets for legs already decided this bar.
        """
        gross = 0.0
        for sym in self.symbols:
            if sym == symbol:
                continue
            qty = self._bar_targets.get(sym, self.positions[sym])
            if qty == 0:
                continue
            gross += abs(qty) * self.data.get_latest_bar_value(sym)
        return gross

    def _target_position(self, symbol: str, signal: SignalEvent) -> int:
        if signal.signal_type is SignalType.EXIT:
            return 0
        price = self.data.get_latest_bar_value(symbol)
        if price <= 0:
            return 0
        strength = max(0.0, min(1.0, signal.strength))
        desired_notional = self.current_equity * self.risk_per_position * strength

        # Book-level gross cap: only as much notional as remains in the budget.
        budget = self.current_equity * self.max_gross_leverage
        remaining = max(0.0, budget - self._planned_gross_excluding(symbol))
        notional = min(desired_notional, remaining)

        qty = math.floor(notional / price)
        return qty if signal.signal_type is SignalType.LONG else -qty

    def on_signal(self, signal: SignalEvent, events: deque[Event]) -> None:
        target = self._target_position(signal.symbol, signal)
        self._bar_targets[signal.symbol] = target
        current = self.positions[signal.symbol]
        delta = target - current
        if delta == 0:
            return
        direction = Direction.BUY if delta > 0 else Direction.SELL
        events.append(
            OrderEvent(
                symbol=signal.symbol,
                timestamp=signal.timestamp,
                order_type=OrderType.MARKET,
                quantity=abs(delta),
                direction=direction,
            )
        )

    # ---- fill -> book update --------------------------------------------- #
    def on_fill(self, fill: FillEvent) -> None:
        signed = fill.signed_quantity
        self.positions[fill.symbol] += signed
        cash_flow = -signed * fill.fill_price  # buys cost cash, sells raise cash
        self.cash += cash_flow - fill.commission
        self.num_trades += 1
        logger.debug(
            "BOOK %s pos=%d cash=%.2f", fill.symbol, self.positions[fill.symbol], self.cash
        )

    # ---- reporting -------------------------------------------------------- #
    def gross_exposure(self) -> float:
        """Current gross notional of all open positions."""
        return sum(
            abs(qty) * self.data.get_latest_bar_value(sym)
            for sym, qty in self.positions.items()
            if qty != 0
        )

    def equity_series(self) -> np.ndarray:
        """Return the equity curve as a NumPy array of equity values."""
        if not self.equity_curve:
            return np.asarray([self.initial_capital], dtype=float)
        return np.asarray([e for _, e in self.equity_curve], dtype=float)
