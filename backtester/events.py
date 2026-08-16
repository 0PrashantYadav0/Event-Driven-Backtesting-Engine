"""Typed events that flow through the backtesting engine's event queue.

The engine is *event-driven*: components communicate exclusively by pushing
immutable events onto a shared FIFO queue. This module defines the four event
types that make up the full trade lifecycle:

    MarketEvent  -> new market data is available (a new bar)
    SignalEvent  -> a strategy has decided on a direction/strength
    OrderEvent   -> the portfolio wants to send an order to the market
    FillEvent    -> an order was executed (with costs + slippage applied)

Keeping events as small frozen dataclasses makes the data flow explicit,
immutable, and trivial to reason about (and to test).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    """Discriminator for the kind of event on the queue."""

    MARKET = "MARKET"
    SIGNAL = "SIGNAL"
    ORDER = "ORDER"
    FILL = "FILL"


class SignalType(str, Enum):
    """Direction a strategy wants to express for a symbol."""

    LONG = "LONG"
    SHORT = "SHORT"
    EXIT = "EXIT"  # flatten any open position


class OrderType(str, Enum):
    """Supported order types. We keep it to MARKET for a realistic sim."""

    MARKET = "MARKET"


class Direction(str, Enum):
    """Order/fill side."""

    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True, slots=True)
class Event:
    """Base class so isinstance / type checks read cleanly."""

    type: EventType


@dataclass(frozen=True, slots=True)
class MarketEvent(Event):
    """Emitted by the DataHandler when a new bar is available.

    Attributes:
        timestamp: The bar timestamp that just became available.
    """

    timestamp: datetime

    def __init__(self, timestamp: datetime) -> None:  # noqa: D401
        object.__setattr__(self, "type", EventType.MARKET)
        object.__setattr__(self, "timestamp", timestamp)


@dataclass(frozen=True, slots=True)
class SignalEvent(Event):
    """Emitted by a Strategy expressing a desired direction for a symbol.

    Attributes:
        symbol: The ticker the signal refers to.
        timestamp: Bar timestamp the signal was generated on.
        signal_type: LONG / SHORT / EXIT.
        strength: Suggested sizing weight in [0, 1]; portfolio may scale it.
    """

    symbol: str
    timestamp: datetime
    signal_type: SignalType
    strength: float = 1.0

    def __init__(
        self,
        symbol: str,
        timestamp: datetime,
        signal_type: SignalType,
        strength: float = 1.0,
    ) -> None:
        object.__setattr__(self, "type", EventType.SIGNAL)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "signal_type", signal_type)
        object.__setattr__(self, "strength", float(strength))


@dataclass(frozen=True, slots=True)
class OrderEvent(Event):
    """Emitted by the Portfolio requesting execution of a trade.

    Attributes:
        symbol: The ticker to trade.
        timestamp: Bar timestamp the order was created on.
        order_type: MARKET (only supported type).
        quantity: Absolute number of units (always positive).
        direction: BUY or SELL.
    """

    symbol: str
    timestamp: datetime
    order_type: OrderType
    quantity: int
    direction: Direction

    def __init__(
        self,
        symbol: str,
        timestamp: datetime,
        order_type: OrderType,
        quantity: int,
        direction: Direction,
    ) -> None:
        if quantity < 0:
            raise ValueError("OrderEvent quantity must be non-negative")
        object.__setattr__(self, "type", EventType.ORDER)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "order_type", order_type)
        object.__setattr__(self, "quantity", int(quantity))
        object.__setattr__(self, "direction", direction)


@dataclass(frozen=True, slots=True)
class FillEvent(Event):
    """Emitted by the ExecutionHandler once an order is (simulated) filled.

    Attributes:
        symbol: The ticker traded.
        timestamp: Bar timestamp of the fill.
        direction: BUY or SELL.
        quantity: Absolute units filled.
        fill_price: Per-unit price *after* slippage.
        commission: Total commission charged for this fill.
        slippage: Total slippage cost (price impact) for this fill, in cash.
    """

    symbol: str
    timestamp: datetime
    direction: Direction
    quantity: int
    fill_price: float
    commission: float
    slippage: float

    def __init__(
        self,
        symbol: str,
        timestamp: datetime,
        direction: Direction,
        quantity: int,
        fill_price: float,
        commission: float,
        slippage: float,
    ) -> None:
        object.__setattr__(self, "type", EventType.FILL)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timestamp", timestamp)
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "quantity", int(quantity))
        object.__setattr__(self, "fill_price", float(fill_price))
        object.__setattr__(self, "commission", float(commission))
        object.__setattr__(self, "slippage", float(slippage))

    @property
    def signed_quantity(self) -> int:
        """Quantity with sign: +q for BUY, -q for SELL."""
        return self.quantity if self.direction is Direction.BUY else -self.quantity
