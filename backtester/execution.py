"""Order execution simulation with transaction costs and slippage.

A realistic backtest must penalise trading. We model two frictions:

* **Commission** — a per-trade cost of ``commission_bps`` basis points of
  notional plus a flat ``min_commission`` floor (mirrors broker fee schedules).
* **Slippage** — adverse price movement between decision and fill. We charge
  ``slippage_bps`` basis points of price, pushing the fill price *up* for buys
  and *down* for sells (you always trade at the worse price).

The handler consumes :class:`OrderEvent` s and emits :class:`FillEvent` s.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque

from .data import DataHandler
from .events import Direction, Event, FillEvent, OrderEvent

logger = logging.getLogger(__name__)


class ExecutionHandler(ABC):
    """Turns :class:`OrderEvent` s into :class:`FillEvent` s."""

    @abstractmethod
    def execute_order(self, order: OrderEvent, events: deque[Event]) -> None:
        """Simulate execution of ``order`` and push a :class:`FillEvent`."""


class SimulatedExecutionHandler(ExecutionHandler):
    """Fills market orders at the current bar close, adjusted for frictions.

    Args:
        data: The data handler, used to read the current market price.
        commission_bps: Commission in basis points of traded notional.
        min_commission: Minimum commission per fill (cash).
        slippage_bps: Slippage in basis points of price, applied adversely.
        price_field: Which bar field to fill against (``close`` by default).
    """

    def __init__(
        self,
        data: DataHandler,
        *,
        commission_bps: float = 1.0,
        min_commission: float = 1.0,
        slippage_bps: float = 2.0,
        price_field: str = "close",
    ) -> None:
        if commission_bps < 0 or min_commission < 0 or slippage_bps < 0:
            raise ValueError("cost parameters must be non-negative")
        self.data = data
        self.commission_bps = commission_bps
        self.min_commission = min_commission
        self.slippage_bps = slippage_bps
        self.price_field = price_field

    def _slipped_price(self, base_price: float, direction: Direction) -> float:
        slip = base_price * self.slippage_bps / 10_000.0
        return base_price + slip if direction is Direction.BUY else base_price - slip

    def _commission(self, notional: float) -> float:
        return max(self.min_commission, notional * self.commission_bps / 10_000.0)

    def execute_order(self, order: OrderEvent, events: deque[Event]) -> None:
        if order.quantity <= 0:
            return  # nothing to fill
        base_price = self.data.get_latest_bar_value(order.symbol, self.price_field)
        fill_price = self._slipped_price(base_price, order.direction)
        notional = fill_price * order.quantity
        commission = self._commission(notional)
        slippage_cost = abs(fill_price - base_price) * order.quantity

        fill = FillEvent(
            symbol=order.symbol,
            timestamp=order.timestamp,
            direction=order.direction,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_cost,
        )
        logger.debug(
            "FILL %s %s x%d @ %.4f (comm=%.2f slip=%.2f)",
            order.direction.value,
            order.symbol,
            order.quantity,
            fill_price,
            commission,
            slippage_cost,
        )
        events.append(fill)
