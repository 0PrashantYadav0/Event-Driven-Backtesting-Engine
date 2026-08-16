"""The event-driven backtest engine.

This is the heart of the system: a single FIFO event queue drives four
decoupled components. Each pass of the outer loop pulls the next bar; the inner
loop then drains every event that bar produced, cascading
``MARKET -> SIGNAL -> ORDER -> FILL`` until the queue is empty.

    +------------+   MarketEvent   +----------+   SignalEvent  +-----------+
    | DataHandler| --------------> | Strategy | -------------> | Portfolio |
    +------------+                 +----------+                +-----------+
          ^                                                        |
          |                                                 OrderEvent
          |                                                        v
          |                     FillEvent               +-------------------+
          +-------------------------------------------- | ExecutionHandler  |
                        (updates book)                   +-------------------+
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

from .data import DataHandler
from .events import (
    Event,
    EventType,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
)
from .execution import ExecutionHandler, SimulatedExecutionHandler
from .performance import PerformanceReport, summarize
from .portfolio import Portfolio
from .strategy import Strategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Outcome of a backtest run."""

    report: PerformanceReport
    equity_curve: list[tuple]  # (timestamp, equity)
    trade_counts: list[int]  # cumulative trades snapshotted per bar (pre-fill)
    bars_processed: int
    events_processed: int
    elapsed_seconds: float

    @property
    def events_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return float("inf")
        return self.events_processed / self.elapsed_seconds


class Backtest:
    """Assembles components and runs the event loop.

    Args:
        data: Data handler feeding bars.
        strategy: Strategy generating signals.
        portfolio: Portfolio sizing/accounting.
        execution: Execution handler simulating fills.
        periods_per_year: Annualisation factor (252 for daily bars).
        risk_free_rate: Annual risk-free rate used in the Sharpe ratio.
    """

    def __init__(
        self,
        data: DataHandler,
        strategy: Strategy,
        portfolio: Portfolio,
        execution: ExecutionHandler,
        *,
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0,
    ) -> None:
        self.data = data
        self.strategy = strategy
        self.portfolio = portfolio
        self.execution = execution
        self.periods_per_year = periods_per_year
        self.risk_free_rate = risk_free_rate

    def run(self) -> BacktestResult:
        events: deque[Event] = deque()
        bars = 0
        processed = 0
        start = time.perf_counter()

        while self.data.continue_backtest:
            self.data.update_bars(events)
            while events:
                event = events.popleft()
                processed += 1
                if event.type is EventType.MARKET:
                    assert isinstance(event, MarketEvent)
                    bars += 1
                    self.portfolio.update_timeindex(event)
                    self.strategy.calculate_signals(event, events)
                elif event.type is EventType.SIGNAL:
                    assert isinstance(event, SignalEvent)
                    self.portfolio.on_signal(event, events)
                elif event.type is EventType.ORDER:
                    assert isinstance(event, OrderEvent)
                    self.execution.execute_order(event, events)
                elif event.type is EventType.FILL:
                    assert isinstance(event, FillEvent)
                    self.portfolio.on_fill(event)

        elapsed = time.perf_counter() - start
        equity = self.portfolio.equity_series()
        report = summarize(
            equity,
            num_trades=self.portfolio.num_trades,
            risk_free_rate=self.risk_free_rate,
            periods_per_year=self.periods_per_year,
        )
        logger.info(
            "Backtest done: %d bars, %d events in %.4fs (%.0f events/s)",
            bars,
            processed,
            elapsed,
            processed / elapsed if elapsed > 0 else float("inf"),
        )
        return BacktestResult(
            report=report,
            equity_curve=list(self.portfolio.equity_curve),
            trade_counts=list(self.portfolio.trade_count_curve),
            bars_processed=bars,
            events_processed=processed,
            elapsed_seconds=elapsed,
        )


def default_execution(data: DataHandler, **kwargs) -> SimulatedExecutionHandler:
    """Convenience factory for the standard simulated execution handler."""
    return SimulatedExecutionHandler(data, **kwargs)
