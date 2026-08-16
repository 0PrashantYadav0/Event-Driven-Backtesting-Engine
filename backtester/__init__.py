"""backtester: an event-driven backtesting engine for quant strategies."""

from __future__ import annotations

from .backtest import Backtest, BacktestResult
from .data import (
    Bar,
    DataHandler,
    HistoricCSVDataHandler,
    InMemoryDataHandler,
    build_sample_dataset,
    generate_cointegrated_pair,
    generate_gbm_series,
    load_aligned_bars,
)
from .events import (
    Direction,
    FillEvent,
    MarketEvent,
    OrderEvent,
    OrderType,
    SignalEvent,
    SignalType,
)
from .execution import ExecutionHandler, SimulatedExecutionHandler
from .performance import PerformanceReport, summarize
from .portfolio import Portfolio
from .strategy import MeanReversionPairsStrategy, MomentumStrategy, Strategy
from .walkforward import WalkForwardResult, walk_forward

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Backtest",
    "BacktestResult",
    "Bar",
    "DataHandler",
    "HistoricCSVDataHandler",
    "InMemoryDataHandler",
    "build_sample_dataset",
    "generate_cointegrated_pair",
    "generate_gbm_series",
    "load_aligned_bars",
    "Direction",
    "FillEvent",
    "MarketEvent",
    "OrderEvent",
    "OrderType",
    "SignalEvent",
    "SignalType",
    "ExecutionHandler",
    "SimulatedExecutionHandler",
    "PerformanceReport",
    "summarize",
    "Portfolio",
    "MeanReversionPairsStrategy",
    "MomentumStrategy",
    "Strategy",
    "WalkForwardResult",
    "walk_forward",
]
