"""Regression tests for issues raised in the VP-of-Engineering review.

Each test maps to a specific finding (B1 / M1 / M2 / M3 / M4 and coverage gaps)
so a reviewer can trace fix -> test directly.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from backtester.data import (
    InMemoryDataHandler,
    build_sample_dataset,
    load_aligned_bars,
)
from backtester.events import Direction, FillEvent, MarketEvent, SignalEvent, SignalType
from backtester.performance import summarize
from backtester.portfolio import Portfolio
from backtester.strategy import MeanReversionPairsStrategy, MomentumStrategy
from backtester.walkforward import walk_forward

from .conftest import make_bars

TS = datetime(2020, 1, 1)


# --- B1: anchored walk-forward must terminate with a finite, growing train --- #
def test_anchored_walk_forward_terminates(tmp_path: Path):
    build_sample_dataset(tmp_path, n=800, seed=7)
    _, bars = load_aligned_bars(tmp_path, ["MOMO", "TREND"])
    result = walk_forward(
        ["MOMO", "TREND"],
        bars,
        lambda data, p: MomentumStrategy(data, ["MOMO", "TREND"], **p),
        {"fast_window": [10, 20], "slow_window": [50]},
        train_size=300,
        test_size=100,
        warmup=60,
        anchored=True,
        portfolio_kwargs={"risk_per_position": 0.5},
    )
    assert 1 <= len(result.folds) <= 20  # finite
    # Anchored => train always starts at 0 and the train window expands.
    assert all(f.train_span[0] == 0 for f in result.folds)
    train_ends = [f.train_span[1] for f in result.folds]
    assert train_ends == sorted(train_ends)
    assert train_ends[-1] > train_ends[0]


# --- M1: gross leverage is a book-level cap, not per-name --------------------- #
def test_book_level_gross_leverage_cap():
    symbols = ["A", "B", "C"]
    data = InMemoryDataHandler(symbols, {s: make_bars([100.0] * 5) for s in symbols})
    events: deque = deque()
    data.update_bars(events)
    market = events.popleft()

    pf = Portfolio(
        data, symbols, initial_capital=100_000.0,
        risk_per_position=0.5, max_gross_leverage=1.0,
    )
    pf.update_timeindex(market)

    orders: deque = deque()
    for s in symbols:
        pf.on_signal(SignalEvent(s, market.timestamp, SignalType.LONG), orders)
    gross = sum(o.quantity * 100.0 for o in orders)
    # Three 50%-of-equity requests, but the book cap is 100% of equity.
    assert gross <= 100_000.0 + 1e-6


def test_leverage_cap_allows_full_size_when_headroom():
    symbols = ["A", "B"]
    data = InMemoryDataHandler(symbols, {s: make_bars([100.0] * 5) for s in symbols})
    events: deque = deque()
    data.update_bars(events)
    market = events.popleft()
    pf = Portfolio(
        data, symbols, initial_capital=100_000.0,
        risk_per_position=0.3, max_gross_leverage=1.0,
    )
    pf.update_timeindex(market)
    orders: deque = deque()
    for s in symbols:
        pf.on_signal(SignalEvent(s, market.timestamp, SignalType.LONG), orders)
    # 2 * 30% = 60% < 100% cap -> both full size (300 shares each).
    assert [o.quantity for o in orders] == [300, 300]


# --- M2: pairs sizing honors the estimated hedge ratio ----------------------- #
def test_hedge_weights_are_beta_weighted():
    strat = MeanReversionPairsStrategy.__new__(MeanReversionPairsStrategy)
    assert strat._hedge_weights(0.5) == (1.0, 0.5)
    assert strat._hedge_weights(2.0) == (0.5, 1.0)
    # degenerate / non-positive beta falls back to equal notionals
    assert strat._hedge_weights(-1.0) == (1.0, 1.0)
    assert strat._hedge_weights(float("nan")) == (1.0, 1.0)


# --- M3: walk-forward OOS trade attribution excludes warmup, sums correctly --- #
def test_oos_trade_attribution_consistent(tmp_path: Path):
    build_sample_dataset(tmp_path, n=800, seed=7)
    _, bars = load_aligned_bars(tmp_path, ["MOMO", "TREND"])
    result = walk_forward(
        ["MOMO", "TREND"],
        bars,
        lambda data, p: MomentumStrategy(data, ["MOMO", "TREND"], **p),
        {"fast_window": [20], "slow_window": [60]},
        train_size=300,
        test_size=100,
        warmup=100,
        portfolio_kwargs={"risk_per_position": 0.5},
    )
    assert result.aggregate is not None
    per_fold = sum(f.oos_report.num_trades for f in result.folds)
    assert result.aggregate.num_trades == per_fold
    assert all(f.oos_report.num_trades >= 0 for f in result.folds)


# --- M4: non-finite data is rejected by the library, not silently poisoned --- #
def test_summarize_rejects_nan_equity():
    with pytest.raises(ValueError):
        summarize(np.array([100.0, np.nan, 120.0]))


def test_summarize_rejects_inf_equity():
    with pytest.raises(ValueError):
        summarize(np.array([100.0, np.inf, 120.0]))


def test_csv_with_non_finite_value_rejected(tmp_path: Path):
    csv = tmp_path / "BAD.csv"
    csv.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2020-01-01T00:00:00,100,101,99,nan,1000\n"
    )
    with pytest.raises(ValueError):
        load_aligned_bars(tmp_path, ["BAD"])


# --- coverage gap: short-position mark-to-market ----------------------------- #
def test_short_position_mark_to_market(stub_data):
    pf = Portfolio(stub_data, ["AAPL"], initial_capital=100_000.0)
    pf.on_fill(FillEvent("AAPL", TS, Direction.SELL, 100, 100.0, 0.0, 0.0))
    assert pf.positions["AAPL"] == -100
    # price falls -> short gains
    stub_data.set_price("AAPL", 90.0)
    pf.update_timeindex(MarketEvent(TS))
    assert pf.current_equity == pytest.approx(pf.cash + (-100) * 90.0)
    assert pf.current_equity > 100_000.0  # profitable short


def test_gross_exposure_counts_shorts(stub_data):
    pf = Portfolio(stub_data, ["AAPL"], initial_capital=100_000.0)
    pf.on_fill(FillEvent("AAPL", TS, Direction.SELL, 50, 100.0, 0.0, 0.0))
    assert pf.gross_exposure() == pytest.approx(50 * 100.0)
