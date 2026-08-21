"""Unit tests for data handling and synthetic generation."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import pytest

from backtester.data import (
    HistoricCSVDataHandler,
    InMemoryDataHandler,
    build_sample_dataset,
    generate_cointegrated_pair,
    generate_gbm_series,
    load_aligned_bars,
)

from .conftest import make_bars


def test_gbm_is_deterministic():
    a = generate_gbm_series(200, seed=42)
    b = generate_gbm_series(200, seed=42)
    assert [x.close for x in a] == [x.close for x in b]


def test_gbm_seed_changes_output():
    a = generate_gbm_series(200, seed=1)
    b = generate_gbm_series(200, seed=2)
    assert [x.close for x in a] != [x.close for x in b]


def test_gbm_prices_positive():
    bars = generate_gbm_series(500, seed=7)
    assert all(b.close > 0 and b.high >= b.low for b in bars)


def test_cointegrated_pair_aligned_and_positive():
    x, y = generate_cointegrated_pair(300, seed=11)
    assert len(x) == len(y) == 300
    assert all(b.close > 0 for b in x + y)


def test_in_memory_handler_no_lookahead():
    handler = InMemoryDataHandler(["A"], {"A": make_bars([1, 2, 3, 4, 5])})
    events: deque = deque()
    for expected_count, expected_last in enumerate([1, 2, 3, 4, 5], start=1):
        handler.update_bars(events)
        seen = handler.get_latest_bars("A", 100)
        assert len(seen) == expected_count  # never more than bars revealed so far
        assert seen[-1].close == expected_last


def test_in_memory_handler_stops_when_exhausted():
    handler = InMemoryDataHandler(["A"], {"A": make_bars([1, 2])})
    events: deque = deque()
    handler.update_bars(events)
    handler.update_bars(events)
    handler.update_bars(events)  # one past the end
    assert handler.continue_backtest is False


def test_in_memory_requires_equal_lengths():
    with pytest.raises(ValueError):
        InMemoryDataHandler(
            ["A", "B"], {"A": make_bars([1, 2, 3]), "B": make_bars([1, 2])}
        )


def test_csv_round_trip_and_alignment(tmp_path: Path):
    build_sample_dataset(tmp_path, n=120, seed=3)
    files = {p.name for p in tmp_path.glob("*.csv")}
    assert files == {"MOMO.csv", "TREND.csv", "PAIR_X.csv", "PAIR_Y.csv"}

    timeline, bars = load_aligned_bars(tmp_path, ["PAIR_X", "PAIR_Y"])
    assert len(timeline) == 120
    assert len(bars["PAIR_X"]) == len(bars["PAIR_Y"]) == 120

    handler = HistoricCSVDataHandler(tmp_path, ["MOMO", "TREND"])
    events: deque = deque()
    handler.update_bars(events)
    assert handler.get_latest_bar_value("MOMO") > 0


def test_missing_csv_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_aligned_bars(tmp_path, ["NOPE"])
