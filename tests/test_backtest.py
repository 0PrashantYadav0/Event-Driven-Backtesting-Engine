"""End-to-end engine tests and walk-forward validation tests."""

from __future__ import annotations

from pathlib import Path

from backtester.backtest import Backtest
from backtester.data import (
    InMemoryDataHandler,
    build_sample_dataset,
    generate_gbm_series,
    load_aligned_bars,
)
from backtester.execution import SimulatedExecutionHandler
from backtester.portfolio import Portfolio
from backtester.strategy import MomentumStrategy
from backtester.walkforward import walk_forward


def _run_momentum(bars):
    data = InMemoryDataHandler(["A"], {"A": bars})
    strat = MomentumStrategy(data, ["A"], fast_window=10, slow_window=30)
    pf = Portfolio(data, ["A"], initial_capital=100_000.0, risk_per_position=0.5)
    ex = SimulatedExecutionHandler(data)
    return Backtest(data, strat, pf, ex).run()


def test_end_to_end_smoke_runs():
    bars = generate_gbm_series(500, seed=7)
    result = _run_momentum(bars)
    assert result.bars_processed == 500
    assert result.events_processed >= 500
    assert result.report.num_periods == 499
    assert result.events_per_second > 0


def test_backtest_is_deterministic():
    bars = generate_gbm_series(400, seed=7)
    r1 = _run_momentum(bars)
    r2 = _run_momentum(bars)
    assert r1.report.as_dict() == r2.report.as_dict()
    assert [e for _, e in r1.equity_curve] == [e for _, e in r2.equity_curve]


def test_costs_reduce_returns():
    bars = generate_gbm_series(500, seed=7)

    def run(slippage_bps, commission_bps):
        data = InMemoryDataHandler(["A"], {"A": bars})
        strat = MomentumStrategy(data, ["A"], fast_window=10, slow_window=30)
        pf = Portfolio(data, ["A"], risk_per_position=0.5)
        ex = SimulatedExecutionHandler(
            data, slippage_bps=slippage_bps, commission_bps=commission_bps,
            min_commission=0.0,
        )
        return Backtest(data, strat, pf, ex).run().report.total_return

    frictionless = run(0.0, 0.0)
    costly = run(20.0, 10.0)
    assert costly < frictionless  # frictions must be a drag on returns


def test_walk_forward_produces_aggregate(tmp_path: Path):
    build_sample_dataset(tmp_path, n=800, seed=7)
    _, bars = load_aligned_bars(tmp_path, ["MOMO", "TREND"])
    result = walk_forward(
        ["MOMO", "TREND"],
        bars,
        lambda data, p: MomentumStrategy(
            data, ["MOMO", "TREND"], fast_window=p["fast"], slow_window=p["slow"]
        ),
        {"fast": [10, 20], "slow": [50]},
        train_size=300,
        test_size=100,
        warmup=60,
        portfolio_kwargs={"risk_per_position": 0.5},
    )
    assert result.aggregate is not None
    assert len(result.folds) >= 1
    # Each chosen param set must come from the grid we supplied.
    for fold in result.folds:
        assert fold.best_params["fast"] in (10, 20)
        assert fold.best_params["slow"] == 50
    assert len(result.oos_equity) >= 2
    assert "Aggregate" in result.pretty()


def test_walk_forward_rejects_oversized_windows(tmp_path: Path):
    build_sample_dataset(tmp_path, n=200, seed=7)
    _, bars = load_aligned_bars(tmp_path, ["MOMO", "TREND"])
    try:
        walk_forward(
            ["MOMO", "TREND"], bars,
            lambda data, p: MomentumStrategy(data, ["MOMO", "TREND"]),
            {},
            train_size=300, test_size=100, warmup=10,
        )
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for oversized windows")
