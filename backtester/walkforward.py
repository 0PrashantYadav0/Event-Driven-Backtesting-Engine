"""Walk-forward validation.

Walk-forward is the antidote to overfitting: parameters are optimised on an
*in-sample* (training) window, then evaluated on the *immediately following*
out-of-sample (test) window that the optimiser never saw. Rolling this forward
across the whole history yields a stitched out-of-sample equity curve whose
metrics are an honest estimate of live performance.

    |--- train ---|- test -|
             |--- train ---|- test -|
                      |--- train ---|- test -|   ...

Only out-of-sample segments are concatenated into the reported equity curve.
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .backtest import Backtest
from .data import Bar, InMemoryDataHandler
from .execution import SimulatedExecutionHandler
from .performance import PerformanceReport, summarize
from .portfolio import Portfolio
from .strategy import Strategy

logger = logging.getLogger(__name__)

StrategyFactory = Callable[[InMemoryDataHandler, dict[str, Any]], Strategy]


@dataclass(frozen=True, slots=True)
class Fold:
    """A single train/test split with its chosen params and OOS result."""

    index: int
    train_span: tuple[int, int]
    test_span: tuple[int, int]
    best_params: dict[str, Any]
    in_sample_sharpe: float
    oos_report: PerformanceReport


@dataclass(slots=True)
class WalkForwardResult:
    """Aggregate walk-forward outcome across all folds."""

    folds: list[Fold] = field(default_factory=list)
    aggregate: PerformanceReport | None = None
    oos_equity: list[float] = field(default_factory=list)

    def pretty(self) -> str:
        lines = ["Walk-forward folds:"]
        for f in self.folds:
            lines.append(
                f"  fold {f.index}: train[{f.train_span[0]}:{f.train_span[1]}] "
                f"test[{f.test_span[0]}:{f.test_span[1]}] "
                f"params={f.best_params} IS_sharpe={f.in_sample_sharpe:.2f} "
                f"OOS_sharpe={f.oos_report.sharpe:.2f} OOS_ret={f.oos_report.total_return:.2%}"
            )
        if self.aggregate is not None:
            lines.append("\nAggregate OUT-OF-SAMPLE performance:")
            lines.append(self.aggregate.pretty())
        return "\n".join(lines)


def _slice_bars(
    bars_by_symbol: dict[str, list[Bar]], symbols: Sequence[str], lo: int, hi: int
) -> dict[str, list[Bar]]:
    return {s: bars_by_symbol[s][lo:hi] for s in symbols}


def _run(
    symbols: Sequence[str],
    bars: dict[str, list[Bar]],
    strategy_factory: StrategyFactory,
    params: dict[str, Any],
    portfolio_kwargs: dict[str, Any],
    execution_kwargs: dict[str, Any],
    periods_per_year: int,
):
    data = InMemoryDataHandler(symbols, bars)
    strategy = strategy_factory(data, params)
    portfolio = Portfolio(data, list(symbols), **portfolio_kwargs)
    execution = SimulatedExecutionHandler(data, **execution_kwargs)
    engine = Backtest(
        data, strategy, portfolio, execution, periods_per_year=periods_per_year
    )
    return engine.run()


def _param_combinations(param_grid: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    if not param_grid:
        return [{}]
    keys = list(param_grid)
    return [dict(zip(keys, combo)) for combo in itertools.product(*param_grid.values())]


def walk_forward(
    symbols: Sequence[str],
    bars_by_symbol: dict[str, list[Bar]],
    strategy_factory: StrategyFactory,
    param_grid: dict[str, Sequence[Any]],
    *,
    train_size: int,
    test_size: int,
    warmup: int,
    anchored: bool = False,
    portfolio_kwargs: dict[str, Any] | None = None,
    execution_kwargs: dict[str, Any] | None = None,
    periods_per_year: int = 252,
    initial_capital: float = 100_000.0,
) -> WalkForwardResult:
    """Run rolling walk-forward validation.

    Args:
        symbols: Universe traded.
        bars_by_symbol: Aligned bars (from :func:`load_aligned_bars`).
        strategy_factory: ``(data, params) -> Strategy``.
        param_grid: Mapping of param name to candidate values to grid-search.
        train_size: Bars in each training window.
        test_size: Bars in each test window.
        warmup: Bars of history prepended to each test window so the strategy is
            warmed up; these bars are excluded from OOS metrics.
        anchored: If True, the training window always starts at index 0
            (expanding window); otherwise it rolls (fixed-size window).
        portfolio_kwargs / execution_kwargs: Passed to Portfolio / execution.
        periods_per_year: Annualisation factor.
        initial_capital: Notional used to reconstruct the stitched OOS curve.

    Returns:
        A :class:`WalkForwardResult` with per-fold detail and aggregate OOS
        metrics computed from the compounded out-of-sample returns.
    """
    portfolio_kwargs = dict(portfolio_kwargs or {})
    execution_kwargs = dict(execution_kwargs or {})
    # Single source of truth for capital: caller-supplied portfolio_kwargs wins,
    # and the stitched OOS curve is rebuilt from the very same number.
    portfolio_kwargs.setdefault("initial_capital", initial_capital)
    initial_capital = float(portfolio_kwargs["initial_capital"])

    n = len(bars_by_symbol[symbols[0]])
    if train_size + test_size > n:
        raise ValueError("train_size + test_size exceeds available history")
    if warmup > train_size:
        raise ValueError("warmup cannot exceed train_size")

    combos = _param_combinations(param_grid)
    result = WalkForwardResult()
    oos_returns: list[np.ndarray] = []
    total_oos_trades = 0

    fold_idx = 0
    # The test window always marches forward by ``test_size``; only the *start*
    # of the training window is pinned when ``anchored`` (expanding window).
    test_start = train_size
    while test_start + test_size <= n:
        test_end = test_start + test_size
        train_start = 0 if anchored else test_start - train_size
        train_end = test_start

        # ---- 1) optimise on the training window ---------------------------- #
        train_bars = _slice_bars(bars_by_symbol, symbols, train_start, train_end)
        best_params, best_sharpe = {}, -np.inf
        for params in combos:
            res = _run(
                symbols, train_bars, strategy_factory, params,
                portfolio_kwargs, execution_kwargs, periods_per_year,
            )
            if res.report.sharpe > best_sharpe:
                best_sharpe, best_params = res.report.sharpe, params

        # ---- 2) evaluate out-of-sample (with a warmup prefix) ------------- #
        warmup_start = train_end - warmup
        test_bars = _slice_bars(bars_by_symbol, symbols, warmup_start, test_end)
        oos_res = _run(
            symbols, test_bars, strategy_factory, best_params,
            portfolio_kwargs, execution_kwargs, periods_per_year,
        )

        # Include the boundary return (last warmup bar -> first test bar): a
        # position opened in warmup is live, so that P&L belongs to OOS.
        boundary = max(0, warmup - 1)
        oos_equity = np.array(
            [e for _, e in oos_res.equity_curve[boundary:]], dtype=float
        )
        # Attribute only trades executed at/after the test window start.
        trades_in_warmup = (
            oos_res.trade_counts[warmup]
            if warmup < len(oos_res.trade_counts)
            else oos_res.report.num_trades
        )
        fold_oos_trades = oos_res.report.num_trades - trades_in_warmup

        if oos_equity.size >= 2:
            seg_returns = oos_equity[1:] / oos_equity[:-1] - 1.0
            oos_returns.append(seg_returns)
        # Always attribute the (warmup-excluded) trade count so the aggregate
        # invariant `aggregate.num_trades == sum(fold trades)` holds even for a
        # degenerate (sub-2-bar) OOS window.
        seg_equity = oos_equity if oos_equity.size else np.asarray([initial_capital])
        seg_report = summarize(
            seg_equity, num_trades=fold_oos_trades, periods_per_year=periods_per_year
        )
        total_oos_trades += fold_oos_trades

        result.folds.append(
            Fold(
                index=fold_idx,
                train_span=(train_start, train_end),
                test_span=(train_end, test_end),
                best_params=best_params,
                in_sample_sharpe=float(best_sharpe),
                oos_report=seg_report,
            )
        )

        fold_idx += 1
        test_start += test_size

    # ---- 3) stitch OOS returns into one compounded equity curve ----------- #
    if oos_returns:
        stitched = np.concatenate(oos_returns)
        equity = initial_capital * np.cumprod(1.0 + stitched)
        equity = np.insert(equity, 0, initial_capital)
        result.oos_equity = equity.tolist()
        result.aggregate = summarize(
            equity, num_trades=total_oos_trades, periods_per_year=periods_per_year
        )
    return result
