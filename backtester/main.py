"""Command-line interface and end-to-end demo for the backtester.

Run ``python -m backtester.main`` (or the installed ``backtester`` script) to
generate a deterministic synthetic dataset and run both strategies plus
walk-forward validation, printing full performance reports.
"""

from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

import numpy as np

from .backtest import Backtest
from .data import (
    HistoricCSVDataHandler,
    InMemoryDataHandler,
    build_sample_dataset,
    load_aligned_bars,
)
from .execution import SimulatedExecutionHandler
from .portfolio import Portfolio
from .strategy import MeanReversionPairsStrategy, MomentumStrategy
from .walkforward import walk_forward

logger = logging.getLogger(__name__)

EXEC_KWARGS = {"commission_bps": 1.0, "min_commission": 1.0, "slippage_bps": 2.0}


def _print_header(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def run_momentum(csv_dir: Path) -> None:
    symbols = ["MOMO", "TREND"]
    data = HistoricCSVDataHandler(csv_dir, symbols)
    strategy = MomentumStrategy(data, symbols, fast_window=20, slow_window=60)
    portfolio = Portfolio(data, symbols, initial_capital=100_000.0, risk_per_position=0.5)
    execution = SimulatedExecutionHandler(data, **EXEC_KWARGS)
    result = Backtest(data, strategy, portfolio, execution).run()
    _print_header("MOMENTUM (dual-MA trend following)")
    print(result.report.pretty())
    print(f"  Throughput     : {result.events_per_second:>10,.0f} events/s")


def run_pairs(csv_dir: Path) -> None:
    symbols = ["PAIR_X", "PAIR_Y"]
    data = HistoricCSVDataHandler(csv_dir, symbols)
    strategy = MeanReversionPairsStrategy(
        data, "PAIR_X", "PAIR_Y", lookback=60, entry_z=2.0, exit_z=0.5
    )
    portfolio = Portfolio(data, symbols, initial_capital=100_000.0, risk_per_position=0.5)
    execution = SimulatedExecutionHandler(data, **EXEC_KWARGS)
    result = Backtest(data, strategy, portfolio, execution).run()
    _print_header("MEAN-REVERSION PAIRS (z-score)")
    print(result.report.pretty())
    print(f"  Throughput     : {result.events_per_second:>10,.0f} events/s")


def run_walkforward(csv_dir: Path) -> None:
    _, momo_bars = load_aligned_bars(csv_dir, ["MOMO", "TREND"])
    momo_wf = walk_forward(
        ["MOMO", "TREND"],
        momo_bars,
        lambda data, p: MomentumStrategy(
            data, ["MOMO", "TREND"], fast_window=p["fast"], slow_window=p["slow"]
        ),
        {"fast": [10, 20, 30], "slow": [60, 100]},
        train_size=400,
        test_size=150,
        warmup=100,
        portfolio_kwargs={"risk_per_position": 0.5},
        execution_kwargs=EXEC_KWARGS,
    )
    _print_header("WALK-FORWARD: Momentum")
    print(momo_wf.pretty())

    _, pair_bars = load_aligned_bars(csv_dir, ["PAIR_X", "PAIR_Y"])
    pair_wf = walk_forward(
        ["PAIR_X", "PAIR_Y"],
        pair_bars,
        lambda data, p: MeanReversionPairsStrategy(
            data, "PAIR_X", "PAIR_Y", lookback=p["lookback"], entry_z=p["entry"], exit_z=0.5
        ),
        {"lookback": [40, 60, 90], "entry": [1.5, 2.0, 2.5]},
        train_size=400,
        test_size=150,
        warmup=100,
        portfolio_kwargs={"risk_per_position": 0.5},
        execution_kwargs=EXEC_KWARGS,
    )
    _print_header("WALK-FORWARD: Mean-Reversion Pairs")
    print(pair_wf.pretty())


def run_benchmark(csv_dir: Path, repeats: int = 3) -> None:
    symbols = ["MOMO", "TREND"]
    _, bars = load_aligned_bars(csv_dir, symbols)
    best = 0.0
    for _ in range(repeats):
        data = InMemoryDataHandler(symbols, bars)
        strategy = MomentumStrategy(data, symbols)
        portfolio = Portfolio(data, symbols, risk_per_position=0.5)
        execution = SimulatedExecutionHandler(data, **EXEC_KWARGS)
        result = Backtest(data, strategy, portfolio, execution).run()
        best = max(best, result.events_per_second)
    _print_header("BENCHMARK")
    print(f"  Peak throughput: {best:>12,.0f} events/s over {repeats} runs")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Event-driven backtesting engine")
    parser.add_argument(
        "command",
        nargs="?",
        default="demo",
        choices=["demo", "momentum", "pairs", "walkforward", "bench"],
    )
    parser.add_argument("--bars", type=int, default=1000, help="synthetic bars to generate")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--data-dir", default=None, help="directory for CSV data")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    np.seterr(all="raise")  # surface numerical issues loudly

    tmp = None
    if args.data_dir is None:
        tmp = tempfile.TemporaryDirectory(prefix="bt_data_")
        csv_dir = Path(tmp.name)
    else:
        csv_dir = Path(args.data_dir)
    build_sample_dataset(csv_dir, n=args.bars, seed=args.seed)

    try:
        if args.command in ("demo", "momentum"):
            run_momentum(csv_dir)
        if args.command in ("demo", "pairs"):
            run_pairs(csv_dir)
        if args.command in ("demo", "walkforward"):
            run_walkforward(csv_dir)
        if args.command in ("demo", "bench"):
            run_benchmark(csv_dir)
    finally:
        if tmp is not None:
            tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
