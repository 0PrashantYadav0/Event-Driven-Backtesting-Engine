"""Market data handling and synthetic data generation.

The :class:`DataHandler` abstraction lets the rest of the engine stay ignorant
of *where* bars come from (CSV, a live feed, a database). During a backtest we
drip-feed one bar at a time and only ever expose data *up to and including* the
current bar. This is the single most important guarantee for avoiding
look-ahead bias, and it is enforced structurally: strategies simply cannot see
future bars because they have not been loaded yet.

This module also ships a deterministic synthetic-data generator (geometric
Brownian motion plus a cointegrated pair) so the whole project runs fully
offline with no downloads.
"""

from __future__ import annotations

import csv
import logging
import math
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .events import Event, MarketEvent

logger = logging.getLogger(__name__)


class Bar(NamedTuple):
    """A single OHLCV bar for one symbol at one timestamp."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_aligned_bars(
    csv_dir: str | Path, symbols: Sequence[str]
) -> tuple[list[datetime], dict[str, list[Bar]]]:
    """Load per-symbol CSVs and align them on their common timestamps.

    Returns a ``(timeline, bars_by_symbol)`` tuple where ``timeline`` is the
    sorted list of shared timestamps and each symbol maps to a list of
    :class:`Bar` s in that same order.
    """
    csv_dir = Path(csv_dir)
    per_symbol: dict[str, dict[datetime, Bar]] = {}
    for symbol in symbols:
        path = csv_dir / f"{symbol}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing CSV for symbol {symbol!r}: {path}")
        rows: dict[datetime, Bar] = {}
        with path.open(newline="") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                ts = datetime.fromisoformat(raw["timestamp"])
                values = {
                    f: float(raw[f]) for f in ("open", "high", "low", "close", "volume")
                }
                if not all(math.isfinite(v) for v in values.values()):
                    raise ValueError(
                        f"Non-finite value in {path} at {raw['timestamp']!r}"
                    )
                if values["close"] <= 0 or values["open"] <= 0:
                    raise ValueError(
                        f"Non-positive price in {path} at {raw['timestamp']!r}"
                    )
                rows[ts] = Bar(timestamp=ts, **values)
        per_symbol[symbol] = rows

    common = set.intersection(*(set(rows) for rows in per_symbol.values()))
    if not common:
        raise ValueError("No overlapping timestamps across the requested symbols")
    timeline = sorted(common)
    bars_by_symbol = {s: [per_symbol[s][ts] for ts in timeline] for s in symbols}
    return timeline, bars_by_symbol


class DataHandler(ABC):
    """Feeds market data to the engine one timestamp at a time.

    Concrete handlers must expose the *latest* bars seen so far and advance the
    internal cursor via :meth:`update_bars`, pushing a :class:`MarketEvent`.
    """

    symbols: list[str]
    continue_backtest: bool

    @abstractmethod
    def get_latest_bars(self, symbol: str, n: int = 1) -> list[Bar]:
        """Return the last ``n`` bars for ``symbol`` known *up to now*."""

    @abstractmethod
    def get_latest_bar_value(
        self, symbol: str, field: str = "close", n_back: int = 0
    ) -> float:
        """Return one field of a recent bar.

        ``n_back=0`` is the most recent bar, ``n_back=1`` the one before it, etc.
        """

    @abstractmethod
    def update_bars(self, events: deque[Event]) -> None:
        """Advance one timestamp and push a :class:`MarketEvent` if data remains."""


class HistoricCSVDataHandler(DataHandler):
    """Replays historic bars loaded from per-symbol CSV files.

    Each CSV must have a header ``timestamp,open,high,low,close,volume`` and be
    sorted ascending by timestamp. Symbols are aligned on the *intersection* of
    their timestamps so every :class:`MarketEvent` corresponds to a bar that
    exists for all symbols (essential for pairs trading).
    """

    def __init__(self, csv_dir: str | Path, symbols: Sequence[str]) -> None:
        self.csv_dir = Path(csv_dir)
        self.symbols = list(symbols)
        self.continue_backtest = True

        self._timeline, self._full = load_aligned_bars(csv_dir, self.symbols)
        self._cursor = 0
        self._latest: dict[str, list[Bar]] = {s: [] for s in self.symbols}
        logger.info(
            "Loaded %d aligned bars for %d symbols from %s",
            len(self._timeline),
            len(self.symbols),
            self.csv_dir,
        )

    def get_latest_bars(self, symbol: str, n: int = 1) -> list[Bar]:
        bars = self._latest.get(symbol)
        if bars is None:
            raise KeyError(f"Unknown symbol {symbol!r}")
        return bars[-n:]

    def get_latest_bar_value(
        self, symbol: str, field: str = "close", n_back: int = 0
    ) -> float:
        bars = self.get_latest_bars(symbol, n_back + 1)
        if len(bars) < n_back + 1:
            raise IndexError(f"Only {len(bars)} bars for {symbol!r}, need {n_back + 1}")
        return float(getattr(bars[0], field))

    def update_bars(self, events: deque[Event]) -> None:
        if self._cursor >= len(self._timeline):
            self.continue_backtest = False
            return
        for symbol in self.symbols:
            self._latest[symbol].append(self._full[symbol][self._cursor])
        ts = self._timeline[self._cursor]
        self._cursor += 1
        events.append(MarketEvent(ts))


class InMemoryDataHandler(DataHandler):
    """Replays pre-aligned bars held in memory.

    Used by walk-forward validation to run backtests over arbitrary slices of a
    timeline without touching disk. ``bars_by_symbol`` must already be aligned
    (equal length, same timestamps) as produced by :func:`load_aligned_bars`.
    """

    def __init__(self, symbols: Sequence[str], bars_by_symbol: dict[str, list[Bar]]) -> None:
        self.symbols = list(symbols)
        self.continue_backtest = True
        lengths = {len(bars_by_symbol[s]) for s in self.symbols}
        if len(lengths) != 1:
            raise ValueError("all symbols must have the same number of bars")
        self._full = {s: list(bars_by_symbol[s]) for s in self.symbols}
        self._n = lengths.pop()
        self._cursor = 0
        self._latest: dict[str, list[Bar]] = {s: [] for s in self.symbols}

    def get_latest_bars(self, symbol: str, n: int = 1) -> list[Bar]:
        bars = self._latest.get(symbol)
        if bars is None:
            raise KeyError(f"Unknown symbol {symbol!r}")
        return bars[-n:]

    def get_latest_bar_value(
        self, symbol: str, field: str = "close", n_back: int = 0
    ) -> float:
        bars = self.get_latest_bars(symbol, n_back + 1)
        if len(bars) < n_back + 1:
            raise IndexError(f"Only {len(bars)} bars for {symbol!r}, need {n_back + 1}")
        return float(getattr(bars[0], field))

    def update_bars(self, events: deque[Event]) -> None:
        if self._cursor >= self._n:
            self.continue_backtest = False
            return
        for symbol in self.symbols:
            self._latest[symbol].append(self._full[symbol][self._cursor])
        ts = self._full[self.symbols[0]][self._cursor].timestamp
        self._cursor += 1
        events.append(MarketEvent(ts))


# --------------------------------------------------------------------------- #
# Synthetic data generation (deterministic, fully offline)
# --------------------------------------------------------------------------- #
def _bars_from_closes(
    closes: np.ndarray,
    start: datetime,
    rng: np.random.Generator,
    freq: timedelta = timedelta(days=1),
) -> list[Bar]:
    """Turn a close-price path into plausible OHLCV bars."""
    bars: list[Bar] = []
    prev_close = float(closes[0])
    for i, close in enumerate(closes):
        close = float(close)
        open_ = prev_close
        # Intrabar wiggle proportional to price so high/low bracket open & close.
        wiggle = abs(rng.normal(0.0, 0.005)) * close
        high = max(open_, close) + wiggle
        low = min(open_, close) - wiggle
        volume = float(rng.integers(50_000, 150_000))
        bars.append(Bar(start + i * freq, open_, high, low, close, volume))
        prev_close = close
    return bars


def generate_gbm_series(
    n: int,
    *,
    s0: float = 100.0,
    mu: float = 0.08,
    sigma: float = 0.20,
    seed: int = 7,
    start: datetime | None = None,
    periods_per_year: int = 252,
) -> list[Bar]:
    """Generate a geometric-Brownian-motion price path as OHLCV bars.

    Args:
        n: Number of bars.
        s0: Initial price.
        mu: Annualised drift.
        sigma: Annualised volatility.
        seed: RNG seed for full determinism.
        start: First bar timestamp (defaults to 2015-01-01).
        periods_per_year: Trading periods per year (252 for daily bars).
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / periods_per_year
    shocks = rng.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), size=n)
    log_path = np.log(s0) + np.cumsum(shocks)
    closes = np.exp(log_path)
    return _bars_from_closes(closes, start or datetime(2015, 1, 1), rng)


def generate_cointegrated_pair(
    n: int,
    *,
    s0: float = 100.0,
    beta: float = 1.0,
    spread_sigma: float = 1.5,
    spread_kappa: float = 0.05,
    common_sigma: float = 0.20,
    seed: int = 11,
    start: datetime | None = None,
    periods_per_year: int = 252,
) -> tuple[list[Bar], list[Bar]]:
    """Generate two cointegrated price series suitable for pairs trading.

    ``y`` follows a random walk (common factor); ``x`` tracks ``beta * y`` plus
    a mean-reverting (Ornstein-Uhlenbeck) spread, so ``y - beta * x`` is
    stationary and tradable.
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / periods_per_year
    common = rng.normal(0.0, common_sigma * np.sqrt(dt), size=n)
    log_y = np.log(s0) + np.cumsum(common)

    spread = np.empty(n)
    spread[0] = 0.0
    for t in range(1, n):
        spread[t] = (
            spread[t - 1] * (1.0 - spread_kappa)
            + rng.normal(0.0, spread_sigma * np.sqrt(dt))
        )
    log_x = (log_y - np.log(beta)) - spread

    closes_y = np.exp(log_y)
    closes_x = np.exp(log_x)
    start = start or datetime(2015, 1, 1)
    bars_y = _bars_from_closes(closes_y, start, np.random.default_rng(seed + 1))
    bars_x = _bars_from_closes(closes_x, start, np.random.default_rng(seed + 2))
    return bars_x, bars_y


def write_bars_csv(path: str | Path, bars: Iterable[Bar]) -> Path:
    """Persist bars to a CSV consumable by :class:`HistoricCSVDataHandler`."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for b in bars:
            writer.writerow(
                [b.timestamp.isoformat(), b.open, b.high, b.low, b.close, b.volume]
            )
    return path


def build_sample_dataset(csv_dir: str | Path, n: int = 1000, seed: int = 7) -> Path:
    """Write a full sample dataset (2 trend symbols + 1 cointegrated pair).

    Returns the directory the CSVs were written to. Symbols produced:
    ``MOMO`` and ``TREND`` (GBM) plus ``PAIR_X`` / ``PAIR_Y`` (cointegrated).
    """
    csv_dir = Path(csv_dir)
    write_bars_csv(csv_dir / "MOMO.csv", generate_gbm_series(n, mu=0.15, sigma=0.25, seed=seed))
    write_bars_csv(csv_dir / "TREND.csv", generate_gbm_series(n, mu=0.10, sigma=0.30, seed=seed + 5))
    x, y = generate_cointegrated_pair(n, seed=seed + 100)
    write_bars_csv(csv_dir / "PAIR_X.csv", x)
    write_bars_csv(csv_dir / "PAIR_Y.csv", y)
    return csv_dir
