"""Performance analytics for an equity curve.

All functions operate on a *returns* series (period-over-period simple returns)
or an *equity* series. Annualisation uses ``periods_per_year`` (252 for daily
bars). The math here is deliberately explicit and unit-tested against
hand-computed values in ``tests/test_performance.py``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

TRADING_DAYS = 252


@dataclass(frozen=True, slots=True)
class PerformanceReport:
    """Summary statistics for a strategy run."""

    total_return: float
    cagr: float
    ann_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    max_drawdown: float
    hit_rate: float
    num_periods: int
    num_trades: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def pretty(self) -> str:
        return (
            f"  Total return   : {self.total_return:>10.2%}\n"
            f"  CAGR           : {self.cagr:>10.2%}\n"
            f"  Ann. volatility: {self.ann_volatility:>10.2%}\n"
            f"  Sharpe (ann.)  : {self.sharpe:>10.2f}\n"
            f"  Sortino (ann.) : {self.sortino:>10.2f}\n"
            f"  Calmar         : {self.calmar:>10.2f}\n"
            f"  Max drawdown   : {self.max_drawdown:>10.2%}\n"
            f"  Hit rate       : {self.hit_rate:>10.2%}\n"
            f"  Periods        : {self.num_periods:>10d}\n"
            f"  Trades         : {self.num_trades:>10d}"
        )


def equity_to_returns(equity: np.ndarray) -> np.ndarray:
    """Convert an equity curve to simple period returns.

    Raises:
        ValueError: If any equity value is non-positive (returns undefined).
    """
    equity = np.asarray(equity, dtype=float)
    if equity.ndim != 1:
        raise ValueError("equity must be 1-D")
    if not np.all(np.isfinite(equity)):
        raise ValueError("equity curve contains NaN or infinite values")
    if equity.size < 2:
        return np.empty(0, dtype=float)
    if np.any(equity <= 0):
        raise ValueError("equity curve contains non-positive values")
    return equity[1:] / equity[:-1] - 1.0


def sharpe_ratio(
    returns: np.ndarray,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised Sharpe ratio.

    ``risk_free_rate`` is an *annual* rate; it is de-annualised to per-period
    before subtracting. Returns 0.0 when volatility is zero (undefined Sharpe).
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period
    std = excess.std(ddof=1)
    if std == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / std)


def annualized_volatility(
    returns: np.ndarray, *, periods_per_year: int = TRADING_DAYS
) -> float:
    """Annualised standard deviation of returns."""
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return 0.0
    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: np.ndarray,
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> float:
    """Annualised Sortino ratio (Sharpe using downside deviation only).

    Returns 0.0 when there is no downside volatility (undefined).
    """
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return 0.0
    rf_per_period = risk_free_rate / periods_per_year
    excess = returns - rf_per_period
    downside = np.minimum(excess, 0.0)
    downside_dev = np.sqrt(np.mean(downside**2))
    if downside_dev == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / downside_dev)


def calmar_ratio(
    equity: np.ndarray, *, periods_per_year: int = TRADING_DAYS
) -> float:
    """CAGR divided by the absolute max drawdown. 0.0 if there is no drawdown."""
    mdd = max_drawdown(equity)
    if mdd == 0:
        return 0.0
    return float(cagr(equity, periods_per_year=periods_per_year) / abs(mdd))


def max_drawdown(equity: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown as a *negative* fraction.

    A result of ``-0.20`` means the worst decline from a running peak was 20%.
    """
    equity = np.asarray(equity, dtype=float)
    if equity.size == 0:
        return 0.0
    running_peak = np.maximum.accumulate(equity)
    drawdowns = equity / running_peak - 1.0
    return float(drawdowns.min())


def cagr(equity: np.ndarray, *, periods_per_year: int = TRADING_DAYS) -> float:
    """Compound annual growth rate implied by an equity curve."""
    equity = np.asarray(equity, dtype=float)
    if equity.size < 2:
        return 0.0
    if equity[0] <= 0 or equity[-1] <= 0:
        raise ValueError("equity endpoints must be positive to compute CAGR")
    total_growth = equity[-1] / equity[0]
    years = (equity.size - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return float(total_growth ** (1.0 / years) - 1.0)


def hit_rate(returns: np.ndarray) -> float:
    """Fraction of non-zero periods that were positive."""
    returns = np.asarray(returns, dtype=float)
    active = returns[returns != 0.0]
    if active.size == 0:
        return 0.0
    return float((active > 0).sum() / active.size)


def summarize(
    equity: np.ndarray,
    *,
    num_trades: int = 0,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS,
) -> PerformanceReport:
    """Compute a full :class:`PerformanceReport` from an equity curve."""
    equity = np.asarray(equity, dtype=float)
    returns = equity_to_returns(equity)
    total_return = 0.0 if equity.size < 2 else float(equity[-1] / equity[0] - 1.0)
    return PerformanceReport(
        total_return=total_return,
        cagr=cagr(equity, periods_per_year=periods_per_year),
        ann_volatility=annualized_volatility(returns, periods_per_year=periods_per_year),
        sharpe=sharpe_ratio(
            returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        ),
        sortino=sortino_ratio(
            returns, risk_free_rate=risk_free_rate, periods_per_year=periods_per_year
        ),
        calmar=calmar_ratio(equity, periods_per_year=periods_per_year),
        max_drawdown=max_drawdown(equity),
        hit_rate=hit_rate(returns),
        num_periods=int(returns.size),
        num_trades=int(num_trades),
    )
