"""Unit tests for performance analytics, validated against hand computations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from backtester.performance import (
    annualized_volatility,
    cagr,
    equity_to_returns,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    summarize,
)


def test_equity_to_returns_basic():
    r = equity_to_returns(np.array([100.0, 110.0, 121.0]))
    assert r == pytest.approx([0.10, 0.10])


def test_equity_to_returns_rejects_non_positive():
    with pytest.raises(ValueError):
        equity_to_returns(np.array([100.0, 0.0, 50.0]))


def test_equity_to_returns_short_series():
    assert equity_to_returns(np.array([100.0])).size == 0


def test_sharpe_matches_hand_computation():
    returns = np.array([0.10, 0.20])
    # mean=0.15, std(ddof=1)=sqrt(0.005), annualised by sqrt(252)
    expected = math.sqrt(252) * 0.15 / math.sqrt(0.005)
    assert sharpe_ratio(returns) == pytest.approx(expected)


def test_sharpe_zero_when_no_variance():
    assert sharpe_ratio(np.array([0.01, 0.01, 0.01])) == 0.0


def test_sharpe_risk_free_deannualised():
    returns = np.array([0.001, 0.002, 0.0015, 0.0])
    s0 = sharpe_ratio(returns, risk_free_rate=0.0)
    s1 = sharpe_ratio(returns, risk_free_rate=0.252)
    assert s1 < s0  # subtracting a positive rate lowers Sharpe


def test_annualized_volatility():
    returns = np.array([0.01, -0.01, 0.02, -0.02])
    expected = returns.std(ddof=1) * math.sqrt(252)
    assert annualized_volatility(returns) == pytest.approx(expected)


def test_max_drawdown():
    equity = np.array([100.0, 120.0, 90.0, 150.0, 75.0])
    # worst = 150 -> 75 = -0.5
    assert max_drawdown(equity) == pytest.approx(-0.5)


def test_max_drawdown_monotonic_up_is_zero():
    assert max_drawdown(np.array([1.0, 2.0, 3.0])) == pytest.approx(0.0)


def test_cagr_one_year_doubling():
    # 2 points, 1 period, periods_per_year=1 -> exactly 1 year, doubles
    assert cagr(np.array([100.0, 200.0]), periods_per_year=1) == pytest.approx(1.0)


def test_cagr_rejects_non_positive_endpoints():
    with pytest.raises(ValueError):
        cagr(np.array([-1.0, 100.0]))


def test_hit_rate_ignores_zeros():
    returns = np.array([0.1, -0.2, 0.0, 0.3, -0.1])
    assert hit_rate(returns) == pytest.approx(0.5)


def test_hit_rate_all_zero():
    assert hit_rate(np.array([0.0, 0.0])) == 0.0


def test_summarize_consistency():
    equity = np.array([100.0, 110.0, 105.0, 120.0])
    report = summarize(equity, num_trades=3)
    assert report.total_return == pytest.approx(0.20)
    assert report.num_periods == 3
    assert report.num_trades == 3
    assert report.max_drawdown <= 0
    # pretty() must not blow up
    assert "Sharpe" in report.pretty()
