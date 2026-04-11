"""
Portfolio construction strategies for backtesting.

Each strategy is a callable that accepts a window of historical returns
and returns a weight dictionary {ticker: weight}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS_PER_YEAR = 252


def equal_weight(returns: pd.DataFrame) -> dict[str, float]:
    """Equally weight all assets. Baseline benchmark strategy."""
    n = len(returns.columns)
    return {t: 1.0 / n for t in returns.columns}


def inverse_volatility(returns: pd.DataFrame) -> dict[str, float]:
    """
    Weight each asset by the inverse of its historical volatility.

    Simpler variant of risk parity — assets with lower vol receive higher weight.
    """
    vols = returns.std()
    vols = vols.replace(0, np.nan).dropna()
    inv_vol = 1.0 / vols
    total = inv_vol.sum()
    return {t: float(inv_vol[t] / total) for t in vols.index}


def risk_parity(returns: pd.DataFrame) -> dict[str, float]:
    """
    Risk Parity: weight assets so each contributes equally to portfolio variance.

    Uses numerical optimisation to equalise risk contributions.
    """
    cov = returns.cov().values
    n = len(returns.columns)
    tickers = returns.columns.tolist()

    def risk_contributions(w):
        port_var = w @ cov @ w
        marginal = cov @ w
        return w * marginal / port_var if port_var > 0 else np.zeros(n)

    def objective(w):
        rc = risk_contributions(w)
        target = np.ones(n) / n
        return np.sum((rc - target) ** 2)

    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(1e-4, 1.0)] * n

    result = minimize(
        objective, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    w = result.x / result.x.sum()
    return {t: float(v) for t, v in zip(tickers, w)}


def minimum_variance(returns: pd.DataFrame) -> dict[str, float]:
    """
    Minimum Variance: long-only constrained mean-variance optimisation.
    Minimises portfolio variance regardless of expected return.
    """
    cov = returns.cov().values
    n = len(returns.columns)
    tickers = returns.columns.tolist()

    def port_variance(w):
        return w @ cov @ w

    def port_variance_grad(w):
        return 2 * cov @ w

    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, 1.0)] * n

    result = minimize(
        port_variance, w0,
        jac=port_variance_grad,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    w = result.x / result.x.sum()
    return {t: float(v) for t, v in zip(tickers, w)}


def maximum_sharpe(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.05,
) -> dict[str, float]:
    """
    Maximum Sharpe: maximise (Rp - Rf) / σp via scipy.optimize.

    Uses annualised estimates.
    """
    mu = returns.mean().values * TRADING_DAYS_PER_YEAR
    cov = returns.cov().values * TRADING_DAYS_PER_YEAR
    n = len(returns.columns)
    tickers = returns.columns.tolist()
    rf_daily = risk_free_rate

    def neg_sharpe(w):
        port_ret = w @ mu
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-10:
            return 0.0
        return -(port_ret - rf_daily) / port_vol

    w0 = np.ones(n) / n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(0.0, 1.0)] * n

    result = minimize(
        neg_sharpe, w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    w = result.x / result.x.sum()
    return {t: float(v) for t, v in zip(tickers, w)}


def momentum(
    returns: pd.DataFrame,
    lookback: int = 252,
    skip: int = 21,
    top_n: int | None = None,
) -> dict[str, float]:
    """
    12-1 Momentum: rank assets by return from t-lookback to t-skip.
    Go long the top quintile (or top_n assets) with equal weight.
    """
    if len(returns) < lookback:
        return equal_weight(returns)

    # 12-1 month return: skip most recent month to avoid short-term reversal
    signal_returns = returns.iloc[-(lookback):-skip]
    scores = (1 + signal_returns).prod() - 1
    scores = scores.sort_values(ascending=False)

    n = top_n or max(1, len(scores) // 5)  # top quintile
    top = scores.head(n).index.tolist()
    w = 1.0 / len(top)
    return {t: (w if t in top else 0.0) for t in returns.columns}


# Registry mapping strategy name → callable
STRATEGY_REGISTRY: dict[str, callable] = {
    "equal_weight": equal_weight,
    "inverse_volatility": inverse_volatility,
    "risk_parity": risk_parity,
    "minimum_variance": minimum_variance,
    "maximum_sharpe": maximum_sharpe,
    "momentum": momentum,
}
