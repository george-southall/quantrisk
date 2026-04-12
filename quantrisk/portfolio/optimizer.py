"""
Mean-variance portfolio optimisation.

Provides:
  - min_variance_portfolio  — global minimum variance (GMV)
  - max_sharpe_portfolio    — tangency portfolio (maximum Sharpe ratio)
  - target_return_portfolio — minimum variance at a given target return
  - efficient_frontier      — parameterised sweep of the efficient frontier

All functions accept a pd.DataFrame of *daily* asset returns and return
annualised figures (×252 for returns/variances).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

TRADING_DAYS = 252


# ── Internal helpers ──────────────────────────────────────────────────────────

def _prep(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (mu_annual, cov_annual, tickers)."""
    clean = returns.dropna()
    mu = clean.mean().values * TRADING_DAYS
    cov = clean.cov().values * TRADING_DAYS
    tickers = clean.columns.tolist()
    return mu, cov, tickers


def _port_stats(
    w: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    risk_free_rate: float = 0.0,
) -> tuple[float, float, float]:
    """Return (annualised_return, annualised_vol, sharpe)."""
    ret = float(w @ mu)
    vol = float(np.sqrt(np.maximum(w @ cov @ w, 0.0)))
    sharpe = (ret - risk_free_rate) / vol if vol > 1e-10 else float("nan")
    return ret, vol, sharpe


def _base_problem(
    n: int,
    max_weight: float,
    extra_constraints: list[dict],
) -> tuple[list[tuple], list[dict]]:
    """Return (bounds, constraints) common to all problems."""
    bounds = [(0.0, float(max_weight))] * n
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}] + extra_constraints
    return bounds, constraints


# ── Public API ────────────────────────────────────────────────────────────────

def min_variance_portfolio(
    returns: pd.DataFrame,
    max_weight: float = 1.0,
) -> dict:
    """
    Global Minimum Variance (GMV) portfolio.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily asset returns.
    max_weight : float
        Maximum weight per asset (0–1).  Default 1.0 = no concentration limit.

    Returns
    -------
    dict with keys:
        weights               — {ticker: weight}
        annualised_return     — float
        annualised_volatility — float
        sharpe_ratio          — float
    """
    mu, cov, tickers = _prep(returns)
    n = len(tickers)

    bounds, constraints = _base_problem(n, max_weight, [])

    result = minimize(
        fun=lambda w: w @ cov @ w,
        x0=np.ones(n) / n,
        jac=lambda w: 2.0 * cov @ w,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1_000},
    )

    w = result.x
    total = w.sum()
    if total > 1e-10:
        w = w / total

    ret, vol, sharpe = _port_stats(w, mu, cov)
    return {
        "weights": dict(zip(tickers, w.tolist())),
        "annualised_return": ret,
        "annualised_volatility": vol,
        "sharpe_ratio": sharpe,
        "success": result.success,
    }


def max_sharpe_portfolio(
    returns: pd.DataFrame,
    risk_free_rate: float = 0.05,
    max_weight: float = 1.0,
) -> dict:
    """
    Tangency / Maximum Sharpe Ratio portfolio.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily asset returns.
    risk_free_rate : float
        Annualised risk-free rate.
    max_weight : float
        Maximum weight per asset.

    Returns
    -------
    Same structure as min_variance_portfolio.
    """
    mu, cov, tickers = _prep(returns)
    n = len(tickers)

    bounds, constraints = _base_problem(n, max_weight, [])

    def neg_sharpe(w: np.ndarray) -> float:
        port_ret = float(w @ mu)
        port_vol = float(np.sqrt(np.maximum(w @ cov @ w, 0.0)))
        if port_vol < 1e-10:
            return 0.0
        return -(port_ret - risk_free_rate) / port_vol

    result = minimize(
        fun=neg_sharpe,
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1_000},
    )

    w = result.x
    total = w.sum()
    if total > 1e-10:
        w = w / total

    ret, vol, sharpe = _port_stats(w, mu, cov, risk_free_rate)
    return {
        "weights": dict(zip(tickers, w.tolist())),
        "annualised_return": ret,
        "annualised_volatility": vol,
        "sharpe_ratio": sharpe,
        "success": result.success,
    }


def target_return_portfolio(
    returns: pd.DataFrame,
    target_return: float,
    max_weight: float = 1.0,
) -> dict | None:
    """
    Minimum-variance portfolio constrained to hit *target_return* (annualised).

    Returns None if the optimiser cannot find a feasible solution.
    """
    mu, cov, tickers = _prep(returns)
    n = len(tickers)

    return_constraint = {
        "type": "eq",
        "fun": lambda w: float(w @ mu) - target_return,
    }
    bounds, constraints = _base_problem(n, max_weight, [return_constraint])

    result = minimize(
        fun=lambda w: w @ cov @ w,
        x0=np.ones(n) / n,
        jac=lambda w: 2.0 * cov @ w,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1_000},
    )

    if not result.success:
        return None

    w = result.x
    total = w.sum()
    if total > 1e-10:
        w = w / total

    ret, vol, sharpe = _port_stats(w, mu, cov)
    return {
        "weights": dict(zip(tickers, w.tolist())),
        "annualised_return": ret,
        "annualised_volatility": vol,
        "sharpe_ratio": sharpe,
        "success": True,
    }


def efficient_frontier(
    returns: pd.DataFrame,
    n_points: int = 80,
    risk_free_rate: float = 0.05,
    max_weight: float = 1.0,
) -> pd.DataFrame:
    """
    Sweep the efficient frontier from the GMV portfolio to the maximum-return asset.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily asset returns.
    n_points : int
        Number of points on the frontier.
    risk_free_rate : float
        Annualised risk-free rate (used to compute Sharpe ratios).
    max_weight : float
        Maximum weight per asset.

    Returns
    -------
    pd.DataFrame with columns:
        annualised_return, annualised_volatility, sharpe_ratio
    Rows are sorted by increasing volatility (the upper frontier half).
    """
    mu, _, _ = _prep(returns)

    gmv = min_variance_portfolio(returns, max_weight=max_weight)
    min_ret = gmv["annualised_return"]
    max_ret = float(mu.max())

    # Ensure there is a range to sweep
    if max_ret <= min_ret:
        max_ret = min_ret + 0.005

    target_returns = np.linspace(min_ret, max_ret, n_points)

    rows: list[dict] = []
    for tr in target_returns:
        res = target_return_portfolio(returns, tr, max_weight=max_weight)
        if res is not None:
            rows.append({
                "annualised_return": res["annualised_return"],
                "annualised_volatility": res["annualised_volatility"],
                "sharpe_ratio": (
                    (res["annualised_return"] - risk_free_rate)
                    / res["annualised_volatility"]
                    if res["annualised_volatility"] > 1e-10
                    else float("nan")
                ),
            })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("annualised_volatility").reset_index(drop=True)
    return df
