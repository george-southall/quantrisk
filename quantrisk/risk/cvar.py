"""Conditional VaR (Expected Shortfall) implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from quantrisk.risk.var import historical_var, parametric_var
from quantrisk.utils.logger import get_logger

logger = get_logger(__name__)


def historical_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
) -> float:
    """
    Historical CVaR (Expected Shortfall).

    The average of all losses that exceed the VaR threshold — i.e. the
    expected loss given that a VaR breach has occurred.

    Returns a positive number representing the expected tail loss.
    """
    clean = returns.dropna()
    percentile = (1 - confidence) * 100
    threshold = np.percentile(clean, percentile)
    tail = clean[clean <= threshold]
    if len(tail) == 0:
        return float("nan")
    daily_cvar = float(-tail.mean())
    return daily_cvar * np.sqrt(horizon)


def parametric_cvar(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
    distribution: str = "normal",
) -> float:
    """
    Parametric CVaR / Expected Shortfall.

    For normal distribution: ES = μ - σ * φ(z) / (1 - c)
    where φ is the standard normal PDF evaluated at the VaR z-score.

    For Student-t: computed numerically.

    Returns a positive number.
    """
    clean = returns.dropna()
    mu = clean.mean()
    sigma = clean.std(ddof=1)

    if distribution == "normal":
        z = stats.norm.ppf(1 - confidence)
        # E[X | X < VaR] = μ - σ * φ(z) / (1 - c)
        es = -(mu - sigma * stats.norm.pdf(z) / (1 - confidence))
    elif distribution == "t":
        df, loc, scale = stats.t.fit(clean)
        q = stats.t.ppf(1 - confidence, df=df, loc=loc, scale=scale)
        # Numerical integral of tail expectation
        tail_prob = stats.t.cdf(q, df=df, loc=loc, scale=scale)
        # E[X | X < q] = integral numerically via scipy
        from scipy.integrate import quad
        def integrand(x):
            return x * stats.t.pdf(x, df=df, loc=loc, scale=scale)
        integral, _ = quad(integrand, -np.inf, q)
        if tail_prob > 0:
            es = -(integral / tail_prob)
        else:
            es = float("nan")
    else:
        raise ValueError(f"Unknown distribution: {distribution!r}")

    return float(es * np.sqrt(horizon))


def monte_carlo_cvar(
    simulated_losses: np.ndarray,
    confidence: float = 0.95,
) -> float:
    """
    CVaR from a pre-simulated array of portfolio losses (positive = loss).

    Parameters
    ----------
    simulated_losses : np.ndarray
        Array of simulated portfolio P&L (NOT yet negated). Pass returns, not losses.
    """
    percentile = (1 - confidence) * 100
    threshold = np.percentile(simulated_losses, percentile)
    tail = simulated_losses[simulated_losses <= threshold]
    if len(tail) == 0:
        return float("nan")
    return float(-tail.mean())


def cvar_summary(
    returns: pd.Series,
    confidence_levels: list[float] | None = None,
    horizon: int = 1,
) -> pd.DataFrame:
    """Return a comparison DataFrame of CVaR across methods and confidence levels."""
    conf_levels = confidence_levels or [0.95, 0.99]
    rows = []
    for conf in conf_levels:
        row = {"confidence": conf, "horizon_days": horizon}
        try:
            row["var_historical"] = historical_var(returns, conf, horizon)
            row["cvar_historical"] = historical_cvar(returns, conf, horizon)
        except Exception:
            row["var_historical"] = row["cvar_historical"] = float("nan")
        try:
            row["var_parametric"] = parametric_var(returns, conf, horizon)
            row["cvar_parametric"] = parametric_cvar(returns, conf, horizon)
        except Exception:
            row["var_parametric"] = row["cvar_parametric"] = float("nan")
        try:
            row["var_parametric_t"] = parametric_var(returns, conf, horizon, "t")
            row["cvar_parametric_t"] = parametric_cvar(returns, conf, horizon, "t")
        except Exception:
            row["var_parametric_t"] = row["cvar_parametric_t"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)
