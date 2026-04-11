"""Full Monte Carlo simulation engine for portfolio paths."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantrisk.config import settings
from quantrisk.utils.logger import get_logger

logger = get_logger(__name__)


def simulate_portfolio_paths(
    asset_returns: pd.DataFrame,
    weights: dict[str, float],
    horizon: int | None = None,
    n_simulations: int | None = None,
    random_seed: int | None = 42,
) -> np.ndarray:
    """
    Simulate correlated portfolio return paths using Cholesky decomposition.

    Parameters
    ----------
    asset_returns : pd.DataFrame
        Historical daily returns used to estimate mu and covariance.
    weights : dict[str, float]
        Portfolio weights.
    horizon : int
        Number of trading days to simulate (default 252).
    n_simulations : int
        Number of independent paths (default from settings).

    Returns
    -------
    np.ndarray of shape (n_simulations, horizon)
        Simulated cumulative portfolio wealth paths (starting at 1.0).
    """
    horizon = horizon or settings.mc_horizon_days
    n_sims = n_simulations or settings.mc_num_simulations

    tickers = [t for t in weights if t in asset_returns.columns]
    clean = asset_returns[tickers].dropna()
    w = np.array([weights[t] for t in tickers])
    w = w / w.sum()

    mu = clean.mean().values
    cov = clean.cov().values
    n_assets = len(tickers)

    # Cholesky with regularisation
    cov_reg = cov + np.eye(n_assets) * 1e-8
    try:
        L = np.linalg.cholesky(cov_reg)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(cov_reg)
        eigvals = np.maximum(eigvals, 0)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    rng = np.random.default_rng(random_seed)

    # Simulate all paths at once: (n_sims, horizon, n_assets)
    Z = rng.standard_normal((n_sims, horizon, n_assets))
    # Apply correlation: each day's Z is correlated via L
    correlated = mu + Z @ L.T  # (n_sims, horizon, n_assets)

    # Portfolio daily returns
    port_daily = correlated @ w  # (n_sims, horizon)

    # Cumulative wealth paths
    wealth = np.cumprod(1 + port_daily, axis=1)  # (n_sims, horizon)

    logger.info(
        "MC simulation: %d paths × %d days, %d assets", n_sims, horizon, n_assets
    )
    return wealth


def mc_var_cvar(
    paths: np.ndarray,
    confidence: float = 0.95,
) -> dict[str, float]:
    """
    Compute VaR and CVaR from simulated terminal wealth.

    Parameters
    ----------
    paths : np.ndarray shape (n_sims, horizon)
        Wealth paths starting at 1.0.
    confidence : float
        Confidence level.
    """
    terminal_returns = paths[:, -1] - 1.0  # terminal P&L relative to start
    percentile = (1 - confidence) * 100
    var_threshold = np.percentile(terminal_returns, percentile)
    cvar = terminal_returns[terminal_returns <= var_threshold].mean()
    return {
        "var": float(-var_threshold),
        "cvar": float(-cvar),
        "prob_loss": float((terminal_returns < 0).mean()),
        "prob_ruin_10pct": float((terminal_returns < -0.10).mean()),
        "prob_ruin_20pct": float((terminal_returns < -0.20).mean()),
        "median_return": float(np.median(terminal_returns)),
        "mean_return": float(terminal_returns.mean()),
        "p5_return": float(np.percentile(terminal_returns, 5)),
        "p95_return": float(np.percentile(terminal_returns, 95)),
    }


def mc_summary(
    asset_returns: pd.DataFrame,
    weights: dict[str, float],
    horizon: int | None = None,
    n_simulations: int | None = None,
    confidence_levels: list[float] | None = None,
    random_seed: int | None = 42,
) -> dict:
    """
    Run a full MC simulation and return a comprehensive summary dict.
    """
    conf_levels = confidence_levels or settings.var_confidence_levels
    paths = simulate_portfolio_paths(
        asset_returns, weights, horizon, n_simulations, random_seed
    )
    result = {"paths": paths}
    for conf in conf_levels:
        result[f"metrics_{conf:.0%}"] = mc_var_cvar(paths, conf)
    return result
