"""Value at Risk (VaR) implementations: Historical, Parametric, Monte Carlo."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from quantrisk.config import settings
from quantrisk.utils.logger import get_logger

logger = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252


def historical_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
) -> float:
    """
    Historical (non-parametric) VaR.

    Sorts the empirical return distribution and reads off the loss at the
    given confidence percentile. No distributional assumption.

    Returns a positive number representing the loss (e.g. 0.02 = 2% loss).
    """
    if len(returns.dropna()) < 30:
        raise ValueError("Insufficient data for historical VaR (need ≥ 30 observations)")

    clean = returns.dropna()
    percentile = (1 - confidence) * 100
    daily_var = float(-np.percentile(clean, percentile))

    # Scale to horizon using square-root-of-time rule
    return daily_var * np.sqrt(horizon)


def parametric_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon: int = 1,
    distribution: str = "normal",
) -> float:
    """
    Parametric VaR.

    Fits the specified distribution to returns and computes VaR analytically.

    Parameters
    ----------
    distribution : str
        'normal' or 't' (Student-t, which captures fat tails better).

    Returns a positive number representing the loss.
    """
    clean = returns.dropna()
    mu = clean.mean()
    sigma = clean.std(ddof=1)

    if distribution == "normal":
        z = stats.norm.ppf(1 - confidence)
        daily_var = -(mu + z * sigma)
    elif distribution == "t":
        df, loc, scale = stats.t.fit(clean)
        q = stats.t.ppf(1 - confidence, df=df, loc=loc, scale=scale)
        daily_var = -q
    else:
        raise ValueError(f"Unknown distribution: {distribution!r}. Use 'normal' or 't'.")

    return float(daily_var * np.sqrt(horizon))


def monte_carlo_var(
    returns: pd.DataFrame,
    weights: dict[str, float] | None = None,
    confidence: float = 0.95,
    horizon: int = 1,
    n_simulations: int | None = None,
    random_seed: int | None = 42,
) -> float:
    """
    Monte Carlo VaR using Cholesky decomposition to preserve correlations.

    Simulates correlated asset returns using a multivariate normal, then
    aggregates to portfolio level using the given weights.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset return time series (columns = tickers).
    weights : dict[str, float] | None
        Portfolio weights. Equal-weighted if None.
    n_simulations : int | None
        Number of simulated paths. Defaults to settings.mc_num_simulations.

    Returns a positive number representing the portfolio loss.
    """
    n_sims = n_simulations or settings.mc_num_simulations

    clean = returns.dropna()
    tickers = clean.columns.tolist()

    if weights is None:
        w = np.ones(len(tickers)) / len(tickers)
    else:
        w = np.array([weights.get(t, 0.0) for t in tickers])
        w = w / w.sum()

    mu = clean.mean().values
    cov = clean.cov().values

    # Cholesky decomposition of covariance matrix
    # Add small regularisation for numerical stability
    cov_reg = cov + np.eye(len(tickers)) * 1e-8
    try:
        L = np.linalg.cholesky(cov_reg)
    except np.linalg.LinAlgError:
        logger.warning("Cholesky failed, falling back to eigenvalue decomposition")
        eigvals, eigvecs = np.linalg.eigh(cov_reg)
        eigvals = np.maximum(eigvals, 0)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    rng = np.random.default_rng(random_seed)

    # Simulate: Z ~ N(0, I), correlated = mu + L @ Z
    Z = rng.standard_normal((n_sims, len(tickers)))
    sim_returns = mu + Z @ L.T  # shape (n_sims, n_assets)

    # Scale to horizon
    if horizon > 1:
        # Simulate multi-day paths by repeating and compounding
        path_returns = np.zeros(n_sims)
        for _ in range(horizon):
            Z_h = rng.standard_normal((n_sims, len(tickers)))
            day_returns = mu + Z_h @ L.T
            path_returns += day_returns @ w
        port_sim = path_returns
    else:
        port_sim = sim_returns @ w  # shape (n_sims,)

    percentile = (1 - confidence) * 100
    var = float(-np.percentile(port_sim, percentile))
    return var


def rolling_var(
    returns: pd.Series,
    window: int = 252,
    confidence: float = 0.95,
    method: str = "historical",
) -> pd.Series:
    """
    Compute rolling VaR over time using a fixed lookback window.

    Parameters
    ----------
    method : str
        'historical' or 'parametric' (normal).
    """
    results = {}
    clean = returns.dropna()

    for i in range(window, len(clean) + 1):
        window_returns = clean.iloc[i - window: i]
        dt = clean.index[i - 1]
        try:
            if method == "historical":
                results[dt] = historical_var(window_returns, confidence)
            else:
                results[dt] = parametric_var(window_returns, confidence)
        except ValueError:
            results[dt] = float("nan")

    return pd.Series(results, name=f"rolling_var_{confidence:.0%}_{method}")


def var_summary(
    returns: pd.Series,
    confidence_levels: list[float] | None = None,
    horizons: list[int] | None = None,
    asset_returns: pd.DataFrame | None = None,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Build a comparison table of all three VaR methods at multiple
    confidence levels and horizons.
    """
    conf_levels = confidence_levels or settings.var_confidence_levels
    hor_list = horizons or [1, 10]

    rows = []
    for conf in conf_levels:
        for h in hor_list:
            row = {"confidence": conf, "horizon_days": h}
            try:
                row["historical"] = historical_var(returns, conf, h)
            except Exception:
                row["historical"] = float("nan")
            try:
                row["parametric_normal"] = parametric_var(returns, conf, h, "normal")
            except Exception:
                row["parametric_normal"] = float("nan")
            try:
                row["parametric_t"] = parametric_var(returns, conf, h, "t")
            except Exception:
                row["parametric_t"] = float("nan")
            if asset_returns is not None:
                try:
                    row["monte_carlo"] = monte_carlo_var(
                        asset_returns, weights, conf, h
                    )
                except Exception:
                    row["monte_carlo"] = float("nan")
            rows.append(row)

    return pd.DataFrame(rows)
