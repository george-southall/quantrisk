"""Return calculation utilities: simple, log, excess, rolling stats."""

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def simple_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Compute period-over-period simple (arithmetic) returns."""
    return prices.pct_change()


def log_returns(prices: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Compute period-over-period log (continuously compounded) returns."""
    return np.log(prices / prices.shift(1))


def cumulative_returns(returns: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """
    Compute cumulative wealth index from simple returns.
    Returns a series/dataframe starting at 1.0.
    """
    return (1 + returns).cumprod()


def total_return(returns: pd.Series) -> float:
    """Total compounded return over the full period."""
    return float((1 + returns).prod() - 1)


def annualised_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Geometrically annualised return."""
    n = len(returns.dropna())
    if n == 0:
        return float("nan")
    compound = (1 + returns.dropna()).prod()
    return float(compound ** (periods_per_year / n) - 1)


def annualised_volatility(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> float:
    """Annualised standard deviation of returns."""
    return float(returns.std() * np.sqrt(periods_per_year))


def excess_returns(
    portfolio_returns: pd.Series,
    risk_free_rate: pd.Series | float,
) -> pd.Series:
    """
    Compute excess returns over the risk-free rate.

    If risk_free_rate is a Series it is resampled/aligned to match portfolio_returns.
    If it is a float it is assumed to be an annualised rate and divided by
    TRADING_DAYS_PER_YEAR to get the daily rate.
    """
    if isinstance(risk_free_rate, (int, float)):
        daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
        return portfolio_returns - daily_rf

    # Align series
    rf_daily = risk_free_rate.reindex(portfolio_returns.index, method="ffill")
    # If RF is annual percentage, convert to daily decimal
    if rf_daily.mean() > 0.2:  # looks like percentage rather than decimal annual
        rf_daily = rf_daily / 100
    rf_daily = rf_daily / TRADING_DAYS_PER_YEAR
    return portfolio_returns - rf_daily


def rolling_annualised_return(
    returns: pd.Series, window: int = 252, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> pd.Series:
    """Rolling geometrically annualised return."""
    return (
        (1 + returns)
        .rolling(window)
        .apply(lambda x: (x.prod() ** (periods_per_year / window)) - 1, raw=True)
    )


def rolling_annualised_volatility(
    returns: pd.Series, window: int = 21, periods_per_year: int = TRADING_DAYS_PER_YEAR
) -> pd.Series:
    """Rolling annualised volatility."""
    return returns.rolling(window).std() * np.sqrt(periods_per_year)


def downside_deviation(
    returns: pd.Series,
    threshold: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised downside deviation below threshold."""
    downside = returns[returns < threshold]
    if len(downside) == 0:
        return 0.0
    return float(downside.std() * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown as a negative fraction."""
    wealth = cumulative_returns(returns.dropna())
    peak = wealth.cummax()
    dd = (wealth - peak) / peak
    return float(dd.min())


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Full time-series of drawdown from running peak."""
    wealth = cumulative_returns(returns.dropna())
    peak = wealth.cummax()
    return (wealth - peak) / peak


def max_drawdown_duration(returns: pd.Series) -> int:
    """Number of periods (days) for the longest drawdown recovery."""
    dd = drawdown_series(returns)
    in_drawdown = dd < 0
    duration = 0
    current = 0
    for val in in_drawdown:
        if val:
            current += 1
            duration = max(duration, current)
        else:
            current = 0
    return duration
