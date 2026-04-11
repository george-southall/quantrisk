"""FRED macroeconomic data fetcher (interest rates, VIX, inflation, etc.)."""

from datetime import date

import pandas as pd

from quantrisk.config import settings
from quantrisk.utils.logger import get_logger

logger = get_logger(__name__)

# FRED series IDs for commonly used macro variables
FRED_SERIES = {
    "fed_funds_rate": "FEDFUNDS",          # Effective Federal Funds Rate (monthly)
    "treasury_3m": "TB3MS",                 # 3-Month Treasury Bill Rate
    "treasury_10y": "GS10",                 # 10-Year Treasury Constant Maturity
    "treasury_2y": "GS2",                   # 2-Year Treasury Constant Maturity
    "vix": "VIXCLS",                        # CBOE Volatility Index (daily)
    "cpi": "CPIAUCSL",                      # Consumer Price Index (monthly)
    "unemployment": "UNRATE",               # Unemployment Rate (monthly)
    "term_spread": "T10Y2Y",               # 10Y-2Y Treasury Spread
    "credit_spread": "BAA10Y",             # Moody's Baa - 10Y Treasury Spread
    "real_gdp_growth": "A191RL1Q225SBEA",  # Real GDP growth (quarterly)
}


def fetch_fred_series(
    series_id: str,
    start: str,
    end: str | None = None,
    frequency: str | None = None,
) -> pd.Series:
    """
    Fetch a single FRED series by ID.

    Args:
        series_id: FRED series identifier (e.g. 'FEDFUNDS').
        start: Start date string 'YYYY-MM-DD'.
        end: End date string (defaults to today).
        frequency: Optionally resample — 'd', 'W', 'ME', 'QE', 'YE'.

    Returns:
        pd.Series indexed by date, named by series_id.
    """
    if end is None:
        end = date.today().isoformat()

    if not settings.fred_api_key:
        logger.warning(
            "FRED_API_KEY not set. Returning empty series for %s. "
            "Set it in .env to enable macro data.",
            series_id,
        )
        return pd.Series(name=series_id, dtype=float)

    try:
        from fredapi import Fred
        fred = Fred(api_key=settings.fred_api_key)
        series = fred.get_series(series_id, observation_start=start, observation_end=end)
        series.name = series_id
        series.index = pd.to_datetime(series.index)

        if frequency:
            series = series.resample(frequency).last().dropna()

        logger.info("Fetched FRED series %s (%d observations)", series_id, len(series))
        return series

    except Exception as exc:
        logger.error("Failed to fetch FRED series %s: %s", series_id, exc)
        return pd.Series(name=series_id, dtype=float)


def fetch_risk_free_rate(start: str, end: str | None = None) -> pd.Series:
    """
    Fetch the daily risk-free rate (3-Month T-Bill, annualised %).
    Falls back to the configured constant rate if FRED is unavailable.
    """
    series = fetch_fred_series("TB3MS", start, end, frequency="ME")
    if series.empty:
        logger.warning(
            "Using fallback risk-free rate of %.2f%%", settings.risk_free_rate_fallback * 100
        )
        return pd.Series(name="risk_free_rate", dtype=float)
    series.name = "risk_free_rate"
    return series / 100  # convert percentage to decimal


def fetch_vix(start: str, end: str | None = None) -> pd.Series:
    """Fetch daily VIX closing levels."""
    series = fetch_fred_series("VIXCLS", start, end)
    series.name = "vix"
    return series


def fetch_macro_panel(start: str, end: str | None = None) -> pd.DataFrame:
    """
    Fetch a panel of key macro indicators and align them into a single DataFrame.
    Monthly/quarterly series are forward-filled to daily frequency.
    """
    if end is None:
        end = date.today().isoformat()

    panel = {}
    for name, series_id in FRED_SERIES.items():
        s = fetch_fred_series(series_id, start, end)
        if not s.empty:
            panel[name] = s

    if not panel:
        logger.warning("No macro data fetched — check FRED_API_KEY")
        return pd.DataFrame()

    df = pd.DataFrame(panel)
    # Build a complete daily date range and forward-fill lower-frequency series
    daily_idx = pd.date_range(start=start, end=end, freq="B")  # business days
    df = df.reindex(daily_idx).ffill()
    return df
