"""Market data fetcher with SQLite-backed caching and retry logic."""

import time
from datetime import date, datetime

import pandas as pd
import yfinance as yf
from sqlalchemy import create_engine, text

from quantrisk.config import settings
from quantrisk.utils.logger import get_logger

logger = get_logger(__name__)

_ENGINE = None


def _get_engine():
    global _ENGINE
    if _ENGINE is None:
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        db_url = f"sqlite:///{settings.db_path}"
        _ENGINE = create_engine(db_url, echo=False)
        _init_schema(_ENGINE)
    return _ENGINE


def _init_schema(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS price_cache (
                ticker      TEXT    NOT NULL,
                date        TEXT    NOT NULL,
                open        REAL,
                high        REAL,
                low         REAL,
                close       REAL,
                adj_close   REAL,
                volume      REAL,
                fetched_at  TEXT    NOT NULL,
                PRIMARY KEY (ticker, date)
            )
        """))


def _fetch_from_cache(ticker: str, start: str, end: str) -> pd.DataFrame:
    engine = _get_engine()
    query = text("""
        SELECT date, open, high, low, close, adj_close, volume
        FROM price_cache
        WHERE ticker = :ticker
          AND date BETWEEN :start AND :end
        ORDER BY date
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"ticker": ticker, "start": start, "end": end})
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    return df


def _save_to_cache(ticker: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    engine = _get_engine()
    fetched_at = datetime.utcnow().isoformat()
    records = []
    for dt, row in df.iterrows():
        records.append({
            "ticker": ticker,
            "date": str(dt.date()),
            "open": float(row.get("Open", row.get("open", None) or 0)),
            "high": float(row.get("High", row.get("high", None) or 0)),
            "low": float(row.get("Low", row.get("low", None) or 0)),
            "close": float(row.get("Close", row.get("close", None) or 0)),
            "adj_close": float(row.get("Adj Close", row.get("adj_close", None) or 0)),
            "volume": float(row.get("Volume", row.get("volume", None) or 0)),
            "fetched_at": fetched_at,
        })
    insert_sql = text("""
        INSERT OR REPLACE INTO price_cache
            (ticker, date, open, high, low, close, adj_close, volume, fetched_at)
        VALUES
            (:ticker, :date, :open, :high, :low, :close, :adj_close, :volume, :fetched_at)
    """)
    with engine.begin() as conn:
        conn.execute(insert_sql, records)
    logger.debug("Cached %d rows for %s", len(records), ticker)


def _cache_is_stale(ticker: str, end: str) -> bool:
    """Return True if the cache is missing data for the requested end date."""
    engine = _get_engine()
    query = text("""
        SELECT MAX(date) FROM price_cache WHERE ticker = :ticker
    """)
    with engine.connect() as conn:
        result = conn.execute(query, {"ticker": ticker}).scalar()
    if result is None:
        return True
    # Consider stale if the latest cached date is more than 2 trading days before end
    latest = datetime.strptime(result, "%Y-%m-%d").date()
    end_date = datetime.strptime(end, "%Y-%m-%d").date() if isinstance(end, str) else end
    return (end_date - latest).days > 2


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str | None = None,
    use_cache: bool = True,
    retries: int = 3,
) -> pd.DataFrame:
    """
    Fetch adjusted closing prices for a list of tickers.

    Returns a DataFrame with dates as index and tickers as columns.
    Uses SQLite cache to avoid repeated API calls.
    """
    if end is None:
        end = date.today().isoformat()

    results: dict[str, pd.Series] = {}

    for ticker in tickers:
        series = _fetch_ticker(ticker, start, end, use_cache, retries)
        if series is not None:
            results[ticker] = series
        else:
            logger.warning("No data returned for %s — skipping", ticker)

    if not results:
        raise ValueError(f"Could not fetch data for any of: {tickers}")

    prices = pd.DataFrame(results)
    prices.index = pd.to_datetime(prices.index)
    prices.sort_index(inplace=True)
    return prices


def _fetch_ticker(
    ticker: str,
    start: str,
    end: str,
    use_cache: bool,
    retries: int,
) -> pd.Series | None:
    # Try cache first
    if use_cache and not _cache_is_stale(ticker, end):
        cached = _fetch_from_cache(ticker, start, end)
        if not cached.empty:
            logger.debug("Cache hit for %s (%s → %s)", ticker, start, end)
            return cached["adj_close"].rename(ticker)

    # Fetch from yfinance with exponential backoff
    for attempt in range(retries):
        try:
            logger.info("Fetching %s from yfinance (attempt %d)", ticker, attempt + 1)
            raw = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if raw.empty:
                logger.warning("yfinance returned empty data for %s", ticker)
                return None

            # Flatten MultiIndex columns if present (yfinance >=0.2.x)
            if isinstance(raw.columns, pd.MultiIndex):
                # Level 0 may be price type or ticker depending on yfinance version
                level0 = raw.columns.get_level_values(0).tolist()
                price_types = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
                if all(isinstance(v, str) and v in price_types for v in level0):
                    raw.columns = level0  # price types are level 0
                else:
                    raw.columns = raw.columns.get_level_values(1)  # tickers are level 1

            if use_cache:
                _save_to_cache(ticker, raw)

            # Support both auto_adjust=False (Adj Close) and auto_adjust=True (Close)
            price_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
            return raw[price_col].rename(ticker)

        except Exception as exc:
            wait = 2 ** attempt
            logger.warning("Error fetching %s: %s — retrying in %ds", ticker, exc, wait)
            time.sleep(wait)

    logger.error("All %d attempts failed for %s", retries, ticker)
    return None


def fetch_ohlcv(
    ticker: str,
    start: str,
    end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch full OHLCV data for a single ticker.

    Returns a DataFrame with columns: open, high, low, close, adj_close, volume.
    """
    if end is None:
        end = date.today().isoformat()

    if use_cache and not _cache_is_stale(ticker, end):
        cached = _fetch_from_cache(ticker, start, end)
        if not cached.empty:
            return cached

    raw = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if raw.empty:
        raise ValueError(f"No OHLCV data found for {ticker}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    if use_cache:
        _save_to_cache(ticker, raw)

    df = raw.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
    })
    return df[["open", "high", "low", "close", "adj_close", "volume"]]
