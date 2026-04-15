"""
Trading 212 CSV export parser.

Trading 212 exports a CSV with columns:
    Action, Time, ISIN, Ticker, Name, Notes, ID,
    No. of shares, Price / share, Currency (Price / share),
    Exchange rate, Total, Currency (Total)

Cash-only accounts (e.g. Cash ISA) have a reduced schema:
    Action, Time, Notes, ID, Total, Currency (Total)

This loader handles both formats and returns a list of Transaction objects
compatible with TransactionPortfolio.
"""

from __future__ import annotations

import io
import os
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from quantrisk.portfolio.transactions import Transaction
from quantrisk.utils.logger import get_logger

logger = get_logger(__name__)


def load_trading212_csv(
    source: str | Path | io.BytesIO | io.StringIO,
) -> list[Transaction]:
    """
    Parse a Trading 212 export CSV into Transaction objects.

    Parameters
    ----------
    source : path, BytesIO, or StringIO
        The CSV file or in-memory buffer.

    Returns
    -------
    list[Transaction] sorted oldest → newest.
    """
    df = pd.read_csv(source)
    logger.info("Loaded Trading 212 CSV: %d rows, columns: %s", len(df), list(df.columns))

    transactions: list[Transaction] = []

    for _, row in df.iterrows():
        action = _safe_str(row.get("Action")) or ""
        if not action:
            continue

        date = pd.to_datetime(row.get("Time"), errors="coerce")
        if pd.isna(date):
            logger.warning("Skipping row with unparseable date: %s", row.get("Time"))
            continue

        total = _safe_float(row.get("Total"), default=0.0)

        # Normalise withdrawals to negative
        if action == "Withdrawal" and total is not None and total > 0:
            total = -total

        tx = Transaction(
            date=date.to_pydatetime(),
            action=action,
            ticker=_safe_str(row.get("Ticker")),
            isin=_safe_str(row.get("ISIN")),
            name=_safe_str(row.get("Name")),
            shares=_safe_float(row.get("No. of shares")),
            price_per_share=_safe_float(row.get("Price / share")),
            price_currency=_safe_str(row.get("Currency (Price / share)")),
            exchange_rate=_safe_float(row.get("Exchange rate"), default=1.0) or 1.0,
            total=total or 0.0,
            transaction_id=_safe_str(row.get("ID")),
        )
        transactions.append(tx)

    transactions.sort(key=lambda t: t.date)
    logger.info("Parsed %d transactions (%d trades)",
                len(transactions),
                sum(1 for t in transactions if t.ticker))
    return transactions


def load_multiple_csvs(
    sources: list[str | Path | io.BytesIO | io.StringIO],
) -> list[Transaction]:
    """
    Merge transactions from multiple CSV exports (e.g. Stocks ISA + Cash ISA).

    De-duplicates by transaction_id where available.
    """
    all_txs: list[Transaction] = []
    seen_ids: set[str] = set()

    for src in sources:
        for tx in load_trading212_csv(src):
            if tx.transaction_id and tx.transaction_id in seen_ids:
                continue
            if tx.transaction_id:
                seen_ids.add(tx.transaction_id)
            all_txs.append(tx)

    all_txs.sort(key=lambda t: t.date)
    return all_txs


# ── yfinance ticker resolution ─────────────────────────────────────────────────

_TICKER_CACHE: dict[str, str] = {}


@contextmanager
def _silence():
    """Suppress stdout/stderr — used to hide expected yfinance 404 noise."""
    with open(os.devnull, "w") as devnull:
        import sys
        old_out, old_err = sys.stdout, sys.stderr
        try:
            sys.stdout = sys.stderr = devnull
            yield
        finally:
            sys.stdout, sys.stderr = old_out, old_err

_LSE_SUFFIXES = [".L", ".IL"]
_US_EXCHANGES = {"USD"}


def resolve_yf_ticker(ticker: str, price_currency: str | None = None) -> str:
    """
    Attempt to find the correct yfinance ticker symbol.

    Strategy:
      1. Return cached result if seen before.
      2. Try the ticker as-is.
      3. If currency is GBP/GBX, try appending .L (London Stock Exchange).
      4. Try .IL (international LSE).
      5. Fall back to original ticker.
    """
    if ticker in _TICKER_CACHE:
        return _TICKER_CACHE[ticker]

    import yfinance as yf

    candidates = [ticker]
    if price_currency and price_currency.upper() in {"GBP", "GBX"}:
        candidates += [f"{ticker}.L", f"{ticker}.IL"]
    else:
        candidates += [f"{ticker}.L"]   # try LSE anyway as fallback

    for candidate in candidates:
        try:
            with _silence():
                hist = yf.download(
                    candidate,
                    period="5d",
                    auto_adjust=True,
                    progress=False,
                )
            if not hist.empty:
                _TICKER_CACHE[ticker] = candidate
                logger.info("Resolved %s -> %s", ticker, candidate)
                return candidate
        except Exception:
            continue

    logger.warning("Could not resolve yfinance ticker for %s; using as-is.", ticker)
    _TICKER_CACHE[ticker] = ticker
    return ticker


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    price_currencies: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for a list of portfolio tickers.

    Handles LSE ticker resolution automatically.
    Returns a DataFrame with original ticker names as columns.
    """
    import yfinance as yf

    currencies = price_currencies or {}
    resolved = {t: resolve_yf_ticker(t, currencies.get(t)) for t in tickers}

    yf_tickers = list(resolved.values())
    raw = yf.download(
        yf_tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        return pd.DataFrame()

    # Extract close prices
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        close = raw[["Close"]] if "Close" in raw.columns else raw

    # Rename resolved tickers back to original names
    reverse = {v: k for k, v in resolved.items()}
    close = close.rename(columns=reverse)

    # Only keep requested tickers
    available = [t for t in tickers if t in close.columns]
    return close[available].dropna(how="all")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _safe_float(val, default=None) -> float | None:
    if val is None:
        return default
    try:
        f = float(val)
        return default if pd.isna(f) else f
    except (TypeError, ValueError):
        return default


def _safe_str(val) -> str | None:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    s = str(val).strip()
    return s or None
