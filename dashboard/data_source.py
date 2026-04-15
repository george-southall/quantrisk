"""
data_source.py — single source of truth for portfolio data.

Bridges TransactionPortfolio (actual trades) → Portfolio (analytics engine).
Used by sidebar.py; no page should import this directly.
"""

from __future__ import annotations

import io
from pathlib import Path

from quantrisk.ingestion.trading212 import (
    load_multiple_csvs,
    resolve_yf_ticker,
)
from quantrisk.portfolio.portfolio import Portfolio
from quantrisk.portfolio.transactions import TransactionPortfolio

# Path to the synthetic demo CSV shipped with the repo
DEMO_PATH = Path(__file__).resolve().parent.parent / "data" / "demo_transactions.csv"


def load_transactions(
    sources: list[str | Path | io.StringIO],
) -> TransactionPortfolio:
    """
    Parse one or more Trading 212 CSV sources into a TransactionPortfolio.

    Parameters
    ----------
    sources : list of file paths or StringIO objects.
    """
    transactions = load_multiple_csvs(sources)
    return TransactionPortfolio(transactions, base_currency="GBP")


def tx_portfolio_to_portfolio(
    tx_portfolio: TransactionPortfolio,
    benchmark: str = "SPY",
    name: str = "My Portfolio",
) -> Portfolio:
    """
    Derive a Portfolio (static-weight analytics object) from a TransactionPortfolio.

    Strategy
    --------
    - Weights  : proportional to cost basis (total_cost per ticker). Avoids needing
                 live prices at sidebar render time.
    - Tickers  : pre-resolved to yfinance symbols (e.g. VWRP → VWRP.L) so that
                 market_data.fetch_prices can find them without extra resolution.
    - Start    : date of the first trade transaction.

    Returns
    -------
    An unloaded Portfolio ready for `.load()`.
    """
    holdings = tx_portfolio.holdings()

    if not holdings:
        raise ValueError("No open positions — deposit and buy assets first.")

    # Build price_currency map from transactions so we know which exchange each
    # ticker trades on (GBP/GBX → try .L suffix; USD → use as-is).
    currencies: dict[str, str] = {}
    for tx in tx_portfolio.transactions:
        if tx.ticker and tx.price_currency:
            currencies[tx.ticker] = tx.price_currency

    # Resolve each holding's ticker to the correct yfinance symbol.
    resolved: dict[str, str] = {}
    for ticker in holdings:
        resolved[ticker] = resolve_yf_ticker(ticker, currencies.get(ticker))

    # Cost-basis weights keyed on the RESOLVED symbol.
    weights: dict[str, float] = {
        resolved[t]: h.total_cost
        for t, h in holdings.items()
        if h.total_cost > 0
    }

    if not weights:
        raise ValueError("All holdings have zero cost basis — cannot build portfolio.")

    # Derive start date from first trade (not deposits).
    trade_actions = TransactionPortfolio.TRADE_ACTIONS
    trade_dates = [
        tx.date
        for tx in tx_portfolio.transactions
        if tx.action in trade_actions and tx.ticker
    ]
    if not trade_dates:
        raise ValueError("No trade transactions found.")
    start_date = min(trade_dates).strftime("%Y-%m-%d")

    return Portfolio(
        weights=weights,
        start_date=start_date,
        benchmark=benchmark,
        name=name,
    )
