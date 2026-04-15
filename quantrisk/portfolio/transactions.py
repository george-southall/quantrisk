"""
Transaction-based portfolio model.

Derives portfolio state (holdings, cost basis, P&L) entirely from a ledger of
buy/sell/deposit/withdrawal transactions — the way real investors track money.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import pandas as pd

# ── Transaction dataclass ──────────────────────────────────────────────────────

@dataclass
class Transaction:
    """A single ledger entry — trade, deposit, or withdrawal."""

    date: datetime
    action: str                         # "Market buy" | "Market sell" | "Deposit" | "Withdrawal"
    ticker: str | None = None
    isin: str | None = None
    name: str | None = None
    shares: float | None = None         # positive for both buys and sells (sign from action)
    price_per_share: float | None = None
    price_currency: str | None = None
    exchange_rate: float = 1.0
    total: float = 0.0                  # in base currency; negative for cash outflows
    transaction_id: str | None = None


# ── Holding ────────────────────────────────────────────────────────────────────

@dataclass
class Holding:
    """Current state of a single position."""

    ticker: str
    isin: str | None
    name: str | None
    shares: float
    avg_cost: float          # average cost per share in base currency
    total_cost: float        # total amount paid for current shares


# ── TransactionPortfolio ───────────────────────────────────────────────────────

class TransactionPortfolio:
    """
    Portfolio state derived from a sorted list of Transaction objects.

    All monetary values are in the base currency (default: GBP).
    """

    TRADE_ACTIONS = {"Market buy", "Market sell", "Limit buy", "Limit sell",
                     "Stop buy", "Stop sell"}
    CASH_IN_ACTIONS = {"Deposit"}
    CASH_OUT_ACTIONS = {"Withdrawal"}

    def __init__(
        self,
        transactions: list[Transaction],
        base_currency: str = "GBP",
    ) -> None:
        self.transactions = sorted(transactions, key=lambda t: t.date)
        self.base_currency = base_currency

    # ── Holdings ───────────────────────────────────────────────────────────────

    def holdings(self) -> dict[str, Holding]:
        """
        Current open positions keyed by ticker.

        Uses the **average cost** method: when you buy more shares the avg cost
        is recalculated; when you sell, the remaining cost basis is reduced pro-rata.
        """
        state: dict[str, dict] = {}

        for tx in self.transactions:
            if not tx.ticker or tx.shares is None:
                continue

            ticker = tx.ticker
            is_buy = tx.action in {"Market buy", "Limit buy", "Stop buy"}
            is_sell = tx.action in {"Market sell", "Limit sell", "Stop sell"}

            if is_buy:
                cost = abs(tx.total)
                if ticker not in state:
                    state[ticker] = {
                        "isin": tx.isin,
                        "name": tx.name,
                        "shares": 0.0,
                        "total_cost": 0.0,
                    }
                state[ticker]["shares"] += tx.shares
                state[ticker]["total_cost"] += cost

            elif is_sell and ticker in state:
                held = state[ticker]["shares"]
                if held > 0:
                    sell_ratio = min(tx.shares / held, 1.0)
                    state[ticker]["total_cost"] *= (1.0 - sell_ratio)
                    state[ticker]["shares"] -= tx.shares
                    state[ticker]["shares"] = max(state[ticker]["shares"], 0.0)

        result = {}
        for ticker, s in state.items():
            if s["shares"] < 1e-8:
                continue
            avg = s["total_cost"] / s["shares"] if s["shares"] > 0 else 0.0
            result[ticker] = Holding(
                ticker=ticker,
                isin=s["isin"],
                name=s["name"],
                shares=s["shares"],
                avg_cost=avg,
                total_cost=s["total_cost"],
            )
        return result

    # ── Realised P&L ───────────────────────────────────────────────────────────

    def realised_pnl(self) -> dict[str, float]:
        """
        Realised P&L per ticker from closed/partial positions.

        Uses the average cost at the time of each sale.
        """
        state: dict[str, dict] = {}
        realised: dict[str, float] = {}

        for tx in self.transactions:
            if not tx.ticker or tx.shares is None:
                continue

            ticker = tx.ticker
            is_buy = tx.action in {"Market buy", "Limit buy", "Stop buy"}
            is_sell = tx.action in {"Market sell", "Limit sell", "Stop sell"}

            if is_buy:
                if ticker not in state:
                    state[ticker] = {"shares": 0.0, "total_cost": 0.0}
                state[ticker]["shares"] += tx.shares
                state[ticker]["total_cost"] += abs(tx.total)

            elif is_sell and ticker in state and state[ticker]["shares"] > 0:
                held = state[ticker]["shares"]
                avg_cost = state[ticker]["total_cost"] / held
                sell_proceeds = abs(tx.total)
                cost_of_sold = avg_cost * tx.shares

                realised[ticker] = realised.get(ticker, 0.0) + (sell_proceeds - cost_of_sold)

                sell_ratio = min(tx.shares / held, 1.0)
                state[ticker]["total_cost"] *= (1.0 - sell_ratio)
                state[ticker]["shares"] -= tx.shares
                state[ticker]["shares"] = max(state[ticker]["shares"], 0.0)

        return realised

    # ── Cash ───────────────────────────────────────────────────────────────────

    def total_deposited(self) -> float:
        """Net cash deposited (deposits minus withdrawals)."""
        total = 0.0
        for tx in self.transactions:
            if tx.action in self.CASH_IN_ACTIONS:
                total += abs(tx.total)
            elif tx.action in self.CASH_OUT_ACTIONS:
                total -= abs(tx.total)
        return total

    def cash_balance(self) -> float:
        """
        Uninvested cash remaining in the account.

        = deposits - withdrawals - net spend on buys + net proceeds from sells
        """
        cash = 0.0
        for tx in self.transactions:
            if tx.action in self.CASH_IN_ACTIONS:
                cash += abs(tx.total)
            elif tx.action in self.CASH_OUT_ACTIONS:
                cash -= abs(tx.total)
            elif tx.action in {"Market buy", "Limit buy", "Stop buy"}:
                cash -= abs(tx.total)
            elif tx.action in {"Market sell", "Limit sell", "Stop sell"}:
                cash += abs(tx.total)
        return cash

    # ── P&L with live prices ───────────────────────────────────────────────────

    def unrealised_pnl(
        self,
        current_prices: dict[str, float],
    ) -> dict[str, float]:
        """
        Unrealised P&L per ticker given a dict of {ticker: current_price}.
        """
        pnl = {}
        for ticker, h in self.holdings().items():
            price = current_prices.get(ticker)
            if price is not None:
                pnl[ticker] = (price - h.avg_cost) * h.shares
        return pnl

    def current_value(self, current_prices: dict[str, float]) -> float:
        """Total portfolio value = equity value + cash."""
        equity = sum(
            current_prices.get(t, 0.0) * h.shares
            for t, h in self.holdings().items()
        )
        return equity + self.cash_balance()

    def total_return(self, current_prices: dict[str, float]) -> float:
        """
        Simple total return vs net cash deposited.

        Returns 0.0 if nothing has been deposited.
        """
        deposited = self.total_deposited()
        if deposited == 0:
            return 0.0
        return (self.current_value(current_prices) / deposited) - 1.0

    # ── Holdings DataFrame ─────────────────────────────────────────────────────

    def holdings_df(self, current_prices: dict[str, float]) -> pd.DataFrame:
        """
        Return a tidy DataFrame summarising all open positions.

        Columns: ticker, name, shares, avg_cost, current_price,
                 current_value, unrealised_pnl, pnl_pct, weight
        """
        rows = []
        total_equity = sum(
            current_prices.get(t, 0.0) * h.shares
            for t, h in self.holdings().items()
        )

        for ticker, h in self.holdings().items():
            price = current_prices.get(ticker)
            if price is None:
                current_val = None
                upnl = None
                pnl_pct = None
                weight = None
            else:
                current_val = price * h.shares
                upnl = (price - h.avg_cost) * h.shares
                pnl_pct = (price / h.avg_cost - 1.0) if h.avg_cost > 0 else None
                weight = current_val / total_equity if total_equity > 0 else None

            rows.append({
                "Ticker": ticker,
                "Name": h.name or ticker,
                "Shares": h.shares,
                "Avg Cost": h.avg_cost,
                "Current Price": price,
                "Current Value": current_val,
                "Unrealised P&L": upnl,
                "P&L %": pnl_pct,
                "Weight": weight,
            })

        return pd.DataFrame(rows)

    # ── Transaction history DataFrame ─────────────────────────────────────────

    def transaction_df(self) -> pd.DataFrame:
        """All transactions as a formatted DataFrame."""
        rows = []
        for tx in self.transactions:
            rows.append({
                "Date": tx.date,
                "Action": tx.action,
                "Ticker": tx.ticker or "—",
                "Name": tx.name or "—",
                "Shares": tx.shares,
                "Price / Share": tx.price_per_share,
                "Total (GBP)": tx.total,
            })
        return pd.DataFrame(rows)

    # ── Historical portfolio value ─────────────────────────────────────────────

    def value_history(
        self,
        price_fetcher: Callable[[list[str], str, str], pd.DataFrame],
    ) -> pd.Series:
        """
        Reconstruct portfolio value (equity + cash) over time.

        Parameters
        ----------
        price_fetcher : callable
            Function(tickers, start_date_str, end_date_str) → pd.DataFrame
            with columns = tickers and DatetimeIndex of trading days.

        Returns
        -------
        pd.Series indexed by date with daily portfolio value in base currency.
        """
        tickers = list({
            tx.ticker for tx in self.transactions
            if tx.ticker and tx.action in self.TRADE_ACTIONS
        })
        if not tickers:
            return pd.Series(dtype=float)

        first_date = self.transactions[0].date.strftime("%Y-%m-%d")
        today = pd.Timestamp.today().strftime("%Y-%m-%d")

        prices = price_fetcher(tickers, first_date, today)
        if prices.empty:
            return pd.Series(dtype=float)

        # Build daily shares-held per ticker
        shares_held = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        for tx in self.transactions:
            if not tx.ticker or tx.shares is None:
                continue
            if tx.action not in self.TRADE_ACTIONS:
                continue
            tx_date = pd.Timestamp(tx.date.date())
            mask = shares_held.index >= tx_date
            is_buy = tx.action in {"Market buy", "Limit buy", "Stop buy"}
            delta = tx.shares if is_buy else -tx.shares
            if tx.ticker in shares_held.columns:
                shares_held.loc[mask, tx.ticker] += delta

        shares_held = shares_held.clip(lower=0)

        # Build daily cash balance
        cash_series = pd.Series(0.0, index=prices.index)
        for tx in self.transactions:
            tx_date = pd.Timestamp(tx.date.date())
            mask = cash_series.index >= tx_date
            if tx.action in self.CASH_IN_ACTIONS:
                cash_series.loc[mask] += abs(tx.total)
            elif tx.action in self.CASH_OUT_ACTIONS:
                cash_series.loc[mask] -= abs(tx.total)
            elif tx.action in {"Market buy", "Limit buy", "Stop buy"}:
                cash_series.loc[mask] -= abs(tx.total)
            elif tx.action in {"Market sell", "Limit sell", "Stop sell"}:
                cash_series.loc[mask] += abs(tx.total)

        equity_series = (shares_held * prices).sum(axis=1)
        return (equity_series + cash_series).rename("portfolio_value")
