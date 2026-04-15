"""
Transaction Portfolio — actual P&L from real broker transactions.

Data source is selected in the sidebar (demo portfolio or your own
Trading 212 CSV upload). This page reads the shared TransactionPortfolio
from session state and shows transaction-level detail that the other
analytics pages don't need.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Transaction Portfolio | QuantRisk",
    page_icon="💷",
    layout="wide",
)

from dashboard.export_utils import chart_download_button, csv_download_button
from dashboard.sidebar import render_sidebar
from quantrisk.ingestion.trading212 import fetch_prices, resolve_yf_ticker

# Render sidebar (populates st.session_state["tx_portfolio"])
render_sidebar()

tx_portfolio = st.session_state.get("tx_portfolio")
if not tx_portfolio:
    st.info("Select a data source in the sidebar to get started.")
    st.stop()

transactions = tx_portfolio.transactions

st.title("Transaction Portfolio")
st.markdown(
    "Track real P&L from actual broker transactions — "
    "not model weights, but what you actually bought and at what price."
)
st.markdown("---")

holdings = tx_portfolio.holdings()

# ── Fetch current prices ───────────────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner="Fetching live prices…")
def _get_prices(tickers: tuple[str, ...]) -> dict[str, float]:
    """Fetch latest close price for each ticker. Cached for 5 minutes."""
    import yfinance as yf
    prices = {}
    for ticker in tickers:
        yf_ticker = resolve_yf_ticker(ticker)
        try:
            hist = yf.download(
                yf_ticker, period="5d", auto_adjust=True,
                progress=False,
            )
            if not hist.empty:
                close = hist["Close"]
                if isinstance(close, pd.DataFrame):
                    close = close.iloc[:, 0]
                prices[ticker] = float(close.dropna().iloc[-1])
        except Exception:
            pass
    return prices


tickers_tuple = tuple(sorted(holdings.keys()))
current_prices = _get_prices(tickers_tuple) if holdings else {}

# ── Summary metrics ────────────────────────────────────────────────────────────
st.subheader("Summary")

total_deposited = tx_portfolio.total_deposited()
portfolio_value = tx_portfolio.current_value(current_prices)
cash_balance = tx_portfolio.cash_balance()
total_upnl = sum(tx_portfolio.unrealised_pnl(current_prices).values())
total_realised = sum(tx_portfolio.realised_pnl().values())
total_return_pct = tx_portfolio.total_return(current_prices)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Deposited", f"£{total_deposited:,.0f}")
c2.metric(
    "Portfolio Value",
    f"£{portfolio_value:,.0f}",
    delta=f"£{portfolio_value - total_deposited:+,.0f}",
)
c3.metric("Cash Balance", f"£{cash_balance:,.0f}")
c4.metric(
    "Unrealised P&L",
    f"£{total_upnl:+,.0f}",
    delta=f"{total_upnl / (total_deposited or 1):.2%}",
    delta_color="normal",
)
c5.metric(
    "Total Return",
    f"{total_return_pct:.2%}",
    delta_color="normal",
)

if total_realised != 0:
    st.caption(f"Realised P&L from closed/trimmed positions: £{total_realised:+,.2f}")

st.markdown("---")

# ── Holdings table ─────────────────────────────────────────────────────────────
st.subheader("Holdings")

if not holdings:
    st.info("No open positions.")
else:
    df = tx_portfolio.holdings_df(current_prices)

    def _colour_pnl(val):
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return ""
        return "color: #00CC96" if val >= 0 else "color: #EF553B"

    format_map = {
        "Shares": "{:.4f}",
        "Avg Cost": "£{:.4f}",
        "Current Price": "£{:.4f}",
        "Current Value": "£{:,.2f}",
        "Unrealised P&L": "£{:+,.2f}",
        "P&L %": "{:.2%}",
        "Weight": "{:.1%}",
    }

    styled = (
        df.style
        .format(format_map, na_rep="—")
        .map(_colour_pnl, subset=["Unrealised P&L", "P&L %"])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)
    csv_download_button(df, "holdings.csv", "Download Holdings CSV", key="dl_holdings")

st.markdown("---")

# ── Portfolio value history ────────────────────────────────────────────────────
st.subheader("Portfolio Value Over Time")

@st.cache_data(ttl=300, show_spinner="Building value history…")
def _value_history(tx_hash: int) -> pd.Series:
    currencies = {}
    for tx in transactions:
        if tx.ticker and tx.price_currency:
            currencies[tx.ticker] = tx.price_currency

    def price_fetcher(tickers, start, end):
        return fetch_prices(tickers, start, end, price_currencies=currencies)

    return tx_portfolio.value_history(price_fetcher)


tx_hash = hash(tuple((t.transaction_id or t.date.isoformat()) for t in transactions))
value_series = _value_history(tx_hash)

if not value_series.empty and len(value_series) > 1:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=value_series.index,
        y=value_series.values,
        mode="lines",
        name="Portfolio Value",
        line=dict(color="#2196F3", width=2),
        fill="tozeroy",
        fillcolor="rgba(33,150,243,0.08)",
    ))

    fig.add_hline(
        y=total_deposited,
        line=dict(color="grey", dash="dash", width=1),
        annotation_text=f"Deposited: £{total_deposited:,.0f}",
        annotation_position="bottom right",
    )

    buy_txs = [tx for tx in transactions if tx.action in {"Market buy", "Limit buy"}]
    sell_txs = [tx for tx in transactions if tx.action in {"Market sell", "Limit sell"}]

    for tx_list, symbol, colour, label in [
        (buy_txs,  "triangle-up",   "#00CC96", "Buy"),
        (sell_txs, "triangle-down", "#EF553B", "Sell"),
    ]:
        if tx_list:
            dates = [pd.Timestamp(tx.date.date()) for tx in tx_list]
            vals = [value_series.asof(d) for d in dates]
            fig.add_trace(go.Scatter(
                x=dates, y=vals, mode="markers", name=label,
                marker=dict(symbol=symbol, size=7, color=colour),
            ))

    fig.update_layout(
        xaxis=dict(title="Date"),
        yaxis=dict(title="Value (£)", tickprefix="£", tickformat=",.0f"),
        height=420,
        margin=dict(t=30, b=40),
        legend=dict(orientation="h", y=1.05),
    )

    st.plotly_chart(fig, use_container_width=True)
    chart_download_button(fig, "portfolio_value.html", "Download Chart", key="dl_val_chart")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Starting Value", f"£{value_series.iloc[0]:,.0f}")
    col_b.metric("Current Value",  f"£{value_series.iloc[-1]:,.0f}")
    n_days = (value_series.index[-1] - value_series.index[0]).days
    col_c.metric("Period", f"{n_days} days ({n_days // 365}y {(n_days % 365) // 30}m)")

elif len(value_series) <= 1:
    st.info("Not enough price history to draw a chart yet.")
else:
    st.warning("Could not fetch price history for chart.")

st.markdown("---")

# ── Transaction history ────────────────────────────────────────────────────────
st.subheader("Transaction History")

tx_df = tx_portfolio.transaction_df()
n_buys  = tx_df["Action"].str.contains("buy",  case=False).sum()
n_sells = tx_df["Action"].str.contains("sell", case=False).sum()
n_deps  = (tx_df["Action"] == "Deposit").sum()
st.caption(
    f"{len(tx_df)} total — {n_buys} buys, {n_sells} sells, {n_deps} deposits"
)

st.dataframe(
    tx_df.style.format({
        "Shares": lambda x: f"{x:.4f}" if pd.notna(x) else "—",
        "Price / Share": lambda x: f"£{x:.4f}" if pd.notna(x) else "—",
        "Total (GBP)": lambda x: f"£{x:+,.2f}" if pd.notna(x) else "—",
    }),
    use_container_width=True,
    hide_index=True,
)
csv_download_button(tx_df, "transactions.csv", "Download Transaction History CSV", key="dl_tx_csv")
