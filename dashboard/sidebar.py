"""Shared sidebar: portfolio configuration and cached data loading."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from quantrisk.portfolio.portfolio import Portfolio

DEFAULT_TICKERS = "AAPL,MSFT,JPM,XOM,GLD,TLT,EEM,VNQ"
DEFAULT_WEIGHTS = "0.15,0.15,0.10,0.10,0.15,0.15,0.10,0.10"
DEFAULT_START   = pd.to_datetime("2015-01-01").date()
DEFAULT_BENCH   = "SPY"


@st.cache_resource(show_spinner="Fetching price data…")
def _load_portfolio(
    weights_key: tuple[tuple[str, float], ...],
    start_date: str,
    benchmark: str,
    name: str,
) -> Portfolio:
    return Portfolio(
        weights=dict(weights_key),
        start_date=start_date,
        benchmark=benchmark,
        name=name,
    ).load()


def render_sidebar() -> Portfolio:
    """Render the sidebar controls and return a loaded Portfolio."""
    with st.sidebar:
        st.title("QuantRisk")
        st.caption("Portfolio Risk Analytics")
        st.divider()

        st.subheader("Portfolio")
        tickers_raw = st.text_input("Tickers (comma-separated)", value=DEFAULT_TICKERS)
        weights_raw = st.text_input("Weights (comma-separated)", value=DEFAULT_WEIGHTS)
        start_date  = st.date_input("Start date", value=DEFAULT_START)
        benchmark   = st.text_input("Benchmark", value=DEFAULT_BENCH).upper().strip()
        port_name   = st.text_input("Portfolio name", value="Demo Portfolio")

        if st.button("Reload data", use_container_width=True):
            st.cache_resource.clear()

        # Parse
        try:
            tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
            weights = [float(w.strip()) for w in weights_raw.split(",") if w.strip()]
        except ValueError:
            st.error("Weights must be numbers.")
            st.stop()

        if len(tickers) != len(weights):
            st.error(f"Got {len(tickers)} tickers but {len(weights)} weights.")
            st.stop()

        if sum(weights) <= 0:
            st.error("Weights must sum to a positive number.")
            st.stop()

        weights_key = tuple(sorted(zip(tickers, weights)))

        portfolio = _load_portfolio(
            weights_key=weights_key,
            start_date=start_date.isoformat(),
            benchmark=benchmark,
            name=port_name,
        )

        st.divider()
        st.caption(
            f"**{portfolio.name}**  \n"
            f"{portfolio.returns.index[0].date()} → "
            f"{portfolio.returns.index[-1].date()}  \n"
            f"{len(portfolio.returns):,} trading days"
        )

    return portfolio
