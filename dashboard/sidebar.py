"""Shared sidebar: transaction-based data source and cached portfolio loading."""

from __future__ import annotations

import io

import streamlit as st

from dashboard.data_source import DEMO_PATH, load_transactions, tx_portfolio_to_portfolio
from quantrisk.portfolio.portfolio import Portfolio


@st.cache_resource(show_spinner="Fetching price data…")
def _load_portfolio(
    weights_key: tuple[tuple[str, float], ...],
    start_date: str,
    benchmark: str,
    name: str,
) -> Portfolio:
    """Load and cache a Portfolio. Signature preserved for cache key stability."""
    return Portfolio(
        weights=dict(weights_key),
        start_date=start_date,
        benchmark=benchmark,
        name=name,
    ).load()


@st.cache_data(ttl=3600, show_spinner="Parsing transactions…")
def _load_tx_portfolio(source_key: tuple, is_demo: bool, sources_payload: tuple[str, ...]):
    """
    Parse transaction CSV(s) and return a TransactionPortfolio.

    source_key    — hashable cache key (content hashes or demo mtime)
    is_demo       — True if loading from the on-disk demo file
    sources_payload — tuple of file-content strings (uploaded) or a single path (demo)
    """
    if is_demo:
        sources = [sources_payload[0]]          # single path string
    else:
        sources = [io.StringIO(s) for s in sources_payload]
    return load_transactions(sources)


def render_sidebar() -> Portfolio:
    """Render the sidebar data-source selector and return a loaded Portfolio."""
    demo_available = DEMO_PATH.exists()

    with st.sidebar:
        st.title("QuantRisk")
        st.caption("Portfolio Risk Analytics")
        st.divider()

        # ── Data source ────────────────────────────────────────────────────────
        st.subheader("Data Source")

        use_demo = st.toggle(
            "Use demo portfolio",
            value=demo_available,
            disabled=not demo_available,
            help=(
                "Synthetic 3-year, 16-asset portfolio generated from real historical prices. "
                "Toggle off to upload your own Trading 212 export."
            ),
        )

        if use_demo and demo_available:
            source_key = (int(DEMO_PATH.stat().st_mtime),)
            sources_payload = (str(DEMO_PATH),)   # single path; is_demo=True in loader
            source_label = "Demo portfolio"
        else:
            uploaded = st.file_uploader(
                "Trading 212 CSV export(s)",
                type="csv",
                accept_multiple_files=True,
                help="Go to History → Download CSV in the Trading 212 app.",
            )
            if not uploaded:
                st.info("Upload a Trading 212 CSV to get started.")
                st.stop()
            source_key = tuple(hash(f.getvalue()) for f in uploaded)
            sources_payload = tuple(f.getvalue().decode("utf-8") for f in uploaded)
            source_label = f"{len(uploaded)} file(s) uploaded"

        # ── Settings ───────────────────────────────────────────────────────────
        st.divider()
        st.subheader("Settings")
        benchmark = st.text_input("Benchmark", value="SPY").upper().strip()

        if st.button("Reload data", use_container_width=True):
            st.cache_resource.clear()
            st.cache_data.clear()
            st.rerun()

        # ── Load transactions ──────────────────────────────────────────────────
        is_demo = use_demo and demo_available
        try:
            tx_portfolio = _load_tx_portfolio(source_key, is_demo, sources_payload)
        except Exception as exc:
            st.error(f"Could not parse CSV: {exc}")
            st.stop()

        # Store in session state so page 11 can access transaction-level detail
        st.session_state["tx_portfolio"] = tx_portfolio

        # ── Build Portfolio from holdings ──────────────────────────────────────
        holdings = tx_portfolio.holdings()
        if not holdings:
            st.error("No open positions found. Deposit and buy assets to populate the portfolio.")
            st.stop()

        try:
            bridge = tx_portfolio_to_portfolio(tx_portfolio, benchmark=benchmark)
        except ValueError as exc:
            st.error(str(exc))
            st.stop()

        # Cache key uses the same signature as the original sidebar so existing
        # Streamlit cache hits survive this change wherever tickers are the same.
        weights_key = tuple(sorted(bridge.weights.items()))

        try:
            portfolio = _load_portfolio(
                weights_key=weights_key,
                start_date=bridge.start_date,
                benchmark=benchmark,
                name=source_label,
            )
        except Exception as exc:
            st.error(f"Could not load price data: {exc}")
            st.stop()

        # ── Info caption ───────────────────────────────────────────────────────
        st.divider()
        st.caption(
            f"**{portfolio.name}**  \n"
            f"{portfolio.returns.index[0].date()} → "
            f"{portfolio.returns.index[-1].date()}  \n"
            f"{len(portfolio.tickers)} assets · "
            f"{len(portfolio.returns):,} trading days"
        )

    return portfolio
