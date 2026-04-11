"""Backtesting page — walk-forward strategy comparison."""

import streamlit as st

st.set_page_config(page_title="Backtesting | QuantRisk", page_icon="🔄", layout="wide")

from dashboard.sidebar import render_sidebar
from quantrisk.backtesting.engine import BacktestEngine
from quantrisk.backtesting.evaluation import TearsheetEvaluator
from quantrisk.backtesting.strategies import STRATEGY_REGISTRY
from quantrisk.utils.plotting import (
    plot_annual_returns_bar,
    plot_cumulative_returns,
    plot_monthly_returns_heatmap,
    plot_rolling_stats,
    plot_weights_history,
)

portfolio = render_sidebar()

st.title("Backtesting")
st.markdown("---")

# ── Configuration ──────────────────────────────────────────────────────────────
with st.expander("Engine configuration", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    est_window = col1.number_input("Estimation window (days)", 63, 504, 252)
    rb_freq    = col2.selectbox("Rebalance frequency", ["ME", "QE", "W"], index=0)
    tc_bps     = col3.number_input("Transaction cost (bps)", 0.0, 100.0, 10.0, step=1.0)
    slip_bps   = col4.number_input("Slippage (bps)", 0.0, 50.0, 5.0, step=1.0)

strategy_options = list(STRATEGY_REGISTRY.keys())
selected = st.multiselect(
    "Strategies to backtest",
    options=strategy_options,
    default=strategy_options,
)

if not selected:
    st.warning("Select at least one strategy.")
    st.stop()

if st.button("Run Backtest", type="primary", use_container_width=True):
    engine = BacktestEngine(
        estimation_window=est_window,
        rebalance_freq=rb_freq,
        transaction_cost_bps=tc_bps,
        slippage_bps=slip_bps,
    )

    with st.spinner("Running walk-forward backtest…"):
        results = {}
        progress = st.progress(0)
        for i, name in enumerate(selected):
            try:
                results[name] = engine.run(
                    portfolio.asset_returns,
                    STRATEGY_REGISTRY[name],
                    strategy_name=name,
                )
            except Exception as exc:
                st.warning(f"{name} failed: {exc}")
            progress.progress((i + 1) / len(selected))
        progress.empty()

    if not results:
        st.error("All strategies failed.")
        st.stop()

    st.session_state["bt_results"]   = results
    st.session_state["bt_benchmark"] = portfolio.benchmark_returns

# ── Display results ────────────────────────────────────────────────────────────
if "bt_results" not in st.session_state:
    st.info("Configure and click **Run Backtest** above.")
    st.stop()

results   = st.session_state["bt_results"]
bench_ret = st.session_state.get("bt_benchmark")
evaluator = TearsheetEvaluator(results, benchmark_returns=bench_ret)

# Cumulative returns
cum_dict = {name: r.returns for name, r in results.items()}
if bench_ret is not None:
    cum_dict[portfolio.benchmark] = bench_ret

st.plotly_chart(
    plot_cumulative_returns(cum_dict, title="Strategy Cumulative Returns"),
    use_container_width=True,
)

# Metrics table
st.subheader("Performance Comparison")
table = evaluator.comparison_table()
st.dataframe(
    table.style.format({
        "annualised_return": "{:.2%}", "annualised_volatility": "{:.2%}",
        "sharpe_ratio": "{:.2f}", "sortino_ratio": "{:.2f}",
        "calmar_ratio": "{:.2f}", "max_drawdown": "{:.2%}",
        "max_drawdown_duration_days": "{:.0f}", "avg_monthly_turnover": "{:.2%}",
        "win_rate": "{:.2%}", "best_day": "{:.2%}", "worst_day": "{:.2%}",
    }),
    use_container_width=True,
)

# Annual returns
st.subheader("Annual Returns")
annual = evaluator.annual_returns()
st.plotly_chart(
    plot_annual_returns_bar(annual, title="Calendar Year Returns"),
    use_container_width=True,
)

# Per-strategy deep dive
st.subheader("Strategy Deep Dive")
chosen = st.selectbox("Strategy", list(results.keys()))

tab_monthly, tab_rolling, tab_weights = st.tabs(
    ["Monthly Returns", "Rolling Metrics", "Weights History"]
)

with tab_monthly:
    pivot = evaluator.monthly_returns_heatmap(chosen)
    st.plotly_chart(
        plot_monthly_returns_heatmap(
            results[chosen].returns, title=f"{chosen} — Monthly Returns"
        ),
        use_container_width=True,
    )

with tab_rolling:
    rolling = evaluator.rolling_metrics(chosen)
    st.plotly_chart(
        plot_rolling_stats(rolling, title=f"{chosen} — Rolling Metrics"),
        use_container_width=True,
    )

with tab_weights:
    wh = results[chosen].weights_history
    if not wh.empty:
        st.plotly_chart(
            plot_weights_history(wh, title=f"{chosen} — Weight History"),
            use_container_width=True,
        )
    else:
        st.info("No weight history available for this strategy.")
