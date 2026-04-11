"""Factor Analysis page — Fama-French regression and PCA factor model."""

import streamlit as st

st.set_page_config(page_title="Factor Analysis | QuantRisk", page_icon="🔬", layout="wide")

from dashboard.sidebar import render_sidebar
from quantrisk.factor_models.attribution import PerformanceAttribution
from quantrisk.factor_models.fama_french import FamaFrenchModel
from quantrisk.factor_models.pca_factors import PCAFactorModel
from quantrisk.utils.plotting import (
    plot_attribution_waterfall,
    plot_factor_loadings,
    plot_monthly_returns_heatmap,
    plot_pca_explained_variance,
)

portfolio = render_sidebar()

st.title("Factor Analysis")
st.markdown("---")

tab_ff, tab_pca = st.tabs(["Fama-French", "PCA Factor Model"])

# ══════════════════════════════════════════════════════════════════════════════
# Fama-French
# ══════════════════════════════════════════════════════════════════════════════
with tab_ff:
    n_factors = st.radio("Model", [3, 5], horizontal=True, key="ff_factors")

    with st.spinner(f"Downloading FF{n_factors} factors and fitting model…"):
        try:
            ff_model = FamaFrenchModel(n_factors=n_factors).fit(portfolio.returns)
            report_df = ff_model.report()
            fitted = True
        except Exception as exc:
            st.error(f"Could not fit Fama-French model: {exc}")
            fitted = False

    if fitted:
        c1, c2, c3 = st.columns(3)
        c1.metric("R²",       f"{ff_model.r_squared:.4f}")
        c2.metric("Daily α",  f"{ff_model.alpha:.6f}")
        c3.metric("Ann. α",   f"{(1 + ff_model.alpha) ** 252 - 1:.2%}")

        st.plotly_chart(
            plot_factor_loadings(report_df, title=f"FF{n_factors} Factor Loadings"),
            use_container_width=True,
        )

        st.subheader("Regression Output")
        st.dataframe(
            report_df[["loading", "t_stat", "p_value", "significant", "description"]],
            use_container_width=True,
        )

        st.subheader("Performance Attribution")
        attr = PerformanceAttribution(ff_model).compute()
        summary = attr.summary()
        st.plotly_chart(
            plot_attribution_waterfall(summary, title="Annualised Return Attribution"),
            use_container_width=True,
        )

        st.subheader("Monthly Attribution")
        monthly_attr = attr.periodic(freq="ME")
        display_cols = [c for c in monthly_attr.columns
                        if c not in ("actual_excess_return", "rf", "actual_total_return")]
        st.dataframe(
            monthly_attr[display_cols].style.format("{:.2%}"),
            use_container_width=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# PCA
# ══════════════════════════════════════════════════════════════════════════════
with tab_pca:
    n_components = st.slider("Number of components", 2, min(10, len(portfolio.tickers)), 5)
    var_threshold = st.slider("Variance threshold", 0.50, 0.99, 0.90, step=0.05)

    pca = PCAFactorModel(n_components=n_components, variance_threshold=var_threshold)
    pca.fit(portfolio.asset_returns)

    n_needed = pca.n_components_for_variance()

    c1, c2, c3 = st.columns(3)
    c1.metric("Components fitted",       n_components)
    c2.metric(f"Components for {var_threshold:.0%} var", n_needed)
    c3.metric("Total variance explained",
              f"{pca.explained_variance_ratio[:n_components].sum():.2%}")

    st.plotly_chart(
        plot_pca_explained_variance(
            pca.explained_variance_ratio,
            title="PCA Explained Variance",
        ),
        use_container_width=True,
    )

    st.subheader("Factor Loadings (asset × PC)")
    st.dataframe(
        pca.loadings.style.background_gradient(cmap="RdBu_r", axis=None),
        use_container_width=True,
    )

    st.subheader("Portfolio Factor Exposures")
    w = portfolio.weight_series()
    exposures = pca.portfolio_factor_exposures(w)
    st.dataframe(
        exposures.to_frame("Exposure").style.format("{:.4f}"),
        use_container_width=True,
    )
