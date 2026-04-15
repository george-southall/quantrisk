"""Options / Greeks — Black-Scholes pricer, Greeks dashboard, and P&L surface."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Options / Greeks | QuantRisk",
    page_icon="📐",
    layout="wide",
)

from dashboard.sidebar import render_sidebar
from quantrisk.config import settings
from quantrisk.derivatives.black_scholes import bs_all_greeks, pnl_surface
from quantrisk.utils.plotting import plot_pnl_surface

portfolio = render_sidebar()

st.title("Options / Greeks")
st.caption("Black-Scholes pricer · Delta · Gamma · Vega · Theta · Rho · P&L Surface")
st.markdown("---")

# ── Inputs ────────────────────────────────────────────────────────────────────
col_opt, col_surf = st.columns([1, 1])

with col_opt:
    st.subheader("Option Parameters")
    c1, c2 = st.columns(2)
    spot = c1.number_input("Spot price (S)", min_value=0.01, value=100.0, step=1.0)
    strike = c2.number_input("Strike price (K)", min_value=0.01, value=100.0, step=1.0)

    c3, c4 = st.columns(2)
    expiry_days = c3.number_input(
        "Days to expiry", min_value=1, max_value=1825, value=90, step=1
    )
    T = expiry_days / 365.0

    impl_vol_pct = c4.number_input(
        "Implied volatility (%)", min_value=1.0, max_value=300.0, value=20.0, step=1.0
    )
    sigma = impl_vol_pct / 100.0

    c5, c6 = st.columns(2)
    rf_pct = c5.number_input(
        "Risk-free rate (%)", min_value=0.0, max_value=20.0,
        value=round(settings.risk_free_rate_fallback * 100, 2), step=0.25,
    )
    r = rf_pct / 100.0
    option_type = c6.radio("Option type", ["call", "put"], horizontal=True)

with col_surf:
    st.subheader("Surface Settings")
    surface_x = st.selectbox(
        "Second axis", ["vol", "time"],
        format_func=lambda v: "Spot × Implied Vol" if v == "vol" else "Spot × Time to Expiry",
    )

    s_lo, s_hi = st.slider(
        "Spot range (× strike)", 0.5, 2.0, (0.6, 1.4), step=0.05
    )
    if surface_x == "vol":
        v_lo, v_hi = st.slider(
            "Vol range (%)", 1, 150, (5, 80), step=5
        )
        vol_range = (v_lo / 100, v_hi / 100)
        t_range = (0.01, 2.0)
    else:
        t_lo, t_hi = st.slider(
            "Time range (years)", 0.01, 3.0, (0.05, 2.0), step=0.05
        )
        t_range = (t_lo, t_hi)
        vol_range = (max(sigma * 0.5, 0.01), sigma * 2.0)

    n_pts = st.slider("Grid resolution", 20, 80, 50, step=5)

st.markdown("---")

# ── Greeks ────────────────────────────────────────────────────────────────────
greeks = bs_all_greeks(spot, strike, T, sigma, r, option_type)

st.subheader("Price & Greeks")

g1, g2, g3, g4, g5, g6 = st.columns(6)
g1.metric("Price", f"${greeks['price']:.4f}")
g2.metric("Delta", f"{greeks['delta']:.4f}")
g3.metric("Gamma", f"{greeks['gamma']:.6f}")
g4.metric("Vega (per 1%)", f"${greeks['vega']:.4f}")
g5.metric("Theta (per day)", f"${greeks['theta']:.4f}")
g6.metric("Rho (per 1%)", f"${greeks['rho']:.4f}")

iv1, iv2 = st.columns(2)
iv1.metric("Intrinsic Value", f"${greeks['intrinsic_value']:.4f}")
iv2.metric("Time Value", f"${greeks['time_value']:.4f}")

# Moneyness helper
moneyness = spot / strike
if abs(moneyness - 1) < 0.02:
    money_label = "At-the-money"
elif (option_type == "call" and moneyness > 1) or (option_type == "put" and moneyness < 1):
    money_label = f"In-the-money ({moneyness:.2%})"
else:
    money_label = f"Out-of-the-money ({moneyness:.2%})"
st.caption(f"Moneyness: **{money_label}**")

with st.expander("Greeks interpretation"):
    st.markdown(f"""
| Greek | Value | Meaning |
|-------|-------|---------|
| **Delta** | `{greeks['delta']:.4f}` | Portfolio gains/loses **${abs(greeks['delta']):.4f}** for each $1 move in spot |
| **Gamma** | `{greeks['gamma']:.6f}` | Delta changes by **{greeks['gamma']:.6f}** for each $1 move in spot |
| **Vega** | `{greeks['vega']:.4f}` | Value changes by **${greeks['vega']:.4f}** for each 1% move in implied vol |
| **Theta** | `{greeks['theta']:.4f}` | Value decays by **${abs(greeks['theta']):.4f}** per calendar day |
| **Rho** | `{greeks['rho']:.4f}` | Value changes by **${abs(greeks['rho']):.4f}** for each 1% move in risk-free rate |
""")

st.markdown("---")

# ── P&L surface ────────────────────────────────────────────────────────────────
st.subheader("P&L Surface")

with st.spinner("Computing surface…"):
    spots_arr, x_vals_arr, Z = pnl_surface(
        K=strike,
        T=T,
        r=r,
        option_type=option_type,
        spot_range=(s_lo, s_hi),
        vol_range=vol_range,
        n_points=n_pts,
        surface_x=surface_x,
        T_range=t_range,
    )

x_label = "Implied Volatility" if surface_x == "vol" else "Time to Expiry (years)"
x_display = x_vals_arr * 100 if surface_x == "vol" else x_vals_arr
x_axis_label = "Implied Volatility (%)" if surface_x == "vol" else "Time to Expiry (years)"

st.plotly_chart(
    plot_pnl_surface(
        spots=spots_arr,
        x_vals=x_display,
        price_grid=Z,
        x_label=x_axis_label,
        title=f"{option_type.capitalize()} Price Surface — Strike ${strike:.0f}",
    ),
    width='stretch',
)

# ── Sensitivity table ──────────────────────────────────────────────────────────
st.subheader("Sensitivity at Different Spot Prices")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

spot_range_abs = np.linspace(spot * 0.8, spot * 1.2, 9)
rows = []
for s in spot_range_abs:
    g = bs_all_greeks(s, strike, T, sigma, r, option_type)
    rows.append({
        "Spot": f"${s:.2f}",
        "Price": f"${g['price']:.4f}",
        "Delta": f"{g['delta']:.4f}",
        "Gamma": f"{g['gamma']:.6f}",
        "Vega": f"${g['vega']:.4f}",
        "Theta/day": f"${g['theta']:.4f}",
    })

st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
