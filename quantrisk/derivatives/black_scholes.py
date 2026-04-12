"""
Black-Scholes option pricing and Greeks.

All functions are scalar; use numpy.vectorize or explicit loops for grids.

Conventions
-----------
- T       : time to expiry in *years*
- sigma   : implied volatility as a decimal (e.g. 0.20 for 20%)
- r       : continuously compounded risk-free rate as a decimal
- vega    : returned per 1 vol-point (1%), i.e. raw_vega / 100
- rho     : returned per 1 rate-point (1%), i.e. raw_rho / 100
- theta   : returned per *calendar day* (negative for long options)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(
    S: float, K: float, T: float, sigma: float, r: float
) -> tuple[float, float]:
    """Compute d1 and d2. Raises ValueError if inputs are non-positive."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        raise ValueError("S, K, T, sigma must all be positive")
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(d1), float(d2)


def bs_price(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    option_type: str = "call",
) -> float:
    """
    Black-Scholes option price.

    At expiry (T <= 0) returns intrinsic value.
    """
    if T <= 1e-6:
        return float(max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0))
    d1, d2 = _d1_d2(S, K, T, sigma, r)
    if option_type == "call":
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def bs_delta(
    S: float, K: float, T: float, sigma: float, r: float, option_type: str = "call"
) -> float:
    """Delta (dV/dS). Call: [0, 1], Put: [-1, 0]."""
    if T <= 1e-6:
        itm = S > K
        return float(1.0 if (itm and option_type == "call") else
                     -1.0 if (itm and option_type == "put") else 0.0)
    d1, _ = _d1_d2(S, K, T, sigma, r)
    return float(norm.cdf(d1) if option_type == "call" else norm.cdf(d1) - 1)


def bs_gamma(S: float, K: float, T: float, sigma: float, r: float) -> float:
    """Gamma (d²V/dS²). Identical for calls and puts."""
    if T <= 1e-6:
        return 0.0
    d1, _ = _d1_d2(S, K, T, sigma, r)
    return float(norm.pdf(d1) / (S * sigma * np.sqrt(T)))


def bs_vega(S: float, K: float, T: float, sigma: float, r: float) -> float:
    """Vega per 1 vol-point (1% move in implied vol). Identical for calls and puts."""
    if T <= 1e-6:
        return 0.0
    d1, _ = _d1_d2(S, K, T, sigma, r)
    return float(S * norm.pdf(d1) * np.sqrt(T) / 100)


def bs_theta(
    S: float, K: float, T: float, sigma: float, r: float, option_type: str = "call"
) -> float:
    """
    Theta per calendar day (time decay).

    Negative for long options (value decays with time).
    """
    if T <= 1e-6:
        return 0.0
    d1, d2 = _d1_d2(S, K, T, sigma, r)
    term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
    if option_type == "call":
        raw = term1 - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        raw = term1 + r * K * np.exp(-r * T) * norm.cdf(-d2)
    return float(raw / 365)


def bs_rho(
    S: float, K: float, T: float, sigma: float, r: float, option_type: str = "call"
) -> float:
    """Rho per 1 rate-point (1% move in risk-free rate)."""
    if T <= 1e-6:
        return 0.0
    _, d2 = _d1_d2(S, K, T, sigma, r)
    if option_type == "call":
        return float(K * T * np.exp(-r * T) * norm.cdf(d2) / 100)
    return float(-K * T * np.exp(-r * T) * norm.cdf(-d2) / 100)


def bs_all_greeks(
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float,
    option_type: str = "call",
) -> dict[str, float]:
    """
    Return all pricing outputs in one call.

    Keys: price, delta, gamma, vega, theta, rho,
          intrinsic_value, time_value.
    """
    price = bs_price(S, K, T, sigma, r, option_type)
    intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
    return {
        "price": price,
        "delta": bs_delta(S, K, T, sigma, r, option_type),
        "gamma": bs_gamma(S, K, T, sigma, r),
        "vega": bs_vega(S, K, T, sigma, r),
        "theta": bs_theta(S, K, T, sigma, r, option_type),
        "rho": bs_rho(S, K, T, sigma, r, option_type),
        "intrinsic_value": float(intrinsic),
        "time_value": float(max(price - intrinsic, 0.0)),
    }


def pnl_surface(
    K: float,
    T: float,
    r: float,
    option_type: str = "call",
    spot_range: tuple[float, float] = (0.5, 1.5),
    vol_range: tuple[float, float] = (0.05, 0.80),
    n_points: int = 50,
    surface_x: str = "vol",
    T_range: tuple[float, float] = (0.01, 2.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute option price grid for a 2D surface chart.

    Parameters
    ----------
    K            : strike price
    T            : base time-to-expiry in years (used when surface_x='vol')
    r            : risk-free rate
    option_type  : 'call' or 'put'
    spot_range   : (lo, hi) multipliers relative to K for spot axis
    vol_range    : (lo, hi) vol range when surface_x='vol'
    n_points     : grid resolution per axis
    surface_x    : 'vol' → x-axis is implied vol; 'time' → x-axis is time-to-expiry
    T_range      : (lo, hi) time range in years when surface_x='time'

    Returns
    -------
    spots    : 1D array of spot prices (y-axis)
    x_vals   : 1D array of x-axis values (vol or time)
    Z        : 2D price grid, shape (len(spots), len(x_vals))
    """
    spots = np.linspace(K * spot_range[0], K * spot_range[1], n_points)

    if surface_x == "vol":
        x_vals = np.linspace(vol_range[0], vol_range[1], n_points)
        _bs_vec = np.vectorize(
            lambda s, v: bs_price(s, K, max(T, 1e-6), v, r, option_type)
        )
        S_grid, V_grid = np.meshgrid(spots, x_vals, indexing="ij")
        Z = _bs_vec(S_grid, V_grid)
    else:
        x_vals = np.linspace(T_range[0], T_range[1], n_points)
        _bs_vec = np.vectorize(
            lambda s, t: bs_price(s, K, max(t, 1e-6), vol_range[0], r, option_type)
        )
        S_grid, T_grid = np.meshgrid(spots, x_vals, indexing="ij")
        Z = _bs_vec(S_grid, T_grid)

    return spots, x_vals, Z
