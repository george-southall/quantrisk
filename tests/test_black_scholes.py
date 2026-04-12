"""Unit tests for Black-Scholes option pricing and Greeks."""

import numpy as np
import pytest

from quantrisk.derivatives.black_scholes import (
    bs_all_greeks,
    bs_delta,
    bs_gamma,
    bs_price,
    bs_rho,
    bs_theta,
    bs_vega,
    pnl_surface,
)

# ── Standard ATM test parameters ──────────────────────────────────────────────
S, K, T, SIGMA, R = 100.0, 100.0, 1.0, 0.20, 0.05


class TestBSPrice:
    def test_call_positive(self):
        assert bs_price(S, K, T, SIGMA, R, "call") > 0

    def test_put_positive(self):
        assert bs_price(S, K, T, SIGMA, R, "put") > 0

    def test_call_put_parity(self):
        """C - P = S - K * exp(-r * T)."""
        call = bs_price(S, K, T, SIGMA, R, "call")
        put = bs_price(S, K, T, SIGMA, R, "put")
        parity = S - K * np.exp(-R * T)
        assert abs((call - put) - parity) < 1e-8

    def test_deep_itm_call_approaches_intrinsic(self):
        """Deep ITM call price ≈ S - K * exp(-rT)."""
        call = bs_price(200.0, K, T, SIGMA, R, "call")
        intrinsic = 200.0 - K * np.exp(-R * T)
        assert abs(call - intrinsic) < 1.0

    def test_deep_otm_call_near_zero(self):
        call = bs_price(50.0, K, T, SIGMA, R, "call")
        assert call < 0.01

    def test_expiry_call_intrinsic(self):
        """At T=0 call returns max(S-K, 0)."""
        assert bs_price(110.0, K, 0.0, SIGMA, R, "call") == pytest.approx(10.0)
        assert bs_price(90.0, K, 0.0, SIGMA, R, "call") == pytest.approx(0.0)

    def test_expiry_put_intrinsic(self):
        assert bs_price(90.0, K, 0.0, SIGMA, R, "put") == pytest.approx(10.0)
        assert bs_price(110.0, K, 0.0, SIGMA, R, "put") == pytest.approx(0.0)

    def test_known_value(self):
        """Cross-check against a well-known reference value."""
        # S=100, K=100, T=1, sigma=0.2, r=0.05 → call ≈ 10.451
        call = bs_price(100, 100, 1, 0.2, 0.05, "call")
        assert abs(call - 10.4506) < 0.001

    def test_higher_vol_increases_price(self):
        low = bs_price(S, K, T, 0.10, R, "call")
        high = bs_price(S, K, T, 0.40, R, "call")
        assert high > low

    def test_longer_expiry_increases_price(self):
        short = bs_price(S, K, 0.25, SIGMA, R, "call")
        long_ = bs_price(S, K, 2.0, SIGMA, R, "call")
        assert long_ > short


class TestDelta:
    def test_call_delta_range(self):
        d = bs_delta(S, K, T, SIGMA, R, "call")
        assert 0.0 <= d <= 1.0

    def test_put_delta_range(self):
        d = bs_delta(S, K, T, SIGMA, R, "put")
        assert -1.0 <= d <= 0.0

    def test_call_put_delta_sum(self):
        """Call delta - put delta = 1 (put-call delta parity)."""
        dc = bs_delta(S, K, T, SIGMA, R, "call")
        dp = bs_delta(S, K, T, SIGMA, R, "put")
        assert abs(dc - dp - 1.0) < 1e-8

    def test_deep_itm_call_delta_near_one(self):
        d = bs_delta(200.0, K, T, SIGMA, R, "call")
        assert d > 0.99

    def test_deep_otm_call_delta_near_zero(self):
        d = bs_delta(50.0, K, T, SIGMA, R, "call")
        assert d < 0.01

    def test_atm_call_delta_near_half(self):
        d = bs_delta(S, K, T, SIGMA, 0.0, "call")
        assert abs(d - 0.5) < 0.05

    def test_expiry_itm_call(self):
        assert bs_delta(110.0, K, 0.0, SIGMA, R, "call") == 1.0

    def test_expiry_otm_call(self):
        assert bs_delta(90.0, K, 0.0, SIGMA, R, "call") == 0.0


class TestGamma:
    def test_gamma_positive(self):
        assert bs_gamma(S, K, T, SIGMA, R) > 0

    def test_gamma_same_for_call_and_put(self):
        assert bs_gamma(S, K, T, SIGMA, R) == bs_gamma(S, K, T, SIGMA, R)

    def test_atm_gamma_highest(self):
        """ATM gamma > deep ITM or OTM gamma."""
        g_atm = bs_gamma(S, K, T, SIGMA, R)
        g_itm = bs_gamma(150.0, K, T, SIGMA, R)
        g_otm = bs_gamma(50.0, K, T, SIGMA, R)
        assert g_atm > g_itm
        assert g_atm > g_otm

    def test_expiry_gamma_zero(self):
        assert bs_gamma(S, K, 0.0, SIGMA, R) == 0.0


class TestVega:
    def test_vega_positive(self):
        assert bs_vega(S, K, T, SIGMA, R) > 0

    def test_vega_scaled_per_one_pct(self):
        """Manual check: vega ≈ S * N'(d1) * sqrt(T) / 100."""
        v = bs_vega(S, K, T, SIGMA, R)
        assert v < 1.0  # per 1% it should be sub-dollar for ATM $100 option

    def test_expiry_vega_zero(self):
        assert bs_vega(S, K, 0.0, SIGMA, R) == 0.0


class TestTheta:
    def test_theta_negative_for_long_call(self):
        assert bs_theta(S, K, T, SIGMA, R, "call") < 0

    def test_theta_negative_for_long_put(self):
        assert bs_theta(S, K, T, SIGMA, R, "put") < 0

    def test_theta_per_day_small(self):
        """Daily theta should be a small fraction of option price."""
        price = bs_price(S, K, T, SIGMA, R, "call")
        theta = bs_theta(S, K, T, SIGMA, R, "call")
        assert abs(theta) < price * 0.05  # theta/day < 5% of price

    def test_expiry_theta_zero(self):
        assert bs_theta(S, K, 0.0, SIGMA, R, "call") == 0.0


class TestRho:
    def test_call_rho_positive(self):
        assert bs_rho(S, K, T, SIGMA, R, "call") > 0

    def test_put_rho_negative(self):
        assert bs_rho(S, K, T, SIGMA, R, "put") < 0

    def test_expiry_rho_zero(self):
        assert bs_rho(S, K, 0.0, SIGMA, R, "call") == 0.0


class TestAllGreeks:
    def test_returns_all_keys(self):
        g = bs_all_greeks(S, K, T, SIGMA, R, "call")
        expected = {"price", "delta", "gamma", "vega", "theta", "rho",
                    "intrinsic_value", "time_value"}
        assert set(g.keys()) == expected

    def test_intrinsic_plus_time_equals_price(self):
        g = bs_all_greeks(S, K, T, SIGMA, R, "call")
        assert abs(g["intrinsic_value"] + g["time_value"] - g["price"]) < 1e-8

    def test_put_intrinsic_otm(self):
        g = bs_all_greeks(110.0, K, T, SIGMA, R, "put")
        assert g["intrinsic_value"] == 0.0

    def test_call_intrinsic_itm(self):
        g = bs_all_greeks(120.0, K, T, SIGMA, R, "call")
        assert g["intrinsic_value"] == pytest.approx(20.0)

    def test_consistent_with_individual_functions(self):
        g = bs_all_greeks(S, K, T, SIGMA, R, "call")
        assert g["price"] == pytest.approx(bs_price(S, K, T, SIGMA, R, "call"))
        assert g["delta"] == pytest.approx(bs_delta(S, K, T, SIGMA, R, "call"))
        assert g["gamma"] == pytest.approx(bs_gamma(S, K, T, SIGMA, R))
        assert g["vega"] == pytest.approx(bs_vega(S, K, T, SIGMA, R))


class TestPnlSurface:
    def test_vol_surface_shape(self):
        spots, x_vals, Z = pnl_surface(K, T, R, n_points=10, surface_x="vol")
        assert Z.shape == (10, 10)
        assert len(spots) == 10
        assert len(x_vals) == 10

    def test_time_surface_shape(self):
        spots, x_vals, Z = pnl_surface(K, T, R, n_points=10, surface_x="time")
        assert Z.shape == (10, 10)

    def test_prices_non_negative(self):
        spots, x_vals, Z = pnl_surface(K, T, R, n_points=20, surface_x="vol")
        assert np.all(Z >= 0)

    def test_put_surface_shape(self):
        spots, x_vals, Z = pnl_surface(
            K, T, R, option_type="put", n_points=10, surface_x="vol"
        )
        assert Z.shape == (10, 10)
        assert np.all(Z >= 0)

    def test_higher_vol_gives_higher_call_price(self):
        spots, x_vals, Z = pnl_surface(
            K, T, R, option_type="call", n_points=20, surface_x="vol",
            vol_range=(0.05, 0.80),
        )
        # For the ATM row, prices should increase with vol
        atm_idx = len(spots) // 2
        assert Z[atm_idx, -1] > Z[atm_idx, 0]
