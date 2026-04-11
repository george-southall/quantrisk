"""
Historical stress scenarios: apply known crisis shocks to the current portfolio.

Shocks are expressed as fractional price changes (e.g. -0.40 = -40%).
Asset class mapping is used to assign shocks when individual tickers don't have
explicit overrides.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS: dict[str, dict] = {
    "2008_gfc": {
        "name": "2008 Global Financial Crisis",
        "description": "Lehman collapse, Sep–Nov 2008. Credit crisis, equity rout.",
        "shocks": {
            # Equities
            "AAPL": -0.48, "MSFT": -0.45, "JPM": -0.65, "XOM": -0.40,
            "EEM": -0.55, "VNQ": -0.60,
            # Bonds / safe haven
            "GLD": 0.05, "TLT": 0.25,
            # SPY as reference
            "SPY": -0.42,
        },
        "asset_class_defaults": {
            "equity": -0.40, "bond": 0.15, "gold": 0.05,
            "real_estate": -0.55, "emerging": -0.50,
        },
    },
    "covid_crash": {
        "name": "COVID-19 Crash",
        "description": "Feb 19 – Mar 23 2020. Fastest bear market in history.",
        "shocks": {
            "AAPL": -0.32, "MSFT": -0.28, "JPM": -0.45, "XOM": -0.55,
            "EEM": -0.32, "VNQ": -0.42,
            "GLD": -0.03, "TLT": 0.15,
            "SPY": -0.34,
        },
        "asset_class_defaults": {
            "equity": -0.34, "bond": 0.08, "gold": 0.02,
            "real_estate": -0.40, "emerging": -0.32,
        },
    },
    "dotcom_bust": {
        "name": "Dot-com Bust",
        "description": "Mar 2000 – Oct 2002. Tech valuations collapse.",
        "shocks": {
            "AAPL": -0.80, "MSFT": -0.65, "JPM": -0.25, "XOM": -0.10,
            "EEM": -0.30, "VNQ": 0.05,
            "GLD": 0.02, "TLT": 0.20,
            "SPY": -0.49,
        },
        "asset_class_defaults": {
            "equity": -0.49, "bond": 0.15, "gold": 0.00,
            "real_estate": 0.05, "emerging": -0.35,
        },
    },
    "black_monday_1987": {
        "name": "Black Monday 1987",
        "description": "Oct 19, 1987. Single-day equities fall ~22%.",
        "shocks": {
            "AAPL": -0.22, "MSFT": -0.22, "JPM": -0.22, "XOM": -0.22,
            "EEM": -0.20, "VNQ": -0.18,
            "GLD": 0.02, "TLT": 0.05,
            "SPY": -0.22,
        },
        "asset_class_defaults": {
            "equity": -0.22, "bond": 0.03, "gold": 0.02,
            "real_estate": -0.18, "emerging": -0.20,
        },
    },
    "2022_rate_shock": {
        "name": "2022 Rate Shock",
        "description": "Jan–Oct 2022. Fed hikes, tech and bonds fall together.",
        "shocks": {
            "AAPL": -0.25, "MSFT": -0.30, "JPM": -0.20, "XOM": 0.60,
            "EEM": -0.30, "VNQ": -0.35,
            "GLD": -0.05, "TLT": -0.32,
            "SPY": -0.25,
        },
        "asset_class_defaults": {
            "equity": -0.25, "bond": -0.20, "gold": -0.03,
            "real_estate": -0.30, "emerging": -0.28,
        },
    },
    "sovereign_debt_2011": {
        "name": "European Sovereign Debt Crisis",
        "description": "2011 Eurozone crisis, US debt ceiling standoff.",
        "shocks": {
            "AAPL": -0.15, "MSFT": -0.18, "JPM": -0.25, "XOM": -0.15,
            "EEM": -0.25, "VNQ": -0.15,
            "GLD": 0.15, "TLT": 0.12,
            "SPY": -0.19,
        },
        "asset_class_defaults": {
            "equity": -0.19, "bond": 0.10, "gold": 0.15,
            "real_estate": -0.12, "emerging": -0.25,
        },
    },
}


@dataclass
class ScenarioResult:
    scenario_key: str
    scenario_name: str
    description: str
    ticker_shocks: dict[str, float]       # shock applied per ticker
    ticker_pl: dict[str, float]           # P&L contribution per ticker
    portfolio_shock: float                # weighted average shock
    portfolio_pl: float                   # absolute P&L (assuming $1 portfolio)
    weights: dict[str, float]
    issues: list[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for ticker in self.weights:
            rows.append({
                "ticker": ticker,
                "weight": self.weights[ticker],
                "shock": self.ticker_shocks.get(ticker, 0.0),
                "pl_contribution": self.ticker_pl.get(ticker, 0.0),
            })
        df = pd.DataFrame(rows).sort_values("pl_contribution")
        df["pl_pct"] = df["pl_contribution"] / df["weight"].where(df["weight"] > 0)
        return df


def apply_scenario(
    weights: dict[str, float],
    scenario_key: str,
) -> ScenarioResult:
    """
    Apply a named historical stress scenario to a portfolio.

    For tickers not explicitly listed in the scenario, applies the equity
    default shock as a conservative assumption.

    Parameters
    ----------
    weights : dict[str, float]
        Current portfolio weights (should sum to ~1).
    scenario_key : str
        Key from SCENARIOS dict (e.g. '2008_gfc').
    """
    if scenario_key not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario '{scenario_key}'. Available: {list(SCENARIOS.keys())}"
        )

    scenario = SCENARIOS[scenario_key]
    shocks = scenario["shocks"]
    defaults = scenario["asset_class_defaults"]
    equity_default = defaults.get("equity", -0.25)

    ticker_shocks: dict[str, float] = {}
    ticker_pl: dict[str, float] = {}
    issues: list[str] = []

    for ticker, weight in weights.items():
        if ticker in shocks:
            shock = shocks[ticker]
        else:
            shock = equity_default
            issues.append(f"{ticker}: no explicit shock; using equity default {equity_default:.0%}")
        ticker_shocks[ticker] = shock
        ticker_pl[ticker] = weight * shock

    portfolio_shock = sum(ticker_pl.values())
    portfolio_pl = portfolio_shock  # fractional P&L for a $1 notional portfolio

    return ScenarioResult(
        scenario_key=scenario_key,
        scenario_name=scenario["name"],
        description=scenario["description"],
        ticker_shocks=ticker_shocks,
        ticker_pl=ticker_pl,
        portfolio_shock=portfolio_shock,
        portfolio_pl=portfolio_pl,
        weights=weights,
        issues=issues,
    )


def run_all_scenarios(weights: dict[str, float]) -> pd.DataFrame:
    """
    Run all historical scenarios and return a summary DataFrame.
    """
    rows = []
    for key in SCENARIOS:
        result = apply_scenario(weights, key)
        rows.append({
            "scenario": result.scenario_name,
            "key": key,
            "portfolio_loss": result.portfolio_pl,
        })
    df = pd.DataFrame(rows).sort_values("portfolio_loss").reset_index(drop=True)
    return df
