#!/usr/bin/env python3
"""
QuantRisk — Demo Portfolio Data Generator
==========================================

Creates a synthetic 3-year transaction history that looks and behaves like a
real ISA portfolio. All buy/sell prices are fetched from actual yfinance
historical data so the P&L figures are genuine.

EDIT THE CONFIG SECTION BELOW to customise:
  - Portfolio composition and target weights
  - Monthly contribution amounts
  - Rebalancing frequency and aggressiveness
  - Start/end dates and random seed

Usage
-----
    cd <project root>
    python scripts/generate_demo_data.py

Output
------
    data/demo_transactions.csv   — Trading 212-format CSV, committed to git
"""

from __future__ import annotations

import random
import sys
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# ── path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit freely
# ══════════════════════════════════════════════════════════════════════════════

CONFIG: dict = {
    # ── Date range ─────────────────────────────────────────────────────────────
    "start_date": "2023-04-03",
    "end_date":   "2026-04-11",   # last possible trade date
    "base_currency": "GBP",

    # ── Reproducibility ────────────────────────────────────────────────────────
    # Change seed to generate a completely different transaction history
    "random_seed": 42,

    # ── Cash contributions ─────────────────────────────────────────────────────
    "initial_deposit":     20_000,   # one-off lump sum at start (£)
    "monthly_deposit_min":  1_500,   # minimum monthly ISA contribution (£)
    "monthly_deposit_max":  3_500,   # maximum monthly ISA contribution (£)

    # ── Trading strategy ───────────────────────────────────────────────────────
    "buys_per_month":      2,     # number of separate buy orders per month
    "invest_ratio":        0.90,  # fraction of available cash to invest each month
    "rebalance_every_n_months": 3,     # 0 = never rebalance
    "rebalance_threshold":      0.04,  # only sell if weight > target + this

    # ── Output ─────────────────────────────────────────────────────────────────
    "output_path": str(ROOT / "data" / "demo_transactions.csv"),

    # ── Portfolio ──────────────────────────────────────────────────────────────
    # target_weight values must sum to exactly 1.0.
    # yf_ticker  = the symbol used in yfinance (may differ from display ticker)
    # currency   = "GBP" (LSE-listed, no FX conversion) or "USD" (converted)
    # isin       = used in the output CSV for realism
    #
    "portfolio": {
        # ── Core global ETFs (LSE, priced in GBP) ─────────────────────────────
        "VWRP": {
            "name": "Vanguard FTSE All-World (Acc)",
            "isin": "IE00BK5BQT80",
            "currency": "GBP",
            "yf_ticker": "VWRP.L",
            "target_weight": 0.18,
        },
        "VUAG": {
            "name": "Vanguard S&P 500 UCITS ETF (Acc)",
            "isin": "IE00BFMXXD54",
            "currency": "GBP",
            "yf_ticker": "VUAG.L",
            "target_weight": 0.10,
        },
        # ── NASDAQ-100 ETF (US-listed, USD) ────────────────────────────────────
        "QQQ": {
            "name": "Invesco QQQ Trust Series 1",
            "isin": "US46090E1038",
            "currency": "USD",
            "yf_ticker": "QQQ",
            "target_weight": 0.07,
        },
        # ── US Mega-cap Technology ─────────────────────────────────────────────
        "AAPL": {
            "name": "Apple Inc",
            "isin": "US0378331005",
            "currency": "USD",
            "yf_ticker": "AAPL",
            "target_weight": 0.08,
        },
        "MSFT": {
            "name": "Microsoft Corporation",
            "isin": "US5949181045",
            "currency": "USD",
            "yf_ticker": "MSFT",
            "target_weight": 0.08,
        },
        "NVDA": {
            "name": "NVIDIA Corporation",
            "isin": "US67066G1040",
            "currency": "USD",
            "yf_ticker": "NVDA",
            "target_weight": 0.07,
        },
        "GOOGL": {
            "name": "Alphabet Inc Class A",
            "isin": "US02079K3059",
            "currency": "USD",
            "yf_ticker": "GOOGL",
            "target_weight": 0.06,
        },
        "AMZN": {
            "name": "Amazon.com Inc",
            "isin": "US0231351067",
            "currency": "USD",
            "yf_ticker": "AMZN",
            "target_weight": 0.05,
        },
        "META": {
            "name": "Meta Platforms Inc",
            "isin": "US30303M1027",
            "currency": "USD",
            "yf_ticker": "META",
            "target_weight": 0.05,
        },
        # ── US Financials / Diversified ────────────────────────────────────────
        "JPM": {
            "name": "JPMorgan Chase & Co",
            "isin": "US46625H1005",
            "currency": "USD",
            "yf_ticker": "JPM",
            "target_weight": 0.05,
        },
        "BRK-B": {
            "name": "Berkshire Hathaway Inc Class B",
            "isin": "US0846701086",
            "currency": "USD",
            "yf_ticker": "BRK-B",
            "target_weight": 0.04,
        },
        "V": {
            "name": "Visa Inc Class A",
            "isin": "US92826C8394",
            "currency": "USD",
            "yf_ticker": "V",
            "target_weight": 0.04,
        },
        # ── US Consumer / Healthcare ───────────────────────────────────────────
        "COST": {
            "name": "Costco Wholesale Corporation",
            "isin": "US22160K1051",
            "currency": "USD",
            "yf_ticker": "COST",
            "target_weight": 0.04,
        },
        "UNH": {
            "name": "UnitedHealth Group Inc",
            "isin": "US91324P1021",
            "currency": "USD",
            "yf_ticker": "UNH",
            "target_weight": 0.03,
        },
        # ── High-volatility / Speculative ──────────────────────────────────────
        "TSLA": {
            "name": "Tesla Inc",
            "isin": "US88160R1014",
            "currency": "USD",
            "yf_ticker": "TSLA",
            "target_weight": 0.04,
        },
        "PLTR": {
            "name": "Palantir Technologies Inc",
            "isin": "US69608A1088",
            "currency": "USD",
            "yf_ticker": "PLTR",
            "target_weight": 0.02,
        },
    },
}

# Weights must sum to 1 — validate at import time
_total_weight = sum(v["target_weight"] for v in CONFIG["portfolio"].values())
assert abs(_total_weight - 1.0) < 1e-9, (
    f"Target weights sum to {_total_weight:.6f}, must be 1.0"
)


# ══════════════════════════════════════════════════════════════════════════════
# Price fetching
# ══════════════════════════════════════════════════════════════════════════════

def fetch_prices(config: dict) -> tuple[pd.DataFrame, pd.Series]:
    """
    Download adjusted-close prices for every asset and the GBP/USD FX rate.

    Returns
    -------
    prices : pd.DataFrame  — columns are portfolio tickers (not yf_tickers)
    gbpusd : pd.Series     — daily GBP/USD exchange rate
    """
    yf_tickers = [v["yf_ticker"] for v in config["portfolio"].values()]
    yf_tickers.append("GBPUSD=X")

    print(f"Downloading price history for {len(yf_tickers)} symbols…")
    raw = yf.download(
        yf_tickers,
        start=config["start_date"],
        end=config["end_date"],
        auto_adjust=True,
        progress=False,
    )

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw

    # Map yf_ticker → portfolio ticker
    yf_to_tick = {v["yf_ticker"]: k for k, v in config["portfolio"].items()}

    prices = {}
    missing = []
    for col in close.columns:
        if col == "GBPUSD=X":
            continue
        ticker = yf_to_tick.get(col, col)
        s = close[col].dropna()
        if s.empty:
            missing.append(col)
        else:
            prices[ticker] = s

    if missing:
        print(f"  WARNING: no data returned for: {missing}")

    gbpusd = close["GBPUSD=X"].dropna() if "GBPUSD=X" in close.columns else pd.Series(
        dtype=float
    )
    prices_df = pd.DataFrame(prices)
    print(f"  Got data for {len(prices_df.columns)} tickers "
          f"({prices_df.index[0].date()} to {prices_df.index[-1].date()})")
    return prices_df, gbpusd


# ══════════════════════════════════════════════════════════════════════════════
# Price / FX helpers
# ══════════════════════════════════════════════════════════════════════════════

def _nearest_price(series: pd.Series, date: pd.Timestamp) -> float | None:
    """Last available price on or before *date*."""
    past = series.index[series.index <= date]
    if past.empty:
        return None
    return float(series.loc[past[-1]])


def _gbpusd(gbpusd: pd.Series, date: pd.Timestamp, fallback: float = 1.27) -> float:
    v = _nearest_price(gbpusd, date)
    return v if v else fallback


def _price_gbp(
    config: dict,
    prices: pd.DataFrame,
    gbpusd: pd.Series,
    ticker: str,
    date: pd.Timestamp,
) -> float | None:
    """Price in GBP for *ticker* on *date*."""
    if ticker not in prices.columns:
        return None
    raw = _nearest_price(prices[ticker], date)
    if raw is None:
        return None
    if config["portfolio"][ticker]["currency"] == "USD":
        fx = _gbpusd(gbpusd, date)
        return raw / fx
    return raw


def _portfolio_equity(
    config: dict,
    prices: pd.DataFrame,
    gbpusd: pd.Series,
    holdings: dict[str, float],
    date: pd.Timestamp,
) -> float:
    """Total equity value of all holdings in GBP."""
    total = 0.0
    for ticker, shares in holdings.items():
        p = _price_gbp(config, prices, gbpusd, ticker, date)
        if p:
            total += shares * p
    return total


def _nearest_trading_day(
    prices: pd.DataFrame,
    target: pd.Timestamp,
    direction: str = "forward",
) -> pd.Timestamp:
    """Return the nearest trading day to *target* (searching forward or backward)."""
    idx = prices.index
    if direction == "forward":
        candidates = idx[idx >= target]
        return candidates[0] if len(candidates) else idx[-1]
    candidates = idx[idx <= target]
    return candidates[-1] if len(candidates) else idx[0]


# ══════════════════════════════════════════════════════════════════════════════
# Transaction generation
# ══════════════════════════════════════════════════════════════════════════════

def generate_transactions(
    config: dict,
    prices: pd.DataFrame,
    gbpusd: pd.Series,
    rng: random.Random,
) -> list[dict]:
    """
    Walk forward month by month, generating deposits, DCA buys,
    and periodic rebalancing sells.
    """

    available_tickers = [t for t in config["portfolio"] if t in prices.columns]
    if not available_tickers:
        raise RuntimeError("No price data available for any portfolio ticker.")

    holdings: dict[str, float] = {t: 0.0 for t in available_tickers}
    cash: float = 0.0
    rows: list[dict] = []

    # ── helpers ────────────────────────────────────────────────────────────────

    def _uid() -> str:
        return str(uuid.uuid4())

    def _row_deposit(dt: pd.Timestamp, amount: float) -> dict:
        return {
            "Action": "Deposit",
            "Time": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "ISIN": "", "Ticker": "", "Name": "",
            "Notes": f"Transaction ID: {_uid()}",
            "ID": _uid(),
            "No. of shares": "", "Price / share": "",
            "Currency (Price / share)": "", "Exchange rate": "",
            "Total": round(amount, 2), "Currency (Total)": "GBP",
        }

    def _row_buy(dt: pd.Timestamp, ticker: str, amount_gbp: float) -> dict | None:
        cfg = config["portfolio"][ticker]
        p_raw = _nearest_price(prices[ticker], dt)
        if p_raw is None or p_raw <= 0:
            return None
        fx = _gbpusd(gbpusd, dt) if cfg["currency"] == "USD" else 1.0
        p_gbp = p_raw / fx if cfg["currency"] == "USD" else p_raw
        shares = amount_gbp / p_gbp
        if shares <= 0:
            return None
        return {
            "Action": "Market buy",
            "Time": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "ISIN": cfg["isin"], "Ticker": ticker, "Name": cfg["name"],
            "Notes": "", "ID": _uid(),
            "No. of shares": round(shares, 8),
            "Price / share": round(p_raw, 6),
            "Currency (Price / share)": cfg["currency"],
            "Exchange rate": round(fx, 6),
            "Total": round(amount_gbp, 2), "Currency (Total)": "GBP",
            # Internals — stripped before saving
            "_ticker": ticker, "_shares": shares, "_amount_gbp": amount_gbp,
        }

    def _row_sell(dt: pd.Timestamp, ticker: str, shares_to_sell: float) -> dict | None:
        cfg = config["portfolio"][ticker]
        p_raw = _nearest_price(prices[ticker], dt)
        if p_raw is None or p_raw <= 0:
            return None
        shares_to_sell = min(shares_to_sell, holdings[ticker])
        if shares_to_sell < 1e-6:
            return None
        fx = _gbpusd(gbpusd, dt) if cfg["currency"] == "USD" else 1.0
        p_gbp = p_raw / fx if cfg["currency"] == "USD" else p_raw
        proceeds = round(shares_to_sell * p_gbp, 2)
        return {
            "Action": "Market sell",
            "Time": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "ISIN": cfg["isin"], "Ticker": ticker, "Name": cfg["name"],
            "Notes": "", "ID": _uid(),
            "No. of shares": round(shares_to_sell, 8),
            "Price / share": round(p_raw, 6),
            "Currency (Price / share)": cfg["currency"],
            "Exchange rate": round(fx, 6),
            "Total": proceeds, "Currency (Total)": "GBP",
            "_ticker": ticker, "_shares": -shares_to_sell, "_amount_gbp": proceeds,
        }

    def _commit(row: dict) -> None:
        nonlocal cash
        if "_ticker" in row:
            t, delta_shares, delta_cash = (
                row["_ticker"], row["_shares"], row["_amount_gbp"]
            )
            if row["Action"] == "Market buy":
                holdings[t] += delta_shares
                cash -= delta_cash
            else:
                holdings[t] += delta_shares   # delta_shares is negative
                cash += delta_cash
        else:
            cash += row["Total"]
        # Strip internal keys
        for k in ["_ticker", "_shares", "_amount_gbp"]:
            row.pop(k, None)
        rows.append(row)

    def _current_weights(date: pd.Timestamp) -> dict[str, float]:
        equity = _portfolio_equity(config, prices, gbpusd, holdings, date)
        total = equity + cash
        if total <= 0:
            return {t: 0.0 for t in available_tickers}
        weights = {}
        for t in available_tickers:
            p = _price_gbp(config, prices, gbpusd, t, date)
            weights[t] = (holdings[t] * p / total) if p else 0.0
        return weights

    def _underweight_order(date: pd.Timestamp) -> list[str]:
        """Tickers sorted by (target_weight - current_weight) descending."""
        cw = _current_weights(date)
        deficit = {
            t: config["portfolio"][t]["target_weight"] - cw.get(t, 0.0)
            for t in available_tickers
        }
        return sorted(deficit, key=deficit.get, reverse=True)

    # ── Month loop ─────────────────────────────────────────────────────────────

    start = pd.Timestamp(config["start_date"])
    end = pd.Timestamp(config["end_date"])
    month = start
    month_idx = 0

    while month <= end:
        label = month.strftime("%Y-%m")
        print(f"  {label}…", end="\r", flush=True)

        # Deposit day: 1st–7th of month, on a trading day
        raw_dep = month + pd.Timedelta(days=rng.randint(0, 6))
        dep_day = _nearest_trading_day(prices, raw_dep, "forward")
        if dep_day > end:
            break
        dep_dt = dep_day + pd.Timedelta(hours=rng.randint(8, 11), minutes=rng.randint(0, 59))

        # Deposit amount
        if month_idx == 0:
            deposit_amount = float(config["initial_deposit"])
        else:
            deposit_amount = round(
                rng.uniform(config["monthly_deposit_min"], config["monthly_deposit_max"]), 2
            )

        _commit(_row_deposit(dep_dt, deposit_amount))

        # ── DCA buys — proportional deficit allocation ─────────────────────────
        # Two buy rounds per month (mid-month and end of month).
        # Each round distributes proportionally across the top-N most underweight
        # tickers, guaranteeing the full portfolio gets funded over time.
        investable = cash * config["invest_ratio"]

        for round_i in range(2):
            if cash < 50 or investable < 50:
                break

            buy_offset = rng.randint(8 + round_i * 7, 12 + round_i * 7)
            raw_buy = dep_day + pd.Timedelta(days=buy_offset)
            buy_day_r = _nearest_trading_day(prices, raw_buy, "forward")
            if buy_day_r > end:
                break
            buy_dt = buy_day_r + pd.Timedelta(
                hours=rng.randint(9, 15), minutes=rng.randint(0, 59)
            )

            # Proportional allocation across all underweight tickers
            cw = _current_weights(buy_dt)
            deficits = {
                t: max(0.0, config["portfolio"][t]["target_weight"] - cw.get(t, 0.0))
                for t in available_tickers
            }
            total_deficit = sum(deficits.values()) or 1.0

            round_budget = min(investable * 0.5, cash * 0.9)
            min_buy_amount = 60.0

            # Sort by deficit desc; buy proportionally, skipping tiny amounts
            sorted_tickers = sorted(deficits, key=deficits.get, reverse=True)
            for ticker in sorted_tickers:
                if deficits[ticker] <= 0:
                    continue
                alloc = (deficits[ticker] / total_deficit) * round_budget
                if alloc < min_buy_amount or cash < min_buy_amount:
                    continue
                row = _row_buy(buy_dt, ticker, min(alloc, cash * 0.85))
                if row:
                    _commit(row)
                buy_dt += pd.Timedelta(minutes=rng.randint(3, 18))

        # ── Periodic rebalancing + profit-taking ──────────────────────────────
        n = config["rebalance_every_n_months"]
        if n > 0 and month_idx > 0 and month_idx % n == 0:
            raw_reb = month + pd.Timedelta(days=rng.randint(10, 20))
            reb_day = _nearest_trading_day(prices, raw_reb, "forward")
            if reb_day <= end:
                reb_dt = reb_day + pd.Timedelta(hours=10, minutes=rng.randint(0, 45))
                cw = _current_weights(reb_dt)
                thresh = config["rebalance_threshold"]
                total_val = (
                    _portfolio_equity(config, prices, gbpusd, holdings, reb_dt) + cash
                )

                # Sell overweight positions back to target
                for ticker in available_tickers:
                    excess = cw.get(ticker, 0.0) - config["portfolio"][ticker]["target_weight"]
                    if excess > thresh and holdings[ticker] > 1e-6 and total_val > 0:
                        p = _price_gbp(config, prices, gbpusd, ticker, reb_dt)
                        if p and p > 0:
                            shares_to_sell = (excess * total_val) / p
                            row = _row_sell(reb_dt, ticker, shares_to_sell)
                            if row:
                                _commit(row)
                        reb_dt += pd.Timedelta(minutes=rng.randint(2, 8))

                # Profit-taking: if a stock has risen >60% vs avg cost, trim 30%
                for ticker in list(available_tickers):
                    if holdings[ticker] < 1e-6:
                        continue
                    p = _price_gbp(config, prices, gbpusd, ticker, reb_dt)
                    if p is None:
                        continue
                    # Estimate avg cost from total_cost / shares (approximation)
                    # Use a heuristic: if current price > 1.6× the price 18 months ago
                    eighteen_m_ago = reb_dt - pd.Timedelta(days=540)
                    old_p = _price_gbp(config, prices, gbpusd, ticker, eighteen_m_ago)
                    if old_p and old_p > 0 and p / old_p > 1.6 and rng.random() < 0.35:
                        trim_shares = holdings[ticker] * rng.uniform(0.20, 0.35)
                        row = _row_sell(reb_dt, ticker, trim_shares)
                        if row:
                            _commit(row)
                        reb_dt += pd.Timedelta(minutes=rng.randint(5, 15))

                # Reinvest rebalancing proceeds into most underweight
                if cash > 300:
                    cw2 = _current_weights(reb_dt)
                    deficits2 = {
                        t: max(0.0, config["portfolio"][t]["target_weight"] - cw2.get(t, 0.0))
                        for t in available_tickers
                    }
                    total_d = sum(deficits2.values()) or 1.0
                    reinvest = cash * 0.80
                    reb_dt2 = reb_dt + pd.Timedelta(minutes=20)
                    for ticker, deficit in sorted(
                        deficits2.items(), key=lambda x: x[1], reverse=True
                    )[:4]:
                        alloc = (deficit / total_d) * reinvest
                        if alloc >= 100 and cash >= 100:
                            row = _row_buy(reb_dt2, ticker, min(alloc, cash * 0.4))
                            if row:
                                _commit(row)
                            reb_dt2 += pd.Timedelta(minutes=rng.randint(3, 12))

        # Advance one calendar month
        y, m = month.year, month.month
        if m == 12:
            month = pd.Timestamp(year=y + 1, month=1, day=1)
        else:
            month = pd.Timestamp(year=y, month=m + 1, day=1)
        month_idx += 1

    print(f"\n  Generated {len(rows)} transactions over {month_idx} months.")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Save + summary
# ══════════════════════════════════════════════════════════════════════════════

_CSV_COLUMNS = [
    "Action", "Time", "ISIN", "Ticker", "Name", "Notes", "ID",
    "No. of shares", "Price / share", "Currency (Price / share)",
    "Exchange rate", "Total", "Currency (Total)",
]


def save_csv(rows: list[dict], config: dict) -> Path:
    df = pd.DataFrame(rows, columns=_CSV_COLUMNS)
    out = Path(config["output_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def print_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows, columns=_CSV_COLUMNS)
    deps  = df[df["Action"] == "Deposit"]
    buys  = df[df["Action"] == "Market buy"]
    sells = df[df["Action"] == "Market sell"]

    print("\n" + "=" * 58)
    print("  Demo Portfolio Summary")
    print("=" * 58)
    print(f"  Total transactions : {len(df)}")
    print(f"  Deposits           : {len(deps)}  "
          f"(£{deps['Total'].astype(float).sum():,.0f} total)")
    print(f"  Buys               : {len(buys)}")
    print(f"  Sells              : {len(sells)}")
    print(f"  Unique tickers     : {buys['Ticker'].nunique()}")
    print(f"  Tickers            : {', '.join(sorted(buys['Ticker'].unique()))}")
    print(f"  Date range         : {df['Time'].min()[:10]} to {df['Time'].max()[:10]}")
    print("=" * 58)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    rng = random.Random(CONFIG["random_seed"])

    prices, gbpusd = fetch_prices(CONFIG)

    print("Generating transactions…")
    rows = generate_transactions(CONFIG, prices, gbpusd, rng)

    out_path = save_csv(rows, CONFIG)
    print_summary(rows)
    print(f"\nSaved to: {out_path}")
    print("\nTo load in the dashboard, go to page 11 — Transaction Portfolio.")


if __name__ == "__main__":
    main()
