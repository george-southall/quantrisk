"""
Fama-French 3-factor and 5-factor model regression.

Factor data priority:
  1. Ken French's data library (Dartmouth) — canonical source
  2. Disk cache (data/ff3_factors.parquet / data/ff5_factors.parquet) — used when
     the remote source is unreachable but was successfully downloaded before
  3. ETF-proxy synthetic factors — last resort fallback using yfinance ETFs:
       Mkt-RF : SPY - RF
       SMB    : IWM - SPY  (small-cap minus large-cap)
       HML    : IVE - IVW  (S&P 500 Value minus S&P 500 Growth)
       RMW    : QUAL - SPY (quality/profitability proxy)
       CMA    : USMV - SPY (low-vol / conservative investment proxy)
     Note: ETF proxies are approximate; use official factors when possible.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import statsmodels.api as sm
import yfinance as yf

from quantrisk.config import settings
from quantrisk.utils.logger import get_logger

logger = get_logger(__name__)

# Ken French data library URLs (daily factors)
FF3_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
FF5_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
)

_FF_CACHE: dict[str, pd.DataFrame] = {}


# ── Download from Dartmouth ────────────────────────────────────────────────────

def _download_ff_factors(url: str) -> pd.DataFrame:
    """Attempt to download and parse FF factors from the Dartmouth URL."""
    logger.info("Downloading Fama-French factors from %s", url)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; research-tool/1.0)"}
    resp = requests.get(url, timeout=20, headers=headers)
    resp.raise_for_status()

    z = zipfile.ZipFile(io.BytesIO(resp.content))
    csv_name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
    content = z.read(csv_name).decode("utf-8", errors="ignore")

    lines = content.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip() and line.strip()[0].isdigit():
            start = i
            break

    end = len(lines)
    for i in range(start, len(lines)):
        if lines[i].strip() == "" and i > start + 10:
            end = i
            break

    df = pd.read_csv(
        io.StringIO("\n".join(lines[start:end])),
        header=None,
        index_col=0,
        skipinitialspace=True,
    )
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d", errors="coerce")
    df = df[df.index.notna()]
    df = df.apply(pd.to_numeric, errors="coerce") / 100
    df.index.name = "Date"
    logger.info("Downloaded %d rows of FF factors", len(df))
    return df


# ── Disk cache helpers ─────────────────────────────────────────────────────────

def _cache_path(key: str) -> Path:
    settings.ensure_dirs()
    return settings.processed_data_dir / f"{key}_factors.parquet"


def _save_to_disk(df: pd.DataFrame, key: str) -> None:
    try:
        df.to_parquet(_cache_path(key))
        logger.info("Saved %s factors to disk cache", key)
    except Exception as exc:
        logger.warning("Could not save FF factors to disk: %s", exc)


def _load_from_disk(key: str) -> pd.DataFrame:
    path = _cache_path(key)
    if path.exists():
        logger.info("Loading %s factors from disk cache", key)
        return pd.read_parquet(path)
    return pd.DataFrame()


# ── ETF-proxy synthetic factors ────────────────────────────────────────────────

_ETF_CACHE: dict[str, pd.DataFrame] = {}

_FF3_ETFS = ["SPY", "IWM", "IVE", "IVW"]
_FF5_ETFS = ["SPY", "IWM", "IVE", "IVW", "QUAL", "USMV"]


def _etf_proxy_factors(
    start: str,
    end: str,
    n_factors: int,
    risk_free_rate: float = 0.05,
) -> pd.DataFrame:
    """
    Build synthetic FF factors from ETF returns via yfinance.

    Columns match the official FF naming: Mkt-RF, SMB, HML[, RMW, CMA], RF.
    """
    cache_key = f"etf_{n_factors}_{start}_{end}"
    if cache_key in _ETF_CACHE:
        return _ETF_CACHE[cache_key]

    tickers = _FF5_ETFS if n_factors == 5 else _FF3_ETFS
    logger.info("Fetching ETF proxy factors: %s", tickers)

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if raw.empty:
        return pd.DataFrame()

    # yfinance >=0.2 returns MultiIndex columns: (metric, ticker)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    elif "Close" in raw.columns:
        prices = raw["Close"]
    else:
        prices = raw

    rets = prices.pct_change().dropna()

    rf_daily = risk_free_rate / 252

    factors = pd.DataFrame(index=rets.index)
    factors["Mkt-RF"] = rets["SPY"] - rf_daily
    factors["SMB"] = rets["IWM"] - rets["SPY"]
    factors["HML"] = rets["IVE"] - rets["IVW"]
    if n_factors == 5:
        factors["RMW"] = rets["QUAL"] - rets["SPY"]
        factors["CMA"] = rets["USMV"] - rets["SPY"]
    factors["RF"] = rf_daily

    factors = factors.dropna()
    _ETF_CACHE[cache_key] = factors
    logger.info("Built %d-factor ETF proxies (%d rows)", n_factors, len(factors))
    return factors


# ── Public factor accessors ────────────────────────────────────────────────────

def _get_factors(
    url: str,
    key: str,
    columns: list[str],
    start: str,
    end: str,
    n_factors: int,
) -> tuple[pd.DataFrame, bool]:
    """
    Return (factors_df, using_proxies), trying: remote → disk cache → ETF proxies.
    """
    # 1. In-memory cache
    if key in _FF_CACHE:
        df = _FF_CACHE[key]
        df.columns = columns
        return df.loc[start:end], False

    # 2. Remote download
    try:
        df = _download_ff_factors(url)
        df.columns = columns
        _save_to_disk(df, key)
        _FF_CACHE[key] = df
        return df.loc[start:end], False
    except Exception as exc:
        logger.warning("Remote FF download failed (%s); trying disk cache.", exc)

    # 3. Disk cache
    df = _load_from_disk(key)
    if not df.empty:
        df.columns = columns
        _FF_CACHE[key] = df
        return df.loc[start:end], False

    # 4. ETF proxies
    logger.warning(
        "FF remote and disk cache unavailable — using ETF-proxy synthetic factors. "
        "Results are approximate."
    )
    rf = settings.risk_free_rate_fallback
    return _etf_proxy_factors(start, end, n_factors=n_factors, risk_free_rate=rf), True


def get_ff3_factors(start: str, end: str | None = None) -> pd.DataFrame:
    """Return daily FF3 factors aligned to [start, end]. Columns: Mkt-RF, SMB, HML, RF."""
    _end = end or date.today().isoformat()
    df, _ = _get_factors(FF3_URL, "ff3", ["Mkt-RF", "SMB", "HML", "RF"], start, _end, 3)
    return df


def get_ff5_factors(start: str, end: str | None = None) -> pd.DataFrame:
    """Return daily FF5 factors aligned to [start, end]. Columns: Mkt-RF, SMB, HML, RMW, CMA, RF."""
    _end = end or date.today().isoformat()
    cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    df, _ = _get_factors(FF5_URL, "ff5", cols, start, _end, 5)
    return df


# ── Model ──────────────────────────────────────────────────────────────────────

class FamaFrenchModel:
    """
    Fama-French 3-factor or 5-factor regression.

    Usage
    -----
    model = FamaFrenchModel(n_factors=5).fit(portfolio_returns, start, end)
    model.report()
    """

    FACTOR_DESCRIPTIONS = {
        "Mkt-RF": "Market excess return (systematic market risk)",
        "SMB": "Small-minus-Big (size premium: small cap tilt)",
        "HML": "High-minus-Low (value premium: value stock tilt)",
        "RMW": "Robust-minus-Weak (profitability premium)",
        "CMA": "Conservative-minus-Aggressive (investment premium)",
    }

    def __init__(self, n_factors: int = 3):
        if n_factors not in (3, 5):
            raise ValueError("n_factors must be 3 or 5")
        self.n_factors = n_factors
        self._result = None
        self._factors: pd.DataFrame | None = None
        self._aligned: pd.DataFrame | None = None
        self.using_proxies: bool = False

    def fit(
        self,
        portfolio_returns: pd.Series,
        start: str | None = None,
        end: str | None = None,
    ) -> "FamaFrenchModel":
        """Fit the factor model via OLS regression."""
        _start = start or str(portfolio_returns.index[0].date())
        _end = end or str(portfolio_returns.index[-1].date())

        if self.n_factors == 3:
            factors, self.using_proxies = _get_factors(
                FF3_URL, "ff3", ["Mkt-RF", "SMB", "HML", "RF"], _start, _end, 3
            )
            factor_cols = ["Mkt-RF", "SMB", "HML"]
        else:
            factors, self.using_proxies = _get_factors(
                FF5_URL, "ff5", ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"], _start, _end, 5
            )
            factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

        if factors.empty:
            raise RuntimeError(
                "Could not obtain Fama-French factors from any source "
                "(remote, disk cache, or ETF proxies)."
            )

        self._factors = factors

        excess_returns = portfolio_returns - factors["RF"]
        aligned = pd.concat(
            [excess_returns.rename("Rp-Rf"), factors[factor_cols]], axis=1
        ).dropna()
        self._aligned = aligned

        y = aligned["Rp-Rf"]
        X = sm.add_constant(aligned[factor_cols])
        self._result = sm.OLS(y, X).fit()

        logger.info(
            "FF%d model fitted: R2=%.3f, alpha=%.4f (t=%.2f)",
            self.n_factors,
            self._result.rsquared,
            self._result.params["const"],
            self._result.tvalues["const"],
        )
        return self

    def report(self) -> pd.DataFrame:
        """Return a tidy DataFrame of factor loadings, t-stats, and p-values."""
        if self._result is None:
            raise RuntimeError("Call .fit() first")
        res = self._result
        rows = []
        for param in res.params.index:
            label = "alpha" if param == "const" else param
            rows.append({
                "factor": label,
                "loading": res.params[param],
                "t_stat": res.tvalues[param],
                "p_value": res.pvalues[param],
                "significant": res.pvalues[param] < 0.05,
                "description": self.FACTOR_DESCRIPTIONS.get(param, "Intercept (skill / alpha)"),
            })
        return pd.DataFrame(rows).set_index("factor")

    def print_report(self) -> None:
        if self._result is None:
            raise RuntimeError("Call .fit() first")
        r = self._result
        rep = self.report()
        print(f"\n{'='*60}")
        print(f"  Fama-French {self.n_factors}-Factor Model")
        print(f"  R² = {r.rsquared:.4f}   Adj R² = {r.rsquared_adj:.4f}")
        print(f"  F-stat = {r.fvalue:.2f}   Observations = {int(r.nobs)}")
        print(f"{'='*60}")
        print(f"  {'Factor':<12} {'Loading':>10} {'t-stat':>8} {'p-value':>8} {'Sig':>5}")
        print(f"  {'-'*50}")
        for factor, row in rep.iterrows():
            sig = "***" if row["p_value"] < 0.01 else ("**" if row["p_value"] < 0.05 else "")
            print(
                f"  {factor:<12} {row['loading']:>10.4f} {row['t_stat']:>8.2f}"
                f" {row['p_value']:>8.4f} {sig:>5}"
            )
        print(f"{'='*60}\n")
        print("  Interpretation:")
        for factor, row in rep.iterrows():
            if factor != "alpha" and row["significant"]:
                direction = "positive" if row["loading"] > 0 else "negative"
                print(f"  - {factor}: {direction} tilt → {row['description']}")

    @property
    def alpha(self) -> float:
        """Daily alpha (intercept) from the regression."""
        if self._result is None:
            raise RuntimeError("Call .fit() first")
        return float(self._result.params["const"])

    @property
    def r_squared(self) -> float:
        if self._result is None:
            raise RuntimeError("Call .fit() first")
        return float(self._result.rsquared)
