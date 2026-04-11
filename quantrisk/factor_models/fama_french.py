"""
Fama-French 3-factor and 5-factor model regression.

Factor data is downloaded from Ken French's data library:
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html

The daily CSV zips are parsed directly via pandas.
"""

from __future__ import annotations

import io
import zipfile
from datetime import date

import pandas as pd
import requests
import statsmodels.api as sm

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


def _download_ff_factors(url: str, cache_key: str) -> pd.DataFrame:
    if cache_key in _FF_CACHE:
        return _FF_CACHE[cache_key]

    logger.info("Downloading Fama-French factors from %s", url)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        z = zipfile.ZipFile(io.BytesIO(resp.content))
        csv_name = [n for n in z.namelist() if n.endswith(".CSV") or n.endswith(".csv")][0]
        content = z.read(csv_name).decode("utf-8", errors="ignore")
    except Exception as exc:
        logger.error("Failed to download FF factors: %s", exc)
        return pd.DataFrame()

    # Find the data section (skip header commentary lines)
    lines = content.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if line.strip() and line.strip()[0].isdigit():
            start = i
            break

    # Find the end (annual averages section starts with blank line)
    end = len(lines)
    for i in range(start, len(lines)):
        stripped = lines[i].strip()
        if stripped == "" and i > start + 10:
            end = i
            break

    data_lines = "\n".join(lines[start:end])
    df = pd.read_csv(
        io.StringIO(data_lines),
        header=None,
        index_col=0,
        skipinitialspace=True,
    )
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d", errors="coerce")
    df = df[df.index.notna()]
    df = df.apply(pd.to_numeric, errors="coerce") / 100  # convert from % to decimal

    _FF_CACHE[cache_key] = df
    logger.info("Downloaded %d rows of FF factors", len(df))
    return df


def get_ff3_factors(start: str, end: str | None = None) -> pd.DataFrame:
    """
    Return daily Fama-French 3 factors aligned to [start, end].

    Columns: Mkt-RF, SMB, HML, RF
    """
    if end is None:
        end = date.today().isoformat()
    df = _download_ff_factors(FF3_URL, "ff3")
    if df.empty:
        return df
    df.columns = ["Mkt-RF", "SMB", "HML", "RF"]
    return df.loc[start:end]


def get_ff5_factors(start: str, end: str | None = None) -> pd.DataFrame:
    """
    Return daily Fama-French 5 factors aligned to [start, end].

    Columns: Mkt-RF, SMB, HML, RMW, CMA, RF
    """
    if end is None:
        end = date.today().isoformat()
    df = _download_ff_factors(FF5_URL, "ff5")
    if df.empty:
        return df
    df.columns = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"]
    return df.loc[start:end]


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
            factors = get_ff3_factors(_start, _end)
            factor_cols = ["Mkt-RF", "SMB", "HML"]
        else:
            factors = get_ff5_factors(_start, _end)
            factor_cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]

        if factors.empty:
            raise RuntimeError("Could not download Fama-French factors")

        self._factors = factors

        # Align portfolio returns with factor data
        excess_returns = portfolio_returns - factors["RF"]
        aligned = pd.concat(
            [excess_returns.rename("Rp-Rf"), factors[factor_cols]], axis=1
        ).dropna()
        self._aligned = aligned

        # OLS regression: Rp - Rf = α + β1*Mkt-RF + β2*SMB + ...
        y = aligned["Rp-Rf"]
        X = sm.add_constant(aligned[factor_cols])
        self._result = sm.OLS(y, X).fit()

        logger.info(
            "FF%d model fitted: R²=%.3f, α=%.4f (t=%.2f)",
            self.n_factors,
            self._result.rsquared,
            self._result.params["const"],
            self._result.tvalues["const"],
        )
        return self

    def report(self) -> pd.DataFrame:
        """
        Return a tidy DataFrame of factor loadings, t-stats, and p-values.
        """
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
