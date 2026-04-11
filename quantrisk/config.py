"""Application-wide configuration via pydantic-settings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # FRED
    fred_api_key: str = ""

    # Storage
    cache_dir: Path = Path("data")
    db_path: Path = Path("data/quantrisk_cache.db")

    # Logging
    log_level: str = "INFO"
    log_dir: Path = Path("logs")

    # Portfolio defaults
    default_start_date: str = "2015-01-01"
    default_benchmark: str = "SPY"

    # Risk defaults
    var_confidence_levels: list[float] = [0.95, 0.99]
    var_horizon_days: int = 1
    mc_num_simulations: int = 10_000
    mc_horizon_days: int = 252
    risk_free_rate_fallback: float = 0.05  # used if FRED is unavailable

    # Backtest defaults
    transaction_cost_bps: float = 10.0  # 10 basis points per trade
    slippage_bps: float = 5.0

    @property
    def raw_data_dir(self) -> Path:
        return self.cache_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.cache_dir / "processed"

    def ensure_dirs(self) -> None:
        """Create all required directories if they don't exist."""
        for d in [self.raw_data_dir, self.processed_data_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
