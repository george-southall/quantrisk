"""Structured logging setup for the QuantRisk platform."""

import logging
import sys

from quantrisk.config import settings


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for the given module name."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler (rotated daily)
    try:
        settings.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = settings.log_dir / "quantrisk.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    except OSError:
        pass  # file logging is best-effort

    logger.propagate = False
    return logger
