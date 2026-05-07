"""Data loading utilities for the Bank Marketing dataset."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Default location relative to the project root
_DEFAULT_CSV = Path(__file__).resolve().parents[5] / "data" / "bank-full.csv"


def load_bank_data(path: str | Path | None = None) -> pd.DataFrame:
    """Load bank-full.csv and return a DataFrame.

    Searches (in order):
    1. The explicit ``path`` argument.
    2. ``<project_root>/data/bank-full.csv``.
    3. Any ``bank-full.csv`` found under the current working directory.
    """
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    candidates.append(_DEFAULT_CSV)
    candidates += list(Path.cwd().rglob("bank-full.csv"))

    for p in candidates:
        if p.exists():
            logger.info("Loading data from %s", p)
            return pd.read_csv(p, sep=";")

    raise FileNotFoundError(
        "bank-full.csv not found. Place the file at 'data/bank-full.csv' "
        "relative to the project root or pass an explicit path."
    )


def basic_info(df: pd.DataFrame) -> dict:
    """Return a dict with basic dataset statistics."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "target_counts": df["y"].value_counts().to_dict() if "y" in df.columns else {},
    }
