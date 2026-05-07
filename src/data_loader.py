"""Loaders for the two Vision 2040 CSVs.

Both loaders raise ``ValueError`` on any malformed row rather than silently
dropping or coercing. Callers should let the error propagate so problems in
the source data are surfaced loudly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INDICATORS_PATH = DATA_DIR / "vision2040_indicators.csv"
TIMESERIES_PATH = DATA_DIR / "vision2040_timeseries.csv"

INDICATOR_REQUIRED_COLS = [
    "indicator_id",
    "pillar",
    "priority",
    "indicator_name",
    "unit",
    "direction",
    "target_type",
    "target_2030",
    "target_2040",
    "target_2030_max",
    "target_2040_max",
    "baseline_year",
    "baseline_value",
    "source_notes",
]

TIMESERIES_REQUIRED_COLS = [
    "indicator_id",
    "year",
    "value",
    "source",
    "is_estimated",
]

VALID_DIRECTIONS = {"higher_better", "lower_better", "target_band"}
VALID_TARGET_TYPES = {"value", "floor", "ceiling", "band"}


def _check_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} CSV missing required columns: {missing}")


def load_indicators() -> pd.DataFrame:
    """Load the indicator metadata CSV.

    Raises ``ValueError`` if the file is missing required columns, has duplicate
    indicator IDs, or contains rows where required fields fail to parse.
    """

    if not INDICATORS_PATH.exists():
        raise ValueError(f"Indicator metadata CSV not found at {INDICATORS_PATH}")

    df = pd.read_csv(INDICATORS_PATH)
    _check_columns(df, INDICATOR_REQUIRED_COLS, "indicators")

    if df["indicator_id"].duplicated().any():
        dups = df.loc[df["indicator_id"].duplicated(), "indicator_id"].tolist()
        raise ValueError(f"Duplicate indicator_id values: {dups}")

    for col in ["indicator_id", "pillar", "priority", "indicator_name", "unit", "direction", "target_type"]:
        if df[col].isna().any():
            bad = df.loc[df[col].isna(), "indicator_id"].tolist()
            raise ValueError(f"Column '{col}' has missing values for rows: {bad}")

    bad_dir = df.loc[~df["direction"].isin(VALID_DIRECTIONS), "indicator_id"].tolist()
    if bad_dir:
        raise ValueError(f"Invalid 'direction' values for: {bad_dir}")

    bad_tt = df.loc[~df["target_type"].isin(VALID_TARGET_TYPES), "indicator_id"].tolist()
    if bad_tt:
        raise ValueError(f"Invalid 'target_type' values for: {bad_tt}")

    for col in ["target_2030", "target_2040", "baseline_value", "target_2030_max", "target_2040_max"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    df["baseline_year"] = pd.to_numeric(df["baseline_year"], errors="raise").astype(int)

    band_rows = df[df["target_type"] == "band"]
    if band_rows[["target_2030_max", "target_2040_max"]].isna().any().any():
        bad = band_rows.loc[
            band_rows[["target_2030_max", "target_2040_max"]].isna().any(axis=1),
            "indicator_id",
        ].tolist()
        raise ValueError(f"Band-type indicators missing band upper bounds: {bad}")

    return df


def load_timeseries() -> pd.DataFrame:
    """Load the indicator time series CSV.

    Sorted by ``indicator_id`` then ``year`` so consumers get a deterministic
    order. Raises ``ValueError`` on missing columns or unparseable values.
    """

    if not TIMESERIES_PATH.exists():
        raise ValueError(f"Time series CSV not found at {TIMESERIES_PATH}")

    df = pd.read_csv(TIMESERIES_PATH)
    _check_columns(df, TIMESERIES_REQUIRED_COLS, "timeseries")

    for col in ["indicator_id", "source"]:
        if df[col].isna().any():
            bad_idx = df.index[df[col].isna()].tolist()
            raise ValueError(f"Column '{col}' has missing values at rows: {bad_idx}")

    df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
    df["value"] = pd.to_numeric(df["value"], errors="raise")

    is_est = df["is_estimated"].astype(str).str.strip().str.upper()
    if not is_est.isin({"TRUE", "FALSE"}).all():
        bad_idx = df.index[~is_est.isin({"TRUE", "FALSE"})].tolist()
        raise ValueError(f"is_estimated must be TRUE/FALSE; bad rows: {bad_idx}")
    df["is_estimated"] = is_est == "TRUE"

    if df.duplicated(subset=["indicator_id", "year"]).any():
        dups = df.loc[df.duplicated(subset=["indicator_id", "year"]), ["indicator_id", "year"]]
        raise ValueError(f"Duplicate (indicator_id, year) rows: {dups.to_dict('records')}")

    return df.sort_values(["indicator_id", "year"]).reset_index(drop=True)
