"""Trend, projection and status logic for Vision 2040 indicators.

The two projection methods follow section 3.2 of the concept document:
- ``cagr`` for indicators whose actuals stay strictly positive and at least 5
- ``linear`` (year-on-year delta) otherwise

Both project forward from the latest actual under an effort multiplier that
scales the historic rate.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

Status = Literal["on_track", "at_risk", "off_track"]


def compute_trend(series: pd.DataFrame, indicator_meta: pd.Series) -> dict:
    """Fit a historic trend to the actuals for one indicator.

    Parameters
    ----------
    series:
        Subset of the timeseries dataframe for a single indicator. Must have
        ``year`` and ``value`` columns, sorted by year.
    indicator_meta:
        Indicator metadata row (currently unused but kept on the signature so
        future per-indicator overrides can be slotted in without breaking
        callers).
    """

    del indicator_meta

    if series.empty:
        raise ValueError("Cannot compute trend on empty series")
    if len(series) < 2:
        raise ValueError(
            f"Need at least 2 observations to compute a trend, got {len(series)}"
        )

    s = series.sort_values("year").reset_index(drop=True)
    first_year = int(s["year"].iloc[0])
    last_year = int(s["year"].iloc[-1])
    first_value = float(s["value"].iloc[0])
    last_value = float(s["value"].iloc[-1])
    span = last_year - first_year
    if span <= 0:
        raise ValueError("Series spans zero years; cannot compute trend")

    use_cagr = (s["value"] > 0).all() and s["value"].min() >= 5
    if use_cagr:
        rate = (last_value / first_value) ** (1.0 / span) - 1.0
        method = "cagr"
    else:
        rate = (last_value - first_value) / span
        method = "linear"

    return {
        "method": method,
        "rate": float(rate),
        "first_year": first_year,
        "first_value": first_value,
        "last_year": last_year,
        "last_value": last_value,
    }


def project(trend: dict, effort_multiplier: float, horizon_year: int) -> pd.DataFrame:
    """Project the trend forward from ``last_year + 1`` to ``horizon_year``.

    Returns a dataframe with columns ``year`` and ``value``. If the horizon is
    at or before the last actual year, returns an empty dataframe with those
    columns.
    """

    last_year = int(trend["last_year"])
    last_value = float(trend["last_value"])
    rate = float(trend["rate"]) * float(effort_multiplier)
    method = trend["method"]

    if horizon_year <= last_year:
        return pd.DataFrame({"year": pd.Series(dtype=int), "value": pd.Series(dtype=float)})

    years = np.arange(last_year + 1, horizon_year + 1, dtype=int)
    if method == "linear":
        values = last_value + (years - last_year) * rate
    elif method == "cagr":
        values = last_value * np.power(1.0 + rate, years - last_year)
    else:
        raise ValueError(f"Unknown trend method: {method!r}")

    return pd.DataFrame({"year": years, "value": values.astype(float)})


def _target_for_year(indicator_meta: pd.Series, target_year: int) -> tuple[float, float | None]:
    """Pick the target (and band upper bound) closest to the requested year.

    Returns ``(target_low, target_high)`` where ``target_high`` is ``None``
    unless the indicator is a band-type target.
    """

    if target_year <= 2030:
        target = float(indicator_meta["target_2030"])
        upper = indicator_meta.get("target_2030_max")
    else:
        target = float(indicator_meta["target_2040"])
        upper = indicator_meta.get("target_2040_max")
    upper = float(upper) if pd.notna(upper) else None
    return target, upper


def classify_status(
    projected_value: float,
    indicator_meta: pd.Series,
    target_year: int,
) -> Status:
    """Classify a projected value against the relevant target.

    Implements the rules in section 3.2.3 of the concept document.
    """

    target_type = indicator_meta["target_type"]
    direction = indicator_meta["direction"]
    target, upper = _target_for_year(indicator_meta, target_year)

    if target_type == "value" and direction == "higher_better":
        if projected_value >= target:
            return "on_track"
        if projected_value >= 0.85 * target:
            return "at_risk"
        return "off_track"

    if target_type == "value" and direction == "lower_better":
        if projected_value <= target:
            return "on_track"
        if projected_value <= 1.15 * target:
            return "at_risk"
        return "off_track"

    if target_type == "floor":
        return "on_track" if projected_value >= target else "off_track"

    if target_type == "ceiling":
        if projected_value > target:
            return "off_track"
        if projected_value >= 0.9 * target:
            return "at_risk"
        return "on_track"

    if target_type == "band":
        if upper is None:
            raise ValueError(
                f"Band target for {indicator_meta['indicator_id']} missing upper bound"
            )
        low, high = target, upper
        if low <= projected_value <= high:
            return "on_track"
        if (low - 1) <= projected_value <= (high + 1):
            return "at_risk"
        return "off_track"

    raise ValueError(
        f"Unhandled (target_type={target_type!r}, direction={direction!r}) combination"
    )


def progress_pct(projected_value: float, indicator_meta: pd.Series) -> float:
    """Percent of journey from baseline to 2040 target, clipped to [0, 150]."""

    baseline = float(indicator_meta["baseline_value"])
    target_2040 = float(indicator_meta["target_2040"])
    denom = target_2040 - baseline
    if denom == 0:
        return 0.0
    pct = (projected_value - baseline) / denom * 100.0
    return float(np.clip(pct, 0.0, 150.0))
