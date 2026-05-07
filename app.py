"""Oman Vision 2040 Progress Cockpit — Streamlit demo dashboard."""

from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from src.data_loader import load_indicators, load_timeseries
from src.plots import (
    ACTUAL_COLOUR,
    PROJECTION_COLOUR,
    STATUS_COLOURS,
    STATUS_LABELS,
    progress_bars,
    trajectory_chart,
)
from src.projection import classify_status, compute_trend, progress_pct, project

DEFAULT_INDICATORS = [
    "real_gdp_growth",
    "non_oil_share_gdp",
    "fdi_net_inflow_gdp",
    "omanis_share_pvt_jobs",
]

EFFORT_MIN, EFFORT_MAX, EFFORT_STEP = 0.0, 3.0, 0.1
HORIZON_MIN, HORIZON_MAX = 2025, 2040


@st.cache_data
def _cached_indicators() -> pd.DataFrame:
    return load_indicators()


@st.cache_data
def _cached_timeseries() -> pd.DataFrame:
    return load_timeseries()


def _required_multiplier(trend: dict, target_2040: float, target_year: int = 2040) -> float | None:
    """Solve for the effort multiplier that makes the projection hit ``target_2040``.

    Returns ``None`` when the calibration is not well-defined (zero historic
    rate, divide-by-zero, or negative ratio under CAGR).
    """

    last_year = trend["last_year"]
    last_value = trend["last_value"]
    rate = trend["rate"]
    span = target_year - last_year
    if span <= 0 or rate == 0:
        return None

    if trend["method"] == "linear":
        return (target_2040 - last_value) / (span * rate)

    if last_value <= 0:
        return None
    ratio = target_2040 / last_value
    if ratio <= 0:
        return None
    needed_rate = ratio ** (1.0 / span) - 1.0
    return needed_rate / rate


def _compute_vision_aligned_multiplier(
    indicators_meta: pd.DataFrame,
    timeseries: pd.DataFrame,
    selected_ids: list[str],
) -> float:
    """Average of per-indicator multipliers that would hit each 2040 target."""

    multipliers: list[float] = []
    for ind_id in selected_ids:
        meta_rows = indicators_meta[indicators_meta["indicator_id"] == ind_id]
        if meta_rows.empty:
            continue
        meta = meta_rows.iloc[0]
        series = timeseries[timeseries["indicator_id"] == ind_id]
        if len(series) < 2:
            continue
        try:
            trend = compute_trend(series, meta)
        except ValueError:
            continue
        m = _required_multiplier(trend, float(meta["target_2040"]))
        if m is None or not math.isfinite(m):
            continue
        multipliers.append(max(EFFORT_MIN, min(EFFORT_MAX, m)))

    if not multipliers:
        return 1.0
    return float(sum(multipliers) / len(multipliers))


def _format_value(value: float, unit: str) -> str:
    if unit == "OMR":
        return f"{value:,.0f} OMR"
    if unit == "percent":
        return f"{value:.1f}%"
    if unit.startswith("score"):
        return f"{value:.3f}"
    return f"{value:,.2f}"


def _format_gap(gap: float, unit: str) -> str:
    sign = "+" if gap >= 0 else "−"
    abs_gap = abs(gap)
    if unit == "OMR":
        return f"{sign}{abs_gap:,.0f} OMR"
    if unit == "percent":
        return f"{sign}{abs_gap:.1f} pp"
    if unit.startswith("score"):
        return f"{sign}{abs_gap:.3f}"
    return f"{sign}{abs_gap:,.2f}"


def _status_badge_html(status: str) -> str:
    colour = STATUS_COLOURS.get(status, "#999999")
    label = STATUS_LABELS.get(status, status)
    return (
        f"<span style='display:inline-block;padding:2px 10px;border-radius:10px;"
        f"background-color:{colour};color:white;font-size:0.78rem;font-weight:600;"
        f"letter-spacing:0.02em;'>{label}</span>"
    )


def _build_progress_df(
    indicators_meta: pd.DataFrame,
    timeseries: pd.DataFrame,
    selected_ids: list[str],
    effort_multiplier: float,
    horizon_year: int,
) -> pd.DataFrame:
    rows = []
    for ind_id in selected_ids:
        meta = indicators_meta[indicators_meta["indicator_id"] == ind_id].iloc[0]
        series = timeseries[timeseries["indicator_id"] == ind_id]
        if len(series) < 2:
            continue
        trend = compute_trend(series, meta)
        proj = project(trend, effort_multiplier, horizon_year)
        projected_value = (
            float(proj["value"].iloc[-1]) if not proj.empty else trend["last_value"]
        )
        rows.append(
            {
                "indicator_id": ind_id,
                "indicator_name": meta["indicator_name"],
                "progress_pct": progress_pct(projected_value, meta),
                "status": classify_status(projected_value, meta, horizon_year),
            }
        )
    return pd.DataFrame(rows)


def _render_kpi_card(meta: pd.Series, series: pd.DataFrame, effort: float, horizon: int) -> None:
    if len(series) < 2:
        st.warning(f"Not enough data for {meta['indicator_name']}.")
        return

    trend = compute_trend(series, meta)
    proj = project(trend, effort, horizon)
    projected_value = (
        float(proj["value"].iloc[-1]) if not proj.empty else trend["last_value"]
    )
    last_actual_year = trend["last_year"]
    last_actual_value = trend["last_value"]
    gap_2030 = projected_value - float(meta["target_2030"])
    if meta["direction"] == "lower_better":
        gap_2030 = -gap_2030
    status = classify_status(projected_value, meta, horizon)
    unit = meta["unit"]

    with st.container(border=True):
        st.markdown(
            f"<div style='font-weight:600;color:{ACTUAL_COLOUR};font-size:0.95rem;"
            f"line-height:1.2;min-height:2.4rem;'>"
            f"{meta['indicator_name']}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(_status_badge_html(status), unsafe_allow_html=True)
        st.markdown(
            f"<div style='margin-top:0.6rem;font-size:0.75rem;color:#666;'>"
            f"Latest actual ({last_actual_year})</div>"
            f"<div style='font-size:1.2rem;font-weight:700;color:{ACTUAL_COLOUR};'>"
            f"{_format_value(last_actual_value, unit)}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='margin-top:0.4rem;font-size:0.75rem;color:#666;'>"
            f"Projected ({horizon})</div>"
            f"<div style='font-size:1.2rem;font-weight:700;color:{PROJECTION_COLOUR};'>"
            f"{_format_value(projected_value, unit)}</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='margin-top:0.4rem;font-size:0.75rem;color:#666;'>"
            f"Gap vs 2030 target ({_format_value(float(meta['target_2030']), unit)})</div>"
            f"<div style='font-size:1.0rem;font-weight:600;color:{ACTUAL_COLOUR};'>"
            f"{_format_gap(gap_2030, unit)}</div>",
            unsafe_allow_html=True,
        )


def main() -> None:
    st.set_page_config(
        page_title="Oman Vision 2040 Progress Cockpit",
        layout="wide",
    )

    indicators_meta = _cached_indicators()
    timeseries = _cached_timeseries()

    if "effort_multiplier" not in st.session_state:
        st.session_state["effort_multiplier"] = 1.0
    if "horizon_year" not in st.session_state:
        st.session_state["horizon_year"] = 2030

    with st.sidebar:
        st.header("Filters")
        pillars = ["All"] + sorted(indicators_meta["pillar"].unique().tolist())
        pillar = st.selectbox("Pillar", pillars, index=0)

        if pillar == "All":
            priority_pool = indicators_meta
        else:
            priority_pool = indicators_meta[indicators_meta["pillar"] == pillar]
        priority_options = sorted(priority_pool["priority"].unique().tolist())
        selected_priorities = st.multiselect(
            "Priorities",
            priority_options,
            default=priority_options,
        )

        indicator_pool = priority_pool[priority_pool["priority"].isin(selected_priorities)]
        indicator_options = indicator_pool["indicator_id"].tolist()
        indicator_labels = dict(
            zip(indicator_pool["indicator_id"], indicator_pool["indicator_name"])
        )
        default_indicators = [i for i in DEFAULT_INDICATORS if i in indicator_options]
        if not default_indicators:
            default_indicators = indicator_options[:4]

        selected_indicators = st.multiselect(
            "Indicators",
            indicator_options,
            default=default_indicators,
            format_func=lambda i: indicator_labels.get(i, i),
        )

        st.divider()
        st.header("Scenario")

        st.slider(
            "Effort multiplier (scales historic trend)",
            min_value=EFFORT_MIN,
            max_value=EFFORT_MAX,
            step=EFFORT_STEP,
            key="effort_multiplier",
            help="1.0 = continue at the historic pace. 0.0 = flatline at last actual. >1.0 = accelerate.",
        )
        st.slider(
            "Projection horizon",
            min_value=HORIZON_MIN,
            max_value=HORIZON_MAX,
            step=1,
            key="horizon_year",
        )

        st.markdown("**Scenario presets**")
        c1, c2, c3 = st.columns(3)
        if c1.button("Status Quo", use_container_width=True):
            st.session_state["effort_multiplier"] = 1.0
            st.rerun()
        if c2.button("Accelerated", use_container_width=True):
            st.session_state["effort_multiplier"] = 1.5
            st.rerun()
        if c3.button("Vision-Aligned", use_container_width=True):
            m = _compute_vision_aligned_multiplier(
                indicators_meta, timeseries, selected_indicators
            )
            st.session_state["effort_multiplier"] = round(
                max(EFFORT_MIN, min(EFFORT_MAX, m)), 1
            )
            st.rerun()

    effort_multiplier = float(st.session_state["effort_multiplier"])
    horizon_year = int(st.session_state["horizon_year"])

    st.title("Oman Vision 2040 Progress Cockpit")
    st.markdown(
        f"<div style='color:#666;margin-top:-0.5rem;margin-bottom:1.2rem;'>"
        f"Trajectory of national KPIs against 2030 and 2040 targets, with a "
        f"trend-based projection that responds to the effort multiplier and "
        f"horizon controls in the sidebar."
        f"</div>",
        unsafe_allow_html=True,
    )

    if not selected_indicators:
        st.info("Select at least one indicator from the sidebar to see the cockpit.")
        return

    st.subheader("Headline KPIs")
    st.caption(
        f"Projection at {horizon_year} under effort multiplier × {effort_multiplier:.1f}"
    )
    n_per_row = 4
    for row_start in range(0, len(selected_indicators), n_per_row):
        row_ids = selected_indicators[row_start : row_start + n_per_row]
        cols = st.columns(n_per_row)
        for col, ind_id in zip(cols, row_ids):
            meta = indicators_meta[indicators_meta["indicator_id"] == ind_id].iloc[0]
            series = timeseries[timeseries["indicator_id"] == ind_id]
            with col:
                _render_kpi_card(meta, series, effort_multiplier, horizon_year)

    st.subheader("Trajectories")
    st.caption("Solid: historic actuals. Dashed: trend-based projection. Dotted lines: 2030 / 2040 targets.")

    if len(selected_indicators) == 1:
        ind_id = selected_indicators[0]
        meta = indicators_meta[indicators_meta["indicator_id"] == ind_id].iloc[0]
        series = timeseries[timeseries["indicator_id"] == ind_id]
        trend = compute_trend(series, meta)
        proj = project(trend, effort_multiplier, horizon_year)
        st.plotly_chart(
            trajectory_chart(series, proj, meta),
            use_container_width=True,
        )
    else:
        for row_start in range(0, len(selected_indicators), 2):
            row_ids = selected_indicators[row_start : row_start + 2]
            cols = st.columns(2)
            for col, ind_id in zip(cols, row_ids):
                meta = indicators_meta[indicators_meta["indicator_id"] == ind_id].iloc[0]
                series = timeseries[timeseries["indicator_id"] == ind_id]
                trend = compute_trend(series, meta)
                proj = project(trend, effort_multiplier, horizon_year)
                with col:
                    st.plotly_chart(
                        trajectory_chart(series, proj, meta),
                        use_container_width=True,
                    )

    st.subheader("Progress to 2040 target")
    progress_df = _build_progress_df(
        indicators_meta,
        timeseries,
        selected_indicators,
        effort_multiplier,
        horizon_year,
    )
    st.plotly_chart(progress_bars(progress_df), use_container_width=True)

    st.divider()
    st.markdown(
        "**Sources** — Vision Document; Annual Reports 2021, 2022-2023, 2023-2024, 2024-2025 "
        "(Oman Ministry of Economy / Vision 2040 Implementation Follow-up Unit). "
        "Mid-series readings flagged `is_estimated = TRUE` in `data/vision2040_timeseries.csv` "
        "are linearly interpolated where source reports skipped a year. "
        "Where a newer report revised an older figure, the newer figure is used."
    )


if __name__ == "__main__":
    main()
