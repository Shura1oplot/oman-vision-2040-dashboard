"""Plotly figures for the Vision 2040 cockpit."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

ACTUAL_COLOUR = "#264653"
PROJECTION_COLOUR = "#2A9D8F"
TARGET_COLOUR = "#E9C46A"
OFF_TRACK_COLOUR = "#E76F51"

STATUS_COLOURS = {
    "on_track": "#2A9D8F",
    "at_risk": "#E9C46A",
    "off_track": "#E76F51",
}

STATUS_LABELS = {
    "on_track": "On track",
    "at_risk": "At risk",
    "off_track": "Off track",
}

X_AXIS_RANGE = [2020, 2040]


def _y_range_with_padding(values: list[float]) -> tuple[float, float]:
    finite = [v for v in values if v is not None and pd.notna(v)]
    if not finite:
        return (0.0, 1.0)
    lo = min(finite)
    hi = max(finite)
    if lo == hi:
        pad = abs(lo) * 0.1 if lo != 0 else 1.0
        return (lo - pad, hi + pad)
    pad = (hi - lo) * 0.1
    return (lo - pad, hi + pad)


def trajectory_chart(
    actuals: pd.DataFrame,
    projection: pd.DataFrame,
    indicator_meta: pd.Series,
) -> go.Figure:
    """Time series with historical actuals, dashed projection, and target lines."""

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=actuals["year"],
            y=actuals["value"],
            mode="lines+markers",
            name="Actuals",
            line=dict(color=ACTUAL_COLOUR, width=2.5),
            marker=dict(size=7, color=ACTUAL_COLOUR),
            hovertemplate="%{x}: %{y:.2f}<extra>Actual</extra>",
        )
    )

    if not projection.empty and not actuals.empty:
        last_actual_year = int(actuals["year"].iloc[-1])
        last_actual_value = float(actuals["value"].iloc[-1])
        proj_x = [last_actual_year, *projection["year"].tolist()]
        proj_y = [last_actual_value, *projection["value"].tolist()]
        fig.add_trace(
            go.Scatter(
                x=proj_x,
                y=proj_y,
                mode="lines",
                name="Projection",
                line=dict(color=PROJECTION_COLOUR, width=2.5, dash="dash"),
                hovertemplate="%{x}: %{y:.2f}<extra>Projection</extra>",
            )
        )

    target_type = indicator_meta["target_type"]
    target_2030 = float(indicator_meta["target_2030"])
    target_2040 = float(indicator_meta["target_2040"])
    target_2030_max = indicator_meta.get("target_2030_max")
    target_2040_max = indicator_meta.get("target_2040_max")
    target_2030_max = float(target_2030_max) if pd.notna(target_2030_max) else None
    target_2040_max = float(target_2040_max) if pd.notna(target_2040_max) else None

    extra_y: list[float] = []

    if target_type == "band" and target_2030_max is not None and target_2040_max is not None:
        fig.add_shape(
            type="rect",
            x0=2025,
            x1=2030,
            y0=target_2030,
            y1=target_2030_max,
            line=dict(width=0),
            fillcolor=TARGET_COLOUR,
            opacity=0.18,
            layer="below",
        )
        fig.add_shape(
            type="rect",
            x0=2030,
            x1=2040,
            y0=target_2040,
            y1=target_2040_max,
            line=dict(width=0),
            fillcolor=TARGET_COLOUR,
            opacity=0.18,
            layer="below",
        )
        fig.add_annotation(
            x=2030,
            y=target_2030_max,
            text=f"2030 band: {target_2030:g}–{target_2030_max:g}",
            showarrow=False,
            yshift=10,
            font=dict(color=ACTUAL_COLOUR, size=11),
        )
        fig.add_annotation(
            x=2040,
            y=target_2040_max,
            text=f"2040 band: {target_2040:g}–{target_2040_max:g}",
            showarrow=False,
            yshift=10,
            xanchor="right",
            font=dict(color=ACTUAL_COLOUR, size=11),
        )
        extra_y.extend([target_2030, target_2030_max, target_2040, target_2040_max])
    else:
        fig.add_hline(
            y=target_2030,
            line=dict(color=TARGET_COLOUR, width=2, dash="dot"),
            annotation_text=f"2030 target: {target_2030:g}",
            annotation_position="top left",
            annotation_font_color=ACTUAL_COLOUR,
        )
        if target_2040 != target_2030:
            fig.add_hline(
                y=target_2040,
                line=dict(color=TARGET_COLOUR, width=2, dash="dot"),
                annotation_text=f"2040 target: {target_2040:g}",
                annotation_position="top right",
                annotation_font_color=ACTUAL_COLOUR,
            )
        extra_y.extend([target_2030, target_2040])

    fig.add_vline(
        x=2030,
        line=dict(color="#999999", width=1, dash="dot"),
        opacity=0.6,
    )
    fig.add_vline(
        x=2040,
        line=dict(color="#999999", width=1, dash="dot"),
        opacity=0.6,
    )

    all_y = (
        actuals["value"].tolist()
        + (projection["value"].tolist() if not projection.empty else [])
        + extra_y
    )
    y_lo, y_hi = _y_range_with_padding(all_y)

    unit = indicator_meta.get("unit", "")
    fig.update_layout(
        title=dict(
            text=f"<b>{indicator_meta['indicator_name']}</b>",
            x=0.0,
            xanchor="left",
            font=dict(size=15, color=ACTUAL_COLOUR),
        ),
        xaxis=dict(
            title="Year",
            range=X_AXIS_RANGE,
            dtick=2,
            gridcolor="#EEEEEE",
        ),
        yaxis=dict(
            title=unit,
            range=[y_lo, y_hi],
            gridcolor="#EEEEEE",
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            x=0.0,
        ),
        margin=dict(l=50, r=30, t=60, b=60),
        height=360,
    )
    return fig


def progress_bars(progress_df: pd.DataFrame) -> go.Figure:
    """Horizontal bars showing % of baseline-to-2040 journey completed."""

    if progress_df.empty:
        fig = go.Figure()
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[
                dict(
                    text="No indicators selected",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=14, color=ACTUAL_COLOUR),
                )
            ],
            height=200,
            plot_bgcolor="white",
            paper_bgcolor="white",
        )
        return fig

    df = progress_df.sort_values("progress_pct", ascending=True).reset_index(drop=True)
    colours = [STATUS_COLOURS.get(s, "#999999") for s in df["status"]]
    text_labels = [
        f"{p:.0f}%  ({STATUS_LABELS.get(s, s)})"
        for p, s in zip(df["progress_pct"], df["status"])
    ]

    fig = go.Figure(
        go.Bar(
            x=df["progress_pct"],
            y=df["indicator_name"],
            orientation="h",
            marker=dict(color=colours),
            text=text_labels,
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Progress: %{x:.1f}%<extra></extra>",
        )
    )
    fig.add_vline(
        x=100,
        line=dict(color=ACTUAL_COLOUR, width=1.5, dash="dot"),
        annotation_text="2040 target",
        annotation_position="top",
        annotation_font_color=ACTUAL_COLOUR,
    )
    fig.update_layout(
        title=dict(
            text="<b>Progress to 2040 target</b>",
            x=0.0,
            xanchor="left",
            font=dict(size=15, color=ACTUAL_COLOUR),
        ),
        xaxis=dict(
            title="Percent of baseline-to-2040 journey",
            range=[0, 165],
            gridcolor="#EEEEEE",
            ticksuffix="%",
        ),
        yaxis=dict(title=""),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=80, t=60, b=60),
        height=max(220, 60 + 40 * len(df)),
        showlegend=False,
    )
    return fig
