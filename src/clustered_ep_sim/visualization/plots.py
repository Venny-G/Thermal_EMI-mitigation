"""Plotly visualization helpers."""

from __future__ import annotations

import math

import plotly.graph_objects as go

from clustered_ep_sim.models.layout import Scenario
from clustered_ep_sim.models.risk import FieldPeak, RiskReport

STATUS_COLORS = {
    "safe": "#2e8b57",
    "caution": "#d97706",
    "critical": "#b91c1c",
}


def _subsystem_hover_text(report: RiskReport) -> dict[str, str]:
    hover = {}
    for assessment in report.assessments:
        hover[assessment.name] = (
            f"{assessment.name}<br>"
            f"Overall: {assessment.overall_state.title()}<br>"
            f"Thermal ratio: {assessment.thermal_ratio:.2f}<br>"
            f"EMI ratio: {assessment.emi_ratio:.2f}<br>"
            f"Dominant driver: {assessment.dominant_driver}<br>"
            f"Dominant thruster: {assessment.dominant_thruster}"
        )
    return hover


def make_field_figure(
    scenario: Scenario,
    grid_x,
    grid_y,
    field,
    report: RiskReport,
    *,
    title: str,
    colorbar_title: str,
    colorscale: str,
    peak: FieldPeak,
    contour_thresholds: tuple[float, ...] = (),
) -> go.Figure:
    """Create a spacecraft layout field map with annotated hardware."""

    x_axis = grid_x[0, :]
    y_axis = grid_y[:, 0]
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=x_axis,
            y=y_axis,
            z=field,
            colorscale=colorscale,
            colorbar={"title": colorbar_title},
            hovertemplate="x=%{x:.2f} m<br>y=%{y:.2f} m<br>value=%{z:.2f}<extra></extra>",
        )
    )

    if contour_thresholds:
        start = min(contour_thresholds)
        end = max(contour_thresholds)
        size = contour_thresholds[1] - contour_thresholds[0] if len(contour_thresholds) > 1 else 0.25
        fig.add_trace(
            go.Contour(
                x=x_axis,
                y=y_axis,
                z=field,
                contours={"start": start, "end": end, "size": size, "coloring": "none"},
                line={"color": "white", "width": 1.5},
                showscale=False,
                hoverinfo="skip",
            )
        )

    fig.add_shape(
        type="rect",
        x0=0.0,
        y0=0.0,
        x1=scenario.bus.width_m,
        y1=scenario.bus.height_m,
        line={"color": "#111827", "width": 2.0},
        fillcolor="rgba(0,0,0,0)",
    )

    subsystem_text = _subsystem_hover_text(report)
    fig.add_trace(
        go.Scatter(
            x=[subsystem.x_m for subsystem in scenario.subsystems],
            y=[subsystem.y_m for subsystem in scenario.subsystems],
            mode="markers+text",
            text=[subsystem.name for subsystem in scenario.subsystems],
            textposition="top right",
            marker={
                "size": 11,
                "symbol": "circle",
                "color": [
                    STATUS_COLORS[next(assessment.overall_state for assessment in report.assessments if assessment.name == subsystem.name)]
                    for subsystem in scenario.subsystems
                ],
                "line": {"color": "black", "width": 1.0},
            },
            hovertext=[subsystem_text[subsystem.name] for subsystem in scenario.subsystems],
            hoverinfo="text",
            name="Subsystems",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[thruster.x_m for thruster in scenario.thrusters],
            y=[thruster.y_m for thruster in scenario.thrusters],
            mode="markers+text",
            text=[thruster.name for thruster in scenario.thrusters],
            textposition="bottom center",
            marker={
                "size": 12,
                "symbol": "x",
                "color": "black",
                "line": {"color": "black", "width": 1.0},
            },
            hovertemplate="%{text}<extra>Thruster</extra>",
            name="Thrusters",
        )
    )

    for thruster in scenario.thrusters:
        arrow_length = 0.10
        dx = arrow_length * math.cos(thruster.orientation_rad)
        dy = arrow_length * math.sin(thruster.orientation_rad)
        fig.add_annotation(
            x=thruster.x_m + dx,
            y=thruster.y_m + dy,
            ax=thruster.x_m,
            ay=thruster.y_m,
            axref="x",
            ayref="y",
            arrowhead=3,
            arrowsize=1.0,
            arrowwidth=1.6,
            arrowcolor="black",
            showarrow=True,
            text="",
        )

    fig.add_annotation(
        x=peak.x_m,
        y=peak.y_m,
        text=f"Peak {peak.value:.2f}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1.0,
        arrowcolor="black",
        bgcolor="rgba(255,255,255,0.75)",
    )

    fig.update_layout(
        title=title,
        template="plotly_white",
        height=570,
        margin={"l": 18, "r": 18, "t": 50, "b": 18},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01, "xanchor": "left", "x": 0.0},
    )
    fig.update_xaxes(title="Bus X [m]", range=[0.0, scenario.bus.width_m], constrain="domain")
    fig.update_yaxes(title="Bus Y [m]", range=[0.0, scenario.bus.height_m], scaleanchor="x", scaleratio=1)
    return fig
