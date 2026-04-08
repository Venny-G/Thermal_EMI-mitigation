"""Streamlit dashboard for clustered propulsion thermal and EMI risk mapping."""

from __future__ import annotations

import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
# Allow local src/ imports when running as a standalone Streamlit app.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import streamlit as st

from clustered_ep_sim.config import list_scenarios, load_scenario
from clustered_ep_sim.models.emi import compute_emi_field
from clustered_ep_sim.models.layout import Scenario, adjust_cluster_layout, make_grid
from clustered_ep_sim.models.risk import RiskReport, evaluate_risk
from clustered_ep_sim.models.thermal import compute_thermal_field
from clustered_ep_sim.models.verification import run_physics_checks
from clustered_ep_sim.visualization.plots import make_field_figure

CONFIG_DIR = ROOT / "configs"
DEFAULT_SCENARIO = "2x2 Clustered Configuration"
MISSION_PROFILES = {
    "Balanced": (0.55, 0.45),
    "Thermal-priority": (0.70, 0.30),
    "EMI-sensitive": (0.35, 0.65),
}


def _status_callout(report: RiskReport) -> None:
    if report.overall_state == "critical":
        st.error(report.summary_text)
    elif report.overall_state == "caution":
        st.warning(report.summary_text)
    else:
        st.success(report.summary_text)


def _output_mode_label(report: RiskReport) -> str:
    if report.thermal_calibration.enabled and report.emi_calibration.enabled:
        return "Calibrated estimates (thermal + EMI)"
    if report.thermal_calibration.enabled:
        return "Calibrated estimates (thermal only)"
    if report.emi_calibration.enabled:
        return "Calibrated estimates (EMI only)"
    return "Screening only"


def _format_optional(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def _apply_ui_overrides() -> None:
    st.markdown(
        """
        <style>
        :root {
            --radius-none: 0px;
        }

        * {
            border-radius: 0 !important;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="base-input"] > div,
        div[data-baseweb="select"] > div,
        div[data-baseweb="popover"] > div,
        div[data-baseweb="tag"] {
            border-radius: 0 !important;
        }

        input, textarea,
        button,
        table,
        thead,
        tbody,
        tr,
        td,
        th {
            border-radius: 0 !important;
        }

        [data-testid="stDataFrame"],
        [data-testid="stTable"],
        [data-testid="stMetric"],
        [data-testid="stExpander"],
        [data-testid="stSidebar"],
        [data-testid="stTabs"],
        [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"],
        [data-testid="stAlert"],
        [data-testid="stMarkdownContainer"],
        [data-testid="stPlotlyChart"],
        [data-testid="stForm"] {
            border-radius: 0 !important;
        }

        .stTabs [data-baseweb="tab-list"],
        .stTabs [data-baseweb="tab"],
        .stTabs [aria-selected="true"],
        .stTabs [aria-selected="false"] {
            border-radius: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _edit_calibration(scenario: Scenario) -> None:
    st.sidebar.subheader("Calibration")
    with st.sidebar.expander("Thermal calibration"):
        scenario.thermal_calibration.enabled = st.checkbox(
            "Thermal calibration enabled",
            value=scenario.thermal_calibration.enabled,
            key=f"{scenario.name}_thermal_calibration_enabled",
        )
        thermal_value = scenario.thermal_calibration.reference_value_w_m2
        thermal_x = scenario.thermal_calibration.reference_location_x_m
        thermal_y = scenario.thermal_calibration.reference_location_y_m
        scenario.thermal_calibration.reference_value_w_m2 = st.number_input(
            "Thermal reference value [W/m^2]",
            min_value=0.0,
            max_value=50000.0,
            value=float(thermal_value if thermal_value is not None else 0.0),
            step=10.0,
            format="%.1f",
            key=f"{scenario.name}_thermal_reference_value_w_m2",
        )
        scenario.thermal_calibration.reference_location_x_m = st.number_input(
            "Thermal reference x [m]",
            min_value=0.0,
            max_value=float(scenario.bus.width_m),
            value=float(thermal_x if thermal_x is not None else min(0.5 * scenario.bus.width_m, scenario.bus.width_m)),
            step=0.01,
            format="%.2f",
            key=f"{scenario.name}_thermal_reference_x",
        )
        scenario.thermal_calibration.reference_location_y_m = st.number_input(
            "Thermal reference y [m]",
            min_value=0.0,
            max_value=float(scenario.bus.height_m),
            value=float(thermal_y if thermal_y is not None else min(0.5 * scenario.bus.height_m, scenario.bus.height_m)),
            step=0.01,
            format="%.2f",
            key=f"{scenario.name}_thermal_reference_y",
        )

    with st.sidebar.expander("EMI calibration"):
        scenario.emi_calibration.enabled = st.checkbox(
            "EMI calibration enabled",
            value=scenario.emi_calibration.enabled,
            key=f"{scenario.name}_emi_calibration_enabled",
        )
        emi_value = scenario.emi_calibration.reference_value_uT
        emi_x = scenario.emi_calibration.reference_location_x_m
        emi_y = scenario.emi_calibration.reference_location_y_m
        scenario.emi_calibration.reference_value_uT = st.number_input(
            "EMI reference value [uT]",
            min_value=0.0,
            max_value=1000000.0,
            value=float(emi_value if emi_value is not None else 0.0),
            step=50.0,
            format="%.1f",
            key=f"{scenario.name}_emi_reference_value_ut",
        )
        scenario.emi_calibration.reference_location_x_m = st.number_input(
            "EMI reference x [m]",
            min_value=0.0,
            max_value=float(scenario.bus.width_m),
            value=float(emi_x if emi_x is not None else min(0.5 * scenario.bus.width_m, scenario.bus.width_m)),
            step=0.01,
            format="%.2f",
            key=f"{scenario.name}_emi_reference_x",
        )
        scenario.emi_calibration.reference_location_y_m = st.number_input(
            "EMI reference y [m]",
            min_value=0.0,
            max_value=float(scenario.bus.height_m),
            value=float(emi_y if emi_y is not None else min(0.5 * scenario.bus.height_m, scenario.bus.height_m)),
            step=0.01,
            format="%.2f",
            key=f"{scenario.name}_emi_reference_y",
        )


def _edit_scenario(base_scenario: Scenario) -> Scenario:
    # Copy the preset so sidebar edits do not mutate the config template.
    scenario = copy.deepcopy(base_scenario)

    st.sidebar.header("Layout Controls")
    scenario.bus.width_m = st.sidebar.number_input(
        "Bus width [m]",
        min_value=0.4,
        max_value=3.0,
        value=float(scenario.bus.width_m),
        step=0.05,
    )
    scenario.bus.height_m = st.sidebar.number_input(
        "Bus height [m]",
        min_value=0.4,
        max_value=3.0,
        value=float(scenario.bus.height_m),
        step=0.05,
    )
    scenario.bus.grid_resolution = st.sidebar.number_input(
        "Grid resolution",
        min_value=120,
        max_value=360,
        value=int(scenario.bus.grid_resolution),
        step=20,
    )

    power_scale = st.sidebar.number_input(
        "Global power scale",
        min_value=0.60,
        max_value=1.50,
        value=1.00,
        step=0.05,
        format="%.2f",
    )
    cant_offset_deg = st.sidebar.number_input(
        "Global cant offset [deg]",
        min_value=-15.0,
        max_value=15.0,
        value=0.0,
        step=1.0,
        format="%.1f",
    )
    spacing_scale = st.sidebar.number_input(
        "Cluster spacing scale",
        min_value=0.70,
        max_value=1.50,
        value=1.00,
        step=0.05,
        format="%.2f",
    )
    scenario = adjust_cluster_layout(
        scenario,
        power_scale=power_scale,
        cant_offset_deg=cant_offset_deg,
        spacing_scale=spacing_scale,
    )
    # Rebuild thruster widgets when upstream layout transforms change.
    thruster_widget_seed = (
        f"{scenario.name}_{scenario.bus.width_m:.2f}_{scenario.bus.height_m:.2f}_"
        f"{power_scale:.2f}_{cant_offset_deg:.1f}_{spacing_scale:.2f}"
    )
    subsystem_widget_seed = f"{scenario.name}_{scenario.bus.width_m:.2f}_{scenario.bus.height_m:.2f}"

    st.sidebar.subheader("Thrusters")
    for index, thruster in enumerate(scenario.thrusters):
        with st.sidebar.expander(thruster.name, expanded=index == 0):
            thruster.x_m = st.number_input(
                f"{thruster.name} x [m]",
                min_value=0.0,
                max_value=float(scenario.bus.width_m),
                value=float(min(max(thruster.x_m, 0.0), scenario.bus.width_m)),
                step=0.01,
                format="%.2f",
                key=f"{thruster_widget_seed}_{thruster.name}_x",
            )
            thruster.y_m = st.number_input(
                f"{thruster.name} y [m]",
                min_value=0.0,
                max_value=float(scenario.bus.height_m),
                value=float(min(max(thruster.y_m, 0.0), scenario.bus.height_m)),
                step=0.01,
                format="%.2f",
                key=f"{thruster_widget_seed}_{thruster.name}_y",
            )
            thruster.orientation_deg = st.number_input(
                f"{thruster.name} orientation [deg]",
                min_value=-45.0,
                max_value=45.0,
                value=float(thruster.orientation_deg),
                step=1.0,
                format="%.1f",
                key=f"{thruster_widget_seed}_{thruster.name}_orientation",
            )
            thruster.power_kw = st.number_input(
                f"{thruster.name} power [kW]",
                min_value=1.0,
                max_value=8.0,
                value=float(thruster.power_kw),
                step=0.1,
                format="%.1f",
                key=f"{thruster_widget_seed}_{thruster.name}_power",
            )

    st.sidebar.subheader("Subsystems")
    for subsystem in scenario.subsystems:
        with st.sidebar.expander(subsystem.name):
            subsystem.x_m = st.number_input(
                f"{subsystem.name} x [m]",
                min_value=0.0,
                max_value=float(scenario.bus.width_m),
                value=float(min(max(subsystem.x_m, 0.0), scenario.bus.width_m)),
                step=0.01,
                format="%.2f",
                key=f"{subsystem_widget_seed}_{subsystem.name}_x",
            )
            subsystem.y_m = st.number_input(
                f"{subsystem.name} y [m]",
                min_value=0.0,
                max_value=float(scenario.bus.height_m),
                value=float(min(max(subsystem.y_m, 0.0), scenario.bus.height_m)),
                step=0.01,
                format="%.2f",
                key=f"{subsystem_widget_seed}_{subsystem.name}_y",
            )
            subsystem.thermal_limit = st.number_input(
                f"{subsystem.name} thermal limit",
                min_value=0.8,
                max_value=3.5,
                value=float(subsystem.thermal_limit),
                step=0.05,
                format="%.2f",
                key=f"{subsystem_widget_seed}_{subsystem.name}_thermal_limit",
            )
            subsystem.emi_limit = st.number_input(
                f"{subsystem.name} EMI limit",
                min_value=0.6,
                max_value=3.0,
                value=float(subsystem.emi_limit),
                step=0.05,
                format="%.2f",
                key=f"{subsystem_widget_seed}_{subsystem.name}_emi_limit",
            )
            subsystem.thermal_shielding = st.number_input(
                f"{subsystem.name} thermal shielding",
                min_value=0.0,
                max_value=0.8,
                value=float(subsystem.thermal_shielding),
                step=0.05,
                format="%.2f",
                key=f"{subsystem_widget_seed}_{subsystem.name}_thermal_shielding",
            )
            subsystem.emi_shielding = st.number_input(
                f"{subsystem.name} EMI shielding",
                min_value=0.0,
                max_value=0.8,
                value=float(subsystem.emi_shielding),
                step=0.05,
                format="%.2f",
                key=f"{subsystem_widget_seed}_{subsystem.name}_emi_shielding",
            )
    _edit_calibration(scenario)
    return scenario


def _assessment_table(report: RiskReport) -> pd.DataFrame:
    rows = []
    thermal_calibrated = report.thermal_calibration.enabled
    emi_calibrated = report.emi_calibration.enabled
    for assessment in report.assessments:
        row = {
            "subsystem": assessment.name,
            "q_proxy_sample": round(assessment.q_proxy_sample, 2),
            "B_proxy_sample": round(assessment.b_proxy_sample, 2),
            "thermal_ratio_screening": round(assessment.thermal_ratio_screening, 2),
            "emi_ratio_screening": round(assessment.emi_ratio_screening, 2),
            "combined_score": round(assessment.combined_score, 2),
            "overall_state": assessment.overall_state.title(),
            "dominant_driver": assessment.dominant_driver,
            "dominant_thermal_thruster": assessment.dominant_thermal_thruster,
            "dominant_emi_thruster": assessment.dominant_emi_thruster,
            "dominant_thruster": assessment.dominant_thruster,
            "likely_failure_mode": assessment.likely_failure_mode,
        }
        if thermal_calibrated:
            row["q_incident_W_m2"] = _format_optional(assessment.q_incident_w_m2)
            row["q_after_shield_W_m2"] = _format_optional(assessment.q_after_shield_w_m2)
            row["thermal_ratio_physical"] = _format_optional(assessment.thermal_ratio_physical)
        if emi_calibrated:
            row["B_before_shield_uT"] = _format_optional(assessment.b_before_shield_uT)
            row["B_after_shield_uT"] = _format_optional(assessment.b_after_shield_uT)
            row["emi_ratio_physical"] = _format_optional(assessment.emi_ratio_physical)
        rows.append(row)
    return pd.DataFrame(rows)


def _contribution_table(report: RiskReport, subsystem_name: str) -> pd.DataFrame:
    assessment = next(assessment for assessment in report.assessments if assessment.name == subsystem_name)
    rows = []
    for contribution in assessment.thruster_contributions:
        row = {
            "thruster": contribution.thruster_name,
            "thermal_proxy": round(contribution.thermal_proxy, 3),
            "thermal_share_pct": round(100.0 * contribution.thermal_fraction, 1),
            "emi_proxy": round(contribution.emi_proxy, 3),
            "emi_share_pct": round(100.0 * contribution.emi_fraction, 1),
        }
        if contribution.q_incident_w_m2 is not None:
            row["thermal_incident_W_m2"] = round(contribution.q_incident_w_m2, 2)
        if contribution.b_before_shield_uT is not None:
            row["emi_before_shield_uT"] = round(contribution.b_before_shield_uT, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def _select_screening_weights() -> tuple[str, float, float]:
    st.sidebar.subheader("Integration Score Settings")
    profile = st.sidebar.selectbox(
        "Weight profile",
        ["Balanced", "Thermal-priority", "EMI-sensitive", "Custom"],
        index=0,
    )
    if profile == "Custom":
        thermal_weight = st.sidebar.number_input(
            "Thermal weight",
            min_value=0.0,
            max_value=1.0,
            value=0.55,
            step=0.05,
            format="%.2f",
        )
        emi_weight = st.sidebar.number_input(
            "EMI weight",
            min_value=0.0,
            max_value=1.0,
            value=0.45,
            step=0.05,
            format="%.2f",
        )
        if thermal_weight == 0.0 and emi_weight == 0.0:
            st.sidebar.warning(
                "At least one weight needs to be non-zero. Resetting to balanced weights."
            )
            return "Balanced", *MISSION_PROFILES["Balanced"]
        return profile, thermal_weight, emi_weight
    return profile, *MISSION_PROFILES[profile]


def _physics_checks_table(
    scenario: Scenario,
    thermal_contributions,
    emi_contributions,
    report: RiskReport,
) -> pd.DataFrame:
    checks = run_physics_checks(scenario, thermal_contributions, emi_contributions, report)
    rows = []
    for check in checks:
        rows.append(
            {
                "Check": check.name,
                "Result": "pass" if check.passed else "check",
                "What it means": check.meaning,
                "Current value": check.summary,
            }
        )
    return pd.DataFrame(rows)


def _render_model_overview_tab(report: RiskReport, screening_profile: str) -> None:
    st.subheader("Model overview")
    st.write(
        "This tab summarizes the simplified screening model used in the app. "
        "It is a directional proxy for thermal and EMI exposure in clustered Hall thruster layouts."
    )

    st.markdown("**1. Thruster-aligned coordinates**")
    st.latex(
        r"""
        s_i = (x-x_i)\cos\theta_i + (y-y_i)\sin\theta_i,\qquad
        c_i = -(x-x_i)\sin\theta_i + (y-y_i)\cos\theta_i
        """
    )
    st.write(
        "The bus plane is rotated into a thruster-aligned frame. "
        "`s_i` is downrange and `c_i` is crossrange."
    )
    st.markdown(
        "- `\\theta_i` is the thruster pointing or cant angle.\n"
        "- `\\phi_i` is the plume half-angle used to control spread."
    )

    st.markdown("**2. Thermal loading proxy**")
    st.latex(
        r"""
        q_i(x,y) = P_i\left[
        e^{-\max(s_i,0)/L_{t,i}}
        e^{-c_i^2/(2\sigma_i(s_i)^2)}
        + \beta_{\mathrm{back}}e^{-|\min(s_i,0)|/L_{\mathrm{back},i}}e^{-(c_i/w_b)^2}
        + \beta_{\mathrm{nf}}e^{-(r_i/r_{\mathrm{nf}})^2}
        \right]
        """
    )
    st.write(
        "Thermal loading is strongest near the thruster and decays with downrange distance and crossrange offset."
    )
    st.latex(
        r"""
        \sigma_i(s_i) = \max\left(0.05,\tan(\phi_i)\left[\max(s_i,0)+0.08\right]\right)
        """
    )
    st.write(
        "Plume width grows with distance and is clipped to a minimum value."
    )

    st.markdown("**3. Magnetic / EMI proxy**")
    st.latex(
        r"""
        B_i(x,y) = P_i\left[
        \frac{\alpha_{\mathrm{coil}}}{1+(r_i/r_c)^3}
        + \alpha_{\mathrm{tail}}e^{-\max(s_i,0)/L_{e,i}}e^{-c_i^2/(2\sigma_{e,i}(s_i)^2)}
        + \alpha_{\mathrm{side}}\frac{e^{-c_i^2/(2w_s^2)}}{1+(r_i/r_s)^2}
        \right]
        """
    )
    st.write(
        "The first term is a dipole-like near-field term. The others control directional decay and off-axis spread."
    )
    st.latex(
        r"""
        \sigma_{e,i}(s_i) = \max\left(0.06,0.8\tan(\phi_i)\left[\max(s_i,0)+0.06\right]\right)
        """
    )
    st.write(
        "The EMI envelope widens with distance using a separate width rule."
    )

    st.markdown("**4. Subsystem exposure and failure rule**")
    st.latex(
        r"""
        q_{\mathrm{eff}} = q(1-\eta_t),\qquad
        B_{\mathrm{eff}} = B(1-\eta_e)
        """
    )
    st.latex(
        r"""
        R_t = \frac{q_{\mathrm{eff}}}{q_{\mathrm{limit}}},\qquad
        R_e = \frac{B_{\mathrm{eff}}}{B_{\mathrm{limit}}}
        """
    )
    st.latex(
        r"""
        \text{subsystem critical if } R_t \ge 1 \text{ or } R_e \ge 1
        """
    )
    st.latex(
        r"""
        R_{\mathrm{screen}} = w_t\frac{q}{q_{\mathrm{ref}}} + w_e\frac{B}{B_{\mathrm{ref}}}
        """
    )
    st.write(
        "Subsystems are checked against their own thresholds, and the bus is also shown as a combined score field."
    )
    st.write(
        f"Selected profile: **{screening_profile}**. "
        f"Current weights are "
        f"$w_t = {report.thermal_weight:.2f}$ and $w_e = {report.emi_weight:.2f}$."
    )
    st.markdown(
        f"- `q_ref` is the median subsystem thermal limit for the current scenario: **{report.thermal_reference:.2f}**\n"
        f"- `B_ref` is the median subsystem EMI limit for the current scenario: **{report.emi_reference:.2f}**"
    )
    st.write(
        "The integration score is normalized against the scenario subsystem limits, not against the map maximum."
    )

    st.markdown("**5. Optional calibrated mode**")
    st.latex(
        r"""
        q''_{\mathrm{incident}}(x,y) = C_{\mathrm{th}} q_{\mathrm{proxy}}(x,y),\qquad
        B_{\mathrm{before\ shield}}(x,y) = C_{\mathrm{emi}} B_{\mathrm{proxy}}(x,y)
        """
    )
    st.write(
        "Calibrated mode anchors the proxy fields to user-supplied reference data and reports first-order physical estimates."
    )
    if report.thermal_calibration.enabled:
        st.markdown(
            f"- Thermal anchor: **{report.thermal_calibration.reference_value:.1f} W/m^2** at "
            f"**({report.thermal_calibration.reference_x_m:.2f} m, {report.thermal_calibration.reference_y_m:.2f} m)**"
        )
    if report.emi_calibration.enabled:
        st.markdown(
            f"- EMI anchor: **{report.emi_calibration.reference_value:.1f} uT** at "
            f"**({report.emi_calibration.reference_x_m:.2f} m, {report.emi_calibration.reference_y_m:.2f} m)**"
        )
    if not report.thermal_calibration.enabled and not report.emi_calibration.enabled:
        st.write("Current run is screening-only. Proxy fields remain dimensionless.")
    else:
        st.write(
            "Calibrated outputs are first-order estimates anchored to a single reference point and are most trustworthy near that calibration region."
        )

    st.markdown("**6. Why this screening model is physically motivated**")
    st.markdown(
        "- Hall thruster plumes are directional, so a plume-aligned coordinate frame is a reasonable first step.\n"
        "- Downrange decay and widening spread capture the main layout effect of plume exposure.\n"
        "- Thermal and EMI thresholds are handled at the subsystem level rather than hidden inside the field model."
    )

    with st.expander("Reference notes"):
        st.markdown(
            "- Goebel and Katz: Hall thruster plumes are directional and can affect spacecraft surfaces.\n"
            "- NASA beam-profile data shows centerline structure, so the Gaussian envelope here is used as a simple shape approximation.\n"
            "- EMI/EMC remains a spacecraft integration concern for avionics and communications."
        )
    with st.expander("Calibration terms in the model"):
        st.markdown(
            "- Backflow term: prevents the upstream region from dropping unrealistically to zero.\n"
            "- Near-field term: captures local hotspot behavior near the thruster.\n"
            "- Side-lobe EMI term: broadens the screening envelope away from the centerline.\n"
            "- These are tuning terms, not derived plasma closures."
        )


def main() -> None:
    st.set_page_config(page_title="Clustered EP Risk Mapper", layout="wide")
    _apply_ui_overrides()
    st.title("Clustered Hall Thruster Thermal / EMI Risk Mapper")
    st.write(
        "Interactive screening tool for visualizing thermal loading, "
        "EMI exposure, and subsystem risk in clustered Hall thruster layouts."
    )

    scenario_map = list_scenarios(CONFIG_DIR)
    scenario_names = list(scenario_map)
    default_index = scenario_names.index(DEFAULT_SCENARIO) if DEFAULT_SCENARIO in scenario_names else 0
    selection = st.sidebar.selectbox("Preset scenario", scenario_names, index=default_index)
    scenario = _edit_scenario(load_scenario(scenario_map[selection]))
    screening_profile, thermal_weight, emi_weight = _select_screening_weights()

    grid_x, grid_y = make_grid(scenario.bus)
    thermal_field, thermal_contributions = compute_thermal_field(grid_x, grid_y, scenario.thrusters)
    emi_field, emi_contributions = compute_emi_field(grid_x, grid_y, scenario.thrusters)
    report = evaluate_risk(
        scenario,
        grid_x,
        grid_y,
        thermal_field,
        thermal_contributions,
        emi_field,
        emi_contributions,
        thermal_weight=thermal_weight,
        emi_weight=emi_weight,
    )

    _status_callout(report)
    st.write(scenario.description)
    if scenario.source_note:
        st.caption(f"Source note: {scenario.source_note}")
    output_mode = _output_mode_label(report)
    assessment_df = _assessment_table(report)
    if report.thermal_calibration.enabled or report.emi_calibration.enabled:
        st.info(
            "Calibrated outputs are first-order estimates anchored to a single reference point. "
            "Accuracy is highest near that calibration region; screening ratios still drive layout classification."
        )

    summary_col, metrics_col = st.columns(2)
    with summary_col:
        st.write(f"**Current layout state:** {report.overall_state.title()}")
        st.write(f"**Output mode:** {output_mode}")
        st.write(
            f"**Peak thermal zone:** {report.max_thermal_peak.value:.2f} at "
            f"({report.max_thermal_peak.x_m:.2f} m, {report.max_thermal_peak.y_m:.2f} m)"
        )
        if report.thermal_calibration.scale_factor is not None:
            peak_thermal_w_m2 = report.max_thermal_peak.value * report.thermal_calibration.scale_factor
            st.write(f"**Peak thermal estimate:** {peak_thermal_w_m2:.1f} W/m^2 incident")
        st.write(
            f"**Peak EMI zone:** {report.max_emi_peak.value:.2f} at "
            f"({report.max_emi_peak.x_m:.2f} m, {report.max_emi_peak.y_m:.2f} m)"
        )
        if report.emi_calibration.scale_factor is not None:
            peak_emi_uT = report.max_emi_peak.value * report.emi_calibration.scale_factor
            st.write(f"**Peak EMI estimate:** {peak_emi_uT:.1f} uT before shielding")
    with metrics_col:
        st.write(f"**Score profile:** {screening_profile}")
        st.write(f"**Critical bus area:** {report.critical_area_fraction:.0%}")
        st.write(f"**Caution bus area:** {report.caution_area_fraction:.0%}")
        st.write(
            f"**Worst combined-risk point:** ({report.max_combined_peak.x_m:.2f} m, {report.max_combined_peak.y_m:.2f} m)"
        )
        if report.thermal_calibration.enabled:
            st.write(
                f"**Thermal calibration:** {report.thermal_calibration.scale_factor:.2f} W/m^2 per proxy unit"
            )
        if report.emi_calibration.enabled:
            st.write(
                f"**EMI calibration:** {report.emi_calibration.scale_factor:.2f} uT per proxy unit"
            )

    tab_thermal, tab_emi, tab_combined, tab_physics = st.tabs(
        ["Thermal map", "EMI map", "Integration score", "Model overview"]
    )
    with tab_thermal:
        if report.thermal_calibration.enabled:
            st.write(
                "Thermal loading proxy across the bus. Calibrated incident estimates are listed in the subsystem table."
            )
        else:
            st.write("Thermal loading proxy across the bus.")
        st.plotly_chart(
            make_field_figure(
                scenario,
                grid_x,
                grid_y,
                thermal_field,
                report,
                title="Thermal Load Proxy",
                colorbar_title="Thermal index",
                colorscale="YlOrRd",
                peak=report.max_thermal_peak,
            ),
            use_container_width=True,
        )

    with tab_emi:
        if report.emi_calibration.enabled:
            st.write(
                "EMI exposure proxy across the bus. Calibrated before-shield estimates are listed in the subsystem table."
            )
        else:
            st.write(
                "EMI exposure proxy across the bus. Highest values occur near the thrusters "
                "and along the directional field tail."
            )
        st.plotly_chart(
            make_field_figure(
                scenario,
                grid_x,
                grid_y,
                emi_field,
                report,
                title="EMI / Magnetic Exposure Proxy",
                colorbar_title="EMI index",
                colorscale="Blues",
                peak=report.max_emi_peak,
            ),
            use_container_width=True,
        )

    with tab_combined:
        summary_col, details_col = st.columns([1.3, 1.0])
        with summary_col:
            st.write(
                "Combined score field used for layout screening."
            )
            st.plotly_chart(
                make_field_figure(
                    scenario,
                    grid_x,
                    grid_y,
                    report.combined_field,
                    report,
                    title="Integration Screening Score",
                    colorbar_title="Score",
                    colorscale="Magma",
                    peak=report.max_combined_peak,
                    contour_thresholds=(0.70, 1.00),
                ),
                use_container_width=True,
            )
        with details_col:
            st.write(
                f"Current weights: thermal = {report.thermal_weight:.2f}, EMI = {report.emi_weight:.2f}"
            )
            st.subheader("Failure Triggers")
            st.write(
                "A subsystem is critical when its thermal or EMI ratio exceeds 1. "
                "The layout is also critical if more than 12% of the bus exceeds the critical score threshold."
            )
            critical_assessments = [assessment for assessment in report.assessments if assessment.overall_state == "critical"]
            if critical_assessments:
                for assessment in critical_assessments:
                    st.markdown(
                        f"- **{assessment.name}**: dominant exceedance is {assessment.dominant_driver} from "
                        f"{assessment.dominant_thruster}. Likely outcome: {assessment.likely_failure_mode}"
                    )
            else:
                st.markdown("- No subsystem is currently beyond its configured failure threshold.")

            st.subheader("Recommendations")
            for recommendation in report.recommendations:
                st.markdown(f"- {recommendation}")

    with tab_physics:
        _render_model_overview_tab(report, screening_profile)
        st.subheader("Quick sanity checks for this exact run")
        st.dataframe(
            _physics_checks_table(scenario, thermal_contributions, emi_contributions, report),
            use_container_width=True,
            hide_index=True,
        )
        st.write(
            "These checks verify that the screening model behaves consistently with its intended assumptions."
        )

    bottom_left, bottom_right = st.columns([1.4, 1.0])
    with bottom_left:
        st.subheader("Subsystem Exposure Summary")
        st.dataframe(assessment_df, use_container_width=True, hide_index=True)
        st.download_button(
            "Download subsystem metrics CSV",
            data=assessment_df.to_csv(index=False),
            file_name=f"{scenario.name.lower().replace(' ', '_')}_subsystem_metrics.csv",
            mime="text/csv",
        )
        st.subheader("Thruster Contribution Breakdown")
        selected_subsystem = st.selectbox(
            "Subsystem for breakdown",
            [assessment.name for assessment in report.assessments],
            key="contribution_subsystem",
        )
        st.dataframe(
            _contribution_table(report, selected_subsystem),
            use_container_width=True,
            hide_index=True,
        )
    with bottom_right:
        st.subheader("Model Notes")
        if report.thermal_calibration.enabled or report.emi_calibration.enabled:
            st.write(
                "This run includes calibrated first-order estimates anchored to user-supplied reference data. "
                "The screening ratios are still the primary layout metric."
            )
        else:
            st.write(
                "This is a screening model with user-configured subsystem limits. "
                "Outputs are dimensionless indices."
            )
        with st.expander("Model assumptions"):
            st.markdown(
                "- Thermal loading uses directional plume decay plus local near-field heating.\n"
                "- EMI uses a dipole-like near-field term plus a directional tail.\n"
                "- Overlapping fields accumulate linearly.\n"
                "- Shielding is applied as a local reduction factor on subsystem exposure.\n"
                "- Calibrated mode anchors the proxy field to one reference point for first-order physical estimates."
            )


if __name__ == "__main__":
    main()
