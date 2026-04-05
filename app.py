"""Streamlit dashboard for clustered propulsion thermal and EMI risk mapping."""

from __future__ import annotations

import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
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


def _edit_scenario(base_scenario: Scenario) -> Scenario:
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

    power_scale = st.sidebar.number_input("Global power scale", min_value=0.60, max_value=1.50, value=1.00, step=0.05, format="%.2f")
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
                key=f"{scenario.name}_{thruster.name}_x",
            )
            thruster.y_m = st.number_input(
                f"{thruster.name} y [m]",
                min_value=0.0,
                max_value=float(scenario.bus.height_m),
                value=float(min(max(thruster.y_m, 0.0), scenario.bus.height_m)),
                step=0.01,
                format="%.2f",
                key=f"{scenario.name}_{thruster.name}_y",
            )
            thruster.orientation_deg = st.number_input(
                f"{thruster.name} orientation [deg]",
                min_value=-45.0,
                max_value=45.0,
                value=float(thruster.orientation_deg),
                step=1.0,
                format="%.1f",
                key=f"{scenario.name}_{thruster.name}_orientation",
            )
            thruster.power_kw = st.number_input(
                f"{thruster.name} power [kW]",
                min_value=1.0,
                max_value=8.0,
                value=float(thruster.power_kw),
                step=0.1,
                format="%.1f",
                key=f"{scenario.name}_{thruster.name}_power",
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
                key=f"{scenario.name}_{subsystem.name}_x",
            )
            subsystem.y_m = st.number_input(
                f"{subsystem.name} y [m]",
                min_value=0.0,
                max_value=float(scenario.bus.height_m),
                value=float(min(max(subsystem.y_m, 0.0), scenario.bus.height_m)),
                step=0.01,
                format="%.2f",
                key=f"{scenario.name}_{subsystem.name}_y",
            )
            subsystem.thermal_limit = st.number_input(
                f"{subsystem.name} thermal limit",
                min_value=0.8,
                max_value=3.5,
                value=float(subsystem.thermal_limit),
                step=0.05,
                format="%.2f",
                key=f"{scenario.name}_{subsystem.name}_thermal_limit",
            )
            subsystem.emi_limit = st.number_input(
                f"{subsystem.name} EMI limit",
                min_value=0.6,
                max_value=3.0,
                value=float(subsystem.emi_limit),
                step=0.05,
                format="%.2f",
                key=f"{scenario.name}_{subsystem.name}_emi_limit",
            )
            subsystem.thermal_shielding = st.number_input(
                f"{subsystem.name} thermal shielding",
                min_value=0.0,
                max_value=0.8,
                value=float(subsystem.thermal_shielding),
                step=0.05,
                format="%.2f",
                key=f"{scenario.name}_{subsystem.name}_thermal_shielding",
            )
            subsystem.emi_shielding = st.number_input(
                f"{subsystem.name} EMI shielding",
                min_value=0.0,
                max_value=0.8,
                value=float(subsystem.emi_shielding),
                step=0.05,
                format="%.2f",
                key=f"{scenario.name}_{subsystem.name}_emi_shielding",
            )
    return scenario


def _assessment_table(report: RiskReport) -> pd.DataFrame:
    rows = []
    for assessment in report.assessments:
        rows.append(
            {
                "Subsystem": assessment.name,
                "Thermal ratio": round(assessment.thermal_ratio, 2),
                "EMI ratio": round(assessment.emi_ratio, 2),
                "Combined score": round(assessment.combined_score, 2),
                "Overall": assessment.overall_state.title(),
                "Dominant driver": assessment.dominant_driver,
                "Dominant thruster": assessment.dominant_thruster,
                "Likely failure mode": assessment.likely_failure_mode,
            }
        )
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
            st.sidebar.warning("At least one weight needs to be non-zero. Resetting to balanced weights.")
            return "Balanced", *MISSION_PROFILES["Balanced"]
        return profile, thermal_weight, emi_weight
    return profile, *MISSION_PROFILES[profile]


def _physics_checks_table(scenario: Scenario, thermal_contributions, emi_contributions, report: RiskReport) -> pd.DataFrame:
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


def _render_physics_tab(report: RiskReport, profile_name: str) -> None:
    st.subheader("What the math is doing")
    st.write(
        "This tab is meant to be readable. The model is not pretending to be a full plume / CFD / EM solve. "
        "It is a directional risk proxy built from simple terms that follow the geometry of a clustered Hall thruster layout."
    )

    st.markdown("**1. Thruster-aligned coordinates**")
    st.latex(
        r"""
        s_i = (x-x_i)\cos\theta_i + (y-y_i)\sin\theta_i,\qquad
        c_i = -(x-x_i)\sin\theta_i + (y-y_i)\cos\theta_i
        """
    )
    st.write(
        "For each thruster, the app rotates the spacecraft plane into a local frame. "
        "`s_i` is downrange along the plume direction and `c_i` is crossrange."
    )
    st.markdown(
        "- `\\theta_i` is the thruster pointing or cant angle.\n"
        "- `\\phi_i` below is the plume half-angle used to control spread.\n"
        "- Those are separate quantities in the code and should not be confused."
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
        "This says the heat effect is strongest near the thruster, falls with distance, and stays stronger inside the plume direction than far off-axis."
    )
    st.latex(
        r"""
        \sigma_i(s_i) = \max\left(0.05,\tan(\phi_i)\left[\max(s_i,0)+0.08\right]\right)
        """
    )
    st.write(
        "So the plume width grows with downrange distance, but it is clipped to a minimum width so the map does not collapse into a razor-thin line."
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
        "The first term is a dipole-like near-thruster magnetic field proxy. "
        "The other terms keep some directional plume influence and side-lobe spread."
    )
    st.latex(
        r"""
        \sigma_{e,i}(s_i) = \max\left(0.06,0.8\tan(\phi_i)\left[\max(s_i,0)+0.06\right]\right)
        """
    )
    st.write(
        "The EMI tail also widens with distance, but it uses a slightly different width rule than the thermal map."
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
        "So the app is doing two things: it checks each subsystem directly against its thresholds, "
        "and it also paints a combined screening map for the whole bus."
    )
    st.write(
        f"Right now the selected profile is **{profile_name}**, which means "
        f"$w_t = {report.thermal_weight:.2f}$ and $w_e = {report.emi_weight:.2f}$."
    )
    st.write(
        "Those weights are mission-priority settings, not physics constants. They tell the app how much to care about thermal versus EMI when making one combined picture."
    )
    st.markdown(
        f"- `q_ref` is the median subsystem thermal limit for the current scenario: **{report.thermal_reference:.2f}**\n"
        f"- `B_ref` is the median subsystem EMI limit for the current scenario: **{report.emi_reference:.2f}**"
    )
    st.write(
        "So the integration score is scenario-normalized against the tolerance table you gave the app. "
        "It is not normalized to the map maximum, and it is not a universal physical constant."
    )

    st.markdown("**5. Why this is a fair first-pass physics model**")
    st.markdown(
        "- Hall thruster plumes are directional and have measurable divergence, so using a plume-aligned field envelope is physically reasonable.\n"
        "- Hall thruster plume profiles are not perfectly Gaussian near centerline, so the app treats the Gaussian-like spread only as a simple shape function, not as exact truth.\n"
        "- Spacecraft integration really does care about both plume-driven surface effects and EMI/EMC exposure, so combining thermal and EMI views is the right systems framing."
    )

    with st.expander("Literature grounding used for this draft"):
        st.markdown(
            "- Goebel and Katz, *Ion and Hall Thruster Plumes* (JPL): Hall thruster plumes have large angular divergence and can change spacecraft surface optical, thermal, and electrical properties.\n"
            "- NASA/TM-2018-219948: Hall thruster beam-current profiles show a centerline inflection, which is why a simple Gaussian should be treated as an approximation, not an exact plume law.\n"
            "- NASA Glenn thermal characterization of the NASA-300MS Hall thruster: magnetic topology and shielding materially change thermal/plume behavior.\n"
            "- NASA JSC EMI/EMC guidance: EMI/EMC is a mission-level concern for communications, avionics, guidance, and other spacecraft electronics."
        )
    with st.expander("Calibration terms in the model"):
        st.markdown(
            "- The backflow term in the thermal map is a shaping term that keeps the upstream region from going unrealistically to zero.\n"
            "- The near-field thermal term is a local hotspot term for danger-close loading near the thruster.\n"
            "- The EMI side-lobe term is a shaping term that broadens the map so it behaves more like a screening envelope than a needle-thin source.\n"
            "- These are tuning terms, not first-principles plasma closures."
        )


def main() -> None:
    st.set_page_config(page_title="Clustered EP Risk Mapper", layout="wide")
    st.title("Clustered Hall Thruster Risk Demo")
    st.write("First draft UI. The point is to move thrusters around and see where the hot zones, EMI zones, and failure regions show up.")

    scenario_map = list_scenarios(CONFIG_DIR)
    scenario_names = list(scenario_map)
    default_index = scenario_names.index(DEFAULT_SCENARIO) if DEFAULT_SCENARIO in scenario_names else 0
    selection = st.sidebar.selectbox("Preset scenario", scenario_names, index=default_index)
    scenario = _edit_scenario(load_scenario(scenario_map[selection]))
    profile_name, thermal_weight, emi_weight = _select_screening_weights()

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

    left_info, right_info = st.columns(2)
    with left_info:
        st.write(f"**Current layout state:** {report.overall_state.title()}")
        st.write(
            f"**Peak thermal zone:** {report.max_thermal_peak.value:.2f} at "
            f"({report.max_thermal_peak.x_m:.2f} m, {report.max_thermal_peak.y_m:.2f} m)"
        )
        st.write(
            f"**Peak EMI zone:** {report.max_emi_peak.value:.2f} at "
            f"({report.max_emi_peak.x_m:.2f} m, {report.max_emi_peak.y_m:.2f} m)"
        )
    with right_info:
        st.write(f"**Score profile:** {profile_name}")
        st.write(f"**Critical bus area:** {report.critical_area_fraction:.0%}")
        st.write(f"**Caution bus area:** {report.caution_area_fraction:.0%}")
        st.write(
            f"**Worst combined-risk point:** ({report.max_combined_peak.x_m:.2f} m, {report.max_combined_peak.y_m:.2f} m)"
        )

    tab_thermal, tab_emi, tab_combined, tab_physics = st.tabs(
        ["Thermal map", "EMI map", "Integration score", "Physics notes"]
    )
    with tab_thermal:
        st.write("This is the plume-shaped thermal loading proxy across the spacecraft bus.")
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
        st.write("This is the simplified magnetic / EMI exposure proxy. It is strongest close to the thrusters and then decays away.")
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
                "This is the integration screening score. It is a weighted decision layer built on top of the thermal and EMI proxies, not a direct physics field."
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
                "A subsystem is flagged as critical when its thermal or EMI exposure exceeds its configured limit. "
                "The overall layout is also flagged if more than 12% of the bus crosses the combined critical threshold."
            )
            critical_assessments = [assessment for assessment in report.assessments if assessment.overall_state == "critical"]
            if critical_assessments:
                for assessment in critical_assessments:
                    st.markdown(
                        f"- **{assessment.name}**: {assessment.dominant_driver} driven exceedance from "
                        f"{assessment.dominant_thruster}. Likely outcome: {assessment.likely_failure_mode}"
                    )
            else:
                st.markdown("- No subsystem is currently beyond its configured failure threshold.")

            st.subheader("Recommendations")
            for recommendation in report.recommendations:
                st.markdown(f"- {recommendation}")

    with tab_physics:
        _render_physics_tab(report, profile_name)
        st.subheader("Quick sanity checks for this exact run")
        st.dataframe(
            _physics_checks_table(scenario, thermal_contributions, emi_contributions, report),
            use_container_width=True,
            hide_index=True,
        )
        st.write(
            "These checks do not prove the model is high-fidelity. They just confirm that the simplified model behaves in the physically sensible way we say it does."
        )

    bottom_left, bottom_right = st.columns([1.4, 1.0])
    with bottom_left:
        st.subheader("Subsystem Exposure Summary")
        st.dataframe(_assessment_table(report), use_container_width=True, hide_index=True)
    with bottom_right:
        st.subheader("Model Notes")
        st.write(
            "This is meant to look like a first-pass engineering demo, not a finished mission tool. "
            "The physics is simplified, but the risk logic and cause-and-effect are explicit."
        )
        with st.expander("Model assumptions"):
            st.markdown(
                "- Thermal loading uses directional plume decay plus local near-field heating.\n"
                "- EMI uses a dipole-like near-field term plus a directional tail.\n"
                "- Overlapping fields accumulate linearly.\n"
                "- Shielding is modeled as a local reduction factor on subsystem exposure.\n"
                "- Outputs are dimensionless indices for comparison, not high-fidelity absolute loads.\n"
                "- The integration score is a tunable decision layer, not a physical conservation law."
            )


if __name__ == "__main__":
    main()
