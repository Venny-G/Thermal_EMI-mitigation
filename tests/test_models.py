from __future__ import annotations

from clustered_ep_sim.models.emi import compute_emi_field
from clustered_ep_sim.models.layout import BusConfig, Scenario, SubsystemConfig, ThrusterConfig, make_grid
from clustered_ep_sim.models.risk import classify_ratio, evaluate_risk
from clustered_ep_sim.models.thermal import compute_thermal_field
from clustered_ep_sim.models.layout import adjust_cluster_layout


def _single_thruster() -> ThrusterConfig:
    return ThrusterConfig(
        name="T1",
        x_m=0.20,
        y_m=0.50,
        orientation_deg=0.0,
        power_kw=4.0,
        plume_half_angle_deg=18.0,
    )


def test_thermal_field_decays_downstream() -> None:
    bus = BusConfig(width_m=1.0, height_m=1.0, grid_resolution=220)
    grid_x, grid_y = make_grid(bus)
    thermal_field, _ = compute_thermal_field(grid_x, grid_y, [_single_thruster()])

    y_index = thermal_field.shape[0] // 2
    near_value = thermal_field[y_index, int(0.30 * (thermal_field.shape[1] - 1))]
    far_value = thermal_field[y_index, int(0.75 * (thermal_field.shape[1] - 1))]
    assert near_value > far_value


def test_overlap_raises_centerline_exposure() -> None:
    bus = BusConfig(width_m=1.0, height_m=1.0, grid_resolution=220)
    grid_x, grid_y = make_grid(bus)
    single_field, _ = compute_thermal_field(grid_x, grid_y, [_single_thruster()])
    dual_field, _ = compute_thermal_field(
        grid_x,
        grid_y,
        [
            ThrusterConfig(name="T1", x_m=0.20, y_m=0.42, orientation_deg=0.0, power_kw=4.0),
            ThrusterConfig(name="T2", x_m=0.20, y_m=0.58, orientation_deg=0.0, power_kw=4.0),
        ],
    )

    center_y = dual_field.shape[0] // 2
    center_x = int(0.45 * (dual_field.shape[1] - 1))
    assert dual_field[center_y, center_x] > single_field[center_y, center_x]


def test_shielding_reduces_effective_subsystem_exposure() -> None:
    bus = BusConfig(width_m=1.0, height_m=1.0, grid_resolution=220)
    thruster = _single_thruster()

    unshielded = Scenario(
        name="unshielded",
        description="",
        bus=bus,
        thrusters=[thruster],
        subsystems=[
            SubsystemConfig(
                name="Avionics",
                x_m=0.55,
                y_m=0.50,
                thermal_limit=1.5,
                emi_limit=1.0,
                thermal_shielding=0.0,
                emi_shielding=0.0,
            )
        ],
    )
    shielded = Scenario(
        name="shielded",
        description="",
        bus=bus,
        thrusters=[thruster],
        subsystems=[
            SubsystemConfig(
                name="Avionics",
                x_m=0.55,
                y_m=0.50,
                thermal_limit=1.5,
                emi_limit=1.0,
                thermal_shielding=0.5,
                emi_shielding=0.5,
            )
        ],
    )

    grid_x, grid_y = make_grid(bus)
    thermal_field, thermal_contributions = compute_thermal_field(grid_x, grid_y, [thruster])
    emi_field, emi_contributions = compute_emi_field(grid_x, grid_y, [thruster])
    unshielded_report = evaluate_risk(
        unshielded,
        grid_x,
        grid_y,
        thermal_field,
        thermal_contributions,
        emi_field,
        emi_contributions,
    )
    shielded_report = evaluate_risk(
        shielded,
        grid_x,
        grid_y,
        thermal_field,
        thermal_contributions,
        emi_field,
        emi_contributions,
    )

    assert shielded_report.assessments[0].thermal_ratio < unshielded_report.assessments[0].thermal_ratio
    assert shielded_report.assessments[0].emi_ratio < unshielded_report.assessments[0].emi_ratio


def test_risk_classifier_thresholds() -> None:
    assert classify_ratio(0.45) == "safe"
    assert classify_ratio(0.95) == "caution"
    assert classify_ratio(1.05) == "critical"


def test_integration_weights_are_normalized() -> None:
    bus = BusConfig(width_m=1.0, height_m=1.0, grid_resolution=220)
    thruster = _single_thruster()
    scenario = Scenario(
        name="weights",
        description="",
        bus=bus,
        thrusters=[thruster],
        subsystems=[
            SubsystemConfig(
                name="Avionics",
                x_m=0.55,
                y_m=0.50,
                thermal_limit=1.5,
                emi_limit=1.0,
            )
        ],
    )
    grid_x, grid_y = make_grid(bus)
    thermal_field, thermal_contributions = compute_thermal_field(grid_x, grid_y, [thruster])
    emi_field, emi_contributions = compute_emi_field(grid_x, grid_y, [thruster])
    report = evaluate_risk(
        scenario,
        grid_x,
        grid_y,
        thermal_field,
        thermal_contributions,
        emi_field,
        emi_contributions,
        thermal_weight=7.0,
        emi_weight=3.0,
    )

    assert report.thermal_weight == 0.7
    assert report.emi_weight == 0.3
    assert report.thermal_reference == 1.5
    assert report.emi_reference == 1.0


def test_global_cant_offset_preserves_opposing_cluster_rows() -> None:
    scenario = Scenario(
        name="cant",
        description="",
        bus=BusConfig(width_m=1.0, height_m=1.0, grid_resolution=100),
        thrusters=[
            ThrusterConfig(name="T1", x_m=0.3, y_m=0.3, orientation_deg=6.0, power_kw=4.0),
            ThrusterConfig(name="T2", x_m=0.3, y_m=0.7, orientation_deg=-6.0, power_kw=4.0),
            ThrusterConfig(name="T3", x_m=0.5, y_m=0.3, orientation_deg=6.0, power_kw=4.0),
            ThrusterConfig(name="T4", x_m=0.5, y_m=0.7, orientation_deg=-6.0, power_kw=4.0),
        ],
        subsystems=[],
    )

    adjusted = adjust_cluster_layout(scenario, cant_offset_deg=4.0)
    orientations = {thruster.name: thruster.orientation_deg for thruster in adjusted.thrusters}

    assert orientations["T1"] == 10.0
    assert orientations["T3"] == 10.0
    assert orientations["T2"] == -10.0
    assert orientations["T4"] == -10.0
