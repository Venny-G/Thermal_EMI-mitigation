from __future__ import annotations

import pytest

from clustered_ep_sim.config import scenario_from_dict
from clustered_ep_sim.models.emi import compute_emi_field
from clustered_ep_sim.models.layout import (
    BusConfig,
    EmiCalibrationConfig,
    Scenario,
    SubsystemConfig,
    ThermalCalibrationConfig,
    ThrusterConfig,
    make_grid,
)
from clustered_ep_sim.models.risk import evaluate_risk
from clustered_ep_sim.models.thermal import compute_thermal_field


def test_config_parses_calibration_and_physical_limits() -> None:
    payload = {
        "name": "calibrated",
        "description": "example",
        "thermal_calibration_enabled": True,
        "thermal_reference_value_W_m2": 150.0,
        "thermal_reference_location_x_m": 0.55,
        "thermal_reference_location_y_m": 0.50,
        "emi_calibration_enabled": True,
        "emi_reference_value_uT": 220.0,
        "emi_reference_location_x_m": 0.55,
        "emi_reference_location_y_m": 0.50,
        "bus": {
            "width_m": 1.0,
            "height_m": 1.0,
        },
        "thrusters": [
            {
                "name": "T1",
                "x_m": 0.20,
                "y_m": 0.50,
                "orientation_deg": 0.0,
                "power_kw": 4.0,
            }
        ],
        "subsystems": [
            {
                "name": "Avionics",
                "x_m": 0.55,
                "y_m": 0.50,
                "thermal_limit": 1.4,
                "emi_limit": 1.1,
                "thermal_limit_W_m2": 110.0,
                "emi_limit_uT": 140.0,
            }
        ],
    }

    scenario = scenario_from_dict(payload)

    assert scenario.thermal_calibration.enabled is True
    assert scenario.thermal_calibration.reference_value_w_m2 == 150.0
    assert scenario.emi_calibration.enabled is True
    assert scenario.emi_calibration.reference_value_uT == 220.0
    assert scenario.subsystems[0].thermal_limit_w_m2 == 110.0
    assert scenario.subsystems[0].emi_limit_uT == 140.0


def test_calibrated_mode_reports_physical_subsystem_metrics() -> None:
    bus = BusConfig(width_m=1.0, height_m=1.0, grid_resolution=220)
    subsystem_x = 0.55
    subsystem_y = 0.50
    scenario = Scenario(
        name="calibrated",
        description="",
        bus=bus,
        thrusters=[
            ThrusterConfig(
                name="T1",
                x_m=0.20,
                y_m=0.50,
                orientation_deg=0.0,
                power_kw=4.0,
                plume_half_angle_deg=18.0,
            )
        ],
        subsystems=[
            SubsystemConfig(
                name="Avionics",
                x_m=subsystem_x,
                y_m=subsystem_y,
                thermal_limit=1.5,
                emi_limit=1.0,
                thermal_shielding=0.25,
                emi_shielding=0.50,
                thermal_limit_w_m2=100.0,
                emi_limit_uT=80.0,
            )
        ],
        thermal_calibration=ThermalCalibrationConfig(
            enabled=True,
            reference_value_w_m2=120.0,
            reference_location_x_m=subsystem_x,
            reference_location_y_m=subsystem_y,
        ),
        emi_calibration=EmiCalibrationConfig(
            enabled=True,
            reference_value_uT=160.0,
            reference_location_x_m=subsystem_x,
            reference_location_y_m=subsystem_y,
        ),
    )

    grid_x, grid_y = make_grid(bus)
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
    )

    assessment = report.assessments[0]
    assert report.thermal_calibration.scale_factor is not None
    assert report.emi_calibration.scale_factor is not None
    assert report.thermal_incident_field_w_m2 is not None
    assert report.emi_before_shield_field_uT is not None
    assert assessment.q_incident_w_m2 == pytest.approx(120.0)
    assert assessment.q_after_shield_w_m2 == pytest.approx(90.0)
    assert assessment.thermal_ratio_physical == pytest.approx(0.9)
    assert assessment.b_before_shield_uT == pytest.approx(160.0)
    assert assessment.b_after_shield_uT == pytest.approx(80.0)
    assert assessment.emi_ratio_physical == pytest.approx(1.0)
