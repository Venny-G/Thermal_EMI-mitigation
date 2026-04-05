from __future__ import annotations

from clustered_ep_sim.config import scenario_from_dict
from clustered_ep_sim.models.emi import compute_emi_field
from clustered_ep_sim.models.layout import make_grid
from clustered_ep_sim.models.risk import evaluate_risk
from clustered_ep_sim.models.thermal import compute_thermal_field
from clustered_ep_sim.models.verification import run_physics_checks


def _scenario_payload() -> dict:
    return {
        "name": "verification-case",
        "description": "Simple two-thruster case for model sanity checks.",
        "bus": {"width_m": 1.1, "height_m": 1.0, "grid_resolution": 180},
        "thrusters": [
            {
                "name": "T1",
                "x_m": 0.22,
                "y_m": 0.35,
                "orientation_deg": 0.0,
                "power_kw": 4.0,
                "plume_half_angle_deg": 18.0,
            },
            {
                "name": "T2",
                "x_m": 0.22,
                "y_m": 0.65,
                "orientation_deg": 0.0,
                "power_kw": 4.0,
                "plume_half_angle_deg": 18.0,
            },
        ],
        "subsystems": [
            {
                "name": "Avionics",
                "x_m": 0.70,
                "y_m": 0.50,
                "thermal_limit": 2.0,
                "emi_limit": 1.2,
                "thermal_shielding": 0.20,
                "emi_shielding": 0.20,
            }
        ],
    }


def test_physics_checks_pass_for_nominal_case() -> None:
    scenario = scenario_from_dict(_scenario_payload())
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
    )

    checks = run_physics_checks(scenario, thermal_contributions, emi_contributions, report)
    assert len(checks) == 5
    assert all(check.passed for check in checks)
