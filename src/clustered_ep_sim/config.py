"""Scenario loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from clustered_ep_sim.models.layout import BusConfig, Scenario, SubsystemConfig, ThrusterConfig


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Scenario file {path} did not contain a mapping.")
    return payload


def scenario_from_dict(payload: dict[str, Any]) -> Scenario:
    """Build a typed scenario from a raw mapping."""

    bus_payload = payload["bus"]
    thruster_payloads = payload["thrusters"]
    subsystem_payloads = payload["subsystems"]

    bus = BusConfig(
        width_m=float(bus_payload["width_m"]),
        height_m=float(bus_payload["height_m"]),
        grid_resolution=int(bus_payload.get("grid_resolution", 220)),
    )

    thrusters = [
        ThrusterConfig(
            name=str(item["name"]),
            x_m=float(item["x_m"]),
            y_m=float(item["y_m"]),
            orientation_deg=float(item.get("orientation_deg", 0.0)),
            power_kw=float(item["power_kw"]),
            plume_half_angle_deg=float(item.get("plume_half_angle_deg", 20.0)),
            thermal_decay_m=float(item.get("thermal_decay_m", 0.28)),
            emi_decay_m=float(item.get("emi_decay_m", 0.18)),
            thermal_scale=float(item.get("thermal_scale", 1.0)),
            emi_scale=float(item.get("emi_scale", 1.0)),
        )
        for item in thruster_payloads
    ]

    subsystems = [
        SubsystemConfig(
            name=str(item["name"]),
            x_m=float(item["x_m"]),
            y_m=float(item["y_m"]),
            thermal_limit=float(item["thermal_limit"]),
            emi_limit=float(item["emi_limit"]),
            criticality=float(item.get("criticality", 1.0)),
            thermal_shielding=float(item.get("thermal_shielding", 0.0)),
            emi_shielding=float(item.get("emi_shielding", 0.0)),
            failure_mode=str(item.get("failure_mode", "")),
        )
        for item in subsystem_payloads
    ]

    return Scenario(
        name=str(payload["name"]),
        description=str(payload.get("description", "")),
        bus=bus,
        thrusters=thrusters,
        subsystems=subsystems,
    )


def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario from a YAML file."""

    scenario_path = Path(path)
    payload = _load_yaml(scenario_path)
    return scenario_from_dict(payload)


def list_scenarios(config_dir: str | Path) -> dict[str, Path]:
    """Return scenario names mapped to their YAML paths."""

    base_path = Path(config_dir)
    scenario_paths = sorted(base_path.glob("*.yaml"))
    scenarios = {load_scenario(path).name: path for path in scenario_paths}
    if not scenarios:
        raise FileNotFoundError(f"No YAML scenarios were found under {base_path}.")
    return scenarios
