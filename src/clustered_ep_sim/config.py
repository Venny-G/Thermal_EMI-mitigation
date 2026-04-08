"""Scenario loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from clustered_ep_sim.models.layout import (
    BusConfig,
    EmiCalibrationConfig,
    Scenario,
    SubsystemConfig,
    ThermalCalibrationConfig,
    ThrusterConfig,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Scenario file {path} did not contain a mapping.")
    return payload


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _parse_thermal_calibration(payload: dict[str, Any]) -> ThermalCalibrationConfig:
    calibration_block = payload.get("calibration", {})
    thermal_block = calibration_block.get("thermal", {}) if isinstance(calibration_block, dict) else {}
    enabled = bool(
        thermal_block.get(
            "enabled",
            payload.get("thermal_calibration_enabled", False),
        )
    )
    return ThermalCalibrationConfig(
        enabled=enabled,
        reference_value_w_m2=_optional_float(
            thermal_block.get(
                "reference_value_W_m2",
                payload.get("thermal_reference_value_W_m2"),
            )
        ),
        reference_location_x_m=_optional_float(
            thermal_block.get(
                "reference_location_x_m",
                payload.get("thermal_reference_location_x_m"),
            )
        ),
        reference_location_y_m=_optional_float(
            thermal_block.get(
                "reference_location_y_m",
                payload.get("thermal_reference_location_y_m"),
            )
        ),
    )


def _parse_emi_calibration(payload: dict[str, Any]) -> EmiCalibrationConfig:
    calibration_block = payload.get("calibration", {})
    emi_block = calibration_block.get("emi", {}) if isinstance(calibration_block, dict) else {}
    enabled = bool(
        emi_block.get(
            "enabled",
            payload.get("emi_calibration_enabled", False),
        )
    )
    return EmiCalibrationConfig(
        enabled=enabled,
        reference_value_uT=_optional_float(
            emi_block.get(
                "reference_value_uT",
                payload.get("emi_reference_value_uT"),
            )
        ),
        reference_location_x_m=_optional_float(
            emi_block.get(
                "reference_location_x_m",
                payload.get("emi_reference_location_x_m"),
            )
        ),
        reference_location_y_m=_optional_float(
            emi_block.get(
                "reference_location_y_m",
                payload.get("emi_reference_location_y_m"),
            )
        ),
    )


def scenario_from_dict(payload: dict[str, Any]) -> Scenario:
    """Build a typed scenario from a raw mapping."""

    bus_payload = payload["bus"]
    thruster_payloads = payload["thrusters"]
    subsystem_payloads = payload["subsystems"]
    thermal_calibration = _parse_thermal_calibration(payload)
    emi_calibration = _parse_emi_calibration(payload)

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
            thermal_limit_w_m2=_optional_float(item.get("thermal_limit_W_m2")),
            emi_limit_uT=_optional_float(item.get("emi_limit_uT")),
            failure_mode=str(item.get("failure_mode", "")),
        )
        for item in subsystem_payloads
    ]

    if thermal_calibration.enabled:
        missing = [
            name
            for name, value in (
                ("thermal_reference_value_W_m2", thermal_calibration.reference_value_w_m2),
                ("thermal_reference_location_x_m", thermal_calibration.reference_location_x_m),
                ("thermal_reference_location_y_m", thermal_calibration.reference_location_y_m),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "Thermal calibration was enabled but is missing required values: "
                + ", ".join(missing)
            )

    if emi_calibration.enabled:
        missing = [
            name
            for name, value in (
                ("emi_reference_value_uT", emi_calibration.reference_value_uT),
                ("emi_reference_location_x_m", emi_calibration.reference_location_x_m),
                ("emi_reference_location_y_m", emi_calibration.reference_location_y_m),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                "EMI calibration was enabled but is missing required values: "
                + ", ".join(missing)
            )

    return Scenario(
        name=str(payload["name"]),
        description=str(payload.get("description", "")),
        bus=bus,
        thrusters=thrusters,
        subsystems=subsystems,
        source_note=str(payload.get("source_note", "")),
        thermal_calibration=thermal_calibration,
        emi_calibration=emi_calibration,
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
