"""Geometry and typed configuration models."""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np


@dataclass(slots=True)
class BusConfig:
    """Spacecraft bus definition for the 2D layout plane."""

    width_m: float
    height_m: float
    grid_resolution: int = 220


@dataclass(slots=True)
class ThrusterConfig:
    """Simplified Hall thruster placement and plume-shape controls."""

    name: str
    x_m: float
    y_m: float
    orientation_deg: float
    power_kw: float
    plume_half_angle_deg: float = 20.0
    thermal_decay_m: float = 0.28
    emi_decay_m: float = 0.18
    thermal_scale: float = 1.0
    emi_scale: float = 1.0

    @property
    def orientation_rad(self) -> float:
        """Orientation angle in radians."""

        return math.radians(self.orientation_deg)


@dataclass(slots=True)
class SubsystemConfig:
    """Subsystem placement, tolerance thresholds, and shielding assumptions."""

    name: str
    x_m: float
    y_m: float
    thermal_limit: float
    emi_limit: float
    criticality: float = 1.0
    thermal_shielding: float = 0.0
    emi_shielding: float = 0.0
    failure_mode: str = ""


@dataclass(slots=True)
class Scenario:
    """Complete spacecraft integration scenario."""

    name: str
    description: str
    bus: BusConfig
    thrusters: list[ThrusterConfig]
    subsystems: list[SubsystemConfig]


def make_grid(bus: BusConfig) -> tuple[np.ndarray, np.ndarray]:
    """Create a square-ish computational grid over the bus plane."""

    nx = bus.grid_resolution
    aspect_ratio = bus.height_m / bus.width_m
    ny = max(2, int(round(nx * aspect_ratio)))
    x_axis = np.linspace(0.0, bus.width_m, nx)
    y_axis = np.linspace(0.0, bus.height_m, ny)
    return np.meshgrid(x_axis, y_axis)


def adjust_cluster_layout(
    scenario: Scenario,
    *,
    power_scale: float = 1.0,
    cant_offset_deg: float = 0.0,
    spacing_scale: float = 1.0,
) -> Scenario:
    """Apply simple global cluster edits for quick trade studies."""

    center_x = float(np.mean([thruster.x_m for thruster in scenario.thrusters]))
    center_y = float(np.mean([thruster.y_m for thruster in scenario.thrusters]))

    thrusters = []
    for thruster in scenario.thrusters:
        dx = thruster.x_m - center_x
        dy = thruster.y_m - center_y
        # Preserve mirrored canting for clustered rows instead of rotating every thruster the same way.
        if abs(thruster.orientation_deg) > 1e-9:
            cant_direction = 1.0 if thruster.orientation_deg > 0.0 else -1.0
        elif abs(dy) > 1e-9:
            cant_direction = -1.0 if dy > 0.0 else 1.0
        else:
            cant_direction = 1.0
        thrusters.append(
            replace(
                thruster,
                x_m=center_x + dx * spacing_scale,
                y_m=center_y + dy * spacing_scale,
                power_kw=thruster.power_kw * power_scale,
                orientation_deg=thruster.orientation_deg + cant_direction * cant_offset_deg,
            )
        )

    return replace(scenario, thrusters=thrusters)
