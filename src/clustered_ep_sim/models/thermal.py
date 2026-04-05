"""Simplified directional thermal field model."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from clustered_ep_sim.models.layout import ThrusterConfig

BACKFLOW_FRACTION = 0.12
NEAR_FIELD_RADIUS_M = 0.06


def _thruster_coordinates(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    thruster: ThrusterConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project grid coordinates into a thruster-aligned local frame."""

    dx = grid_x - thruster.x_m
    dy = grid_y - thruster.y_m
    cos_theta = math.cos(thruster.orientation_rad)
    sin_theta = math.sin(thruster.orientation_rad)
    downrange = dx * cos_theta + dy * sin_theta
    crossrange = -dx * sin_theta + dy * cos_theta
    distance = np.hypot(dx, dy)
    return downrange, crossrange, distance


def thermal_contribution(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    thruster: ThrusterConfig,
) -> np.ndarray:
    """Return a thermal load proxy for one thruster.

    This is a heuristic plume-shaped field, not a transient thermal solution.
    It is intended for relative integration studies and layout comparisons.
    """

    downrange, crossrange, distance = _thruster_coordinates(grid_x, grid_y, thruster)
    forward_distance = np.clip(downrange, 0.0, None)
    plume_angle = math.radians(thruster.plume_half_angle_deg)
    plume_sigma = np.maximum(0.05, np.tan(plume_angle) * (forward_distance + 0.08))
    forward_plume = np.exp(-forward_distance / thruster.thermal_decay_m)
    cross_decay = np.exp(-0.5 * (crossrange / plume_sigma) ** 2)
    backflow = BACKFLOW_FRACTION * np.exp(
        -np.abs(np.minimum(downrange, 0.0)) / max(0.05, 0.4 * thruster.thermal_decay_m)
    ) * np.exp(-(crossrange / 0.14) ** 2)
    near_field = 0.60 * np.exp(-(distance / NEAR_FIELD_RADIUS_M) ** 2)
    return thruster.power_kw * thruster.thermal_scale * (forward_plume * cross_decay + backflow + near_field)


def compute_thermal_field(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    thrusters: Sequence[ThrusterConfig],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Superpose all thruster thermal contributions."""

    contributions = [thermal_contribution(grid_x, grid_y, thruster) for thruster in thrusters]
    if not contributions:
        return np.zeros_like(grid_x), []
    return np.sum(contributions, axis=0), contributions
