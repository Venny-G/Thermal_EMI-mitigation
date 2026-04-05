"""Simplified EMI / magnetic interference field model."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from clustered_ep_sim.models.layout import ThrusterConfig

COIL_RADIUS_M = 0.09


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


def emi_contribution(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    thruster: ThrusterConfig,
) -> np.ndarray:
    """Return an EMI exposure proxy for one thruster.

    The field combines a near-thruster magnetic term with a directional tail.
    It is a quick-look integration model, not a full electromagnetic solution.
    """

    downrange, crossrange, distance = _thruster_coordinates(grid_x, grid_y, thruster)
    forward_distance = np.clip(downrange, 0.0, None)
    plume_angle = math.radians(thruster.plume_half_angle_deg)
    plume_sigma = np.maximum(0.06, 0.8 * np.tan(plume_angle) * (forward_distance + 0.06))
    directional_tail = 0.45 * np.exp(-forward_distance / thruster.emi_decay_m) * np.exp(
        -0.5 * (crossrange / plume_sigma) ** 2
    )
    near_field = 1.15 / (1.0 + (distance / COIL_RADIUS_M) ** 3)
    side_lobe = 0.18 * np.exp(-0.5 * (crossrange / 0.12) ** 2) / (1.0 + (distance / 0.25) ** 2)
    return thruster.power_kw * thruster.emi_scale * (near_field + directional_tail + side_lobe)


def compute_emi_field(
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    thrusters: Sequence[ThrusterConfig],
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Superpose all thruster EMI contributions."""

    contributions = [emi_contribution(grid_x, grid_y, thruster) for thruster in thrusters]
    if not contributions:
        return np.zeros_like(grid_x), []
    return np.sum(contributions, axis=0), contributions
