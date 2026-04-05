"""Physics sanity checks for the simplified thermal and EMI models."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from clustered_ep_sim.models.emi import emi_contribution
from clustered_ep_sim.models.layout import BusConfig, Scenario, ThrusterConfig
from clustered_ep_sim.models.risk import RiskReport
from clustered_ep_sim.models.thermal import thermal_contribution


@dataclass(slots=True)
class PhysicsCheck:
    """Small pass/fail verification item for the current scenario."""

    name: str
    passed: bool
    summary: str
    meaning: str


def _point_in_thruster_frame(thruster: ThrusterConfig, downrange_m: float, crossrange_m: float) -> tuple[float, float]:
    cos_theta = math.cos(thruster.orientation_rad)
    sin_theta = math.sin(thruster.orientation_rad)
    x_m = thruster.x_m + downrange_m * cos_theta - crossrange_m * sin_theta
    y_m = thruster.y_m + downrange_m * sin_theta + crossrange_m * cos_theta
    return x_m, y_m


def _sample_scalar(fn, thruster: ThrusterConfig, x_m: float, y_m: float) -> float:
    x = np.array([[x_m]], dtype=float)
    y = np.array([[y_m]], dtype=float)
    return float(fn(x, y, thruster)[0, 0])


def _forward_distance_limit(thruster: ThrusterConfig, bus: BusConfig) -> float:
    cos_theta = math.cos(thruster.orientation_rad)
    sin_theta = math.sin(thruster.orientation_rad)
    candidates: list[float] = []

    if cos_theta > 1e-9:
        candidates.append((bus.width_m - thruster.x_m) / cos_theta)
    elif cos_theta < -1e-9:
        candidates.append((0.0 - thruster.x_m) / cos_theta)

    if sin_theta > 1e-9:
        candidates.append((bus.height_m - thruster.y_m) / sin_theta)
    elif sin_theta < -1e-9:
        candidates.append((0.0 - thruster.y_m) / sin_theta)

    positive = [candidate for candidate in candidates if candidate > 0.0]
    return min(positive) if positive else 0.0


def _thermal_decay_check(scenario: Scenario) -> PhysicsCheck:
    near_values: list[float] = []
    far_values: list[float] = []

    for thruster in scenario.thrusters:
        limit = _forward_distance_limit(thruster, scenario.bus)
        if limit < 0.12:
            continue
        near_distance = min(0.10, 0.35 * limit)
        far_distance = min(0.28, 0.75 * limit)
        if far_distance <= near_distance:
            continue
        near_point = _point_in_thruster_frame(thruster, near_distance, 0.0)
        far_point = _point_in_thruster_frame(thruster, far_distance, 0.0)
        near_values.append(_sample_scalar(thermal_contribution, thruster, *near_point))
        far_values.append(_sample_scalar(thermal_contribution, thruster, *far_point))

    if not near_values:
        return PhysicsCheck(
            name="Thermal axial decay",
            passed=False,
            summary="not enough room on the bus to evaluate",
            meaning="the thermal proxy should drop as you move downrange from a thruster.",
        )

    near_mean = float(np.mean(near_values))
    far_mean = float(np.mean(far_values))
    return PhysicsCheck(
        name="Thermal axial decay",
        passed=near_mean > far_mean,
        summary=f"mean near/far thermal = {near_mean:.2f} / {far_mean:.2f}",
        meaning="the thermal proxy should drop as you move downrange from a thruster.",
    )


def _directionality_check(scenario: Scenario) -> PhysicsCheck:
    on_axis_values: list[float] = []
    off_axis_values: list[float] = []

    for thruster in scenario.thrusters:
        limit = _forward_distance_limit(thruster, scenario.bus)
        if limit < 0.14:
            continue
        downrange = min(0.18, 0.50 * limit)
        plume_sigma = max(0.08, math.tan(math.radians(thruster.plume_half_angle_deg)) * (downrange + 0.08))
        off_axis = min(0.16, 1.2 * plume_sigma)
        on_axis_point = _point_in_thruster_frame(thruster, downrange, 0.0)
        off_axis_point = _point_in_thruster_frame(thruster, downrange, off_axis)
        on_axis_values.append(_sample_scalar(thermal_contribution, thruster, *on_axis_point))
        off_axis_values.append(_sample_scalar(thermal_contribution, thruster, *off_axis_point))

    if not on_axis_values:
        return PhysicsCheck(
            name="Plume directionality",
            passed=False,
            summary="not enough room on the bus to evaluate",
            meaning="on-axis plume loading should be stronger than off-axis loading.",
        )

    on_axis_mean = float(np.mean(on_axis_values))
    off_axis_mean = float(np.mean(off_axis_values))
    return PhysicsCheck(
        name="Plume directionality",
        passed=on_axis_mean > off_axis_mean,
        summary=f"mean on-axis/off-axis thermal = {on_axis_mean:.2f} / {off_axis_mean:.2f}",
        meaning="on-axis plume loading should be stronger than off-axis loading.",
    )


def _emi_decay_check(scenario: Scenario) -> PhysicsCheck:
    near_values: list[float] = []
    far_values: list[float] = []

    for thruster in scenario.thrusters:
        limit = _forward_distance_limit(thruster, scenario.bus)
        if limit < 0.10:
            continue
        near_distance = min(0.05, 0.25 * limit)
        far_distance = min(0.22, 0.70 * limit)
        if far_distance <= near_distance:
            continue
        near_point = _point_in_thruster_frame(thruster, near_distance, 0.0)
        far_point = _point_in_thruster_frame(thruster, far_distance, 0.0)
        near_values.append(_sample_scalar(emi_contribution, thruster, *near_point))
        far_values.append(_sample_scalar(emi_contribution, thruster, *far_point))

    if not near_values:
        return PhysicsCheck(
            name="EMI radial decay",
            passed=False,
            summary="not enough room on the bus to evaluate",
            meaning="the magnetic/EMI proxy should be strongest near the thruster and weaken away from it.",
        )

    near_mean = float(np.mean(near_values))
    far_mean = float(np.mean(far_values))
    return PhysicsCheck(
        name="EMI radial decay",
        passed=near_mean > far_mean,
        summary=f"mean near/far EMI = {near_mean:.2f} / {far_mean:.2f}",
        meaning="the magnetic/EMI proxy should be strongest near the thruster and weaken away from it.",
    )


def _overlap_check(thermal_contributions: Sequence[np.ndarray], emi_contributions: Sequence[np.ndarray]) -> PhysicsCheck:
    if len(thermal_contributions) < 2 or len(emi_contributions) < 2:
        return PhysicsCheck(
            name="Field superposition",
            passed=True,
            summary="single-thruster case, overlap check not needed",
            meaning="multiple thrusters should create a stronger shared zone than any one thruster alone.",
        )

    thermal_stack = np.stack(thermal_contributions)
    emi_stack = np.stack(emi_contributions)
    total_thermal = np.sum(thermal_stack, axis=0)
    total_emi = np.sum(emi_stack, axis=0)
    thermal_overlap_gain = float(np.max(total_thermal - np.max(thermal_stack, axis=0)) / np.max(thermal_stack))
    emi_overlap_gain = float(np.max(total_emi - np.max(emi_stack, axis=0)) / np.max(emi_stack))
    return PhysicsCheck(
        name="Field superposition",
        passed=thermal_overlap_gain >= 0.05 and emi_overlap_gain >= 0.05,
        summary=f"shared-zone gain thermal/EMI = {thermal_overlap_gain:.2f} / {emi_overlap_gain:.2f}",
        meaning="multiple thrusters should create a stronger shared zone than any one thruster alone.",
    )


def _shielding_check(report: RiskReport) -> PhysicsCheck:
    thermal_reduction = [
        assessment.thermal_raw - assessment.thermal_effective
        for assessment in report.assessments
        if assessment.thermal_effective < assessment.thermal_raw
    ]
    emi_reduction = [
        assessment.emi_raw - assessment.emi_effective
        for assessment in report.assessments
        if assessment.emi_effective < assessment.emi_raw
    ]
    passed = bool(thermal_reduction or emi_reduction)
    summary = (
        f"subsystems with reduced exposure = {max(len(thermal_reduction), len(emi_reduction))}"
        if passed
        else "no active shielding reduction in this scenario"
    )
    return PhysicsCheck(
        name="Shielding reduction",
        passed=passed,
        summary=summary,
        meaning="shielding inputs should reduce the effective exposure seen by subsystems.",
    )


def run_physics_checks(
    scenario: Scenario,
    thermal_contributions: Sequence[np.ndarray],
    emi_contributions: Sequence[np.ndarray],
    report: RiskReport,
) -> list[PhysicsCheck]:
    """Return a compact set of sanity checks for the active scenario."""

    return [
        _thermal_decay_check(scenario),
        _directionality_check(scenario),
        _emi_decay_check(scenario),
        _overlap_check(thermal_contributions, emi_contributions),
        _shielding_check(report),
    ]
