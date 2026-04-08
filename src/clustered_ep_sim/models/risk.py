"""Subsystem exposure and combined risk logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from clustered_ep_sim.models.layout import Scenario, SubsystemConfig

STATE_ORDER = {"safe": 0, "caution": 1, "critical": 2}


@dataclass(slots=True)
class FieldPeak:
    """Maximum field location on the bus plane."""

    x_m: float
    y_m: float
    value: float


@dataclass(slots=True)
class CalibrationSummary:
    """Metadata for one proxy-to-physical field calibration."""

    enabled: bool
    reference_value: float | None
    reference_x_m: float | None
    reference_y_m: float | None
    reference_proxy_value: float | None
    scale_factor: float | None


@dataclass(slots=True)
class ThrusterContribution:
    """Per-thruster contribution at one subsystem sample point."""

    thruster_name: str
    thermal_proxy: float
    thermal_fraction: float
    emi_proxy: float
    emi_fraction: float
    q_incident_w_m2: float | None
    b_before_shield_uT: float | None


@dataclass(slots=True)
class SubsystemAssessment:
    """Risk summary for a single subsystem."""

    name: str
    x_m: float
    y_m: float
    thermal_raw: float
    emi_raw: float
    thermal_effective: float
    emi_effective: float
    thermal_ratio: float
    emi_ratio: float
    thermal_state: str
    emi_state: str
    overall_state: str
    combined_score: float
    dominant_driver: str
    dominant_thruster: str
    dominant_thermal_thruster: str
    dominant_emi_thruster: str
    likely_failure_mode: str
    thermal_ratio_screening: float
    emi_ratio_screening: float
    thermal_ratio_physical: float | None
    emi_ratio_physical: float | None
    q_proxy_sample: float
    b_proxy_sample: float
    q_incident_w_m2: float | None
    q_after_shield_w_m2: float | None
    b_before_shield_uT: float | None
    b_after_shield_uT: float | None
    thruster_contributions: list[ThrusterContribution]


@dataclass(slots=True)
class RiskReport:
    """Full scenario assessment."""

    overall_state: str
    combined_field: np.ndarray
    thermal_incident_field_w_m2: np.ndarray | None
    emi_before_shield_field_uT: np.ndarray | None
    thermal_weight: float
    emi_weight: float
    thermal_reference: float
    emi_reference: float
    thermal_calibration: CalibrationSummary
    emi_calibration: CalibrationSummary
    assessments: list[SubsystemAssessment]
    max_thermal_peak: FieldPeak
    max_emi_peak: FieldPeak
    max_combined_peak: FieldPeak
    caution_area_fraction: float
    critical_area_fraction: float
    summary_text: str
    recommendations: list[str]


def classify_ratio(ratio: float) -> str:
    """Translate a threshold ratio into a risk state."""

    if ratio < 0.70:
        return "safe"
    if ratio < 1.00:
        return "caution"
    return "critical"


def worse_state(left: str, right: str) -> str:
    """Return the more severe of two states."""

    return left if STATE_ORDER[left] >= STATE_ORDER[right] else right


def bilinear_sample(
    field: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    x_m: float,
    y_m: float,
) -> float:
    """Sample a field at an arbitrary point using bilinear interpolation."""

    x_clamped = float(np.clip(x_m, x_axis[0], x_axis[-1]))
    y_clamped = float(np.clip(y_m, y_axis[0], y_axis[-1]))
    x_index = int(np.clip(np.searchsorted(x_axis, x_clamped) - 1, 0, len(x_axis) - 2))
    y_index = int(np.clip(np.searchsorted(y_axis, y_clamped) - 1, 0, len(y_axis) - 2))

    x0, x1 = x_axis[x_index], x_axis[x_index + 1]
    y0, y1 = y_axis[y_index], y_axis[y_index + 1]
    tx = 0.0 if x1 == x0 else (x_clamped - x0) / (x1 - x0)
    ty = 0.0 if y1 == y0 else (y_clamped - y0) / (y1 - y0)

    lower = (1.0 - tx) * field[y_index, x_index] + tx * field[y_index, x_index + 1]
    upper = (1.0 - tx) * field[y_index + 1, x_index] + tx * field[y_index + 1, x_index + 1]
    return float((1.0 - ty) * lower + ty * upper)


def _sample_contributions(
    fields: Sequence[np.ndarray],
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    subsystem: SubsystemConfig,
) -> list[float]:
    return [bilinear_sample(field, x_axis, y_axis, subsystem.x_m, subsystem.y_m) for field in fields]


def _field_peak(field: np.ndarray, x_axis: np.ndarray, y_axis: np.ndarray) -> FieldPeak:
    index = int(np.argmax(field))
    y_index, x_index = np.unravel_index(index, field.shape)
    return FieldPeak(x_m=float(x_axis[x_index]), y_m=float(y_axis[y_index]), value=float(field[y_index, x_index]))


def _calibrate_field(
    field: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    *,
    enabled: bool,
    reference_value: float | None,
    reference_x_m: float | None,
    reference_y_m: float | None,
) -> tuple[CalibrationSummary, np.ndarray | None]:
    if not enabled:
        return CalibrationSummary(False, None, None, None, None, None), None
    if reference_value is None or reference_x_m is None or reference_y_m is None:
        raise ValueError("Calibration is enabled but the reference value or location is missing.")

    reference_proxy_value = bilinear_sample(
        field,
        x_axis,
        y_axis,
        reference_x_m,
        reference_y_m,
    )
    if reference_proxy_value <= 0.0:
        raise ValueError("Calibration reference point must sample a positive proxy field value.")

    scale_factor = reference_value / reference_proxy_value
    calibration = CalibrationSummary(
        enabled=True,
        reference_value=reference_value,
        reference_x_m=reference_x_m,
        reference_y_m=reference_y_m,
        reference_proxy_value=reference_proxy_value,
        scale_factor=scale_factor,
    )
    return calibration, field * scale_factor


def _dominant_thruster_name(
    scenario: Scenario,
    thermal_samples: Sequence[float],
    emi_samples: Sequence[float],
    thermal_ratio: float,
    emi_ratio: float,
) -> str:
    if not scenario.thrusters:
        return "N/A"
    scores = thermal_samples if thermal_ratio >= emi_ratio else emi_samples
    dominant_index = int(np.argmax(scores))
    return scenario.thrusters[dominant_index].name


def _build_thruster_contributions(
    scenario: Scenario,
    thermal_samples: Sequence[float],
    emi_samples: Sequence[float],
    thermal_scale_factor_w_m2: float | None,
    emi_scale_factor_uT: float | None,
) -> list[ThrusterContribution]:
    thermal_total = float(np.sum(thermal_samples))
    emi_total = float(np.sum(emi_samples))
    contributions: list[ThrusterContribution] = []

    for thruster, thermal_value, emi_value in zip(
        scenario.thrusters,
        thermal_samples,
        emi_samples,
        strict=False,
    ):
        contributions.append(
            ThrusterContribution(
                thruster_name=thruster.name,
                thermal_proxy=float(thermal_value),
                thermal_fraction=0.0 if thermal_total <= 0.0 else float(thermal_value) / thermal_total,
                emi_proxy=float(emi_value),
                emi_fraction=0.0 if emi_total <= 0.0 else float(emi_value) / emi_total,
                q_incident_w_m2=(
                    None if thermal_scale_factor_w_m2 is None else float(thermal_value) * thermal_scale_factor_w_m2
                ),
                b_before_shield_uT=(
                    None if emi_scale_factor_uT is None else float(emi_value) * emi_scale_factor_uT
                ),
            )
        )

    return contributions


def _assessment_for_subsystem(
    scenario: Scenario,
    subsystem: SubsystemConfig,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    thermal_field: np.ndarray,
    thermal_contributions: Sequence[np.ndarray],
    emi_field: np.ndarray,
    emi_contributions: Sequence[np.ndarray],
    thermal_scale_factor_w_m2: float | None,
    emi_scale_factor_uT: float | None,
) -> SubsystemAssessment:
    thermal_raw = bilinear_sample(thermal_field, x_axis, y_axis, subsystem.x_m, subsystem.y_m)
    emi_raw = bilinear_sample(emi_field, x_axis, y_axis, subsystem.x_m, subsystem.y_m)
    thermal_effective = thermal_raw * (1.0 - subsystem.thermal_shielding)
    emi_effective = emi_raw * (1.0 - subsystem.emi_shielding)
    thermal_ratio = thermal_effective / subsystem.thermal_limit
    emi_ratio = emi_effective / subsystem.emi_limit
    thermal_state = classify_ratio(thermal_ratio)
    emi_state = classify_ratio(emi_ratio)
    overall_state = worse_state(thermal_state, emi_state)
    combined_score = subsystem.criticality * (0.5 * thermal_ratio + 0.5 * emi_ratio)
    if thermal_state == "caution" and emi_state == "caution" and combined_score >= 1.0:
        overall_state = "critical"
    dominant_driver = "thermal" if thermal_ratio >= emi_ratio else "EMI"
    thermal_samples = _sample_contributions(thermal_contributions, x_axis, y_axis, subsystem)
    emi_samples = _sample_contributions(emi_contributions, x_axis, y_axis, subsystem)
    dominant_thermal_thruster = _dominant_thruster_name(
        scenario,
        thermal_samples,
        emi_samples,
        1.0,
        0.0,
    )
    dominant_emi_thruster = _dominant_thruster_name(
        scenario,
        thermal_samples,
        emi_samples,
        0.0,
        1.0,
    )
    dominant_thruster = dominant_thermal_thruster if thermal_ratio >= emi_ratio else dominant_emi_thruster
    failure_mode = subsystem.failure_mode or "elevated integration risk."
    q_incident_w_m2 = None if thermal_scale_factor_w_m2 is None else thermal_raw * thermal_scale_factor_w_m2
    q_after_shield_w_m2 = None if thermal_scale_factor_w_m2 is None else thermal_effective * thermal_scale_factor_w_m2
    b_before_shield_uT = None if emi_scale_factor_uT is None else emi_raw * emi_scale_factor_uT
    b_after_shield_uT = None if emi_scale_factor_uT is None else emi_effective * emi_scale_factor_uT
    thermal_ratio_physical = None
    emi_ratio_physical = None
    if q_after_shield_w_m2 is not None and subsystem.thermal_limit_w_m2 is not None:
        thermal_ratio_physical = q_after_shield_w_m2 / subsystem.thermal_limit_w_m2
    if b_after_shield_uT is not None and subsystem.emi_limit_uT is not None:
        emi_ratio_physical = b_after_shield_uT / subsystem.emi_limit_uT
    thruster_contributions = _build_thruster_contributions(
        scenario,
        thermal_samples,
        emi_samples,
        thermal_scale_factor_w_m2,
        emi_scale_factor_uT,
    )
    return SubsystemAssessment(
        name=subsystem.name,
        x_m=subsystem.x_m,
        y_m=subsystem.y_m,
        thermal_raw=thermal_raw,
        emi_raw=emi_raw,
        thermal_effective=thermal_effective,
        emi_effective=emi_effective,
        thermal_ratio=thermal_ratio,
        emi_ratio=emi_ratio,
        thermal_state=thermal_state,
        emi_state=emi_state,
        overall_state=overall_state,
        combined_score=combined_score,
        dominant_driver=dominant_driver,
        dominant_thruster=dominant_thruster,
        dominant_thermal_thruster=dominant_thermal_thruster,
        dominant_emi_thruster=dominant_emi_thruster,
        likely_failure_mode=failure_mode,
        thermal_ratio_screening=thermal_ratio,
        emi_ratio_screening=emi_ratio,
        thermal_ratio_physical=thermal_ratio_physical,
        emi_ratio_physical=emi_ratio_physical,
        q_proxy_sample=thermal_raw,
        b_proxy_sample=emi_raw,
        q_incident_w_m2=q_incident_w_m2,
        q_after_shield_w_m2=q_after_shield_w_m2,
        b_before_shield_uT=b_before_shield_uT,
        b_after_shield_uT=b_after_shield_uT,
        thruster_contributions=thruster_contributions,
    )


def _build_summary(
    overall_state: str,
    assessments: Sequence[SubsystemAssessment],
    max_combined_peak: FieldPeak,
) -> str:
    critical_names = [assessment.name for assessment in assessments if assessment.overall_state == "critical"]
    caution_names = [assessment.name for assessment in assessments if assessment.overall_state == "caution"]

    if overall_state == "critical":
        if critical_names:
            names = ", ".join(critical_names)
            return (
                f"Critical integration risk. {names} exceed allowable exposure, and the worst combined-risk "
                f"zone is centered near x={max_combined_peak.x_m:.2f} m, y={max_combined_peak.y_m:.2f} m."
            )
        return "Critical integration risk. The bus contains a broad combined-risk zone above the failure threshold."

    if overall_state == "caution":
        if caution_names:
            names = ", ".join(caution_names)
            return (
                f"Cautionary layout. {names} are operating inside reduced margin, with the highest combined-risk "
                f"zone near x={max_combined_peak.x_m:.2f} m, y={max_combined_peak.y_m:.2f} m."
            )
        return "Cautionary layout. Combined-risk contours are starting to encroach on usable spacecraft area."

    return "Nominal layout. All configured subsystems remain below caution thresholds in this simplified model."


def _build_recommendations(
    assessments: Sequence[SubsystemAssessment],
    critical_area_fraction: float,
) -> list[str]:
    recommendations: list[str] = []
    critical_assessments = [assessment for assessment in assessments if assessment.overall_state == "critical"]

    for assessment in critical_assessments[:3]:
        move_distance_cm = int(np.clip(np.ceil(12.0 * max(assessment.thermal_ratio, assessment.emi_ratio)), 10.0, 35.0))
        if assessment.dominant_driver == "thermal":
            recommendations.append(
                f"{assessment.name} is {assessment.thermal_ratio:.2f}x over its thermal limit; move it about "
                f"{move_distance_cm} cm away from {assessment.dominant_thruster} or add at least 35% local thermal shielding."
            )
        else:
            recommendations.append(
                f"{assessment.name} is {assessment.emi_ratio:.2f}x over its EMI limit; move it about "
                f"{move_distance_cm} cm away from {assessment.dominant_thruster} or add at least 35% local EMI shielding."
            )

    if critical_area_fraction >= 0.12:
        recommendations.append(
            f"{critical_area_fraction:.0%} of the bus is above the combined critical threshold; increase cluster spacing "
            "or reduce total thruster power density before placing additional avionics near the cluster centerline."
        )

    if not recommendations:
        recommendations.append(
            "Current geometry keeps all listed subsystems below caution limits. Use this layout as a baseline before tightening spacing."
        )

    return recommendations


def evaluate_risk(
    scenario: Scenario,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    thermal_field: np.ndarray,
    thermal_contributions: Sequence[np.ndarray],
    emi_field: np.ndarray,
    emi_contributions: Sequence[np.ndarray],
    *,
    thermal_weight: float = 0.55,
    emi_weight: float = 0.45,
) -> RiskReport:
    """Evaluate subsystem exposure and layout-level risk."""

    x_axis = grid_x[0, :]
    y_axis = grid_y[:, 0]
    weight_sum = thermal_weight + emi_weight
    if weight_sum <= 0.0:
        raise ValueError("At least one integration-screening weight must be positive.")
    thermal_weight /= weight_sum
    emi_weight /= weight_sum
    thermal_reference = float(np.median([subsystem.thermal_limit for subsystem in scenario.subsystems]))
    emi_reference = float(np.median([subsystem.emi_limit for subsystem in scenario.subsystems]))
    combined_field = thermal_weight * (thermal_field / thermal_reference) + emi_weight * (emi_field / emi_reference)
    thermal_calibration, thermal_incident_field_w_m2 = _calibrate_field(
        thermal_field,
        x_axis,
        y_axis,
        enabled=scenario.thermal_calibration.enabled,
        reference_value=scenario.thermal_calibration.reference_value_w_m2,
        reference_x_m=scenario.thermal_calibration.reference_location_x_m,
        reference_y_m=scenario.thermal_calibration.reference_location_y_m,
    )
    emi_calibration, emi_before_shield_field_uT = _calibrate_field(
        emi_field,
        x_axis,
        y_axis,
        enabled=scenario.emi_calibration.enabled,
        reference_value=scenario.emi_calibration.reference_value_uT,
        reference_x_m=scenario.emi_calibration.reference_location_x_m,
        reference_y_m=scenario.emi_calibration.reference_location_y_m,
    )

    assessments = [
        _assessment_for_subsystem(
            scenario,
            subsystem,
            x_axis,
            y_axis,
            thermal_field,
            thermal_contributions,
            emi_field,
            emi_contributions,
            thermal_calibration.scale_factor,
            emi_calibration.scale_factor,
        )
        for subsystem in scenario.subsystems
    ]

    caution_area_fraction = float(np.mean(combined_field >= 0.70))
    critical_area_fraction = float(np.mean(combined_field >= 1.00))

    overall_state = "safe"
    if critical_area_fraction >= 0.12 or any(assessment.overall_state == "critical" for assessment in assessments):
        overall_state = "critical"
    elif caution_area_fraction >= 0.05 or any(assessment.overall_state == "caution" for assessment in assessments):
        overall_state = "caution"

    max_thermal_peak = _field_peak(thermal_field, x_axis, y_axis)
    max_emi_peak = _field_peak(emi_field, x_axis, y_axis)
    max_combined_peak = _field_peak(combined_field, x_axis, y_axis)
    summary_text = _build_summary(overall_state, assessments, max_combined_peak)
    recommendations = _build_recommendations(assessments, critical_area_fraction)

    return RiskReport(
        overall_state=overall_state,
        combined_field=combined_field,
        thermal_incident_field_w_m2=thermal_incident_field_w_m2,
        emi_before_shield_field_uT=emi_before_shield_field_uT,
        thermal_weight=thermal_weight,
        emi_weight=emi_weight,
        thermal_reference=thermal_reference,
        emi_reference=emi_reference,
        thermal_calibration=thermal_calibration,
        emi_calibration=emi_calibration,
        assessments=assessments,
        max_thermal_peak=max_thermal_peak,
        max_emi_peak=max_emi_peak,
        max_combined_peak=max_combined_peak,
        caution_area_fraction=caution_area_fraction,
        critical_area_fraction=critical_area_fraction,
        summary_text=summary_text,
        recommendations=recommendations,
    )
