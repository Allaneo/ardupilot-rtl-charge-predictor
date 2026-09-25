"""Validation helpers for centred-payload SITL engineering flights."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Dict, Iterable, List, Sequence

from pymavlink import mavutil


@dataclass(frozen=True)
class PayloadRunMetrics:
    """Decision-window and RTL metrics from one engineering flight."""

    run_name: str
    payload_mass_kg: float | None
    rtl_charge_mah: int
    rtl_duration_s: float
    automatic_disarm_observed: bool
    rtl_start_boot_s: float
    window_duration_s: float
    current_sample_count: int
    hover_current_mean_a: float
    hover_current_std_a: float
    hover_throttle_mean: float
    motor_pwm_mean: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WindRunMetrics:
    """Truth-versus-EKF wind metrics at the RTL decision window."""

    run_name: str
    payload_mass_kg: float
    true_wind_speed_m_s: float
    true_wind_direction_deg: float
    true_wind_north_m_s: float
    true_wind_east_m_s: float
    estimated_wind_north_mean_m_s: float
    estimated_wind_east_mean_m_s: float
    estimated_wind_north_std_m_s: float
    estimated_wind_east_std_m_s: float
    estimated_wind_speed_m_s: float
    estimated_wind_direction_deg: float | None
    wind_vector_error_m_s: float
    wind_direction_error_deg: float | None
    sample_count: int
    rtl_start_boot_s: float
    window_duration_s: float
    automatic_disarm_observed: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _mean_and_std(values: Sequence[float], label: str) -> tuple[float, float]:
    if not values:
        raise ValueError(f"no {label} samples in the decision window")
    return fmean(values), pstdev(values)


def extract_payload_run_metrics(
    run_dir: Path,
    *,
    window_duration_s: float = 10.0,
) -> PayloadRunMetrics:
    """Extract pre-RTL hover metrics from one raw flight directory."""

    result_path = run_dir / "result.json"
    log_path = run_dir / "logs" / "00000001.BIN"
    result = json.loads(result_path.read_text())

    log = mavutil.mavlink_connection(str(log_path))
    rtl_start_us: int | None = None
    relevant_messages: List[Any] = []
    while True:
        message = log.recv_match()
        if message is None:
            break
        message_type = message.get_type()
        if (
            message_type == "MODE"
            and int(message.ModeNum) == 6
            and rtl_start_us is None
        ):
            rtl_start_us = int(message.TimeUS)
        elif message_type in {"BAT", "CTUN", "RCOU"}:
            relevant_messages.append(message)

    if rtl_start_us is None:
        raise ValueError(f"no RTL mode transition found in {log_path}")

    window_start_us = rtl_start_us - int(window_duration_s * 1_000_000)

    def in_window(message: Any) -> bool:
        return window_start_us <= int(message.TimeUS) < rtl_start_us

    current_values = [
        float(message.Curr)
        for message in relevant_messages
        if message.get_type() == "BAT" and in_window(message)
    ]
    throttle_values = [
        float(message.ThO)
        for message in relevant_messages
        if message.get_type() == "CTUN" and in_window(message)
    ]
    motor_pwm_values = [
        fmean(float(getattr(message, f"C{index}")) for index in range(1, 5))
        for message in relevant_messages
        if message.get_type() == "RCOU" and in_window(message)
    ]

    current_mean, current_std = _mean_and_std(current_values, "battery current")
    throttle_mean, _ = _mean_and_std(throttle_values, "throttle")
    motor_pwm_mean, _ = _mean_and_std(motor_pwm_values, "motor PWM")

    return PayloadRunMetrics(
        run_name=run_dir.name,
        payload_mass_kg=result.get("payload_mass_kg"),
        rtl_charge_mah=int(result["rtl_charge_mah"]),
        rtl_duration_s=float(result["rtl_duration_s"]),
        automatic_disarm_observed=bool(result["automatic_disarm_observed"]),
        rtl_start_boot_s=rtl_start_us / 1_000_000.0,
        window_duration_s=window_duration_s,
        current_sample_count=len(current_values),
        hover_current_mean_a=current_mean,
        hover_current_std_a=current_std,
        hover_throttle_mean=throttle_mean,
        motor_pwm_mean=motor_pwm_mean,
    )


def _strictly_increasing(values: Iterable[float]) -> bool:
    sequence = list(values)
    return all(right > left for left, right in zip(sequence, sequence[1:]))


def wind_vector_from_direction(
    speed_m_s: float,
    direction_from_deg: float,
) -> tuple[float, float]:
    """Convert meteorological direction-from wind to North/East motion."""

    radians = math.radians(direction_from_deg)
    return -math.cos(radians) * speed_m_s, -math.sin(radians) * speed_m_s


def wind_direction_from_vector(north_m_s: float, east_m_s: float) -> float | None:
    speed = math.hypot(north_m_s, east_m_s)
    if speed < 1e-6:
        return None
    return math.degrees(math.atan2(-east_m_s, -north_m_s)) % 360.0


def circular_error_deg(estimate_deg: float, truth_deg: float) -> float:
    return abs((estimate_deg - truth_deg + 180.0) % 360.0 - 180.0)


def extract_wind_run_metrics(
    run_dir: Path,
    *,
    window_duration_s: float = 10.0,
) -> WindRunMetrics:
    """Extract primary-core EKF wind estimates during the pre-RTL window."""

    result = json.loads((run_dir / "result.json").read_text())
    speed = float(result["wind_speed_m_s"])
    direction = float(result["wind_direction_deg"])
    payload = float(result["payload_mass_kg"])
    true_north, true_east = wind_vector_from_direction(speed, direction)

    log_path = run_dir / "logs" / "00000001.BIN"
    log = mavutil.mavlink_connection(str(log_path))
    rtl_start_us: int | None = None
    wind_messages: List[Any] = []
    while True:
        message = log.recv_match()
        if message is None:
            break
        message_type = message.get_type()
        if (
            message_type == "MODE"
            and int(message.ModeNum) == 6
            and rtl_start_us is None
        ):
            rtl_start_us = int(message.TimeUS)
        elif message_type == "XKF2" and int(message.C) == 0:
            wind_messages.append(message)

    if rtl_start_us is None:
        raise ValueError(f"no RTL mode transition found in {log_path}")
    window_start_us = rtl_start_us - int(window_duration_s * 1_000_000)
    decision_messages = [
        message
        for message in wind_messages
        if window_start_us <= int(message.TimeUS) < rtl_start_us
    ]
    north_values = [float(message.VWN) for message in decision_messages]
    east_values = [float(message.VWE) for message in decision_messages]
    north_mean, north_std = _mean_and_std(north_values, "EKF north wind")
    east_mean, east_std = _mean_and_std(east_values, "EKF east wind")
    estimated_speed = math.hypot(north_mean, east_mean)
    estimated_direction = wind_direction_from_vector(north_mean, east_mean)
    direction_error = (
        None
        if speed < 0.25 or estimated_direction is None
        else circular_error_deg(estimated_direction, direction)
    )

    return WindRunMetrics(
        run_name=run_dir.name,
        payload_mass_kg=payload,
        true_wind_speed_m_s=speed,
        true_wind_direction_deg=direction,
        true_wind_north_m_s=true_north,
        true_wind_east_m_s=true_east,
        estimated_wind_north_mean_m_s=north_mean,
        estimated_wind_east_mean_m_s=east_mean,
        estimated_wind_north_std_m_s=north_std,
        estimated_wind_east_std_m_s=east_std,
        estimated_wind_speed_m_s=estimated_speed,
        estimated_wind_direction_deg=estimated_direction,
        wind_vector_error_m_s=math.hypot(
            north_mean - true_north,
            east_mean - true_east,
        ),
        wind_direction_error_deg=direction_error,
        sample_count=len(decision_messages),
        rtl_start_boot_s=rtl_start_us / 1_000_000.0,
        window_duration_s=window_duration_s,
        automatic_disarm_observed=bool(result["automatic_disarm_observed"]),
    )


def wind_validation_checks(
    runs: Sequence[WindRunMetrics],
    *,
    minimum_samples: int = 50,
    maximum_vector_error_m_s: float = 1.5,
) -> Dict[str, bool]:
    """Evaluate basic M3 availability, completion and accuracy gates."""

    if not runs:
        raise ValueError("at least one wind-validation run is required")
    numeric_values = [
        value
        for run in runs
        for value in (
            run.estimated_wind_north_mean_m_s,
            run.estimated_wind_east_mean_m_s,
            run.wind_vector_error_m_s,
        )
    ]
    return {
        "all_runs_automatic_disarm": all(
            run.automatic_disarm_observed for run in runs
        ),
        "all_windows_have_samples": all(
            run.sample_count >= minimum_samples for run in runs
        ),
        "all_estimates_are_finite": all(math.isfinite(value) for value in numeric_values),
        "all_vector_errors_within_engineering_limit": all(
            run.wind_vector_error_m_s <= maximum_vector_error_m_s for run in runs
        ),
    }


def payload_validation_checks(
    control: PayloadRunMetrics,
    patched_runs: Sequence[PayloadRunMetrics],
    *,
    zero_charge_tolerance_mah: int = 5,
    zero_current_tolerance_a: float = 0.10,
    zero_throttle_tolerance: float = 0.002,
) -> Dict[str, bool]:
    """Evaluate the M2 zero-equivalence and monotonicity gates."""

    ordered = sorted(
        patched_runs,
        key=lambda run: -math.inf if run.payload_mass_kg is None else run.payload_mass_kg,
    )
    if not ordered or ordered[0].payload_mass_kg != 0.0:
        raise ValueError("patched_runs must include a zero-payload flight")
    if any(run.payload_mass_kg is None for run in ordered):
        raise ValueError("every patched run must record payload_mass_kg")

    zero = ordered[0]
    return {
        "all_runs_automatic_disarm": control.automatic_disarm_observed
        and all(run.automatic_disarm_observed for run in ordered),
        "zero_charge_matches_unmodified": abs(
            zero.rtl_charge_mah - control.rtl_charge_mah
        )
        <= zero_charge_tolerance_mah,
        "zero_current_matches_unmodified": abs(
            zero.hover_current_mean_a - control.hover_current_mean_a
        )
        <= zero_current_tolerance_a,
        "zero_throttle_matches_unmodified": abs(
            zero.hover_throttle_mean - control.hover_throttle_mean
        )
        <= zero_throttle_tolerance,
        "hover_current_increases_with_payload": _strictly_increasing(
            run.hover_current_mean_a for run in ordered
        ),
        "hover_throttle_increases_with_payload": _strictly_increasing(
            run.hover_throttle_mean for run in ordered
        ),
        "rtl_charge_increases_with_payload": _strictly_increasing(
            float(run.rtl_charge_mah) for run in ordered
        ),
    }
