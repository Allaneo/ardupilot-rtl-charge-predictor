"""DataFlash parsing and one-row-per-flight feature construction."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from pymavlink import mavutil

from rtl_charge.validation import wind_vector_from_direction


class ProcessingError(ValueError):
    """Raised when a completed run cannot produce a trustworthy dataset row."""


def _finite(values: Iterable[float], label: str) -> list[float]:
    result = [float(value) for value in values if math.isfinite(float(value))]
    if not result:
        raise ProcessingError(f"no finite {label} samples")
    return result


def _mean_std(values: Iterable[float], label: str) -> tuple[float, float]:
    finite = _finite(values, label)
    return fmean(finite), pstdev(finite)


def _latest(messages: Sequence[Any], field: str, label: str) -> float:
    if not messages:
        raise ProcessingError(f"no {label} samples")
    value = float(getattr(messages[-1], field))
    if not math.isfinite(value):
        raise ProcessingError(f"latest {label} value is not finite")
    return value


def _parameter(parameters: Mapping[str, float], name: str) -> float:
    try:
        value = float(parameters[name])
    except KeyError as error:
        raise ProcessingError(f"required parameter {name} is absent from the log") from error
    if not math.isfinite(value):
        raise ProcessingError(f"required parameter {name} is not finite")
    return value


def _series(messages: Sequence[Any], field: str) -> tuple[np.ndarray, np.ndarray]:
    points = sorted(
        (float(message.TimeUS) / 1_000_000.0, float(getattr(message, field)))
        for message in messages
        if math.isfinite(float(getattr(message, field)))
    )
    if len(points) < 2:
        raise ProcessingError(f"fewer than two finite {field} samples")
    return np.asarray([point[0] for point in points]), np.asarray([point[1] for point in points])


def _interpolated_difference(
    messages: Sequence[Any],
    field: str,
    start_s: float,
    end_s: float,
) -> float:
    times, values = _series(messages, field)
    if start_s < times[0] or end_s > times[-1]:
        raise ProcessingError(f"{field} samples do not bracket the RTL interval")
    return float(np.interp(end_s, times, values) - np.interp(start_s, times, values))


def _trapezoid_integral(
    messages: Sequence[Any],
    field: str,
    start_s: float,
    end_s: float,
    *,
    multiplier_field: str | None = None,
) -> float:
    times, values = _series(messages, field)
    if start_s < times[0] or end_s > times[-1]:
        raise ProcessingError(f"{field} samples do not bracket the RTL interval")
    interior = times[(times > start_s) & (times < end_s)]
    integration_times = np.concatenate(([start_s], interior, [end_s]))
    integrated_values = np.interp(integration_times, times, values)
    if multiplier_field is not None:
        multiplier_times, multiplier_values = _series(messages, multiplier_field)
        integrated_values = integrated_values * np.interp(
            integration_times,
            multiplier_times,
            multiplier_values,
        )
    return float(np.trapezoid(integrated_values, integration_times))


def read_dataflash(log_path: Path) -> tuple[int, int, dict[str, list[Any]], dict[str, float]]:
    """Read required messages and locate the first complete RTL interval."""

    messages: dict[str, list[Any]] = defaultdict(list)
    parameters: dict[str, float] = {}
    rtl_start_us: int | None = None
    disarm_us: int | None = None
    wanted = {"BAT", "CTUN", "RCOU", "ATT", "RATE", "XKF1", "XKF2"}
    log = mavutil.mavlink_connection(str(log_path))
    while True:
        message = log.recv_match()
        if message is None:
            break
        message_type = message.get_type()
        if message_type == "PARM":
            parameters[str(message.Name)] = float(message.Value)
        elif message_type == "MODE" and rtl_start_us is None and int(message.ModeNum) == 6:
            rtl_start_us = int(message.TimeUS)
        elif (
            message_type == "ARM"
            and rtl_start_us is not None
            and disarm_us is None
            and int(message.ArmState) == 0
            and int(message.TimeUS) > rtl_start_us
        ):
            disarm_us = int(message.TimeUS)
        elif message_type in wanted:
            if message_type == "BAT" and int(message.Inst) != 0:
                continue
            if message_type in {"XKF1", "XKF2"} and int(message.C) != 0:
                continue
            messages[message_type].append(message)

    if rtl_start_us is None:
        raise ProcessingError(f"no RTL mode transition found in {log_path}")
    if disarm_us is None:
        raise ProcessingError(f"no automatic disarm after RTL found in {log_path}")
    return rtl_start_us, disarm_us, dict(messages), parameters


def _window(
    messages: Sequence[Any],
    start_us: int,
    end_us: int,
) -> list[Any]:
    return [message for message in messages if start_us <= int(message.TimeUS) < end_us]


def _motor_features(
    messages: Sequence[Any],
    *,
    pwm_min: float,
    pwm_max: float,
) -> dict[str, float | int]:
    if pwm_max <= pwm_min:
        raise ProcessingError("MOT_PWM_MAX must exceed MOT_PWM_MIN")
    channel_names = [f"C{index}" for index in range(1, 15)]
    active_channels = [
        name
        for name in channel_names
        if any(float(getattr(message, name, 0.0)) >= pwm_min for message in messages)
    ]
    if not active_channels:
        raise ProcessingError("no active motor-output channels in decision window")
    normalized = [
        min(1.0, max(0.0, (float(getattr(message, name)) - pwm_min) / (pwm_max - pwm_min)))
        for message in messages
        for name in active_channels
    ]
    mean, std = _mean_std(normalized, "normalized motor output")
    return {
        "active_motor_count": len(active_channels),
        "motor_output_mean": mean,
        "motor_output_std": std,
        "motor_output_max": max(normalized),
        "motor_saturation_fraction": sum(value >= 0.95 for value in normalized) / len(normalized),
    }


def _route_state(xkf1_window: Sequence[Any]) -> dict[str, float]:
    north_m = _latest(xkf1_window, "PN", "north position")
    east_m = _latest(xkf1_window, "PE", "east position")
    down_m = _latest(xkf1_window, "PD", "down position")
    distance_m = math.hypot(north_m, east_m)
    if distance_m < 1.0:
        raise ProcessingError("decision point is too close to home for a return route")
    return_north = -north_m / distance_m
    return_east = -east_m / distance_m
    return {
        "distance_home_m": distance_m,
        "relative_altitude_m": -down_m,
        "route_bearing_sin": return_east,
        "route_bearing_cos": return_north,
        "return_unit_north": return_north,
        "return_unit_east": return_east,
    }


def _physics_features(
    *,
    route: Mapping[str, float],
    parameters: Mapping[str, float],
    wind_north_m_s: float,
    wind_east_m_s: float,
    current_mean_a: float,
) -> dict[str, float]:
    horizontal_speed = _parameter(parameters, "RTL_SPEED_MS")
    if horizontal_speed <= 0:
        horizontal_speed = _parameter(parameters, "WP_SPD")
    climb_speed = _parameter(parameters, "WP_SPD_UP")
    descent_speed = _parameter(parameters, "WP_SPD_DN")
    land_speed = _parameter(parameters, "LAND_SPD_MS")
    land_high_speed = _parameter(parameters, "LAND_SPD_HIGH_MS") or descent_speed
    land_low_altitude = _parameter(parameters, "LAND_ALT_LOW_M")
    rtl_altitude = _parameter(parameters, "RTL_ALT_M")
    minimum_climb = _parameter(parameters, "RTL_CLIMB_MIN_M")

    altitude = route["relative_altitude_m"]
    return_altitude = max(altitude + minimum_climb, rtl_altitude)
    planned_climb = max(0.0, return_altitude - altitude)
    unit_north = route["return_unit_north"]
    unit_east = route["return_unit_east"]
    along_wind = wind_north_m_s * unit_north + wind_east_m_s * unit_east
    crosswind = wind_north_m_s * unit_east - wind_east_m_s * unit_north
    kinematic_ground_speed = max(1.0, horizontal_speed + along_wind)
    horizontal_time = route["distance_home_m"] / kinematic_ground_speed
    climb_time = planned_climb / climb_speed
    high_descent = max(0.0, return_altitude - land_low_altitude) / land_high_speed
    low_descent = min(return_altitude, land_low_altitude) / land_speed
    descent_time = high_descent + low_descent
    loiter_time = _parameter(parameters, "RTL_LOIT_TIME") / 1000.0
    total_time = climb_time + horizontal_time + loiter_time + descent_time
    desired_ground_north = horizontal_speed * unit_north
    desired_ground_east = horizontal_speed * unit_east
    air_north = desired_ground_north - wind_north_m_s
    air_east = desired_ground_east - wind_east_m_s
    airspeed = math.hypot(air_north, air_east)
    return {
        "planned_rtl_climb_m": planned_climb,
        "headwind_m_s": -along_wind,
        "crosswind_abs_m_s": abs(crosswind),
        "nominal_horizontal_time_s": horizontal_time,
        "nominal_climb_time_s": climb_time,
        "nominal_descent_time_s": descent_time,
        "nominal_total_rtl_time_s": total_time,
        "required_air_velocity_north_m_s": air_north,
        "required_air_velocity_east_m_s": air_east,
        "required_airspeed_m_s": airspeed,
        "required_airspeed_squared_m2_s2": airspeed**2,
        "current_time_charge_baseline_mah": current_mean_a * total_time * 1000.0 / 3600.0,
    }


def build_flight_row(
    run_dir: Path,
    *,
    observation_window_s: float | None = None,
    charge_absolute_tolerance_mah: float = 2.0,
    charge_relative_tolerance: float = 0.01,
) -> dict[str, Any]:
    """Build and quality-check one processed row from a completed run."""

    result_path = run_dir / "result.json"
    log_path = run_dir / "logs" / "00000001.BIN"
    if not result_path.exists() or not log_path.exists():
        raise ProcessingError(f"run is missing result.json or DataFlash log: {run_dir}")
    result = json.loads(result_path.read_text())
    if not result.get("automatic_disarm_observed", False):
        raise ProcessingError("result does not confirm automatic disarm")
    if observation_window_s is None:
        observation_window_s = float(result["observation_window_s"])

    rtl_start_us, disarm_us, messages, parameters = read_dataflash(log_path)
    window_start_us = rtl_start_us - int(observation_window_s * 1_000_000)
    decision = {
        name: _window(items, window_start_us, rtl_start_us)
        for name, items in messages.items()
    }
    required_types = {"BAT", "CTUN", "RCOU", "ATT", "RATE", "XKF1", "XKF2"}
    missing = [name for name in sorted(required_types) if not decision.get(name)]
    if missing:
        raise ProcessingError(f"decision window lacks message types: {missing}")

    start_s = rtl_start_us / 1_000_000.0
    end_s = disarm_us / 1_000_000.0
    bat = messages["BAT"]
    logged_charge = _interpolated_difference(bat, "CurrTot", start_s, end_s)
    integrated_charge = _trapezoid_integral(bat, "Curr", start_s, end_s) * 1000.0 / 3600.0
    integrated_energy = _trapezoid_integral(
        bat,
        "Curr",
        start_s,
        end_s,
        multiplier_field="Volt",
    ) / 3600.0
    allowed_error = max(charge_absolute_tolerance_mah, charge_relative_tolerance * logged_charge)
    crosscheck_error = abs(integrated_charge - logged_charge)
    if crosscheck_error > allowed_error:
        raise ProcessingError(
            "charge-label cross-check failed: "
            f"logged={logged_charge:.3f} mAh, integrated={integrated_charge:.3f} mAh, "
            f"allowed={allowed_error:.3f} mAh"
        )

    battery_current = _finite((message.Curr for message in decision["BAT"]), "battery current")
    battery_voltage = _finite((message.Volt for message in decision["BAT"]), "battery voltage")
    current_mean, current_std = _mean_std(battery_current, "battery current")
    voltage_mean, voltage_std = _mean_std(battery_voltage, "battery voltage")
    throttle_mean, throttle_std = _mean_std(
        (message.ThO for message in decision["CTUN"]),
        "throttle",
    )
    wind_north_mean, wind_north_std = _mean_std(
        (message.VWN for message in decision["XKF2"]),
        "EKF north wind",
    )
    wind_east_mean, wind_east_std = _mean_std(
        (message.VWE for message in decision["XKF2"]),
        "EKF east wind",
    )
    route = _route_state(decision["XKF1"])
    vertical_speeds = [-float(message.VD) for message in decision["XKF1"]]
    horizontal_speeds = [math.hypot(float(message.VN), float(message.VE)) for message in decision["XKF1"]]
    vertical_mean, vertical_std = _mean_std(vertical_speeds, "vertical speed")
    horizontal_mean, horizontal_std = _mean_std(horizontal_speeds, "horizontal speed")
    roll_radians = [math.radians(float(message.Roll)) for message in decision["ATT"]]
    pitch_radians = [math.radians(float(message.Pitch)) for message in decision["ATT"]]
    roll_rate = [math.radians(float(message.R)) for message in decision["RATE"]]
    pitch_rate = [math.radians(float(message.P)) for message in decision["RATE"]]
    _, roll_rate_std = _mean_std(roll_rate, "roll rate")
    _, pitch_rate_std = _mean_std(pitch_rate, "pitch rate")
    motor = _motor_features(
        decision["RCOU"],
        pwm_min=_parameter(parameters, "MOT_PWM_MIN"),
        pwm_max=_parameter(parameters, "MOT_PWM_MAX"),
    )
    physics = _physics_features(
        route=route,
        parameters=parameters,
        wind_north_m_s=wind_north_mean,
        wind_east_m_s=wind_east_mean,
        current_mean_a=current_mean,
    )
    wind_speed = float(result["wind_speed_m_s"])
    wind_direction = float(result["wind_direction_deg"])
    true_wind_north, true_wind_east = wind_vector_from_direction(wind_speed, wind_direction)
    telemetry_charge = float(result["rtl_charge_mah"])
    telemetry_error = abs(telemetry_charge - logged_charge)
    if telemetry_error > allowed_error:
        raise ProcessingError(
            "runner/logged charge disagreement exceeds tolerance: "
            f"runner={telemetry_charge:.3f} mAh, logged={logged_charge:.3f} mAh"
        )

    current_quantiles = np.quantile(np.asarray(battery_current), [0.25, 0.5, 0.75, 0.9])
    row: dict[str, Any] = {
        "scenario_id": result.get("scenario_id", run_dir.name),
        "run_id": result.get("run_id", run_dir.name),
        "raw_log_path": str(log_path),
        "rtl_start_boot_s": start_s,
        "disarm_boot_s": end_s,
        "decision_window_s": observation_window_s,
        "decision_battery_sample_count": len(decision["BAT"]),
        "distance_home_m": route["distance_home_m"],
        "relative_altitude_m": route["relative_altitude_m"],
        "route_bearing_sin": route["route_bearing_sin"],
        "route_bearing_cos": route["route_bearing_cos"],
        "battery_voltage_v": _latest(decision["BAT"], "Volt", "battery voltage"),
        "voltage_mean_v": voltage_mean,
        "voltage_min_v": min(battery_voltage),
        "voltage_std_v": voltage_std,
        "ekf_wind_north_m_s": _latest(decision["XKF2"], "VWN", "EKF north wind"),
        "ekf_wind_east_m_s": _latest(decision["XKF2"], "VWE", "EKF east wind"),
        "ekf_wind_north_mean_m_s": wind_north_mean,
        "ekf_wind_north_std_m_s": wind_north_std,
        "ekf_wind_east_mean_m_s": wind_east_mean,
        "ekf_wind_east_std_m_s": wind_east_std,
        "current_mean_a": current_mean,
        "current_std_a": current_std,
        "current_q25_a": float(current_quantiles[0]),
        "current_median_a": float(current_quantiles[1]),
        "current_q75_a": float(current_quantiles[2]),
        "current_q90_a": float(current_quantiles[3]),
        "throttle_mean": throttle_mean,
        "throttle_std": throttle_std,
        **motor,
        "roll_abs_mean_rad": fmean(abs(value) for value in roll_radians),
        "pitch_abs_mean_rad": fmean(abs(value) for value in pitch_radians),
        "roll_rate_std_rad_s": roll_rate_std,
        "pitch_rate_std_rad_s": pitch_rate_std,
        "vertical_speed_mean_m_s": vertical_mean,
        "vertical_speed_std_m_s": vertical_std,
        "horizontal_speed_mean_m_s": horizontal_mean,
        "horizontal_speed_std_m_s": horizontal_std,
        **physics,
        "payload_mass_kg": float(result["payload_mass_kg"]),
        "true_wind_north_m_s": true_wind_north,
        "true_wind_east_m_s": true_wind_east,
        "requested_distance_m": float(result["outbound_distance_m"]),
        "requested_outbound_bearing_deg": float(result.get("outbound_bearing_deg", 0.0)),
        "rtl_charge_mah": logged_charge,
        "rtl_energy_wh": integrated_energy,
        "rtl_duration_s": end_s - start_s,
        "rtl_charge_integrated_mah": integrated_charge,
        "rtl_charge_runner_mah": telemetry_charge,
        "charge_crosscheck_error_mah": crosscheck_error,
        "charge_crosscheck_tolerance_mah": allowed_error,
        "charge_crosscheck_passed": True,
    }
    return row
