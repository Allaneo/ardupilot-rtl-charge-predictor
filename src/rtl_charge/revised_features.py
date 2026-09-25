"""Opt-in feature revisions; frozen M6/M7 inputs remain unchanged."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from rtl_charge.schema import assert_deployable_features


ZERO_CLIMB_FEATURES = frozenset({"planned_rtl_climb_m", "nominal_climb_time_s"})


def consistent_nominal_f2(
    frame: pd.DataFrame,
    original_f2_features: Sequence[str],
    *,
    target_ground_speed_m_s: float = 10.0,
) -> pd.DataFrame:
    """Rebuild F2 for a constant *target* ground speed, not achieved RTL speed.

    The existing 250-flight processed table and frozen model artifacts are not
    overwritten. This returns only deployable F2 columns, without the two
    climb inputs that are zero throughout that dataset.
    """
    if not np.isfinite(target_ground_speed_m_s) or target_ground_speed_m_s <= 0:
        raise ValueError("target_ground_speed_m_s must be positive and finite")
    return consistent_speed_f2(
        frame,
        original_f2_features,
        np.full(len(frame), target_ground_speed_m_s, dtype=np.float64),
    )


def consistent_speed_f2(
    frame: pd.DataFrame,
    original_f2_features: Sequence[str],
    ground_speed_m_s: np.ndarray,
) -> pd.DataFrame:
    """Calculate all speed-dependent F2 inputs from one ground-speed estimate."""
    speed = np.asarray(ground_speed_m_s, dtype=np.float64)
    if speed.shape != (len(frame),) or not np.isfinite(speed).all() or np.any(speed <= 0):
        raise ValueError("ground_speed_m_s must have one positive finite value per row")

    selected = assert_deployable_features(
        name for name in original_f2_features if name not in ZERO_CLIMB_FEATURES
    )
    required = set(selected) | {
        "distance_home_m",
        "route_bearing_cos",
        "route_bearing_sin",
        "ekf_wind_north_mean_m_s",
        "ekf_wind_east_mean_m_s",
        "nominal_horizontal_time_s",
        "nominal_total_rtl_time_s",
        "current_mean_a",
    }
    absent = required.difference(frame.columns)
    if absent:
        raise ValueError(f"missing columns for revised F2: {sorted(absent)}")

    result = frame.loc[:, list(selected)].copy()
    distance = frame["distance_home_m"].to_numpy(dtype=np.float64)
    unit_north = frame["route_bearing_cos"].to_numpy(dtype=np.float64)
    unit_east = frame["route_bearing_sin"].to_numpy(dtype=np.float64)
    wind_north = frame["ekf_wind_north_mean_m_s"].to_numpy(dtype=np.float64)
    wind_east = frame["ekf_wind_east_mean_m_s"].to_numpy(dtype=np.float64)
    current = frame["current_mean_a"].to_numpy(dtype=np.float64)
    old_horizontal = frame["nominal_horizontal_time_s"].to_numpy(dtype=np.float64)
    old_total = frame["nominal_total_rtl_time_s"].to_numpy(dtype=np.float64)
    if not np.isfinite(np.column_stack((distance, unit_north, unit_east, wind_north,
                                        wind_east, current, old_horizontal, old_total))).all():
        raise ValueError("revised F2 inputs must be finite")
    if np.any(distance < 0) or np.any(old_total < old_horizontal):
        raise ValueError("distance and non-horizontal time must be non-negative")

    horizontal_time = distance / speed
    total_time = horizontal_time + (old_total - old_horizontal)
    along_wind = wind_north * unit_north + wind_east * unit_east
    crosswind = wind_north * unit_east - wind_east * unit_north
    air_north = speed * unit_north - wind_north
    air_east = speed * unit_east - wind_east
    airspeed_squared = np.square(air_north) + np.square(air_east)
    result["headwind_m_s"] = -along_wind
    result["crosswind_abs_m_s"] = np.abs(crosswind)
    result["nominal_horizontal_time_s"] = horizontal_time
    result["nominal_total_rtl_time_s"] = total_time
    result["required_air_velocity_north_m_s"] = air_north
    result["required_air_velocity_east_m_s"] = air_east
    result["required_airspeed_m_s"] = np.sqrt(airspeed_squared)
    result["required_airspeed_squared_m2_s2"] = airspeed_squared
    result["current_time_charge_baseline_mah"] = current * total_time * 1000.0 / 3600.0
    return result
