"""Self-consistent cruise speed feature checks."""

import numpy as np
import pandas as pd
import pytest

from scripts.validate_controller_aware_features import make_consistent_features


def test_cruise_time_and_airspeed_share_ground_speed():
    frame = pd.DataFrame({
        "distance_home_m": [700.0],
        "route_bearing_cos": [1.0],
        "route_bearing_sin": [0.0],
        "ekf_wind_north_m_s": [-3.0],
        "ekf_wind_east_m_s": [0.0],
        "headwind_m_s": [3.0],
        "crosswind_abs_m_s": [0.0],
        "voltage_mean_v": [14.0],
        "current_mean_a": [28.0],
        "nominal_horizontal_time_s": [100.0],
        "nominal_total_rtl_time_s": [160.0],
    })
    frame["planned_rtl_climb_m"] = 0.0
    features = make_consistent_features(
        frame, np.array([8.0]), ("distance_home_m", "planned_rtl_climb_m")
    )
    assert features.loc[0, "predicted_horizontal_time_s"] == pytest.approx(87.5)
    assert features.loc[0, "predicted_cruise_airspeed_m_s"] == pytest.approx(11.0)
    assert features.loc[0, "predicted_total_rtl_time_s"] == pytest.approx(147.5)
    assert features.loc[0, "cruise_drag_charge_proxy"] == pytest.approx(87.5 * 11**3 / 14)
