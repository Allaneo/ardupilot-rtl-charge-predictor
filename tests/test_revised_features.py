from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rtl_charge.revised_features import consistent_nominal_f2, consistent_speed_f2
from rtl_charge.schema import load_feature_sets


F2 = load_feature_sets(Path("config/feature_sets.yaml"))["f2_physics_informed"]


def test_revised_f2_uses_one_ground_speed_assumption_and_drops_zero_climb_inputs():
    source = pd.DataFrame({name: [0.0] for name in F2})
    source["distance_home_m"] = 700.0
    source["route_bearing_cos"] = 1.0
    source["route_bearing_sin"] = 0.0
    source["ekf_wind_north_mean_m_s"] = -3.0
    source["ekf_wind_east_mean_m_s"] = 4.0
    source["current_mean_a"] = 36.0
    source["nominal_horizontal_time_s"] = 100.0
    source["nominal_total_rtl_time_s"] = 125.0

    revised = consistent_nominal_f2(source, F2)

    assert "planned_rtl_climb_m" not in revised
    assert "nominal_climb_time_s" not in revised
    assert len(revised.columns) == len(F2) - 2
    assert revised.loc[0, "headwind_m_s"] == 3.0
    assert revised.loc[0, "crosswind_abs_m_s"] == 4.0
    assert revised.loc[0, "nominal_horizontal_time_s"] == 70.0
    assert revised.loc[0, "nominal_total_rtl_time_s"] == 95.0
    assert revised.loc[0, "required_airspeed_m_s"] == pytest.approx(np.hypot(13.0, 4.0))
    assert revised.loc[0, "current_time_charge_baseline_mah"] == pytest.approx(950.0)
    assert source.loc[0, "nominal_horizontal_time_s"] == 100.0


def test_revised_f2_rejects_invalid_target_speed():
    with pytest.raises(ValueError, match="target_ground_speed_m_s"):
        consistent_nominal_f2(pd.DataFrame(), F2, target_ground_speed_m_s=0.0)


def test_revised_f2_updates_time_and_airspeed_together_for_predicted_speed():
    source = pd.DataFrame({name: [0.0] for name in F2})
    source["distance_home_m"] = 700.0
    source["route_bearing_cos"] = 1.0
    source["ekf_wind_north_mean_m_s"] = -3.0
    source["nominal_horizontal_time_s"] = 100.0
    source["nominal_total_rtl_time_s"] = 125.0
    source["current_mean_a"] = 36.0

    revised = consistent_speed_f2(source, F2, np.array([7.0]))

    assert revised.loc[0, "nominal_horizontal_time_s"] == 100.0
    assert revised.loc[0, "nominal_total_rtl_time_s"] == 125.0
    assert revised.loc[0, "required_airspeed_m_s"] == 10.0
    assert revised.loc[0, "current_time_charge_baseline_mah"] == pytest.approx(1250.0)
    assert "nominal_climb_time_s" not in revised
