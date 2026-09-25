import pytest

from pathlib import Path

from rtl_charge.schema import assert_deployable_features, load_feature_sets


def test_deployable_features_accept_decision_time_columns() -> None:
    assert assert_deployable_features(["distance_home_m", "current_mean_a"]) == (
        "distance_home_m",
        "current_mean_a",
    )


@pytest.mark.parametrize(
    "forbidden_column",
    [
        "payload_mass_kg",
        "true_wind_north_m_s",
        "sitl_speedup",
        "rtl_charge_mah",
        "rtl_charge_integrated_mah",
    ],
)
def test_deployable_features_reject_truth_and_targets(forbidden_column: str) -> None:
    with pytest.raises(ValueError, match="non-deployable"):
        assert_deployable_features(["distance_home_m", forbidden_column])


def test_tracked_feature_sets_resolve_without_leakage() -> None:
    feature_sets = load_feature_sets(Path("config/feature_sets.yaml"))

    assert set(feature_sets) == {
        "f0_route_and_wind",
        "f1_vehicle_symptoms",
        "f2_physics_informed",
    }
    assert set(feature_sets["f0_route_and_wind"]).issubset(feature_sets["f1_vehicle_symptoms"])
    assert set(feature_sets["f1_vehicle_symptoms"]).issubset(feature_sets["f2_physics_informed"])
