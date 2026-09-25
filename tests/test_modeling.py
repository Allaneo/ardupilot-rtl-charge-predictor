import numpy as np
import pandas as pd
import pytest

from rtl_charge.modeling import (
    NN_ARCHITECTURES,
    fit_dense_regressor,
    load_train_validation,
    make_true_wind_oracle,
    neural_candidates,
    px4_style_charge_baseline,
    regression_metrics,
    xgboost_candidates,
)
from rtl_charge.schema import assert_deployable_features


def test_test_targets_are_not_converted_or_returned(tmp_path):
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame(
        {
            "split": ["train", "validation", "test"],
            "run_id": ["tr", "va", "te"],
            "feature": [1.0, 2.0, 3.0],
            "rtl_charge_mah": [100.0, 120.0, "DO_NOT_READ_TEST_LABEL"],
        }
    ).to_csv(dataset, index=False)

    split = load_train_validation(dataset, ["feature"])

    assert split.train_y.tolist() == [100.0]
    assert split.validation_y.tolist() == [120.0]
    assert split.held_out_rows == 1
    assert not hasattr(split, "test_y")


def test_feature_allowlist_rejects_target_and_privileged_truth():
    with pytest.raises(ValueError, match="non-deployable"):
        assert_deployable_features(["distance_home_m", "rtl_charge_mah"])
    with pytest.raises(ValueError, match="non-deployable"):
        assert_deployable_features(["true_wind_north_m_s"])


def test_metrics_report_underprediction_in_charge_units():
    metrics = regression_metrics(np.array([100.0, 200.0]), np.array([90.0, 220.0]))
    assert metrics["mae_mah"] == 15.0
    assert metrics["mean_signed_error_mah"] == 5.0
    assert metrics["mean_underprediction_mah"] == 5.0
    assert metrics["p95_underprediction_mah"] == pytest.approx(9.5)


def test_px4_style_baseline_replaces_wind_adjusted_horizontal_time():
    inputs = pd.DataFrame(
        {
            "distance_home_m": [100.0],
            "nominal_total_rtl_time_s": [30.0],
            "nominal_horizontal_time_s": [10.0],
            "current_mean_a": [36.0],
        }
    )
    # 100 m / 10 m/s plus 20 s of shared climb, descent and loiter time.
    assert px4_style_charge_baseline(inputs).tolist() == [300.0]


def test_true_wind_oracle_only_replaces_explicit_wind_fields():
    features = (
        "ekf_wind_north_mean_m_s",
        "ekf_wind_north_std_m_s",
        "ekf_wind_east_mean_m_s",
        "ekf_wind_east_std_m_s",
        "distance_home_m",
    )
    values = pd.DataFrame(
        {
            "ekf_wind_north_mean_m_s": [1.0],
            "ekf_wind_north_std_m_s": [0.2],
            "ekf_wind_east_mean_m_s": [2.0],
            "ekf_wind_east_std_m_s": [0.3],
            "distance_home_m": [500.0],
            "true_wind_north_m_s": [-3.0],
            "true_wind_east_m_s": [4.0],
        }
    )
    oracle = make_true_wind_oracle(values, features)
    assert oracle.loc[0, "ekf_wind_north_mean_m_s"] == -3.0
    assert oracle.loc[0, "ekf_wind_north_std_m_s"] == 0.0
    assert oracle.loc[0, "ekf_wind_east_mean_m_s"] == 4.0
    assert oracle.loc[0, "ekf_wind_east_std_m_s"] == 0.0
    assert oracle.loc[0, "distance_home_m"] == 500.0


def test_true_wind_oracle_recomputes_all_wind_derived_physics_features():
    features = [
        "route_bearing_cos",
        "route_bearing_sin",
        "distance_home_m",
        "nominal_horizontal_time_s",
        "nominal_total_rtl_time_s",
        "current_mean_a",
        "ekf_wind_north_mean_m_s",
        "ekf_wind_north_std_m_s",
        "ekf_wind_east_mean_m_s",
        "ekf_wind_east_std_m_s",
        "headwind_m_s",
        "crosswind_abs_m_s",
        "required_air_velocity_north_m_s",
        "required_air_velocity_east_m_s",
        "required_airspeed_m_s",
        "required_airspeed_squared_m2_s2",
        "current_time_charge_baseline_mah",
    ]
    source = pd.DataFrame(
        {
            "route_bearing_cos": [1.0],
            "route_bearing_sin": [0.0],
            "distance_home_m": [1000.0],
            "nominal_horizontal_time_s": [100.0],
            "nominal_total_rtl_time_s": [125.0],
            "current_mean_a": [36.0],
            "ekf_wind_north_mean_m_s": [0.0],
            "ekf_wind_north_std_m_s": [0.1],
            "ekf_wind_east_mean_m_s": [0.0],
            "ekf_wind_east_std_m_s": [0.2],
            "headwind_m_s": [0.0],
            "crosswind_abs_m_s": [0.0],
            "required_air_velocity_north_m_s": [10.0],
            "required_air_velocity_east_m_s": [0.0],
            "required_airspeed_m_s": [10.0],
            "required_airspeed_squared_m2_s2": [100.0],
            "current_time_charge_baseline_mah": [1250.0],
            "true_wind_north_m_s": [-2.0],
            "true_wind_east_m_s": [3.0],
        }
    )

    oracle = make_true_wind_oracle(source, features)

    assert oracle.loc[0, "headwind_m_s"] == 2.0
    assert oracle.loc[0, "crosswind_abs_m_s"] == 3.0
    assert oracle.loc[0, "nominal_horizontal_time_s"] == pytest.approx(125.0)
    assert oracle.loc[0, "nominal_total_rtl_time_s"] == pytest.approx(150.0)
    assert oracle.loc[0, "required_air_velocity_north_m_s"] == 12.0
    assert oracle.loc[0, "required_air_velocity_east_m_s"] == -3.0
    assert oracle.loc[0, "current_time_charge_baseline_mah"] == pytest.approx(1500.0)


def test_numpy_neural_network_is_reproducible_and_predicts_finite_values():
    train_x = np.arange(48, dtype=float).reshape(24, 2) / 10.0
    train_y = 3.0 * train_x[:, 0] - 2.0 * train_x[:, 1] + 100.0
    validation_x = train_x[:5]
    validation_y = train_y[:5]
    common = {
        "loss": "huber",
        "alpha": 1e-4,
        "seed": 42,
        "max_epochs": 6,
        "patience": 3,
    }

    model_a, epoch_a = fit_dense_regressor(
        train_x, train_y, validation_x, validation_y, **common
    )
    model_b, epoch_b = fit_dense_regressor(
        train_x, train_y, validation_x, validation_y, **common
    )

    assert epoch_a == epoch_b
    assert np.allclose(model_a.predict(validation_x), model_b.predict(validation_x))
    assert np.isfinite(model_a.predict(validation_x)).all()


@pytest.mark.parametrize("architecture", [(16,), (64, 32, 16), (128, 64, 32, 16), (16,) * 10])
def test_neural_network_supports_variable_depth(architecture):
    train_x = np.arange(48, dtype=float).reshape(24, 2) / 10.0
    train_y = 3.0 * train_x[:, 0] - 2.0 * train_x[:, 1] + 100.0
    model, epochs = fit_dense_regressor(
        train_x, train_y, train_x[:5], train_y[:5],
        loss="huber", alpha=1e-4, hidden_layers=architecture,
        seed=42, max_epochs=3, fixed_epochs=True,
    )
    assert epochs == 3
    assert tuple(weight.shape[1] for weight in model.weights[:-1]) == architecture
    assert np.isfinite(model.predict(train_x[:5])).all()


def test_neural_architecture_catalog_includes_agreed_shapes():
    assert len(NN_ARCHITECTURES) == 15
    assert (64, 32, 16) in NN_ARCHITECTURES
    assert (128, 64, 32, 16) in NN_ARCHITECTURES
    assert (1024,) in NN_ARCHITECTURES
    assert (16,) * 10 in NN_ARCHITECTURES


def test_candidate_search_stays_bounded_but_includes_regularized_models():
    nn = neural_candidates()
    trees = xgboost_candidates()
    assert len(nn) == 6
    assert {candidate["alpha"] for candidate in nn} == {1e-3, 1e-2, 1e-1}
    assert len(trees) == 12
    assert min(candidate["max_depth"] for candidate in trees) == 1
    assert min(candidate["n_estimators"] for candidate in trees) == 50
