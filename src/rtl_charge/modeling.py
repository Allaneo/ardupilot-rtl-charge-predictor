"""Leakage-safe M6 model fitting and validation helpers.

The test split is deliberately never selected from the target column here.
M6 uses the fixed training split to fit candidates and the validation split to
select configurations. Final test evaluation belongs to M7.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rtl_charge.schema import assert_deployable_features


SEED = 20260921
TARGET = "rtl_charge_mah"
PX4_STYLE_MULTICOPTER_CRUISE_SPEED_M_S = 10.0
NN_HIDDEN_LAYERS = (64, 32)
NN_ARCHITECTURES = (
    (16,), (32,), (64,), (128,), (256,), (512,), (1024,),
    (32, 16), (64, 32), (128, 64), (128, 64, 32),
    (64, 32, 16), (128, 64, 32, 16),
    (16, 16, 16, 16, 16),
    (16, 16, 16, 16, 16, 16, 16, 16, 16, 16),
)
NN_MAX_EPOCHS = 600
NN_PATIENCE = 45


@dataclass(frozen=True)
class SplitData:
    """Only train and validation targets are materialized for M6."""

    train_x: pd.DataFrame
    train_y: np.ndarray
    validation_x: pd.DataFrame
    validation_y: np.ndarray
    train_run_ids: tuple[str, ...]
    validation_run_ids: tuple[str, ...]
    held_out_rows: int
    train_true_wind_x: pd.DataFrame | None = None
    validation_true_wind_x: pd.DataFrame | None = None
    train_aux: pd.DataFrame | None = None
    validation_aux: pd.DataFrame | None = None


@dataclass
class DenseRegressor:
    """NumPy MLP with normalization for inference."""

    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: float
    y_scale: float
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]

    def predict(self, x: np.ndarray | pd.DataFrame) -> np.ndarray:
        values = np.asarray(x, dtype=np.float64)
        values = (values - self.x_mean) / self.x_scale
        normalized, _ = _forward(values, self.weights, self.biases)
        return normalized.reshape(-1) * self.y_scale + self.y_mean


def load_train_validation(
    dataset_path: Path,
    feature_names: Iterable[str],
    target_name: str = TARGET,
    *,
    include_true_wind_oracle: bool = False,
    auxiliary_columns: Iterable[str] = (),
) -> SplitData:
    """Load only train/validation target values; never index test labels."""

    features = assert_deployable_features(feature_names)
    auxiliary = tuple(auxiliary_columns)
    frame_columns = set(pd.read_csv(dataset_path, nrows=0).columns)
    oracle_columns = {"true_wind_north_m_s", "true_wind_east_m_s"} if include_true_wind_oracle else set()
    required = {"split", "run_id", *features, *oracle_columns, *auxiliary, target_name}
    missing = required.difference(frame_columns)
    if missing:
        raise ValueError(f"dataset is missing required columns: {sorted(missing)}")

    input_columns = sorted(required.difference({target_name}))
    frame = pd.read_csv(dataset_path, usecols=input_columns)
    split = frame["split"].astype(str).to_numpy()
    train_mask = split == "train"
    validation_mask = split == "validation"
    held_out_rows = int(np.count_nonzero(split == "test"))
    if not train_mask.any() or not validation_mask.any():
        raise ValueError("dataset must contain both train and validation rows")

    train_frame = frame.loc[train_mask]
    validation_frame = frame.loc[validation_mask]
    train_x = train_frame.loc[:, features].astype(np.float64)
    validation_x = validation_frame.loc[:, features].astype(np.float64)
    train_true_wind_x = validation_true_wind_x = None
    if include_true_wind_oracle:
        train_true_wind_x = make_true_wind_oracle(train_frame, features)
        validation_true_wind_x = make_true_wind_oracle(validation_frame, features)
    train_aux = train_frame.loc[:, auxiliary].astype(np.float64) if auxiliary else None
    validation_aux = validation_frame.loc[:, auxiliary].astype(np.float64) if auxiliary else None
    targets = {"train": [], "validation": []}
    with dataset_path.open(newline="") as source:
        reader = csv.reader(source)
        header = next(reader)
        split_index = header.index("split")
        target_index = header.index(target_name)
        for row in reader:
            partition = row[split_index]
            if partition in targets:
                targets[partition].append(float(row[target_index]))
    train_y = np.asarray(targets["train"], dtype=np.float64)
    validation_y = np.asarray(targets["validation"], dtype=np.float64)
    if len(train_y) != len(train_frame) or len(validation_y) != len(validation_frame):
        raise ValueError("target stream and feature rows disagree on train/validation membership")
    for label, values in (
        ("training features", train_x.to_numpy()),
        ("validation features", validation_x.to_numpy()),
        ("training targets", train_y),
        ("validation targets", validation_y),
    ):
        if not np.isfinite(values).all():
            raise ValueError(f"{label} contain non-finite values")

    return SplitData(
        train_x=train_x,
        train_y=train_y,
        validation_x=validation_x,
        validation_y=validation_y,
        train_run_ids=tuple(train_frame["run_id"].astype(str)),
        validation_run_ids=tuple(validation_frame["run_id"].astype(str)),
        held_out_rows=held_out_rows,
        train_true_wind_x=train_true_wind_x,
        validation_true_wind_x=validation_true_wind_x,
        train_aux=train_aux,
        validation_aux=validation_aux,
    )


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return primary and safety-relevant residual summaries in mAh."""

    y = np.asarray(actual, dtype=np.float64).reshape(-1)
    estimate = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if y.shape != estimate.shape or y.size == 0:
        raise ValueError("actual and predicted must be non-empty arrays of equal shape")
    if not np.isfinite(y).all() or not np.isfinite(estimate).all():
        raise ValueError("actual and predicted must be finite")
    error = estimate - y
    underprediction = np.maximum(-error, 0.0)
    total_variation = float(np.sum(np.square(y - np.mean(y))))
    r_squared = (
        1.0 - float(np.sum(np.square(error))) / total_variation
        if total_variation > 0.0
        else 0.0
    )
    return {
        "mae_mah": float(np.mean(np.abs(error))),
        "median_absolute_error_mah": float(np.median(np.abs(error))),
        "p95_absolute_error_mah": float(np.quantile(np.abs(error), 0.95)),
        "rmse_mah": float(np.sqrt(np.mean(np.square(error)))),
        "wape": float(np.sum(np.abs(error)) / np.sum(np.abs(y))),
        "r_squared": r_squared,
        "mean_signed_error_mah": float(np.mean(error)),
        "mean_underprediction_mah": float(np.mean(underprediction)),
        "p95_underprediction_mah": float(np.quantile(underprediction, 0.95)),
    }


def px4_style_charge_baseline(frame: pd.DataFrame) -> np.ndarray:
    """Estimate charge from wind-neutral multicopter kinematic return time.

    PX4's multicopter time estimate uses configured multicopter cruise speed;
    its wind correction is for fixed-wing cruise. Vertical, descent and loiter
    terms are taken from the project's shared kinematic estimate, replacing
    only its wind-corrected horizontal time with distance / cruise speed.
    """

    non_horizontal_time = (
        frame["nominal_total_rtl_time_s"].to_numpy(dtype=np.float64)
        - frame["nominal_horizontal_time_s"].to_numpy(dtype=np.float64)
    )
    total_time = (
        frame["distance_home_m"].to_numpy(dtype=np.float64)
        / PX4_STYLE_MULTICOPTER_CRUISE_SPEED_M_S
        + non_horizontal_time
    )
    return frame["current_mean_a"].to_numpy(dtype=np.float64) * total_time * 1000.0 / 3600.0


def ridge_candidates() -> tuple[float, ...]:
    return (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def xgboost_candidates() -> tuple[dict[str, Any], ...]:
    """A small deterministic grid, bounded for the 39-flight validation set."""

    candidates: list[dict[str, Any]] = []
    for depth, rate, trees, subsample, columns, child_weight in (
        (2, 0.03, 100, 0.8, 0.8, 1),
        (2, 0.03, 300, 0.8, 1.0, 1),
        (3, 0.03, 100, 1.0, 0.8, 5),
        (3, 0.03, 300, 0.8, 0.8, 5),
        (2, 0.10, 100, 1.0, 1.0, 1),
        (2, 0.10, 300, 0.8, 0.8, 5),
        (3, 0.10, 100, 0.8, 1.0, 1),
        (3, 0.10, 300, 1.0, 0.8, 5),
        (1, 0.03, 50, 1.0, 1.0, 1),
        (1, 0.03, 100, 0.8, 1.0, 5),
        (1, 0.10, 50, 1.0, 0.8, 5),
        (1, 0.10, 100, 0.8, 0.8, 10),
    ):
        candidates.append(
            {
                "n_estimators": trees,
                "max_depth": depth,
                "learning_rate": rate,
                "subsample": subsample,
                "colsample_bytree": columns,
                "min_child_weight": child_weight,
                "reg_lambda": 1.0,
            }
        )
    return tuple(candidates)


def _forward(
    values: np.ndarray,
    weights: tuple[np.ndarray, ...],
    biases: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]]:
    activations = [values]
    preactivations = []
    for weight, bias in zip(weights[:-1], biases[:-1]):
        z = activations[-1] @ weight + bias
        preactivations.append(z)
        activations.append(np.maximum(z, 0.0))
    output = activations[-1] @ weights[-1] + biases[-1]
    return output, (tuple(activations), tuple(preactivations))


def _copy_parameters(
    weights: tuple[np.ndarray, ...], biases: tuple[np.ndarray, ...]
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    return tuple(value.copy() for value in weights), tuple(value.copy() for value in biases)


def fit_dense_regressor(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    *,
    loss: str,
    alpha: float,
    learning_rate: float = 0.001,
    hidden_layers: tuple[int, ...] = NN_HIDDEN_LAYERS,
    seed: int = SEED,
    max_epochs: int = NN_MAX_EPOCHS,
    patience: int = NN_PATIENCE,
    fixed_epochs: bool = False,
) -> tuple[DenseRegressor, int]:
    """Fit a configurable ReLU network with Adam and Huber/MAE loss."""

    if loss not in {"mae", "huber"}:
        raise ValueError("loss must be 'mae' or 'huber'")
    if not hidden_layers or any(
        not isinstance(width, int) or isinstance(width, bool) or width <= 0
        for width in hidden_layers
    ):
        raise ValueError("hidden_layers must contain positive integer widths")
    x = np.asarray(train_x, dtype=np.float64)
    y = np.asarray(train_y, dtype=np.float64).reshape(-1)
    x_val = np.asarray(validation_x, dtype=np.float64)
    y_val = np.asarray(validation_y, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x_val.ndim != 2 or x.shape[1] != x_val.shape[1]:
        raise ValueError("training and validation features must be 2D with matching columns")
    if len(x) != len(y) or len(x_val) != len(y_val) or not len(y):
        raise ValueError("feature and target row counts must match and training cannot be empty")

    x_mean = x.mean(axis=0)
    x_scale = x.std(axis=0)
    x_scale[x_scale < 1e-12] = 1.0
    x = (x - x_mean) / x_scale
    x_val = (x_val - x_mean) / x_scale
    y_mean = float(y.mean())
    y_scale = float(y.std())
    if y_scale < 1e-12:
        y_scale = 1.0
    y = ((y - y_mean) / y_scale).reshape(-1, 1)

    rng = np.random.default_rng(seed)
    dims = (x.shape[1], *hidden_layers, 1)
    weights = tuple(
        rng.normal(0.0, np.sqrt(2.0 / dims[index]), (dims[index], dims[index + 1]))
        for index in range(len(dims) - 1)
    )
    biases = tuple(
        np.zeros((1, dims[index + 1]), dtype=np.float64)
        for index in range(len(dims) - 1)
    )
    first_m_w = tuple(np.zeros_like(value) for value in weights)
    first_v_w = tuple(np.zeros_like(value) for value in weights)
    first_m_b = tuple(np.zeros_like(value) for value in biases)
    first_v_b = tuple(np.zeros_like(value) for value in biases)

    best_mae = float("inf")
    best_parameters = _copy_parameters(weights, biases)
    best_epoch = 0
    stale_epochs = 0
    update = 0
    batch_size = min(32, len(x))
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8

    for epoch in range(1, max_epochs + 1):
        order = rng.permutation(len(x))
        for start in range(0, len(x), batch_size):
            indices = order[start : start + batch_size]
            batch_x = x[indices]
            batch_y = y[indices]
            output, cache = _forward(batch_x, weights, biases)
            difference = output - batch_y
            if loss == "mae":
                delta = np.sign(difference) / len(batch_x)
            else:
                delta = np.clip(difference, -1.0, 1.0) / len(batch_x)

            activations, preactivations = cache
            gradients_w = [np.empty_like(weight) for weight in weights]
            gradients_b = [np.empty_like(bias) for bias in biases]
            for index in range(len(weights) - 1, -1, -1):
                gradients_w[index] = activations[index].T @ delta + alpha * weights[index]
                gradients_b[index] = delta.sum(axis=0, keepdims=True)
                if index:
                    delta = (delta @ weights[index].T) * (preactivations[index - 1] > 0.0)
            update += 1

            weights_next = []
            biases_next = []
            m_w_next, v_w_next = [], []
            m_b_next, v_b_next = [], []
            for index, gradient in enumerate(gradients_w):
                m = beta1 * first_m_w[index] + (1.0 - beta1) * gradient
                v = beta2 * first_v_w[index] + (1.0 - beta2) * np.square(gradient)
                mhat = m / (1.0 - beta1**update)
                vhat = v / (1.0 - beta2**update)
                weights_next.append(weights[index] - learning_rate * mhat / (np.sqrt(vhat) + epsilon))
                m_w_next.append(m)
                v_w_next.append(v)
            for index, gradient in enumerate(gradients_b):
                m = beta1 * first_m_b[index] + (1.0 - beta1) * gradient
                v = beta2 * first_v_b[index] + (1.0 - beta2) * np.square(gradient)
                mhat = m / (1.0 - beta1**update)
                vhat = v / (1.0 - beta2**update)
                biases_next.append(biases[index] - learning_rate * mhat / (np.sqrt(vhat) + epsilon))
                m_b_next.append(m)
                v_b_next.append(v)
            weights = tuple(weights_next)
            biases = tuple(biases_next)
            first_m_w, first_v_w = tuple(m_w_next), tuple(v_w_next)
            first_m_b, first_v_b = tuple(m_b_next), tuple(v_b_next)

        if fixed_epochs:
            best_parameters = _copy_parameters(weights, biases)
            best_epoch = epoch
            continue

        predicted_normalized, _ = _forward(x_val, weights, biases)
        predicted = predicted_normalized.reshape(-1) * y_scale + y_mean
        validation_mae = float(np.mean(np.abs(predicted - y_val)))
        if validation_mae < best_mae - 1e-10:
            best_mae = validation_mae
            best_parameters = _copy_parameters(weights, biases)
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    best_weights, best_biases = best_parameters
    model = DenseRegressor(
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
        weights=best_weights,
        biases=best_biases,
    )
    return model, best_epoch


def neural_candidates() -> tuple[dict[str, Any], ...]:
    return tuple(
        {"loss": loss, "alpha": alpha, "learning_rate": 0.001}
        for loss, alpha in product(("mae", "huber"), (1e-3, 1e-2, 1e-1))
    )


def fit_ridge(alpha: float) -> Any:
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def fit_xgboost(parameters: dict[str, Any], *, seed: int = SEED) -> Any:
    from xgboost import XGBRegressor

    candidate_parameters = dict(parameters)
    random_state = int(candidate_parameters.pop("random_state", seed))
    return XGBRegressor(
        objective="reg:squarederror",
        eval_metric="mae",
        tree_method="hist",
        n_jobs=1,
        random_state=random_state,
        verbosity=0,
        **candidate_parameters,
    )


def make_true_wind_oracle(frame: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    """Replace only estimated wind features with privileged true-wind values."""

    names = assert_deployable_features(feature_names)
    oracle = frame.loc[:, names].copy()
    replacements = {
        "ekf_wind_north_m_s": "true_wind_north_m_s",
        "ekf_wind_north_mean_m_s": "true_wind_north_m_s",
        "ekf_wind_north_std_m_s": None,
        "ekf_wind_east_m_s": "true_wind_east_m_s",
        "ekf_wind_east_mean_m_s": "true_wind_east_m_s",
        "ekf_wind_east_std_m_s": None,
    }
    for feature, truth_column in replacements.items():
        if feature not in oracle.columns:
            continue
        if truth_column is None:
            oracle.loc[:, feature] = 0.0
        else:
            if truth_column not in frame:
                raise ValueError(f"oracle needs privileged column {truth_column}")
            oracle.loc[:, feature] = frame.loc[:, truth_column].to_numpy(dtype=np.float64)
    physics_columns = {
        "route_bearing_cos",
        "route_bearing_sin",
        "distance_home_m",
        "nominal_horizontal_time_s",
        "nominal_total_rtl_time_s",
        "current_mean_a",
        "headwind_m_s",
        "crosswind_abs_m_s",
        "required_air_velocity_north_m_s",
        "required_air_velocity_east_m_s",
        "required_airspeed_m_s",
        "required_airspeed_squared_m2_s2",
        "current_time_charge_baseline_mah",
    }
    if physics_columns.issubset(oracle.columns):
        unit_north = oracle["route_bearing_cos"].to_numpy(dtype=np.float64)
        unit_east = oracle["route_bearing_sin"].to_numpy(dtype=np.float64)
        wind_north = oracle["ekf_wind_north_mean_m_s"].to_numpy(dtype=np.float64)
        wind_east = oracle["ekf_wind_east_mean_m_s"].to_numpy(dtype=np.float64)
        along_wind = wind_north * unit_north + wind_east * unit_east
        crosswind = wind_north * unit_east - wind_east * unit_north
        non_horizontal_time = (
            oracle["nominal_total_rtl_time_s"].to_numpy(dtype=np.float64)
            - oracle["nominal_horizontal_time_s"].to_numpy(dtype=np.float64)
        )
        horizontal_time = oracle["distance_home_m"].to_numpy(dtype=np.float64) / np.maximum(
            1.0, PX4_STYLE_MULTICOPTER_CRUISE_SPEED_M_S + along_wind
        )
        oracle.loc[:, "headwind_m_s"] = -along_wind
        oracle.loc[:, "crosswind_abs_m_s"] = np.abs(crosswind)
        oracle.loc[:, "nominal_horizontal_time_s"] = horizontal_time
        oracle.loc[:, "nominal_total_rtl_time_s"] = horizontal_time + non_horizontal_time
        desired_north = PX4_STYLE_MULTICOPTER_CRUISE_SPEED_M_S * unit_north
        desired_east = PX4_STYLE_MULTICOPTER_CRUISE_SPEED_M_S * unit_east
        air_north = desired_north - wind_north
        air_east = desired_east - wind_east
        airspeed_squared = np.square(air_north) + np.square(air_east)
        oracle.loc[:, "required_air_velocity_north_m_s"] = air_north
        oracle.loc[:, "required_air_velocity_east_m_s"] = air_east
        oracle.loc[:, "required_airspeed_squared_m2_s2"] = airspeed_squared
        oracle.loc[:, "required_airspeed_m_s"] = np.sqrt(airspeed_squared)
        oracle.loc[:, "current_time_charge_baseline_mah"] = (
            oracle["current_mean_a"].to_numpy(dtype=np.float64)
            * oracle["nominal_total_rtl_time_s"].to_numpy(dtype=np.float64)
            * 1000.0
            / 3600.0
        )
    return oracle


def candidate_row(
    *,
    model_name: str,
    feature_set: str,
    feature_count: int,
    params: dict[str, Any],
    metrics: dict[str, float],
) -> dict[str, Any]:
    return {
        "model": model_name,
        "feature_set": feature_set,
        "feature_count": feature_count,
        "hyperparameters": params,
        **metrics,
    }
