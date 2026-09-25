#!/usr/bin/env python3
"""Exploratory train/validation-only test of self-consistent RTL speed features.

Cruise speed is supervised by post-RTL logs during training only. Training-row
speed estimates are out-of-fold, so charge fitting never sees a speed estimate
from a model trained on that row's post-RTL speed. No test rows are read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtl_charge.modeling import fit_dense_regressor, fit_ridge, load_train_validation, regression_metrics  # noqa: E402
from rtl_charge.schema import load_feature_sets  # noqa: E402


DATASET = ROOT / "data/processed/pipeline_dataset.csv"
SPEED = ROOT / "reports/m9/rtl_speed_train_validation.csv"
FEATURES = ROOT / "config/feature_sets.yaml"
OUTPUT = ROOT / "reports/m9"
SPEED_INPUTS = (
    "headwind_m_s",
    "crosswind_abs_m_s",
    "required_airspeed_m_s",
    "current_mean_a",
    "throttle_mean",
)
SPEED_TARGET = "middle_route_ground_speed_m_s"
SPEED_SETPOINT = 10.0


def speed_model() -> RandomForestRegressor:
    """Fixed small tree ensemble; no validation-based parameter search."""
    return RandomForestRegressor(
        n_estimators=150,
        max_depth=4,
        min_samples_leaf=6,
        max_features=1.0,
        random_state=20260923,
        n_jobs=1,
    )


def make_consistent_features(
    frame: pd.DataFrame, ground_speed: np.ndarray, base_inputs: tuple[str, ...]
) -> pd.DataFrame:
    """Construct cruise time and airspeed from the same predicted ground speed."""
    speed = np.asarray(ground_speed, dtype=np.float64)
    if speed.shape != (len(frame),) or not np.isfinite(speed).all() or np.any(speed <= 0):
        raise ValueError("one positive finite ground speed is required per flight")
    voltage = frame["voltage_mean_v"].to_numpy(dtype=np.float64)
    if np.any(voltage <= 0):
        raise ValueError("battery voltage must be positive")
    north = frame["route_bearing_cos"].to_numpy(dtype=np.float64)
    east = frame["route_bearing_sin"].to_numpy(dtype=np.float64)
    wind_north = frame["ekf_wind_north_m_s"].to_numpy(dtype=np.float64)
    wind_east = frame["ekf_wind_east_m_s"].to_numpy(dtype=np.float64)
    airspeed = np.hypot(speed * north - wind_north, speed * east - wind_east)
    horizontal_time = frame["distance_home_m"].to_numpy(dtype=np.float64) / speed
    nonhorizontal_time = (
        frame["nominal_total_rtl_time_s"].to_numpy(dtype=np.float64)
        - frame["nominal_horizontal_time_s"].to_numpy(dtype=np.float64)
    )
    total_time = horizontal_time + nonhorizontal_time
    f3 = frame.loc[:, [name for name in base_inputs if name != "planned_rtl_climb_m"]].copy()
    f3["headwind_m_s"] = frame["headwind_m_s"]
    f3["crosswind_abs_m_s"] = frame["crosswind_abs_m_s"]
    f3["predicted_cruise_ground_speed_m_s"] = speed
    f3["predicted_horizontal_time_s"] = horizontal_time
    f3["predicted_total_rtl_time_s"] = total_time
    f3["predicted_cruise_airspeed_m_s"] = airspeed
    f3["predicted_current_time_charge_mah"] = (
        frame["current_mean_a"].to_numpy(dtype=np.float64) * total_time * 1000.0 / 3600.0
    )
    f3["cruise_drag_charge_proxy"] = horizontal_time * airspeed**3 / voltage
    if not np.isfinite(f3.to_numpy(dtype=np.float64)).all():
        raise ValueError("non-finite F3 features")
    return f3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--speed", type=Path, default=SPEED)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    feature_sets = load_feature_sets(FEATURES)
    base_inputs = feature_sets["f1_vehicle_symptoms"]
    f2_inputs = feature_sets["f2_physics_informed"]
    split = load_train_validation(args.dataset, f2_inputs)
    train = split.train_x.copy()
    val = split.validation_x.copy()
    train.index = pd.Index(split.train_run_ids, name="run_id")
    val.index = pd.Index(split.validation_run_ids, name="run_id")
    speed = pd.read_csv(args.speed).set_index("run_id", verify_integrity=True)
    expected = set(train.index) | set(val.index)
    if set(speed.index) != expected or set(speed.split) != {"train", "validation"}:
        raise ValueError("speed diagnostics must contain exactly all train/validation flights")
    if not (speed.loc[train.index, "split"] == "train").all():
        raise ValueError("training speed rows are misaligned")
    if not (speed.loc[val.index, "split"] == "validation").all():
        raise ValueError("validation speed rows are misaligned")
    train_speed = speed.loc[train.index, SPEED_TARGET].to_numpy(dtype=np.float64)
    val_speed = speed.loc[val.index, SPEED_TARGET].to_numpy(dtype=np.float64)
    train_speed_x = train.loc[:, list(SPEED_INPUTS)].to_numpy(dtype=np.float64)
    val_speed_x = val.loc[:, list(SPEED_INPUTS)].to_numpy(dtype=np.float64)
    folds = KFold(n_splits=5, shuffle=True, random_state=20260923)
    oof_speed = cross_val_predict(speed_model(), train_speed_x, train_speed, cv=folds, n_jobs=1)
    model = speed_model().fit(train_speed_x, train_speed)
    val_speed_pred = model.predict(val_speed_x)
    oof_speed = np.clip(oof_speed, 1.0, SPEED_SETPOINT)
    val_speed_pred = np.clip(val_speed_pred, 1.0, SPEED_SETPOINT)
    old_speed = np.maximum(1.0, SPEED_SETPOINT - val["headwind_m_s"].to_numpy(dtype=np.float64))

    def speed_errors(prediction: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
        return {
            "n": int(mask.sum()),
            "mae_m_s": float(np.mean(np.abs(prediction[mask] - val_speed[mask]))),
            "mean_error_m_s": float(np.mean(prediction[mask] - val_speed[mask])),
        }

    all_rows = np.ones(len(val), dtype=bool)
    high_headwind = val["headwind_m_s"].to_numpy(dtype=np.float64) >= 2.0
    speed_summary = {
        "training_flights": len(train),
        "validation_flights": len(val),
        "crossfit_train_speed_mae_m_s": float(np.mean(np.abs(oof_speed - train_speed))),
        "all_validation": {
            "constant_10_m_s": speed_errors(np.full(len(val), SPEED_SETPOINT), all_rows),
            "old_wind_added_formula": speed_errors(old_speed, all_rows),
            "learned_speed": speed_errors(val_speed_pred, all_rows),
        },
        "headwind_at_least_2_m_s": {
            "constant_10_m_s": speed_errors(np.full(len(val), SPEED_SETPOINT), high_headwind),
            "old_wind_added_formula": speed_errors(old_speed, high_headwind),
            "learned_speed": speed_errors(val_speed_pred, high_headwind),
        } if high_headwind.any() else None,
        "validation_lean_near_30deg_count": int((speed.loc[val.index, "lean_near_30deg_fraction"] > 0.5).sum()),
    }

    train_f3 = make_consistent_features(train, oof_speed, base_inputs)
    val_f3 = make_consistent_features(val, val_speed_pred, base_inputs)
    train_y = split.train_y
    val_y = split.validation_y
    predictions = pd.DataFrame({
        "run_id": split.validation_run_ids,
        "actual_rtl_charge_mah": val_y,
        "actual_middle_route_ground_speed_m_s": val_speed,
        "predicted_middle_route_ground_speed_m_s": val_speed_pred,
        "headwind_m_s": val["headwind_m_s"].to_numpy(dtype=np.float64),
    })
    results = []
    for variant, use_cubic in (("without_cubic", False), ("with_cubic", True)):
        train_features = train_f3 if use_cubic else train_f3.drop(columns="cruise_drag_charge_proxy")
        val_features = val_f3 if use_cubic else val_f3.drop(columns="cruise_drag_charge_proxy")
        ridge = fit_ridge(alpha=1.0).fit(train_features, train_y)
        ridge_pred = ridge.predict(val_features)
        predictions[f"f3_{variant}_ridge_predicted_mah"] = ridge_pred
        results.append({"model": f"ridge_f3_{variant}", **regression_metrics(val_y, ridge_pred)})

        neural, epoch = fit_dense_regressor(
            train_features.to_numpy(dtype=np.float64), train_y,
            val_features.to_numpy(dtype=np.float64), val_y,
            loss="huber", alpha=0.001, learning_rate=0.001,
            seed=20260924, max_epochs=525, fixed_epochs=True,
        )
        neural_pred = neural.predict(val_features)
        predictions[f"f3_{variant}_neural_predicted_mah"] = neural_pred
        results.append({
            "model": f"neural_f3_{variant}", "fixed_epochs": epoch,
            **regression_metrics(val_y, neural_pred),
        })

    saved = pd.read_csv(ROOT / "reports/m6/selected_validation_predictions.csv")
    for name, label in (
        ("ridge", "ridge_f2_frozen"),
        ("neural_network", "neural_f2_frozen"),
    ):
        candidate = saved.loc[
            (saved.model == name) & (saved.feature_set == "f2_physics_informed")
        ].set_index("run_id")
        if set(candidate.index) != set(val.index):
            raise ValueError(f"frozen validation predictions do not match: {name}")
        values = candidate.loc[val.index, "predicted_rtl_charge_mah"].to_numpy(dtype=np.float64)
        predictions[f"{label}_predicted_mah"] = values
        results.append({"model": label, **regression_metrics(val_y, values)})

    comparisons = []
    rng = np.random.default_rng(20260923)
    pairs = (
        ("f3_with_cubic_ridge", "f3_without_cubic_ridge"),
        ("f3_with_cubic_ridge", "ridge_f2_frozen"),
        ("f3_with_cubic_neural", "f3_without_cubic_neural"),
        ("f3_with_cubic_neural", "neural_f2_frozen"),
    )
    indices = rng.integers(0, len(val_y), size=(10_000, len(val_y)))
    for candidate, reference in pairs:
        candidate_error = np.abs(predictions[f"{candidate}_predicted_mah"].to_numpy() - val_y)
        reference_error = np.abs(predictions[f"{reference}_predicted_mah"].to_numpy() - val_y)
        difference = candidate_error - reference_error
        resampled = difference[indices].mean(axis=1)
        comparisons.append({
            "candidate": candidate,
            "reference": reference,
            "mean_mae_difference_mah": float(difference.mean()),
            "bootstrap_95_low_mah": float(np.quantile(resampled, 0.025)),
            "bootstrap_95_high_mah": float(np.quantile(resampled, 0.975)),
            "n_validation_flights": len(val_y),
        })

    condition = []
    for label, mask in (("headwind_below_2", ~high_headwind), ("headwind_at_least_2", high_headwind)):
        for model in ("f3_with_cubic_ridge", "f3_without_cubic_ridge", "ridge_f2_frozen"):
            predicted = predictions[f"{model}_predicted_mah"].to_numpy(dtype=np.float64)
            condition.append({
                "condition": label,
                "model": model,
                "n_validation_flights": int(mask.sum()),
                "mae_mah": float(np.mean(np.abs(predicted[mask] - val_y[mask]))),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output_dir / "validation_metrics.csv", index=False)
    predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(comparisons).to_csv(args.output_dir / "validation_paired_differences.csv", index=False)
    pd.DataFrame(condition).to_csv(args.output_dir / "validation_headwind_breakdown.csv", index=False)
    (args.output_dir / "speed_summary.json").write_text(json.dumps(speed_summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"speed": speed_summary, "charge_models": results}, indent=2))


if __name__ == "__main__":
    main()
