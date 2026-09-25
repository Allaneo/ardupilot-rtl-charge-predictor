#!/usr/bin/env python3
"""Validation-only test of predicted RTL speed and time in F0/F1."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtl_charge.modeling import fit_dense_regressor, fit_ridge, load_train_validation, regression_metrics  # noqa: E402
from rtl_charge.schema import load_feature_sets  # noqa: E402
from validate_controller_aware_features import SPEED_INPUTS, SPEED_SETPOINT, speed_model  # noqa: E402


OUT = ROOT / "reports/m12_speed_f0_f1"
F0_SPEED_INPUTS = (
    "headwind_m_s",          # derived from F0 route and EKF wind
    "crosswind_abs_m_s",     # derived from F0 route and EKF wind
    "required_airspeed_m_s", # derived from F0 route and EKF wind at the 10 m/s target
    "voltage_mean_v",        # directly in F0
)
SPEED_INPUTS_BY_SET = {
    "f0_route_and_wind": F0_SPEED_INPUTS,
    "f1_vehicle_symptoms": SPEED_INPUTS,
}


def augmented_inputs(
    base: pd.DataFrame,
    speed_m_s: np.ndarray,
    *,
    include_time: bool,
) -> pd.DataFrame:
    result = base.copy()
    result["predicted_cruise_ground_speed_m_s"] = speed_m_s
    if include_time:
        result["predicted_horizontal_time_s"] = result["distance_home_m"] / speed_m_s
    return result


def charge_metrics(actual: np.ndarray, predicted: np.ndarray, headwind_mask: np.ndarray) -> dict[str, float]:
    summary = regression_metrics(actual, predicted)
    return {
        "mae_mah": summary["mae_mah"],
        "rmse_mah": summary["rmse_mah"],
        "max_underprediction_mah": float(np.max(actual - predicted)),
        "high_headwind_mae_mah": float(np.mean(np.abs(actual[headwind_mask] - predicted[headwind_mask]))),
    }


def main() -> None:
    sets = load_feature_sets(ROOT / "config/feature_sets.yaml")
    split = load_train_validation(ROOT / "data/processed/pipeline_dataset.csv", sets["f2_physics_informed"])
    train = split.train_x.copy()
    validation = split.validation_x.copy()
    train.index = pd.Index(split.train_run_ids, name="run_id")
    validation.index = pd.Index(split.validation_run_ids, name="run_id")
    for frame in (train, validation):
        north = frame["route_bearing_cos"].to_numpy(dtype=np.float64)
        east = frame["route_bearing_sin"].to_numpy(dtype=np.float64)
        wind_north = frame["ekf_wind_north_mean_m_s"].to_numpy(dtype=np.float64)
        wind_east = frame["ekf_wind_east_mean_m_s"].to_numpy(dtype=np.float64)
        derived = {
            "headwind_m_s": -(wind_north * north + wind_east * east),
            "crosswind_abs_m_s": np.abs(wind_north * east - wind_east * north),
            "required_airspeed_m_s": np.hypot(SPEED_SETPOINT * north - wind_north,
                                            SPEED_SETPOINT * east - wind_east),
        }
        for column, expected in derived.items():
            if not np.allclose(frame[column], expected, rtol=1e-7, atol=1e-7):
                raise ValueError(f"{column} is not derivable from the base route and wind columns")
    speed_labels = pd.read_csv(ROOT / "reports/m9/rtl_speed_train_validation.csv").set_index(
        "run_id", verify_integrity=True
    )
    if set(speed_labels.index) != set(train.index) | set(validation.index):
        raise ValueError("speed labels do not match the train/validation flights")
    if not (speed_labels.loc[train.index, "split"] == "train").all() or not (
        speed_labels.loc[validation.index, "split"] == "validation"
    ).all():
        raise ValueError("speed labels are assigned to the wrong split")
    actual_train_speed = speed_labels.loc[train.index, "middle_route_ground_speed_m_s"].to_numpy(dtype=np.float64)
    actual_validation_speed = speed_labels.loc[
        validation.index, "middle_route_ground_speed_m_s"
    ].to_numpy(dtype=np.float64)
    headwind_mask = validation["headwind_m_s"].to_numpy(dtype=np.float64) >= 2.0
    folds = KFold(n_splits=5, shuffle=True, random_state=20260923)
    predictions = pd.DataFrame({
        "run_id": validation.index,
        "actual_charge_mah": split.validation_y,
        "actual_speed_m_s": actual_validation_speed,
        "headwind_m_s": validation["headwind_m_s"].to_numpy(dtype=np.float64),
    })
    summary: dict[str, object] = {
        "training_flights": len(train),
        "validation_flights": len(validation),
        "high_headwind_validation_flights": int(headwind_mask.sum()),
        "input_sets": {},
    }
    for set_name, speed_columns in SPEED_INPUTS_BY_SET.items():
        train_speed_x = train.loc[:, speed_columns].to_numpy(dtype=np.float64)
        validation_speed_x = validation.loc[:, speed_columns].to_numpy(dtype=np.float64)
        out_of_fold_speed = np.clip(
            cross_val_predict(speed_model(), train_speed_x, actual_train_speed, cv=folds, n_jobs=1),
            1.0,
            SPEED_SETPOINT,
        )
        predicted_speed = np.clip(
            speed_model().fit(train_speed_x, actual_train_speed).predict(validation_speed_x),
            1.0,
            SPEED_SETPOINT,
        )
        predictions[f"{set_name}_predicted_speed_m_s"] = predicted_speed
        train_base = train.loc[:, sets[set_name]]
        validation_base = validation.loc[:, sets[set_name]]
        variants = {
            "base": (train_base, validation_base),
            "plus_constant_speed_time": (
                train_base.assign(predicted_horizontal_time_s=train_base["distance_home_m"] / SPEED_SETPOINT),
                validation_base.assign(predicted_horizontal_time_s=validation_base["distance_home_m"] / SPEED_SETPOINT),
            ),
            "plus_speed": (
                augmented_inputs(train_base, out_of_fold_speed, include_time=False),
                augmented_inputs(validation_base, predicted_speed, include_time=False),
            ),
            "plus_predicted_speed_time": (
                train_base.assign(predicted_horizontal_time_s=train_base["distance_home_m"] / out_of_fold_speed),
                validation_base.assign(predicted_horizontal_time_s=validation_base["distance_home_m"] / predicted_speed),
            ),
            "plus_speed_and_time": (
                augmented_inputs(train_base, out_of_fold_speed, include_time=True),
                augmented_inputs(validation_base, predicted_speed, include_time=True),
            ),
        }
        set_summary = {
            "speed_inputs": list(speed_columns),
            "speed_validation_mae_m_s": float(np.mean(np.abs(predicted_speed - actual_validation_speed))),
            "speed_high_headwind_mae_m_s": float(np.mean(
                np.abs(predicted_speed[headwind_mask] - actual_validation_speed[headwind_mask])
            )),
            "charge_validation": {},
        }
        for variant_name, (train_x, validation_x) in variants.items():
            if tuple(train_x.columns) != tuple(validation_x.columns):
                raise ValueError("train/validation feature columns differ")
            ridge = fit_ridge(alpha=1.0).fit(train_x, split.train_y)
            ridge_pred = ridge.predict(validation_x)
            neural, _ = fit_dense_regressor(
                train_x.to_numpy(dtype=np.float64), split.train_y,
                validation_x.to_numpy(dtype=np.float64), split.validation_y,
                loss="huber", alpha=0.001, learning_rate=0.001,
                seed=20260924, max_epochs=525, fixed_epochs=True,
            )
            neural_pred = neural.predict(validation_x)
            set_summary["charge_validation"][variant_name] = {
                "ridge": charge_metrics(split.validation_y, ridge_pred, headwind_mask),
                "neural_64_32": charge_metrics(split.validation_y, neural_pred, headwind_mask),
            }
            predictions[f"{set_name}_{variant_name}_ridge_mah"] = ridge_pred
            predictions[f"{set_name}_{variant_name}_neural_mah"] = neural_pred
        summary["input_sets"][set_name] = set_summary

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    predictions.to_csv(OUT / "validation_predictions.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
