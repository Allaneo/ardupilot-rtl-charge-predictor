#!/usr/bin/env python3
"""Validation-only comparison of agreed RTL-charge inputs and NN architectures.

Runs are checkpointed individually. Held-out test labels are never indexed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtl_charge.modeling import (  # noqa: E402
    NN_ARCHITECTURES,
    fit_dense_regressor,
    fit_ridge,
    fit_xgboost,
    load_train_validation,
    regression_metrics,
)
from rtl_charge.revised_features import consistent_speed_f2  # noqa: E402
from rtl_charge.schema import load_feature_sets  # noqa: E402
from compare_speed_features_f0_f1 import F0_SPEED_INPUTS  # noqa: E402
from validate_controller_aware_features import (  # noqa: E402
    SPEED_INPUTS,
    SPEED_SETPOINT,
    make_consistent_features,
    speed_model,
)

OUT = ROOT / "reports/m13_input_architecture_sweep"
SEEDS = (20260924, 20260925, 20260926)
XGB_PARAMETERS = {
    "colsample_bytree": 0.8,
    "learning_rate": 0.1,
    "max_depth": 3,
    "min_child_weight": 5,
    "n_estimators": 300,
    "reg_lambda": 1.0,
    "subsample": 1.0,
}


def speed_predictions(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    speed_labels: pd.DataFrame,
    columns: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
    train_x = train.loc[:, list(columns)].to_numpy(dtype=np.float64)
    validation_x = validation.loc[:, list(columns)].to_numpy(dtype=np.float64)
    train_y = speed_labels.loc[train.index, "middle_route_ground_speed_m_s"].to_numpy(dtype=np.float64)
    folds = KFold(n_splits=5, shuffle=True, random_state=20260923)
    out_of_fold = cross_val_predict(speed_model(), train_x, train_y, cv=folds, n_jobs=1)
    predicted = speed_model().fit(train_x, train_y).predict(validation_x)
    return (
        np.clip(out_of_fold, 1.0, SPEED_SETPOINT),
        np.clip(predicted, 1.0, SPEED_SETPOINT),
    )


def with_time(frame: pd.DataFrame, speed: np.ndarray, *, include_speed: bool) -> pd.DataFrame:
    result = frame.copy()
    if include_speed:
        result["predicted_cruise_ground_speed_m_s"] = speed
    result["predicted_horizontal_time_s"] = result["distance_home_m"].to_numpy(dtype=np.float64) / speed
    return result


def load_variants() -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame]], np.ndarray, np.ndarray, tuple[str, ...], tuple[str, ...], np.ndarray]:
    feature_sets = load_feature_sets(ROOT / "config/feature_sets.yaml")
    split = load_train_validation(ROOT / "data/processed/pipeline_dataset.csv", feature_sets["f2_physics_informed"])
    train, validation = split.train_x.copy(), split.validation_x.copy()
    train.index = pd.Index(split.train_run_ids, name="run_id")
    validation.index = pd.Index(split.validation_run_ids, name="run_id")
    speed_labels = pd.read_csv(ROOT / "reports/m9/rtl_speed_train_validation.csv").set_index("run_id", verify_integrity=True)
    if set(speed_labels.index) != set(train.index) | set(validation.index):
        raise ValueError("speed labels must match exactly the train/validation flights")
    if not (speed_labels.loc[train.index, "split"] == "train").all() or not (
        speed_labels.loc[validation.index, "split"] == "validation"
    ).all():
        raise ValueError("speed labels have incorrect split assignments")

    f0_train, f0_validation = speed_predictions(train, validation, speed_labels, F0_SPEED_INPUTS)
    f1_train, f1_validation = speed_predictions(train, validation, speed_labels, SPEED_INPUTS)
    fixed_train = np.full(len(train), SPEED_SETPOINT)
    fixed_validation = np.full(len(validation), SPEED_SETPOINT)
    f0_columns = feature_sets["f0_route_and_wind"]
    f1_columns = feature_sets["f1_vehicle_symptoms"]
    f2_columns = feature_sets["f2_physics_informed"]
    f0 = (train.loc[:, f0_columns], validation.loc[:, f0_columns])
    f1 = (train.loc[:, f1_columns], validation.loc[:, f1_columns])
    variants = {
        "f0_original": f0,
        "f0_fixed_time": (with_time(f0[0], fixed_train, include_speed=False),
                          with_time(f0[1], fixed_validation, include_speed=False)),
        "f0_learned_time": (with_time(f0[0], f0_train, include_speed=False),
                            with_time(f0[1], f0_validation, include_speed=False)),
        "f1_original": f1,
        "f1_fixed_time": (with_time(f1[0], fixed_train, include_speed=False),
                          with_time(f1[1], fixed_validation, include_speed=False)),
        "f1_learned_speed_time": (with_time(f1[0], f1_train, include_speed=True),
                                  with_time(f1[1], f1_validation, include_speed=True)),
        "f2_corrected_fixed": (consistent_speed_f2(train, f2_columns, fixed_train),
                               consistent_speed_f2(validation, f2_columns, fixed_validation)),
        "f2_corrected_learned": (consistent_speed_f2(train, f2_columns, f1_train),
                                 consistent_speed_f2(validation, f2_columns, f1_validation)),
        "f3_cubic_fixed": (make_consistent_features(train, fixed_train, f1_columns),
                           make_consistent_features(validation, fixed_validation, f1_columns)),
        "f3_cubic_learned": (make_consistent_features(train, f1_train, f1_columns),
                             make_consistent_features(validation, f1_validation, f1_columns)),
    }
    if len(variants) != 10:
        raise AssertionError("the approved comparison has exactly ten variants")
    for name, (train_x, validation_x) in variants.items():
        if tuple(train_x.columns) != tuple(validation_x.columns):
            raise ValueError(f"{name}: train/validation feature columns differ")
        if not np.isfinite(train_x.to_numpy(dtype=np.float64)).all() or not np.isfinite(validation_x.to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{name}: nonfinite charge input")
    headwind = validation["headwind_m_s"].to_numpy(dtype=np.float64)
    return variants, split.train_y, split.validation_y, split.train_run_ids, split.validation_run_ids, headwind


def metrics(actual: np.ndarray, predicted: np.ndarray, headwind_mask: np.ndarray) -> dict[str, float]:
    base = regression_metrics(actual, predicted)
    return {
        "mae_mah": base["mae_mah"],
        "rmse_mah": base["rmse_mah"],
        "max_underprediction_mah": float(np.max(actual - predicted)),
        "high_headwind_mae_mah": float(np.mean(np.abs(actual[headwind_mask] - predicted[headwind_mask]))),
    }


def parameter_count(widths: tuple[int, ...], inputs: int) -> int:
    dims = (inputs, *widths, 1)
    return sum((dims[i] + 1) * dims[i + 1] for i in range(len(dims) - 1))


def write_once(path: Path, data: dict[str, object]) -> None:
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != payload:
            raise ValueError(f"existing checkpoint disagrees: {path}")
        return
    temporary = path.with_suffix(".tmp")
    temporary.write_text(payload)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-runs", type=int, default=None, help="stop after this many new checkpoints")
    args = parser.parse_args()
    if args.max_new_runs is not None and args.max_new_runs <= 0:
        raise ValueError("--max-new-runs must be positive")
    variants, train_y, validation_y, train_ids, validation_ids, headwind = load_variants()
    headwind_mask = headwind >= 2.0
    manifest = {
        "training_run_ids": list(train_ids),
        "validation_run_ids": list(validation_ids),
        "variants": {name: list(pair[0].columns) for name, pair in variants.items()},
        "architectures": [list(shape) for shape in NN_ARCHITECTURES],
        "seeds": list(SEEDS),
        "ridge_alpha": 1.0,
        "xgboost_parameters": XGB_PARAMETERS,
        "neural_settings": {"loss": "huber", "alpha": 0.001, "learning_rate": 0.001,
                            "max_epochs": 600, "patience": 45, "early_stopping": "validation MAE"},
        "speed_model": "150-tree depth-4 random forest, five-fold out-of-fold training predictions",
        "headwind_threshold_m_s": 2.0,
        "test_set_used": False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    runs_dir = OUT / "runs"
    runs_dir.mkdir(exist_ok=True)
    write_once(OUT / "manifest.json", manifest)
    completed = 0
    new = 0
    total = len(variants) * (2 + len(NN_ARCHITECTURES) * len(SEEDS))
    for variant, (train_x, validation_x) in variants.items():
        train_array = train_x.to_numpy(dtype=np.float64)
        validation_array = validation_x.to_numpy(dtype=np.float64)
        jobs = [("ridge", None, None), ("xgboost", None, None)] + [
            ("neural", widths, seed) for widths in NN_ARCHITECTURES for seed in SEEDS
        ]
        for model_name, widths, seed in jobs:
            shape_name = "-".join(map(str, widths)) if widths is not None else "fixed"
            run_id = f"{variant}__{model_name}__{shape_name}" + (f"__{seed}" if seed is not None else "")
            checkpoint = runs_dir / f"{run_id}.json"
            if checkpoint.exists():
                completed += 1
                continue
            if model_name == "ridge":
                model = fit_ridge(alpha=1.0).fit(train_x, train_y)
                selected_epoch = None
                parameters = None
            elif model_name == "xgboost":
                model = fit_xgboost(XGB_PARAMETERS, seed=20260928).fit(train_x, train_y)
                selected_epoch = None
                parameters = None
            else:
                assert widths is not None and seed is not None
                model, selected_epoch = fit_dense_regressor(
                    train_array, train_y, validation_array, validation_y,
                    hidden_layers=widths, seed=seed, loss="huber", alpha=0.001,
                    learning_rate=0.001, max_epochs=600, patience=45,
                )
                parameters = parameter_count(widths, train_array.shape[1])
            train_pred = model.predict(train_x if model_name != "neural" else train_array)
            validation_pred = model.predict(validation_x if model_name != "neural" else validation_array)
            record = {
                "run_id": run_id,
                "variant": variant,
                "model": model_name,
                "architecture": list(widths) if widths is not None else None,
                "seed": seed,
                "selected_epoch": selected_epoch,
                "parameter_count": parameters,
                "train": metrics(train_y, train_pred, np.ones(len(train_y), dtype=bool)),
                "validation": metrics(validation_y, validation_pred, headwind_mask),
                "validation_predictions_mah": [float(value) for value in validation_pred],
            }
            write_once(checkpoint, record)
            completed += 1
            new += 1
            print(f"{completed}/{total} {run_id} val_MAE={record['validation']['mae_mah']:.2f}", flush=True)
            if args.max_new_runs is not None and new >= args.max_new_runs:
                return
    print(f"Completed {completed}/{total} runs; {new} new.")


if __name__ == "__main__":
    main()
