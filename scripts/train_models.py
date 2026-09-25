#!/usr/bin/env python3
"""Train M6 candidates, select only on validation data, and hold out the test set."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rtl_charge.modeling import (  # noqa: E402
    PX4_STYLE_MULTICOPTER_CRUISE_SPEED_M_S,
    SEED,
    fit_dense_regressor,
    fit_ridge,
    fit_xgboost,
    load_train_validation,
    neural_candidates,
    px4_style_charge_baseline,
    regression_metrics,
    ridge_candidates,
    xgboost_candidates,
)
from rtl_charge.schema import load_feature_sets  # noqa: E402


DATASET = PROJECT_ROOT / "data" / "processed" / "pipeline_dataset.csv"
FEATURES = PROJECT_ROOT / "config" / "feature_sets.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "m6"
DEFAULT_MODELS = PROJECT_ROOT / "models" / "m6"
AUXILIARY_COLUMNS = (
    "current_mean_a",
    "current_time_charge_baseline_mah",
    "distance_home_m",
    "nominal_horizontal_time_s",
    "nominal_total_rtl_time_s",
)
LOCKED_BASELINE_PARAMETERS = {
    "RTL_SPEED_MS": 0.0,
    "WP_SPD": 10.0,
    "WP_SPD_UP": 2.5,
    "WP_SPD_DN": 1.5,
    "LAND_SPD_MS": 0.5,
    "LAND_SPD_HIGH_MS": 0.0,
    "LAND_ALT_LOW_M": 10.0,
    "RTL_ALT_M": 15.0,
    "RTL_CLIMB_MIN_M": 0.0,
    "RTL_LOIT_TIME": 5000.0,
}


def describe_numeric(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    quartiles = np.quantile(values, [0.25, 0.5, 0.75])
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "min": float(np.min(values)),
        "q25": float(quartiles[0]),
        "median": float(quartiles[1]),
        "q75": float(quartiles[2]),
        "max": float(np.max(values)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n")


def record_result(
    records: list[dict[str, Any]],
    predictions: list[pd.DataFrame],
    *,
    model: str,
    feature_set: str,
    run_ids: tuple[str, ...],
    actual: np.ndarray,
    estimated: np.ndarray,
    hyperparameters: dict[str, Any],
    n_features: int,
    selected: bool = False,
) -> dict[str, Any]:
    metrics = regression_metrics(actual, estimated)
    encoded_parameters = json.dumps(jsonable(hyperparameters), sort_keys=True)
    result = {
        "model": model,
        "feature_set": feature_set,
        "feature_count": n_features,
        "n_train": None,
        "n_validation": int(len(actual)),
        "selected_on_validation": selected,
        "hyperparameters": encoded_parameters,
        **metrics,
    }
    records.append(result)
    predictions.append(
        pd.DataFrame(
            {
                "run_id": run_ids,
                "model": model,
                "feature_set": feature_set,
                "hyperparameters": encoded_parameters,
                "selected_on_validation": selected,
                "rtl_charge_mah": actual,
                "predicted_rtl_charge_mah": estimated,
                "error_mah": estimated - actual,
                "absolute_error_mah": np.abs(estimated - actual),
                "underprediction_mah": np.maximum(actual - estimated, 0.0),
            }
        )
    )
    return result


def mark_selected_predictions(
    predictions: list[pd.DataFrame],
    selected_result: dict[str, Any],
) -> None:
    """Mark per-flight residuals for the candidate chosen on validation MAE."""

    for frame in predictions:
        mask = (
            (frame["model"] == selected_result["model"])
            & (frame["feature_set"] == selected_result["feature_set"])
            & (frame["hyperparameters"] == selected_result["hyperparameters"])
        )
        frame.loc[mask, "selected_on_validation"] = True


def save_model(path: Path, model: Any, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, **jsonable(metadata)}, path)


def train_ridge_candidates(
    feature_set: str,
    features: tuple[str, ...],
    split: Any,
    records: list[dict[str, Any]],
    predictions: list[pd.DataFrame],
) -> tuple[dict[str, Any], Any]:
    candidates = []
    for alpha in ridge_candidates():
        model = fit_ridge(alpha)
        model.fit(split.train_x, split.train_y)
        result = record_result(
            records,
            predictions,
            model="ridge",
            feature_set=feature_set,
            run_ids=split.validation_run_ids,
            actual=split.validation_y,
            estimated=model.predict(split.validation_x),
            hyperparameters={"alpha": alpha, "scaling": "standardized on training rows"},
            n_features=len(features),
        )
        result["n_train"] = len(split.train_y)
        candidates.append((result, model))
    best_result, best_model = min(
        candidates, key=lambda item: (item[0]["mae_mah"], item[0]["hyperparameters"])
    )
    best_result["selected_on_validation"] = True
    mark_selected_predictions(predictions, best_result)
    return best_result, best_model


def train_neural_candidates(
    feature_set: str,
    features: tuple[str, ...],
    split: Any,
    records: list[dict[str, Any]],
    predictions: list[pd.DataFrame],
    *,
    seed: int,
) -> tuple[dict[str, Any], Any, dict[str, Any], int]:
    candidates = []
    for candidate_index, parameters in enumerate(neural_candidates()):
        model, best_epoch = fit_dense_regressor(
            split.train_x.to_numpy(),
            split.train_y,
            split.validation_x.to_numpy(),
            split.validation_y,
            loss=parameters["loss"],
            alpha=parameters["alpha"],
            learning_rate=parameters["learning_rate"],
            seed=seed + candidate_index,
        )
        result = record_result(
            records,
            predictions,
            model="neural_network",
            feature_set=feature_set,
            run_ids=split.validation_run_ids,
            actual=split.validation_y,
            estimated=model.predict(split.validation_x),
            hyperparameters={
                **parameters,
                "architecture": list((64, 32, 1)),
                "optimizer": "Adam",
                "random_seed": seed + candidate_index,
                "best_epoch": best_epoch,
                "early_stopping_patience": 45,
                "target_scaling": "standardized on training rows",
                "huber_delta_standardized": 1.0 if parameters["loss"] == "huber" else None,
            },
            n_features=len(features),
        )
        result["n_train"] = len(split.train_y)
        candidates.append((result, model, parameters, best_epoch))
    best = min(candidates, key=lambda item: (item[0]["mae_mah"], item[0]["hyperparameters"]))
    best[0]["selected_on_validation"] = True
    mark_selected_predictions(predictions, best[0])
    return best[0], best[1], best[2], best[3]


def train_xgboost_candidates(
    feature_set: str,
    features: tuple[str, ...],
    split: Any,
    records: list[dict[str, Any]],
    predictions: list[pd.DataFrame],
    *,
    seed: int,
) -> tuple[dict[str, Any], Any]:
    candidates = []
    for candidate_index, parameters in enumerate(xgboost_candidates()):
        model = fit_xgboost(parameters, seed=seed + candidate_index)
        model.fit(split.train_x, split.train_y, verbose=False)
        result = record_result(
            records,
            predictions,
            model="xgboost",
            feature_set=feature_set,
            run_ids=split.validation_run_ids,
            actual=split.validation_y,
            estimated=model.predict(split.validation_x),
            hyperparameters={**parameters, "random_state": seed + candidate_index},
            n_features=len(features),
        )
        result["n_train"] = len(split.train_y)
        candidates.append((result, model))
    best_result, best_model = min(
        candidates, key=lambda item: (item[0]["mae_mah"], item[0]["hyperparameters"])
    )
    best_result["selected_on_validation"] = True
    mark_selected_predictions(predictions, best_result)
    return best_result, best_model


def train_oracle(
    features: tuple[str, ...],
    split: Any,
    records: list[dict[str, Any]],
    predictions: list[pd.DataFrame],
    *,
    seed: int,
    neural_parameters: dict[str, Any],
    xgb_parameters: dict[str, Any],
    models_dir: Path,
) -> dict[str, Any]:
    if split.train_true_wind_x is None or split.validation_true_wind_x is None:
        raise ValueError("true-wind oracle features were not loaded")

    neural_model, best_epoch = fit_dense_regressor(
        split.train_true_wind_x.to_numpy(),
        split.train_y,
        split.validation_true_wind_x.to_numpy(),
        split.validation_y,
        loss=neural_parameters["loss"],
        alpha=neural_parameters["alpha"],
        learning_rate=neural_parameters["learning_rate"],
        seed=seed + 901,
    )
    neural_row = record_result(
        records,
        predictions,
        model="neural_network_true_wind_oracle",
        feature_set="f2_true_wind_oracle",
        run_ids=split.validation_run_ids,
        actual=split.validation_y,
        estimated=neural_model.predict(split.validation_true_wind_x),
        hyperparameters={
            **neural_parameters,
            "copied_from": "best deployable f2 neural_network configuration",
            "random_seed": seed + 901,
            "best_epoch_selected_on_validation": best_epoch,
        },
        n_features=len(features),
        selected=True,
    )
    neural_row["n_train"] = len(split.train_y)
    save_model(
        models_dir / "neural_network__f2_true_wind_oracle.joblib",
        neural_model,
        {"features": list(features), "validation_metrics": neural_row},
    )

    xgb_model = fit_xgboost(xgb_parameters, seed=seed + 902)
    xgb_model.fit(split.train_true_wind_x, split.train_y, verbose=False)
    xgb_row = record_result(
        records,
        predictions,
        model="xgboost_true_wind_oracle",
        feature_set="f2_true_wind_oracle",
        run_ids=split.validation_run_ids,
        actual=split.validation_y,
        estimated=xgb_model.predict(split.validation_true_wind_x),
        hyperparameters={
            **xgb_parameters,
            "copied_from": "best deployable f2 xgboost configuration",
        },
        n_features=len(features),
        selected=True,
    )
    xgb_row["n_train"] = len(split.train_y)
    save_model(
        models_dir / "xgboost__f2_true_wind_oracle.joblib",
        xgb_model,
        {"features": list(features), "validation_metrics": xgb_row},
    )
    return {
        "neural_network_true_wind_oracle": neural_row,
        "xgboost_true_wind_oracle": xgb_row,
    }


def fit_learning_curve_model(
    name: str,
    parameters: dict[str, Any],
    train_x: pd.DataFrame,
    train_y: np.ndarray,
    validation_x: pd.DataFrame,
    *,
    seed: int,
    fixed_epochs: int,
) -> Any:
    if name == "ridge":
        model = fit_ridge(float(parameters["alpha"]))
        model.fit(train_x, train_y)
        return model
    if name == "neural_network":
        model, _ = fit_dense_regressor(
            train_x.to_numpy(),
            train_y,
            validation_x.to_numpy(),
            np.zeros(len(validation_x), dtype=np.float64),
            loss=str(parameters["loss"]),
            alpha=float(parameters["alpha"]),
            learning_rate=float(parameters["learning_rate"]),
            seed=seed,
            max_epochs=max(1, fixed_epochs),
            fixed_epochs=True,
        )
        return model
    if name == "xgboost":
        model = fit_xgboost(parameters, seed=seed)
        model.fit(train_x, train_y, verbose=False)
        return model
    raise ValueError(f"unsupported learning curve model: {name}")


def build_learning_curves(
    split: Any,
    selected: dict[str, dict[str, Any]],
    *,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(split.train_y))
    sizes = sorted({max(24, int(round(len(order) * fraction))) for fraction in (0.25, 0.5, 0.75, 1.0)})
    sizes = [size for size in sizes if size <= len(order)]
    rows = []
    for size in sizes:
        indices = order[:size]
        subset_x = split.train_x.iloc[indices]
        subset_y = split.train_y[indices]
        for model_name in ("ridge", "neural_network", "xgboost"):
            choice = selected[model_name]
            model = fit_learning_curve_model(
                model_name,
                choice["parameters"],
                subset_x,
                subset_y,
                split.validation_x,
                seed=seed + size,
                fixed_epochs=int(choice.get("best_epoch", 1)),
            )
            train_prediction = model.predict(subset_x)
            validation_prediction = model.predict(split.validation_x)
            for partition, actual, predicted in (
                ("train_subset", subset_y, train_prediction),
                ("validation", split.validation_y, validation_prediction),
            ):
                row = regression_metrics(actual, predicted)
                rows.append(
                    {
                        "feature_set": "f2_physics_informed",
                        "model": model_name,
                        "training_rows": int(size),
                        "partition": partition,
                        **row,
                    }
                )
    return pd.DataFrame(rows)


def plot_validation(records: list[dict[str, Any]], path: Path) -> None:
    selected = [
        row for row in records
        if row["selected_on_validation"]
    ]
    selected.sort(key=lambda row: row["mae_mah"])
    model_labels = {
        "training_mean": "Training mean",
        "nominal_time_x_current": "Nominal time × current",
        "px4_style_multicopter_time_x_current": "PX4-style time × current",
        "ridge": "Ridge",
        "neural_network": "Neural network",
        "xgboost": "XGBoost",
        "neural_network_true_wind_oracle": "Neural net · true-wind oracle",
        "xgboost_true_wind_oracle": "XGBoost · true-wind oracle",
    }
    feature_labels = {
        "f0_route_and_wind": "F0: route + wind",
        "f1_vehicle_symptoms": "F1: vehicle symptoms",
        "f2_physics_informed": "F2: physics-informed",
        "f2_true_wind_oracle": "F2 oracle",
        "baseline": "baseline",
    }
    labels = [
        model_labels[row["model"]]
        if row["feature_set"] == "baseline"
        else f"{model_labels[row['model']]} · {feature_labels[row['feature_set']]}"
        for row in selected
    ]
    values = [row["mae_mah"] for row in selected]
    colors = ["#2A6F97" if "oracle" not in row["model"] else "#9B5DE5" for row in selected]
    figure, axis = plt.subplots(figsize=(10, max(5.2, len(labels) * 0.35)))
    bars = axis.barh(labels, values, color=colors)
    axis.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    axis.invert_yaxis()
    axis.set_xlabel("Validation MAE (mAh)")
    axis.set_title("M6 model selection on the 39-flight validation split")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def plot_learning_curves(curves: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    styles = {"ridge": "o-", "neural_network": "s-", "xgboost": "^-"}
    for axis, partition, title in (
        (axes[0], "train_subset", "Training MAE"),
        (axes[1], "validation", "Validation MAE"),
    ):
        subset = curves[curves["partition"] == partition]
        for name, values in subset.groupby("model", sort=True):
            axis.plot(values["training_rows"], values["mae_mah"], styles[name], label=name)
        axis.set_title(title)
        axis.set_xlabel("Training flights")
        axis.set_ylabel("MAE (mAh)")
        axis.grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("F2 learning curves; fixed selected settings, same validation flights")
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    feature_sets = load_feature_sets(args.features)
    splits = {
        name: load_train_validation(
            args.dataset,
            columns,
            include_true_wind_oracle=name == "f2_physics_informed",
            auxiliary_columns=AUXILIARY_COLUMNS,
        )
        for name, columns in feature_sets.items()
    }
    f2_name = "f2_physics_informed"
    f2_features = feature_sets[f2_name]
    f2 = splits[f2_name]
    if f2.train_aux is None or f2.validation_aux is None:
        raise ValueError("baseline inputs were not loaded")
    if any(split.held_out_rows != f2.held_out_rows for split in splits.values()):
        raise ValueError("feature-set loaders disagree about held-out split size")
    train_ids = f2.train_run_ids
    validation_ids = f2.validation_run_ids
    if any(splits[name].train_run_ids != train_ids for name in splits):
        raise ValueError("feature-set loaders disagree about training flight membership")
    if any(splits[name].validation_run_ids != validation_ids for name in splits):
        raise ValueError("feature-set loaders disagree about validation flight membership")

    records: list[dict[str, Any]] = []
    validation_predictions: list[pd.DataFrame] = []
    selected_models: dict[str, dict[str, Any]] = {}

    mean_prediction = np.full(len(f2.validation_y), float(np.mean(f2.train_y)))
    mean_row = record_result(
        records,
        validation_predictions,
        model="training_mean",
        feature_set="baseline",
        run_ids=validation_ids,
        actual=f2.validation_y,
        estimated=mean_prediction,
        hyperparameters={"mean_rtl_charge_mah": float(np.mean(f2.train_y))},
        n_features=0,
        selected=True,
    )
    mean_row["n_train"] = len(f2.train_y)

    current_time_prediction = f2.validation_aux["current_time_charge_baseline_mah"].to_numpy()
    current_time_row = record_result(
        records,
        validation_predictions,
        model="nominal_time_x_current",
        feature_set="baseline",
        run_ids=validation_ids,
        actual=f2.validation_y,
        estimated=current_time_prediction,
        hyperparameters={"source_feature": "current_time_charge_baseline_mah"},
        n_features=1,
        selected=True,
    )
    current_time_row["n_train"] = len(f2.train_y)

    px4_prediction = px4_style_charge_baseline(f2.validation_aux)
    px4_row = record_result(
        records,
        validation_predictions,
        model="px4_style_multicopter_time_x_current",
        feature_set="baseline",
        run_ids=validation_ids,
        actual=f2.validation_y,
        estimated=px4_prediction,
        hyperparameters={
            "horizontal_cruise_speed_m_s": PX4_STYLE_MULTICOPTER_CRUISE_SPEED_M_S,
            "wind_correction": "not applied to multicopter cruise-time term",
            "vertical_and_loiter_terms": "shared locked kinematic baseline",
            "current_source": "10-second pre-RTL decision window",
        },
        n_features=4,
        selected=True,
    )
    px4_row["n_train"] = len(f2.train_y)

    for feature_set, features in feature_sets.items():
        split = splits[feature_set]
        ridge_best, ridge_model = train_ridge_candidates(
            feature_set, features, split, records, validation_predictions
        )
        ridge_parameters = json.loads(ridge_best["hyperparameters"])
        selected_models[f"ridge__{feature_set}"] = {
            "row": ridge_best,
            "model": ridge_model,
            "parameters": ridge_parameters,
        }
        save_model(
            args.models_dir / f"ridge__{feature_set}.joblib",
            ridge_model,
            {"model_name": "ridge", "feature_set": feature_set, "features": list(features), "validation": ridge_best},
        )

        neural_best, neural_model, neural_parameters, best_epoch = train_neural_candidates(
            feature_set, features, split, records, validation_predictions, seed=args.seed
        )
        selected_models[f"neural_network__{feature_set}"] = {
            "row": neural_best,
            "model": neural_model,
            "parameters": neural_parameters,
            "best_epoch": best_epoch,
        }
        save_model(
            args.models_dir / f"neural_network__{feature_set}.joblib",
            neural_model,
            {"model_name": "neural_network", "feature_set": feature_set, "features": list(features), "validation": neural_best},
        )

        xgb_best, xgb_model = train_xgboost_candidates(
            feature_set, features, split, records, validation_predictions, seed=args.seed
        )
        xgb_parameters = json.loads(xgb_best["hyperparameters"])
        selected_models[f"xgboost__{feature_set}"] = {
            "row": xgb_best,
            "model": xgb_model,
            "parameters": xgb_parameters,
        }
        save_model(
            args.models_dir / f"xgboost__{feature_set}.joblib",
            xgb_model,
            {"model_name": "xgboost", "feature_set": feature_set, "features": list(features), "validation": xgb_best},
        )

    data_audit = {
        "split_counts": {
            "train": len(f2.train_y),
            "validation": len(f2.validation_y),
            "test_rows_count_only": f2.held_out_rows,
        },
        "test_target_values_accessed": False,
        "target_rtl_charge_mah": {
            "train": describe_numeric(f2.train_y),
            "validation": describe_numeric(f2.validation_y),
        },
        "feature_coverage": {
            name: {
                "features": len(columns),
                "train_rows_all_finite": bool(np.isfinite(splits[name].train_x.to_numpy()).all()),
                "validation_rows_all_finite": bool(np.isfinite(splits[name].validation_x.to_numpy()).all()),
                "train_ranges": {
                    feature: [
                        float(splits[name].train_x[feature].min()),
                        float(splits[name].train_x[feature].max()),
                    ]
                    for feature in columns
                },
                "validation_ranges": {
                    feature: [
                        float(splits[name].validation_x[feature].min()),
                        float(splits[name].validation_x[feature].max()),
                    ]
                    for feature in columns
                },
            }
            for name, columns in feature_sets.items()
        },
    }
    save_json(args.output_dir / "dataset_audit.json", data_audit)

    oracle = train_oracle(
        f2_features,
        f2,
        records,
        validation_predictions,
        seed=args.seed,
        neural_parameters=selected_models[f"neural_network__{f2_name}"]["parameters"],
        xgb_parameters=selected_models[f"xgboost__{f2_name}"]["parameters"],
        models_dir=args.models_dir,
    )

    learning_curves = build_learning_curves(
        f2,
        {
            name: selected_models[f"{name}__{f2_name}"]
            for name in ("ridge", "neural_network", "xgboost")
        },
        seed=args.seed,
    )

    candidate_table = pd.DataFrame(records)
    candidate_table.to_csv(args.output_dir / "candidate_validation_metrics.csv", index=False)
    all_validation_predictions = pd.concat(validation_predictions, ignore_index=True)
    all_validation_predictions.to_csv(
        args.output_dir / "candidate_validation_predictions.csv", index=False
    )
    all_validation_predictions.loc[all_validation_predictions["selected_on_validation"]].to_csv(
        args.output_dir / "selected_validation_predictions.csv", index=False
    )
    learning_curves.to_csv(args.output_dir / "f2_learning_curves.csv", index=False)
    plot_validation(records, args.output_dir / "validation_mae_comparison.png")
    plot_learning_curves(learning_curves, args.output_dir / "f2_learning_curves.png")

    selected_summary = {
        "baseline_training_mean": mean_row,
        "baseline_nominal_time_x_current": current_time_row,
        "baseline_px4_style": px4_row,
        **{
            key: value["row"]
            for key, value in selected_models.items()
        },
        **oracle,
    }
    summary = {
        "milestone": "M6",
        "target": "rtl_charge_mah",
        "primary_metric": "validation MAE in mAh",
        "selection_split": "validation",
        "residual_convention": "prediction minus actual; underprediction is max(actual minus prediction, 0)",
        "test_policy": {
            "rows_held_out": f2.held_out_rows,
            "test_labels_used": False,
            "test_features_scored": False,
            "test_evaluation_deferred_to": "M7",
        },
        "split_rows": {"train": len(f2.train_y), "validation": len(f2.validation_y)},
        "seed": args.seed,
        "feature_counts": {name: len(columns) for name, columns in feature_sets.items()},
        "feature_missing_values_train_validation": {name: 0 for name in feature_sets},
        "dataset_audit": "dataset_audit.json",
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "baseline_configuration": {
            "px4_style_multicopter_horizontal_cruise_speed_m_s": PX4_STYLE_MULTICOPTER_CRUISE_SPEED_M_S,
            "dataflash_parameter_readback": LOCKED_BASELINE_PARAMETERS,
            "interpretation": "wind-neutral multicopter horizontal ETA; shared project climb, descent and loiter terms; multiplied by recent current",
        },
        "selected_validation_results": selected_summary,
        "candidate_count": len(records),
        "candidate_metrics": "candidate_validation_metrics.csv",
        "candidate_predictions": "candidate_validation_predictions.csv",
        "selected_predictions": "selected_validation_predictions.csv",
        "learning_curves": "f2_learning_curves.csv",
        "plots": ["validation_mae_comparison.png", "f2_learning_curves.png"],
        "models_dir": (
            str(args.models_dir.resolve().relative_to(PROJECT_ROOT))
            if args.models_dir.resolve().is_relative_to(PROJECT_ROOT)
            else str(args.models_dir.resolve())
        ),
    }
    save_json(args.output_dir / "summary.json", summary)
    print(
        f"M6 complete: {len(records)} validation candidates; "
        f"train={len(f2.train_y)}, validation={len(f2.validation_y)}, "
        f"held-out test rows={f2.held_out_rows} (labels untouched)"
    )
    for name, result in selected_summary.items():
        print(f"{name}: validation MAE={result['mae_mah']:.1f} mAh")


if __name__ == "__main__":
    main()
