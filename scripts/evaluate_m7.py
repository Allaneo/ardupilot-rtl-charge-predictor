#!/usr/bin/env python3
"""One-time held-out test evaluation of frozen M6 models."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtl_charge.modeling import (  # noqa: E402
    TARGET,
    make_true_wind_oracle,
    px4_style_charge_baseline,
    regression_metrics,
)
from rtl_charge.schema import load_feature_sets  # noqa: E402


SEED = 20260921
BOOTSTRAPS = 10_000
DATASET = ROOT / "data/processed/pipeline_dataset.csv"
MODELS = ROOT / "models/m6"
OUT = ROOT / "reports/m7"


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    draws = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = np.mean(values[draws], axis=1)
    return {
        "estimate": float(np.mean(values)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def bin_from_train(train: pd.Series, test: pd.Series, labels: list[str]):
    edges = np.unique(np.quantile(train.to_numpy(float), np.linspace(0, 1, len(labels) + 1)))
    if len(edges) < 3:
        return pd.Series(["all"] * len(test), index=test.index), [float(x) for x in edges]
    # Expand endpoints so every test value is assigned, while cut points remain train-only.
    edges[0], edges[-1] = -np.inf, np.inf
    used_labels = labels[: len(edges) - 1]
    return pd.cut(test, bins=edges, labels=used_labels, include_lowest=True), [float(x) for x in edges[1:-1]]


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing final-evaluation directory: {OUT}")
    feature_sets = load_feature_sets(ROOT / "config/feature_sets.yaml")
    frame = pd.read_csv(DATASET)
    train = frame.loc[frame["split"] == "train"].copy()
    test = frame.loc[frame["split"] == "test"].copy()
    if len(test) != 39 or len(train) != 172:
        raise ValueError(f"Frozen split changed: train={len(train)}, test={len(test)}")
    if test["run_id"].duplicated().any() or test[TARGET].isna().any():
        raise ValueError("test run ids must be unique and target labels present")

    # From here, M7 intentionally reads the held-out test labels exactly once.
    y = test[TARGET].to_numpy(dtype=float)
    aux = test[["current_mean_a", "current_time_charge_baseline_mah", "distance_home_m",
                "nominal_horizontal_time_s", "nominal_total_rtl_time_s"]]
    predictions: list[pd.DataFrame] = []
    m6 = json.loads((ROOT / "reports/m6/summary.json").read_text())
    selected = m6["selected_validation_results"]

    def add(name: str, feature_set: str, model_label: str, values: np.ndarray, features: tuple[str, ...] = (), oracle=False):
        values = np.asarray(values, dtype=float).reshape(-1)
        if len(values) != len(y) or not np.isfinite(values).all():
            raise ValueError(f"invalid predictions for {name}")
        errors = values - y
        predictions.append(pd.DataFrame({
            "run_id": test["run_id"].to_numpy(), "model": name, "feature_set": feature_set,
            "actual_rtl_charge_mah": y, "predicted_rtl_charge_mah": values,
            "error_mah": errors, "absolute_error_mah": np.abs(errors),
            "underprediction_mah": np.maximum(-errors, 0),
        }))

    add("training_mean", "baseline", "Training mean", np.full(len(test), selected["baseline_training_mean"]["hyperparameters"] and
        json.loads(selected["baseline_training_mean"]["hyperparameters"])["mean_rtl_charge_mah"]))
    add("nominal_time_x_current", "baseline", "Nominal time × current", aux["current_time_charge_baseline_mah"].to_numpy())
    add("px4_style_multicopter_time_x_current", "baseline", "PX4-style time × current", px4_style_charge_baseline(aux))

    for feature_set, features in feature_sets.items():
        x = test.loc[:, features]
        for model_name in ("ridge", "neural_network", "xgboost"):
            artifact = joblib.load(MODELS / f"{model_name}__{feature_set}.joblib")
            expected = selected[f"{model_name}__{feature_set}"]
            stored = artifact.get("validation")
            if stored is None or not stored.get("selected_on_validation"):
                raise ValueError(f"artifact was not selected on validation: {model_name}/{feature_set}")
            if abs(float(stored["mae_mah"]) - float(expected["mae_mah"])) > 1e-9:
                raise ValueError(f"artifact metadata mismatch: {model_name}/{feature_set}")
            add(f"{model_name}__{feature_set}", feature_set, model_name, artifact["model"].predict(x), features)

    f2 = feature_sets["f2_physics_informed"]
    true_wind = make_true_wind_oracle(test, f2)
    for model_name in ("neural_network", "xgboost"):
        artifact = joblib.load(MODELS / f"{model_name}__f2_true_wind_oracle.joblib")
        add(f"{model_name}_true_wind_oracle", "f2_true_wind_oracle", model_name,
            artifact["model"].predict(true_wind), f2, oracle=True)

    pred = pd.concat(predictions, ignore_index=True)
    rng = np.random.default_rng(SEED)
    metric_rows = []
    for name, group in pred.groupby("model", sort=False):
        stats = regression_metrics(group["actual_rtl_charge_mah"].to_numpy(), group["predicted_rtl_charge_mah"].to_numpy())
        metric_rows.append({"model": name, "feature_set": group["feature_set"].iloc[0], "n_test": len(group), **stats,
                            "mae_bootstrap_95": bootstrap_mean(group["absolute_error_mah"].to_numpy(), rng)})
    metrics = pd.DataFrame(metric_rows)

    # Paired flight bootstrap of absolute-error differences, using F2 Ridge as a fixed reference.
    reference = pred[pred["model"] == "ridge__f2_physics_informed"].set_index("run_id")["absolute_error_mah"]
    paired_rows = []
    for name, group in pred.groupby("model", sort=False):
        candidate = group.set_index("run_id")["absolute_error_mah"].reindex(reference.index)
        diff = candidate.to_numpy() - reference.to_numpy()
        interval = bootstrap_mean(diff, rng)
        paired_rows.append({"model": name, "reference": "ridge__f2_physics_informed",
                            "mean_mae_difference_mah": interval["estimate"],
                            "paired_bootstrap_ci95_low": interval["ci95_low"],
                            "paired_bootstrap_ci95_high": interval["ci95_high"],
                            "interpretation": "negative favors candidate; interval is descriptive with 39 flights"})
    paired = pd.DataFrame(paired_rows)

    # Diagnostic strata: thresholds derive from training flights only. Tiny bins are explicitly flagged.
    route_unit_n = test["route_bearing_cos"].to_numpy(float)
    route_unit_e = test["route_bearing_sin"].to_numpy(float)
    true_along_wind = test["true_wind_north_m_s"].to_numpy(float) * route_unit_n + test["true_wind_east_m_s"].to_numpy(float) * route_unit_e
    wind_error = np.hypot(test["true_wind_north_m_s"] - test["ekf_wind_north_mean_m_s"],
                          test["true_wind_east_m_s"] - test["ekf_wind_east_mean_m_s"])
    strata = {
        "true_wind_speed": (np.hypot(test["true_wind_north_m_s"], test["true_wind_east_m_s"]),
                            np.hypot(train["true_wind_north_m_s"], train["true_wind_east_m_s"]), ["low", "mid", "high"]),
        "wind_route_regime": (pd.Series(np.where(true_along_wind < -0.25, "headwind", np.where(true_along_wind > 0.25, "tailwind", "crosswind")), index=test.index), None, None),
        "payload_mass": (test["payload_mass_kg"], train["payload_mass_kg"], ["low", "mid", "high"]),
        "distance_home": (test["distance_home_m"], train["distance_home_m"], ["near", "mid", "far"]),
        "altitude": (test["relative_altitude_m"], train["relative_altitude_m"], ["low", "mid", "high"]),
        "ekf_wind_error": (pd.Series(wind_error, index=test.index),
                            np.hypot(train["true_wind_north_m_s"] - train["ekf_wind_north_mean_m_s"], train["true_wind_east_m_s"] - train["ekf_wind_east_mean_m_s"]),
                            ["low", "mid", "high"]),
    }
    breakdown = []
    thresholds = {}
    for variable, (test_values, train_values, labels) in strata.items():
        if train_values is None:
            cats = test_values
            thresholds[variable] = {"definition": "true along-route wind: < -0.25 m/s headwind, > 0.25 m/s tailwind, otherwise crosswind"}
        else:
            cats, cuts = bin_from_train(train_values, test_values, labels)
            thresholds[variable] = {"cut_points_from_training_only": cuts}
        for model, group in pred.groupby("model", sort=False):
            indexed = group.set_index("run_id")
            # Use positional category values; pandas would otherwise reindex the
            # original dataset index against run IDs and silently produce NaNs.
            categories_by_run = pd.Series(np.asarray(cats), index=test["run_id"].to_numpy())
            for category in categories_by_run.dropna().unique():
                ids = categories_by_run
                ids = ids[ids == category].index
                residuals = indexed.loc[indexed.index.intersection(ids)]
                if residuals.empty:
                    continue
                breakdown.append({"variable": variable, "bin": str(category), "model": model,
                                  "n_flights": len(residuals), "small_sample_warning": len(residuals) < 5,
                                  "mae_mah": float(residuals["absolute_error_mah"].mean()),
                                  "mean_signed_error_mah": float(residuals["error_mah"].mean()),
                                  "mean_underprediction_mah": float(residuals["underprediction_mah"].mean()),
                                  "p95_underprediction_mah": float(residuals["underprediction_mah"].quantile(.95))})

    OUT.mkdir(parents=True)
    pred.to_csv(OUT / "test_predictions.csv", index=False)
    metrics.to_csv(OUT / "test_metrics.csv", index=False)
    paired.to_csv(OUT / "paired_bootstrap_vs_f2_ridge.csv", index=False)
    pd.DataFrame(breakdown).to_csv(OUT / "error_breakdowns.csv", index=False)
    summary = {
        "milestone": "M7", "target": TARGET, "primary_metric": "test MAE in mAh",
        "test_evaluation_count": 1, "split_counts": {"train": len(train), "test": len(test)},
        "test_labels_used": True, "model_selection_on_test": False,
        "test_run_ids": test["run_id"].tolist(), "seed": SEED, "bootstrap_resamples": BOOTSTRAPS,
        "bootstrap_unit": "flight; paired comparisons resample identical run ids",
        "paired_reference": "ridge__f2_physics_informed",
        "subgroup_thresholds": thresholds,
        "subgroup_caution": "39 test flights; every bin reports n; bins below five are flagged and descriptive only",
        "oracle_caution": "true-wind models use privileged simulator truth and are diagnostic, not deployable",
        "artifacts": ["test_predictions.csv", "test_metrics.csv", "paired_bootstrap_vs_f2_ridge.csv", "error_breakdowns.csv"],
        "results": jsonable(metrics.to_dict(orient="records")),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(metrics[["model", "mae_mah", "mean_underprediction_mah", "p95_underprediction_mah"]].to_string(index=False))
    print(f"\nM7 complete: evaluated {len(metrics)} frozen model/baseline candidates on {len(test)} held-out flights; reports: {OUT}")


if __name__ == "__main__":
    main()
