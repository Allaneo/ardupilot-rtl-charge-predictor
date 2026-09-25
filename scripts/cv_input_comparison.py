#!/usr/bin/env python3
"""D033 / M22: input sets x models, five-fold cross-validated on all 349 flights.

Inputs F0, F1, F2 and F3 (tilt-limit speed, fitted inside each fold); models
Ridge, XGBoost and the cross-validated network architecture (three-seed mean).
Answers whether inputs or architecture matter more, on equal footing.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtl_charge.modeling import TARGET, fit_ridge, fit_xgboost  # noqa: E402
from rtl_charge.schema import load_feature_sets  # noqa: E402
from cv_architecture_sweep import fit_network, fold_features, shape_name  # noqa: E402
from evaluate_confirmation import FOLD_SEED, SEEDS  # noqa: E402
from nested_cv_final_model import load_all  # noqa: E402


M20 = ROOT / "reports/m20_cv_architecture_sweep/cv_architecture_metrics.csv"
M21 = ROOT / "reports/m21_nested_cv_final_model/summary.json"
OUT = ROOT / "reports/m22_cv_input_comparison"
XGB = {"colsample_bytree": 0.8, "learning_rate": 0.1, "max_depth": 3, "min_child_weight": 5,
       "n_estimators": 300, "reg_lambda": 1.0, "subsample": 1.0}
XGB_SEED = 20260928
INPUTS = ("F0", "F1", "F2", "F3")
BOOTSTRAPS = 10_000
WORKERS = 4


def architecture() -> tuple[int, ...]:
    """D033: M21's final network, or M20's best network if M21 chose Ridge."""
    choice = json.loads(M21.read_text())["final_choice"]
    if choice == "Ridge":
        table = pd.read_csv(M20)
        choice = table.loc[table["model"] != "Ridge"].sort_values("mae_mah")["model"].iloc[0]
    return tuple(int(width) for width in choice.removeprefix("network ").split("-"))


def fold_inputs(frame, measured, saturated, sets, fit_rows, test_rows) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    inputs = {name: (frame.iloc[fit_rows].loc[:, sets[key]].to_numpy(dtype=np.float64),
                     frame.iloc[test_rows].loc[:, sets[key]].to_numpy(dtype=np.float64))
              for name, key in (("F0", "f0_route_and_wind"), ("F1", "f1_vehicle_symptoms"), ("F2", "f2_physics_informed"))}
    inputs["F3"] = fold_features(frame, measured, saturated, fit_rows, test_rows, sets["f1_vehicle_symptoms"])
    return inputs


def paired(a: np.ndarray, b: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    diff = np.abs(a - y) - np.abs(b - y)
    means = diff[rng.integers(0, len(diff), size=(BOOTSTRAPS, len(diff)))].mean(axis=1)
    return {"estimate": float(diff.mean()), "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975))}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing report: {OUT}")
    hidden = architecture()
    network = f"network {shape_name(hidden)}"
    sets = load_feature_sets(ROOT / "config/feature_sets.yaml")
    frame, measured, saturated = load_all()
    y = frame[TARGET].to_numpy(dtype=np.float64)
    headwind = frame["headwind_m_s"].to_numpy(dtype=np.float64)
    models = ("Ridge", "XGBoost", network)
    predictions = {(i, m): np.empty(len(y)) for i in INPUTS for m in models}
    seed_values: dict[tuple[str, int], list[np.ndarray]] = {}

    jobs, test_rows_by_fold = [], {}
    for fold, (fit_rows, test_rows) in enumerate(KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED).split(frame)):
        test_rows_by_fold[fold] = test_rows
        for name, (x_train, x_test) in fold_inputs(frame, measured, saturated, sets, fit_rows, test_rows).items():
            predictions[(name, "Ridge")][test_rows] = fit_ridge(1.0).fit(x_train, y[fit_rows]).predict(x_test)
            predictions[(name, "XGBoost")][test_rows] = fit_xgboost(XGB, seed=XGB_SEED).fit(
                x_train, y[fit_rows], verbose=False).predict(x_test)
            jobs += [((fold, name), hidden, seed, x_train, y[fit_rows], x_test) for seed in SEEDS]

    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for (fold, name), _, _, _, values in pool.map(fit_network, jobs):
            seed_values.setdefault((name, fold), []).append(values)
    for (name, fold), values in seed_values.items():
        predictions[(name, network)][test_rows_by_fold[fold]] = np.mean(values, axis=0)

    rng = np.random.default_rng(FOLD_SEED)
    strong = headwind >= 4.0
    rows = []
    for (inputs, model), values in predictions.items():
        error = values - y
        abs_error = np.abs(error)
        means = abs_error[rng.integers(0, len(y), size=(BOOTSTRAPS, len(y)))].mean(axis=1)
        rows.append({"inputs": inputs, "model": model, "mae_mah": float(abs_error.mean()),
                     "ci95_low": float(np.quantile(means, 0.025)), "ci95_high": float(np.quantile(means, 0.975)),
                     "mae_percent": float(np.mean(abs_error / y) * 100),
                     "strong_headwind_mae_mah": float(abs_error[strong].mean()),
                     "max_underprediction_mah": float(np.max(-error))})
    comparisons = []
    for model in models:
        for a, b in (("F3", "F0"), ("F3", "F1"), ("F3", "F2"), ("F2", "F0")):
            comparisons.append({"within": model, "first": a, "second": b,
                                **paired(predictions[(a, model)], predictions[(b, model)], y, rng)})
    for inputs in INPUTS:
        for a, b in ((network, "Ridge"), (network, "XGBoost"), ("Ridge", "XGBoost")):
            comparisons.append({"within": inputs, "first": a, "second": b,
                                **paired(predictions[(inputs, a)], predictions[(inputs, b)], y, rng)})

    OUT.mkdir(parents=True)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "cv_input_metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(OUT / "paired_comparisons.csv", index=False)
    per_flight = pd.DataFrame({"run_id": frame["run_id"], "actual_rtl_charge_mah": y, "headwind_m_s": headwind})
    for (inputs, model), values in predictions.items():
        per_flight[f"{inputs}__{model}"] = values
    per_flight.to_csv(OUT / "cv_predictions.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({
        "milestone": "M22_cv_input_comparison", "protocol": "DECISIONS.md D033", "flights": len(y),
        "strong_headwind_flights": int(strong.sum()), "network_architecture": list(hidden),
        "xgboost_parameters": XGB, "results": rows, "comparisons": comparisons}, indent=2) + "\n")
    print(table.pivot(index="inputs", columns="model", values="mae_mah").round(1).to_string())
    print(table.pivot(index="inputs", columns="model", values="strong_headwind_mae_mah").round(1).to_string())
    print(pd.DataFrame(comparisons).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
