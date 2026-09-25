#!/usr/bin/env python3
"""Refit every frozen M7 candidate on the 211 train/validation flights.

M7 scored models fitted on the 172 training flights; M14 refitted its revised
candidates on all 211 train/validation flights. This script gives the 14 M7
candidates the same treatment with their frozen M6 settings: Ridge alpha,
XGBoost parameters and seed, and the neural seed with its validation-selected
epoch as a fixed epoch count. Before each refit, the same settings are fitted
on the 172 training flights and must reproduce the frozen artifact's test
predictions. Nominal time x current and the PX4-style baseline have no fitted
parameters and are carried over unchanged. No candidate is reselected.
"""

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
    fit_dense_regressor,
    fit_ridge,
    fit_xgboost,
    make_true_wind_oracle,
    px4_style_charge_baseline,
)
from rtl_charge.schema import load_feature_sets  # noqa: E402
from evaluate_revised_candidates_test import RNG_SEED, bootstrap, test_metrics  # noqa: E402


DATASET = ROOT / "data/processed/pipeline_dataset.csv"
MODELS = ROOT / "models/m6"
M6_SUMMARY = ROOT / "reports/m6/summary.json"
M14_PREDICTIONS = ROOT / "reports/m14_revised_candidates_test/test_predictions.csv"
OUT = ROOT / "reports/m16_equal_data_all_candidates"
REFERENCE = "ridge__f2_physics_informed"
F2 = "f2_physics_informed"
# Ridge and the NumPy network reproduce to float64 precision. XGBoost predicts
# in float32, and the F2 refit drifts by up to 0.89 mAh per flight (0.03 mAh in
# test MAE) with identical parameters; the drift is recorded in the summary.
REPRODUCTION_TOLERANCE_MAH = 1e-6
XGBOOST_REPRODUCTION_TOLERANCE_MAH = 1.0


def hyperparameters(entry: dict) -> dict:
    value = entry["hyperparameters"]
    return json.loads(value) if isinstance(value, str) else dict(value)


def fit_candidate(name: str, params: dict, x: pd.DataFrame, y: np.ndarray):
    """Fit one frozen candidate on (x, y) and return an object with predict()."""
    if name.startswith("ridge"):
        return fit_ridge(float(params["alpha"])).fit(x, y)
    if name.startswith("xgboost"):
        model_params = {key: value for key, value in params.items() if key != "copied_from"}
        seed = int(model_params.pop("random_state"))
        return fit_xgboost(model_params, seed=seed).fit(x, y, verbose=False)
    if name.startswith("neural_network"):
        epochs = int(params.get("best_epoch", params.get("best_epoch_selected_on_validation")))
        values = x.to_numpy(dtype=np.float64)
        model, _ = fit_dense_regressor(
            values, y, values, y,
            loss=str(params["loss"]), alpha=float(params["alpha"]),
            learning_rate=float(params["learning_rate"]), seed=int(params["random_seed"]),
            max_epochs=epochs, fixed_epochs=True,
        )
        return model
    raise ValueError(f"unsupported candidate: {name}")


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing report: {OUT}")
    feature_sets = load_feature_sets(ROOT / "config/feature_sets.yaml")
    frame = pd.read_csv(DATASET)
    train = frame.loc[frame["split"] == "train"]
    trainval = frame.loc[frame["split"].isin(["train", "validation"])]
    test = frame.loc[frame["split"] == "test"]
    if (len(train), len(trainval), len(test)) != (172, 211, 39):
        raise ValueError("frozen split changed")
    selected = json.loads(M6_SUMMARY.read_text())["selected_validation_results"]
    y_train = train[TARGET].to_numpy(dtype=np.float64)
    y_trainval = trainval[TARGET].to_numpy(dtype=np.float64)
    y_test = test[TARGET].to_numpy(dtype=np.float64)
    headwind = test["headwind_m_s"].to_numpy(dtype=np.float64)

    # Candidate name -> (artifact file, matrices built for train, trainval, test).
    f2_columns = feature_sets[F2]
    matrices = {name: tuple(part.loc[:, columns] for part in (train, trainval, test)) for name, columns in feature_sets.items()}
    oracle_matrices = tuple(make_true_wind_oracle(part, f2_columns) for part in (train, trainval, test))
    candidates = {}
    for feature_set in feature_sets:
        for model in ("ridge", "neural_network", "xgboost"):
            name = f"{model}__{feature_set}"
            candidates[name] = (MODELS / f"{name}.joblib", matrices[feature_set], hyperparameters(selected[name]))
    for model in ("neural_network", "xgboost"):
        candidates[f"{model}_true_wind_oracle"] = (
            MODELS / f"{model}__f2_true_wind_oracle.joblib", oracle_matrices,
            hyperparameters(selected[f"{model}_true_wind_oracle"]),
        )

    frozen_predictions = {}
    refit_predictions = {}
    reproduction = {}
    for name, (artifact, (x_train, x_trainval, x_test), params) in candidates.items():
        frozen = joblib.load(artifact)["model"].predict(x_test)
        reproduced = fit_candidate(name, params, x_train, y_train).predict(x_test)
        error = float(np.max(np.abs(np.asarray(reproduced).reshape(-1) - np.asarray(frozen).reshape(-1))))
        tolerance = XGBOOST_REPRODUCTION_TOLERANCE_MAH if name.startswith("xgboost") else REPRODUCTION_TOLERANCE_MAH
        if error > tolerance:
            raise ValueError(f"{name}: frozen settings do not reproduce the artifact (max {error} mAh)")
        reproduction[name] = error
        frozen_predictions[name] = np.asarray(frozen, dtype=np.float64).reshape(-1)
        refit_predictions[name] = np.asarray(
            fit_candidate(name, params, x_trainval, y_trainval).predict(x_test), dtype=np.float64
        ).reshape(-1)

    for name, values in (
        ("training_mean", None),
        ("nominal_time_x_current", test["current_time_charge_baseline_mah"].to_numpy(dtype=np.float64)),
        ("px4_style_multicopter_time_x_current", px4_style_charge_baseline(test)),
    ):
        if name == "training_mean":
            frozen_predictions[name] = np.full(len(test), y_train.mean())
            refit_predictions[name] = np.full(len(test), y_trainval.mean())
        else:
            frozen_predictions[name] = refit_predictions[name] = np.asarray(values, dtype=np.float64)

    # Carry in the M14 revised candidates (already fitted on 211 flights) for one common table.
    saved = pd.read_csv(M14_PREDICTIONS)
    if not np.array_equal(saved["run_id"].to_numpy(), test["run_id"].to_numpy()):
        raise ValueError("M14 predictions are not aligned with the test rows")
    revised = [name for name in saved.columns if name.startswith(("f3_learned", "f2_learned"))]
    for name in revised:
        refit_predictions[name] = saved[name].to_numpy(dtype=np.float64)
    refit_predictions["f3_learned_neural_seed_mean"] = np.mean(
        [refit_predictions[name] for name in revised if name.startswith("f3_learned_neural_")], axis=0
    )
    refit_predictions["f2_learned_neural_seed_mean"] = np.mean(
        [refit_predictions[name] for name in revised if name.startswith("f2_learned_neural_")], axis=0
    )

    rng = np.random.default_rng(RNG_SEED)
    reference_error = np.abs(refit_predictions[REFERENCE] - y_test)
    rows = []
    for name, values in refit_predictions.items():
        stats = test_metrics(y_test, values, headwind)
        diff = np.abs(values - y_test) - reference_error
        rows.append({
            "name": name,
            "source": "M14 revised" if name.startswith(("f3_learned", "f2_learned")) else "M7 frozen settings",
            "frozen_172_mae_mah": float(np.mean(np.abs(frozen_predictions[name] - y_test))) if name in frozen_predictions else None,
            **stats,
            "mae_bootstrap_95": bootstrap(np.abs(values - y_test), rng),
            "paired_vs_reference": {**bootstrap(diff, rng), "flights_closer_than_reference": int(np.sum(diff < 0))},
        })

    OUT.mkdir(parents=True)
    predictions = test.loc[:, ["run_id", TARGET, "headwind_m_s"]].rename(columns={TARGET: "actual_rtl_charge_mah"})
    for name, values in refit_predictions.items():
        predictions[name] = values
    predictions.to_csv(OUT / "test_predictions_211.csv", index=False)
    flat = pd.json_normalize(rows, sep="__")
    flat.to_csv(OUT / "test_metrics_211.csv", index=False)
    summary = {
        "milestone": "M16_equal_data_all_candidates",
        "training_flights": len(trainval), "test_flights": len(test),
        "paired_reference": f"{REFERENCE} refitted on 211 flights",
        "reproduction_max_abs_mah": reproduction,
        "candidate_selection_on_test": False,
        "unchanged_baselines": ["nominal_time_x_current", "px4_style_multicopter_time_x_current"],
        "posthoc_caveat": "Same 39 test flights as M7/M14/M15; equal-data refits and seed means were added after earlier test results were seen.",
        "results": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"max reproduction difference: {max(reproduction.values()):.2e} mAh")
    view = flat[["name", "frozen_172_mae_mah", "mae_mah", "max_underprediction_mah", "high_headwind_mae_mah",
                 "paired_vs_reference__estimate", "paired_vs_reference__ci95_low", "paired_vs_reference__ci95_high"]]
    print(view.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
