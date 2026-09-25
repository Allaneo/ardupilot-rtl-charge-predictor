#!/usr/bin/env python3
"""One-time D029 confirmation evaluation.

Trains the predeclared candidates on 289 flights (the 250 development flights
plus 39 completed D031 strong-headwind training flights) and scores them once on the
60 new confirmation flights. Candidates, speed models, epoch
rule and comparisons are fixed in DECISIONS.md (D029, D031); nothing here is tuned on
confirmation results. The script refuses to overwrite its output directory.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtl_charge.modeling import TARGET, fit_dense_regressor, fit_ridge, regression_metrics  # noqa: E402
from rtl_charge.schema import load_feature_sets  # noqa: E402
from tilt_limit_speed_check import SATURATION_FRACTION, TiltLimitSpeed  # noqa: E402
from validate_controller_aware_features import SPEED_INPUTS, SPEED_SETPOINT, SPEED_TARGET, make_consistent_features, speed_model  # noqa: E402


DEVELOPMENT = (ROOT / "data/processed/pipeline_dataset.csv", ROOT / "data/processed/headwind_training_dataset.csv")
CONFIRMATION = ROOT / "data/processed/confirmation_dataset.csv"
SPEED_LABELS = (ROOT / "reports/m19_training_speed_labels/rtl_speed_250.csv",
                ROOT / "reports/m19_training_speed_labels/rtl_speed_headwind_training.csv")
OUT = ROOT / "reports/m19_confirmation"
FOLD_SEED = 20260923
SEEDS = (20260924, 20260925, 20260926)
BOOTSTRAPS = 10_000
RNG_SEED = 20260930
# 250 development + 39 D031 flights; train-h-0025 failed automatic disarm twice and is excluded.
EXPECTED_DEVELOPMENT = 289
NEURAL = {"loss": "huber", "alpha": 0.001, "learning_rate": 0.001}
PRIMARY = (("C_f3_tilt_ridge", "A_f2_ridge"), ("D_f3_tilt_neural_mean", "B_f2_neural_mean"),
           ("C_f3_tilt_ridge", "E_f3_forest_ridge"))


def cross_fit_speed(make_model, fit, frame: pd.DataFrame, measured: np.ndarray, saturated: np.ndarray) -> np.ndarray:
    """Five-fold out-of-fold speed predictions for the training rows."""
    predicted = np.empty(len(frame))
    for fit_rows, predict_rows in KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED).split(frame):
        model = fit(make_model(), frame.iloc[fit_rows], measured[fit_rows], saturated[fit_rows])
        predicted[predict_rows] = predict(model, frame.iloc[predict_rows])
    return predicted


def fit_forest(model, frame, measured, _saturated):
    return model.fit(frame.loc[:, list(SPEED_INPUTS)].to_numpy(dtype=np.float64), measured)


def fit_tilt(model, frame, measured, saturated):
    return model.fit(frame, measured, saturated)


def predict(model, frame: pd.DataFrame) -> np.ndarray:
    if isinstance(model, TiltLimitSpeed):
        return model.predict(frame)
    values = model.predict(frame.loc[:, list(SPEED_INPUTS)].to_numpy(dtype=np.float64))
    return np.clip(values, 1.0, SPEED_SETPOINT)


def cv_epochs(x: np.ndarray, y: np.ndarray, hidden: tuple[int, ...], seed: int) -> tuple[int, list[int]]:
    """Median early-stopping epoch over five training folds (no confirmation data)."""
    best = []
    for fit_rows, stop_rows in KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED).split(x):
        _, epoch = fit_dense_regressor(x[fit_rows], y[fit_rows], x[stop_rows], y[stop_rows],
                                       hidden_layers=hidden, seed=seed, max_epochs=600, patience=45, **NEURAL)
        best.append(int(epoch))
    return max(1, int(np.median(best))), best


def neural_seeds(train_x: pd.DataFrame, y: np.ndarray, test_x: pd.DataFrame, hidden: tuple[int, ...]):
    x = train_x.to_numpy(dtype=np.float64)
    predictions, epochs = {}, {}
    for seed in SEEDS:
        epoch, folds = cv_epochs(x, y, hidden, seed)
        model, _ = fit_dense_regressor(x, y, x, y, hidden_layers=hidden, seed=seed,
                                       max_epochs=epoch, fixed_epochs=True, **NEURAL)
        predictions[seed] = model.predict(test_x.to_numpy(dtype=np.float64))
        epochs[seed] = {"final_epochs": epoch, "fold_best_epochs": folds}
    return predictions, epochs


def bootstrap(values: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    draws = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[draws].mean(axis=1)
    return {"estimate": float(values.mean()), "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975))}


def summarize(actual, predicted, groups, headwind, rng) -> dict[str, object]:
    error = predicted - actual
    under = np.maximum(actual - predicted, 0.0)
    row = {**regression_metrics(actual, predicted),
           "mean_signed_error_mah": float(error.mean()),
           "mean_underprediction_mah": float(under.mean()),
           "p95_underprediction_mah": float(np.quantile(under, 0.95)),
           "max_underprediction_mah": float(under.max()),
           "mae_bootstrap_95": bootstrap(np.abs(error), rng)}
    for name, mask in (("group_h", groups == "h"), ("group_g", groups == "g"), ("est_headwind_ge_3", headwind >= 3.0)):
        row[f"{name}_n"] = int(mask.sum())
        row[f"{name}_mae_mah"] = float(np.mean(np.abs(error[mask])))
        row[f"{name}_max_underprediction_mah"] = float(under[mask].max())
    return row


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing report: {OUT}")
    sets = load_feature_sets(ROOT / "config/feature_sets.yaml")
    f1_columns, f2_columns = sets["f1_vehicle_symptoms"], sets["f2_physics_informed"]
    train = pd.concat([pd.read_csv(path) for path in DEVELOPMENT], ignore_index=True)
    test = pd.read_csv(CONFIRMATION)
    if len(train) != EXPECTED_DEVELOPMENT or train["run_id"].duplicated().any():
        raise ValueError(f"expected {EXPECTED_DEVELOPMENT} unique development flights")
    if not test["run_id"].str.startswith("confirm-").all() or test["run_id"].duplicated().any():
        raise ValueError("confirmation rows must be unique confirm-* flights")
    if set(train["run_id"]) & set(test["run_id"]):
        raise ValueError("development and confirmation flights overlap")
    labels = pd.concat([pd.read_csv(path) for path in SPEED_LABELS]).set_index("run_id", verify_integrity=True)
    if set(labels.index) != set(train["run_id"]):
        raise ValueError("speed labels must cover exactly the training flights")
    labels = labels.loc[train["run_id"]]
    measured = labels[SPEED_TARGET].to_numpy(dtype=np.float64)
    saturated = labels["lean_near_30deg_fraction"].to_numpy() > SATURATION_FRACTION
    y_train = train[TARGET].to_numpy(dtype=np.float64)
    y_test = test[TARGET].to_numpy(dtype=np.float64)
    groups = test["run_id"].str.slice(8, 9).to_numpy()
    headwind = test["headwind_m_s"].to_numpy(dtype=np.float64)

    # Speed models: out-of-fold for training rows, full fit for confirmation rows.
    speeds = {}
    for name, make_model, fit in (("forest", speed_model, fit_forest), ("tilt", TiltLimitSpeed, fit_tilt)):
        train_speed = cross_fit_speed(make_model, fit, train, measured, saturated)
        full = fit(make_model(), train, measured, saturated)
        speeds[name] = (train_speed, predict(full, test), full)
    tilt_model = speeds["tilt"][2].model

    f3_tilt = (make_consistent_features(train, speeds["tilt"][0], f1_columns),
               make_consistent_features(test, speeds["tilt"][1], f1_columns))
    f3_forest = (make_consistent_features(train, speeds["forest"][0], f1_columns),
                 make_consistent_features(test, speeds["forest"][1], f1_columns))
    f2 = (train.loc[:, f2_columns], test.loc[:, f2_columns])

    predictions = {
        "A_f2_ridge": fit_ridge(1.0).fit(f2[0], y_train).predict(f2[1]),
        "C_f3_tilt_ridge": fit_ridge(1.0).fit(f3_tilt[0], y_train).predict(f3_tilt[1]),
        "E_f3_forest_ridge": fit_ridge(1.0).fit(f3_forest[0], y_train).predict(f3_forest[1]),
        "nominal_time_x_current": test["current_time_charge_baseline_mah"].to_numpy(dtype=np.float64),
    }
    b_seeds, b_epochs = neural_seeds(f2[0], y_train, f2[1], (64, 32))
    d_seeds, d_epochs = neural_seeds(f3_tilt[0], y_train, f3_tilt[1], (1024,))
    predictions["B_f2_neural_mean"] = np.mean(list(b_seeds.values()), axis=0)
    predictions["D_f3_tilt_neural_mean"] = np.mean(list(d_seeds.values()), axis=0)
    for seed in SEEDS:
        predictions[f"B_f2_neural_seed_{seed}"] = b_seeds[seed]
        predictions[f"D_f3_tilt_neural_seed_{seed}"] = d_seeds[seed]
    for name, values in predictions.items():
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(values) != len(test) or not np.isfinite(values).all():
            raise ValueError(f"invalid predictions for {name}")
        predictions[name] = values

    rng = np.random.default_rng(RNG_SEED)
    metrics = [{"name": name, **summarize(y_test, values, groups, headwind, rng)} for name, values in predictions.items()]
    paired = []
    for candidate, reference in PRIMARY:
        diff = np.abs(predictions[candidate] - y_test) - np.abs(predictions[reference] - y_test)
        paired.append({"candidate": candidate, "reference": reference, "primary": True, **bootstrap(diff, rng),
                       "group_h": bootstrap(diff[groups == "h"], rng), "group_g": bootstrap(diff[groups == "g"], rng),
                       "flights_candidate_closer": int(np.sum(diff < 0))})

    OUT.mkdir(parents=True)
    frame = test.loc[:, ["run_id", TARGET, "headwind_m_s"]].rename(columns={TARGET: "actual_rtl_charge_mah"})
    frame["group"] = groups
    frame["forest_speed_m_s"] = speeds["forest"][1]
    frame["tilt_speed_m_s"] = speeds["tilt"][1]
    for name, values in predictions.items():
        frame[name] = values
    frame.to_csv(OUT / "confirmation_predictions.csv", index=False)
    pd.json_normalize(metrics, sep="__").to_csv(OUT / "confirmation_metrics.csv", index=False)
    pd.json_normalize(paired, sep="__").to_csv(OUT / "primary_comparisons.csv", index=False)
    summary = {
        "milestone": "M19_confirmation", "protocol": "DECISIONS.md D029",
        "training_flights": len(train), "confirmation_flights": len(test),
        "group_counts": {"h": int((groups == "h").sum()), "g": int((groups == "g").sum())},
        "tilt_model": {"intercept_m_s": float(tilt_model.intercept_), "throttle_coefficient_m_s": float(tilt_model.coef_[0]),
                       "saturated_training_flights": int(saturated.sum())},
        "neural_epochs": {"B_f2_64_32": b_epochs, "D_f3_tilt_1024": d_epochs},
        "evaluation_count": 1, "selection_on_confirmation": False,
        "metrics": metrics, "primary_comparisons": paired,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    view = pd.json_normalize(metrics, sep="__")[["name", "mae_mah", "rmse_mah", "max_underprediction_mah", "group_h_mae_mah", "group_g_mae_mah"]]
    print(view.round(1).to_string(index=False))
    print(pd.json_normalize(paired, sep="__")[["candidate", "reference", "estimate", "ci95_low", "ci95_high"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
