#!/usr/bin/env python3
"""D032 / M20: five-fold cross-validated architecture sweep on the 289 training flights.

F3 tilt-limit inputs; the 15 network shapes x 3 seeds plus Ridge. Everything
that learns from data (speed model, scaling, early stopping) is fitted inside
each outer fold, so each flight's prediction comes from models that never saw
it. Confirmation flights are not read.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtl_charge.modeling import NN_ARCHITECTURES, TARGET, fit_dense_regressor, fit_ridge  # noqa: E402
from rtl_charge.schema import load_feature_sets  # noqa: E402
from evaluate_confirmation import DEVELOPMENT, EXPECTED_DEVELOPMENT, FOLD_SEED, NEURAL, SEEDS, SPEED_LABELS  # noqa: E402
from tilt_limit_speed_check import SATURATION_FRACTION, TiltLimitSpeed  # noqa: E402
from validate_controller_aware_features import SPEED_TARGET, make_consistent_features  # noqa: E402


OUT = ROOT / "reports/m20_cv_architecture_sweep"
STOP_FRACTION = 0.2
BOOTSTRAPS = 10_000
WORKERS = 4


def shape_name(hidden: tuple[int, ...]) -> str:
    if len(set(hidden)) == 1 and len(hidden) > 1:
        return f"{len(hidden)}x{hidden[0]}"
    return "-".join(str(width) for width in hidden)


def load_training() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.concat([pd.read_csv(path) for path in DEVELOPMENT], ignore_index=True)
    if len(frame) != EXPECTED_DEVELOPMENT or frame["run_id"].duplicated().any():
        raise ValueError("expected the 289 unique training flights")
    if frame["run_id"].str.startswith("confirm-").any():
        raise ValueError("confirmation flights must not be in the sweep")
    labels = pd.concat([pd.read_csv(path) for path in SPEED_LABELS]).set_index("run_id", verify_integrity=True)
    labels = labels.loc[frame["run_id"]]
    return frame, labels[SPEED_TARGET].to_numpy(dtype=np.float64), \
        labels["lean_near_30deg_fraction"].to_numpy() > SATURATION_FRACTION


def fold_features(frame, measured, saturated, fit_rows, test_rows, f1_columns):
    """F3 tilt-limit inputs for one outer fold, fitted on the outer-training flights only."""
    train, test = frame.iloc[fit_rows], frame.iloc[test_rows]
    train_speed = np.empty(len(fit_rows))
    for inner_fit, inner_predict in KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED).split(train):
        model = TiltLimitSpeed().fit(train.iloc[inner_fit], measured[fit_rows][inner_fit], saturated[fit_rows][inner_fit])
        train_speed[inner_predict] = model.predict(train.iloc[inner_predict])
    test_speed = TiltLimitSpeed().fit(train, measured[fit_rows], saturated[fit_rows]).predict(test)
    return (make_consistent_features(train, train_speed, f1_columns).to_numpy(dtype=np.float64),
            make_consistent_features(test, test_speed, f1_columns).to_numpy(dtype=np.float64))


def fit_network(job):
    """Early-stop on a fixed 20% of the outer-training flights, then refit on all of them."""
    fold, hidden, seed, x_train, y_train, x_test = job
    fit_idx, stop_idx = train_test_split(np.arange(len(y_train)), test_size=STOP_FRACTION, random_state=FOLD_SEED)
    _, epoch = fit_dense_regressor(x_train[fit_idx], y_train[fit_idx], x_train[stop_idx], y_train[stop_idx],
                                   hidden_layers=hidden, seed=seed, max_epochs=600, patience=45, **NEURAL)
    epoch = max(1, int(epoch))
    model, _ = fit_dense_regressor(x_train, y_train, x_train, y_train, hidden_layers=hidden, seed=seed,
                                   max_epochs=epoch, fixed_epochs=True, **NEURAL)
    return fold, hidden, seed, epoch, model.predict(x_test)


def paired(errors_a: np.ndarray, errors_b: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    diff = np.abs(errors_a) - np.abs(errors_b)
    means = diff[rng.integers(0, len(diff), size=(BOOTSTRAPS, len(diff)))].mean(axis=1)
    return {"estimate": float(diff.mean()), "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975))}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing report: {OUT}")
    f1_columns = load_feature_sets(ROOT / "config/feature_sets.yaml")["f1_vehicle_symptoms"]
    frame, measured, saturated = load_training()
    y = frame[TARGET].to_numpy(dtype=np.float64)
    headwind = frame["headwind_m_s"].to_numpy(dtype=np.float64)
    folds = list(KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED).split(frame))

    ridge = np.empty(len(y))
    jobs = []
    fold_index = {}
    for fold, (fit_rows, test_rows) in enumerate(folds):
        x_train, x_test = fold_features(frame, measured, saturated, fit_rows, test_rows, f1_columns)
        ridge[test_rows] = fit_ridge(1.0).fit(x_train, y[fit_rows]).predict(x_test)
        fold_index[fold] = test_rows
        jobs += [(fold, hidden, seed, x_train, y[fit_rows], x_test) for hidden in NN_ARCHITECTURES for seed in SEEDS]

    predictions = {(shape_name(h), s): np.empty(len(y)) for h in NN_ARCHITECTURES for s in SEEDS}
    epochs = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for fold, hidden, seed, epoch, values in pool.map(fit_network, jobs):
            predictions[(shape_name(hidden), seed)][fold_index[fold]] = values
            epochs.setdefault(shape_name(hidden), []).append(epoch)

    rng = np.random.default_rng(FOLD_SEED)
    strong = headwind >= 4.0
    per_flight = pd.DataFrame({"run_id": frame["run_id"], "actual_rtl_charge_mah": y, "headwind_m_s": headwind,
                               "ridge": ridge})
    seed_means = {}
    rows = []
    for hidden in NN_ARCHITECTURES:
        name = shape_name(hidden)
        seeds = [predictions[(name, s)] for s in SEEDS]
        mean = np.mean(seeds, axis=0)
        seed_means[name] = mean
        per_flight[f"net_{name}"] = mean
        seed_maes = [float(np.mean(np.abs(values - y))) for values in seeds]
        rows.append({"model": f"network {name}", "shape": name,
                     "parameters": int(sum(a * b + b for a, b in zip((43, *hidden), (*hidden, 1)))),
                     "mae_mah": float(np.mean(np.abs(mean - y))),
                     "seed_mae_min": min(seed_maes), "seed_mae_max": max(seed_maes),
                     "strong_headwind_mae_mah": float(np.mean(np.abs(mean - y)[strong])),
                     "max_underprediction_mah": float(np.max(y - mean)),
                     "median_epochs": float(np.median(epochs[name]))})
    rows.append({"model": "Ridge", "shape": "linear", "parameters": 44,
                 "mae_mah": float(np.mean(np.abs(ridge - y))),
                 "strong_headwind_mae_mah": float(np.mean(np.abs(ridge - y)[strong])),
                 "max_underprediction_mah": float(np.max(y - ridge))})
    table = pd.DataFrame(rows)
    for reference, values in (("vs_1024", seed_means["1024"]), ("vs_ridge", ridge)):
        comparisons = []
        for row in rows:
            candidate = ridge if row["model"] == "Ridge" else seed_means[row["shape"]]
            comparisons.append(paired(candidate - y, values - y, rng))
        table[f"{reference}_estimate"] = [c["estimate"] for c in comparisons]
        table[f"{reference}_ci95_low"] = [c["ci95_low"] for c in comparisons]
        table[f"{reference}_ci95_high"] = [c["ci95_high"] for c in comparisons]

    OUT.mkdir(parents=True)
    table.to_csv(OUT / "cv_architecture_metrics.csv", index=False)
    per_flight.to_csv(OUT / "cv_predictions.csv", index=False)
    summary = {"milestone": "M20_cv_architecture_sweep", "protocol": "DECISIONS.md D032",
               "flights": len(y), "strong_headwind_flights": int(strong.sum()), "folds": 5,
               "seeds": list(SEEDS), "confirmation_flights_read": 0,
               "results": table.to_dict(orient="records")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    view = table[["model", "parameters", "mae_mah", "seed_mae_min", "seed_mae_max", "strong_headwind_mae_mah",
                  "max_underprediction_mah", "vs_1024_estimate", "vs_1024_ci95_low", "vs_1024_ci95_high",
                  "vs_ridge_estimate", "vs_ridge_ci95_low", "vs_ridge_ci95_high"]]
    print(view.sort_values("mae_mah").round(1).to_string(index=False))


if __name__ == "__main__":
    main()
