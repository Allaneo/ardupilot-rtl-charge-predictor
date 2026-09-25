#!/usr/bin/env python3
"""D032 / M21: final model on all 349 flights with nested cross-validation.

Procedure being evaluated: among Ridge and three networks (three-seed means) on
F3 tilt-limit inputs, pick the candidate with the lowest inner five-fold
cross-validated MAE, then refit it. The outer five-fold loop estimates how well
that whole procedure predicts flights it never saw. The final model is the same
procedure applied to all 349 flights and is saved for later use.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtl_charge.modeling import TARGET, fit_dense_regressor, fit_ridge  # noqa: E402
from rtl_charge.schema import load_feature_sets  # noqa: E402
from cv_architecture_sweep import fit_network, fold_features, shape_name  # noqa: E402
from evaluate_confirmation import CONFIRMATION, DEVELOPMENT, FOLD_SEED, NEURAL, SEEDS, SPEED_LABELS  # noqa: E402
from tilt_limit_speed_check import SATURATION_FRACTION, TiltLimitSpeed  # noqa: E402
from validate_controller_aware_features import SPEED_TARGET, make_consistent_features  # noqa: E402


CONFIRMATION_SPEED = ROOT / "reports/m19_confirmation_speed/rtl_speed_confirmation.csv"
OUT = ROOT / "reports/m21_nested_cv_final_model"
MODEL_OUT = ROOT / "models/m21_final"
SHAPES = ((256,), (1024,), (64, 32, 16))
CANDIDATES = ("Ridge", *(f"network {shape_name(shape)}" for shape in SHAPES))
BOOTSTRAPS = 10_000
WORKERS = 4


def load_all() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    frame = pd.concat([pd.read_csv(path) for path in (*DEVELOPMENT, CONFIRMATION)], ignore_index=True)
    if len(frame) != 349 or frame["run_id"].duplicated().any():
        raise ValueError("expected 349 unique flights")
    labels = pd.concat([pd.read_csv(path) for path in (*SPEED_LABELS, CONFIRMATION_SPEED)])
    labels = labels.set_index("run_id", verify_integrity=True).loc[frame["run_id"]]
    return frame, labels[SPEED_TARGET].to_numpy(dtype=np.float64), \
        labels["lean_near_30deg_fraction"].to_numpy() > SATURATION_FRACTION


def predict_candidates(pool, frame, measured, saturated, y, f1_columns, train_idx, test_idx) -> dict[str, np.ndarray]:
    """Fit every candidate on train_idx and predict test_idx (indices into the full frame)."""
    x_train, x_test = fold_features(frame, measured, saturated, train_idx, test_idx, f1_columns)
    result = {"Ridge": fit_ridge(1.0).fit(x_train, y[train_idx]).predict(x_test)}
    jobs = [(0, shape, seed, x_train, y[train_idx], x_test) for shape in SHAPES for seed in SEEDS]
    by_shape: dict[str, list[np.ndarray]] = {}
    for _, shape, _, _, values in pool.map(fit_network, jobs):
        by_shape.setdefault(f"network {shape_name(shape)}", []).append(values)
    result.update({name: np.mean(values, axis=0) for name, values in by_shape.items()})
    return result


def select(pool, frame, measured, saturated, y, f1_columns, idx) -> tuple[str, dict[str, float]]:
    """Inner five-fold CV over idx; return the candidate with the lowest MAE."""
    oof = {name: np.empty(len(idx)) for name in CANDIDATES}
    for inner_fit, inner_test in KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED).split(idx):
        predictions = predict_candidates(pool, frame, measured, saturated, y, f1_columns, idx[inner_fit], idx[inner_test])
        for name, values in predictions.items():
            oof[name][inner_test] = values
    maes = {name: float(np.mean(np.abs(values - y[idx]))) for name, values in oof.items()}
    return min(maes, key=maes.get), maes


def bootstrap_mae(errors: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    values = np.abs(errors)
    means = values[rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))].mean(axis=1)
    return {"estimate": float(values.mean()), "ci95_low": float(np.quantile(means, 0.025)),
            "ci95_high": float(np.quantile(means, 0.975))}


def summarize(y, predicted, headwind, source, rng) -> dict[str, object]:
    error = predicted - y
    under = np.maximum(-error, 0.0)
    row = {"mae": bootstrap_mae(error, rng), "rmse_mah": float(np.sqrt(np.mean(error**2))),
           "mean_signed_error_mah": float(error.mean()), "p95_underprediction_mah": float(np.quantile(under, 0.95)),
           "max_underprediction_mah": float(under.max()), "mae_percent_of_charge": float(np.mean(np.abs(error) / y) * 100)}
    bands = {"tailwind": headwind < 0, "0-2": (headwind >= 0) & (headwind < 2), "2-4": (headwind >= 2) & (headwind < 4),
             "4+": headwind >= 4}
    row["by_headwind"] = {name: {"n": int(mask.sum()), "mae_mah": float(np.mean(np.abs(error[mask])))}
                          for name, mask in bands.items()}
    row["by_source"] = {name: {"n": int((source == name).sum()), "mae_mah": float(np.mean(np.abs(error[source == name])))}
                        for name in np.unique(source)}
    return row


def main() -> None:
    if OUT.exists() or MODEL_OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {OUT} or {MODEL_OUT}")
    f1_columns = load_feature_sets(ROOT / "config/feature_sets.yaml")["f1_vehicle_symptoms"]
    frame, measured, saturated = load_all()
    y = frame[TARGET].to_numpy(dtype=np.float64)
    headwind = frame["headwind_m_s"].to_numpy(dtype=np.float64)
    source = np.where(frame["run_id"].str.startswith("confirm-"), "confirmation",
                      np.where(frame["run_id"].str.startswith("train-h-"), "headwind_training", "development"))
    all_idx = np.arange(len(y))

    nested = np.empty(len(y))
    fixed = {name: np.empty(len(y)) for name in CANDIDATES}
    folds = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for fold, (outer_train, outer_test) in enumerate(KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED).split(all_idx)):
            chosen, inner_maes = select(pool, frame, measured, saturated, y, f1_columns, outer_train)
            outer = predict_candidates(pool, frame, measured, saturated, y, f1_columns, outer_train, outer_test)
            nested[outer_test] = outer[chosen]
            for name, values in outer.items():
                fixed[name][outer_test] = values
            folds.append({"fold": fold, "chosen": chosen, "inner_cv_mae_mah": inner_maes,
                          "outer_mae_mah": float(np.mean(np.abs(outer[chosen] - y[outer_test])))})
            print(f"outer fold {fold}: chose {chosen}; outer MAE {folds[-1]['outer_mae_mah']:.1f}", flush=True)

        final_choice, final_inner = select(pool, frame, measured, saturated, y, f1_columns, all_idx)

        # Refit the chosen candidate on all 349 flights and save it with its speed model.
        speed_oof = np.empty(len(y))
        for fit_rows, predict_rows in KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED).split(all_idx):
            model = TiltLimitSpeed().fit(frame.iloc[fit_rows], measured[fit_rows], saturated[fit_rows])
            speed_oof[predict_rows] = model.predict(frame.iloc[predict_rows])
        speed_model = TiltLimitSpeed().fit(frame, measured, saturated)
        x_all = make_consistent_features(frame, speed_oof, f1_columns)
        if final_choice == "Ridge":
            charge_models = [fit_ridge(1.0).fit(x_all, y)]
        else:
            shape = SHAPES[[f"network {shape_name(s)}" for s in SHAPES].index(final_choice)]
            x = x_all.to_numpy(dtype=np.float64)
            charge_models = []
            for seed in SEEDS:
                # Same recipe as the CV folds: early-stopping epoch from a fixed 20%, then refit on all flights.
                _, hidden, _, epoch, _ = fit_network((0, shape, seed, x, y, x[:1]))
                model, _ = fit_dense_regressor(x, y, x, y, hidden_layers=hidden, seed=seed,
                                               max_epochs=epoch, fixed_epochs=True, **NEURAL)
                charge_models.append(model)

    rng = np.random.default_rng(FOLD_SEED)
    results = {"nested_selected": summarize(y, nested, headwind, source, rng)}
    results.update({f"fixed_{name}": summarize(y, values, headwind, source, rng) for name, values in fixed.items()})

    OUT.mkdir(parents=True)
    MODEL_OUT.mkdir(parents=True)
    joblib.dump({"speed_model": speed_model, "charge_models": charge_models, "choice": final_choice,
                 "feature_columns": list(x_all.columns), "f1_columns": list(f1_columns),
                 "usage": "features = make_consistent_features(frame, speed_model.predict(frame), f1_columns); "
                          "prediction = mean of charge_models' predictions"},
                MODEL_OUT / "final_model.joblib")
    predictions = pd.DataFrame({"run_id": frame["run_id"], "source": source, "actual_rtl_charge_mah": y,
                                "headwind_m_s": headwind, "nested_selected": nested,
                                **{f"fixed_{name}": values for name, values in fixed.items()}})
    predictions.to_csv(OUT / "outer_predictions.csv", index=False)
    summary = {"milestone": "M21_nested_cv_final_model", "protocol": "DECISIONS.md D032",
               "flights": len(y), "candidates": list(CANDIDATES), "seeds": list(SEEDS),
               "outer_folds": folds, "final_choice": final_choice, "final_inner_cv_mae_mah": final_inner,
               "tilt_model": {"intercept_m_s": float(speed_model.model.intercept_),
                              "throttle_coefficient_m_s": float(speed_model.model.coef_[0])},
               "caveat": "Candidate set chosen after M20, which used 289 of these 349 flights; nested estimate slightly optimistic.",
               "results": results}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"final choice on all 349 flights: {final_choice}")
    print(json.dumps({k: {"mae": v["mae"], "max_under": v["max_underprediction_mah"], "by_headwind": v["by_headwind"]}
                      for k, v in results.items()}, indent=1))


if __name__ == "__main__":
    main()
