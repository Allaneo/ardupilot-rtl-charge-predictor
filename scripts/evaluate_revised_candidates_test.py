#!/usr/bin/env python3
"""Post-hoc test check for predeclared learned-speed candidates.

The original M7 test set has already been used for the historical models. This
script therefore reports an informative post-hoc check and refuses to overwrite
the frozen reports/m7 directory.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtl_charge.modeling import fit_dense_regressor, fit_ridge, regression_metrics  # noqa: E402
from rtl_charge.revised_features import consistent_speed_f2  # noqa: E402
from rtl_charge.schema import load_feature_sets  # noqa: E402
from validate_controller_aware_features import SPEED_INPUTS, SPEED_SETPOINT, speed_model, make_consistent_features  # noqa: E402


DATASET = ROOT / "data/processed/pipeline_dataset.csv"
SPEED_LABELS = ROOT / "reports/m9/rtl_speed_train_validation.csv"
OUT = ROOT / "reports/m14_revised_candidates_test"
SEEDS = (20260924, 20260925, 20260926)
F3_EPOCHS = {20260924: 321, 20260925: 409, 20260926: 421}
F2_EPOCHS = {20260924: 467, 20260925: 441, 20260926: 467}
BOOTSTRAPS = 10_000
RNG_SEED = 20260923


def bootstrap(values: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    draws = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[draws].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def test_metrics(actual: np.ndarray, predicted: np.ndarray, headwind: np.ndarray) -> dict[str, object]:
    result = regression_metrics(actual, predicted)
    errors = predicted - actual
    high = headwind >= 2.0
    return {
        **result,
        "max_underprediction_mah": float(np.max(actual - predicted)),
        "mean_underprediction_mah": float(np.mean(np.maximum(actual - predicted, 0.0))),
        "p95_underprediction_mah": float(np.quantile(np.maximum(actual - predicted, 0.0), 0.95)),
        "high_headwind_n": int(high.sum()),
        "high_headwind_mae_mah": float(np.mean(np.abs(errors[high] if high.any() else errors))),
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing report: {OUT}")
    sets = load_feature_sets(ROOT / "config/feature_sets.yaml")
    frame = pd.read_csv(DATASET)
    trainval = frame.loc[frame["split"].isin(["train", "validation"])].copy()
    test = frame.loc[frame["split"] == "test"].copy()
    if len(trainval) != 211 or len(test) != 39:
        raise ValueError(f"frozen split changed: trainval={len(trainval)} test={len(test)}")
    if trainval["run_id"].duplicated().any() or test["run_id"].duplicated().any():
        raise ValueError("run IDs must be unique")

    speed = pd.read_csv(SPEED_LABELS).set_index("run_id", verify_integrity=True)
    if set(speed.index) != set(trainval["run_id"]):
        raise ValueError("speed labels must contain exactly the 211 train/validation flights")
    speed = speed.loc[trainval["run_id"]]
    speed_x = trainval.loc[:, list(SPEED_INPUTS)].to_numpy(dtype=np.float64)
    speed_y = speed["middle_route_ground_speed_m_s"].to_numpy(dtype=np.float64)
    folds = KFold(n_splits=5, shuffle=True, random_state=RNG_SEED)
    train_speed_oof = np.clip(cross_val_predict(speed_model(), speed_x, speed_y, cv=folds, n_jobs=1), 1.0, SPEED_SETPOINT)
    test_speed_model = speed_model().fit(speed_x, speed_y)
    test_speed = np.clip(test_speed_model.predict(test.loc[:, list(SPEED_INPUTS)].to_numpy(dtype=np.float64)), 1.0, SPEED_SETPOINT)

    f1_columns = sets["f1_vehicle_symptoms"]
    f2_columns = sets["f2_physics_informed"]
    train_f2 = consistent_speed_f2(trainval, f2_columns, train_speed_oof)
    test_f2 = consistent_speed_f2(test, f2_columns, test_speed)
    train_f3 = make_consistent_features(trainval, train_speed_oof, f1_columns)
    test_f3 = make_consistent_features(test, test_speed, f1_columns)
    for name, pair in (("f2", (train_f2, test_f2)), ("f3", (train_f3, test_f3))):
        if tuple(pair[0].columns) != tuple(pair[1].columns) or not np.isfinite(pair[0].to_numpy()).all() or not np.isfinite(pair[1].to_numpy()).all():
            raise ValueError(f"invalid {name} revised inputs")

    y_trainval = trainval["rtl_charge_mah"].to_numpy(dtype=np.float64)
    y_test = test["rtl_charge_mah"].to_numpy(dtype=np.float64)
    headwind = test["headwind_m_s"].to_numpy(dtype=np.float64)
    predictions: list[dict[str, object]] = []

    def add(name: str, predicted: np.ndarray, metadata: dict[str, object]) -> None:
        predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
        if len(predicted) != len(y_test) or not np.isfinite(predicted).all():
            raise ValueError(f"invalid predictions for {name}")
        predictions.append({"name": name, "predicted": predicted, **metadata})

    # Frozen historical reference, scored by the original M7 artifact.
    import joblib
    frozen = joblib.load(ROOT / "models/m6/ridge__f2_physics_informed.joblib")["model"]
    add("frozen_m7_f2_ridge", frozen.predict(test.loc[:, f2_columns]), {"feature_set": "historical_f2", "model": "ridge"})

    for label, train_x, test_x, epochs in (
        ("f3_learned", train_f3, test_f3, F3_EPOCHS),
        ("f2_learned", train_f2, test_f2, F2_EPOCHS),
    ):
        ridge = fit_ridge(alpha=1.0).fit(train_x, y_trainval)
        add(f"{label}_ridge", ridge.predict(test_x), {"feature_set": label, "model": "ridge"})
        for seed in SEEDS:
            model, _ = fit_dense_regressor(
                train_x.to_numpy(dtype=np.float64), y_trainval,
                train_x.to_numpy(dtype=np.float64), y_trainval,
                hidden_layers=(1024,) if label == "f3_learned" else (256,),
                seed=seed, loss="huber", alpha=0.001, learning_rate=0.001,
                max_epochs=epochs[seed], fixed_epochs=True,
            )
            add(f"{label}_neural_{seed}", model.predict(test_x.to_numpy(dtype=np.float64)), {
                "feature_set": label, "model": "neural", "seed": seed,
                "architecture": [1024] if label == "f3_learned" else [256],
                "fixed_epochs": epochs[seed],
            })

    rng = np.random.default_rng(RNG_SEED)
    reference = next(item for item in predictions if item["name"] == "frozen_m7_f2_ridge")
    rows = []
    paired = []
    prediction_frame = pd.DataFrame({"run_id": test["run_id"].to_numpy(), "actual_rtl_charge_mah": y_test,
                                     "headwind_m_s": headwind})
    for item in predictions:
        values = item["predicted"]
        stats = test_metrics(y_test, values, headwind)
        rows.append({"name": item["name"], **{key: value for key, value in item.items() if key not in {"name", "predicted"}}, **stats,
                     "mae_bootstrap_95": bootstrap(np.abs(values - y_test), rng)})
        prediction_frame[item["name"]] = values
        diff = np.abs(values - y_test) - np.abs(reference["predicted"] - y_test)
        paired.append({"name": item["name"], "reference": reference["name"], **bootstrap(diff, rng),
                       "interpretation": "negative favors candidate; descriptive post-hoc comparison"})

    OUT.mkdir(parents=True)
    prediction_frame.to_csv(OUT / "test_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "test_metrics.csv", index=False)
    pd.DataFrame(paired).to_csv(OUT / "paired_vs_frozen_m7_f2_ridge.csv", index=False)
    summary = {
        "milestone": "M14_posthoc_revised_candidate_test",
        "train_validation_flights": len(trainval), "test_flights": len(test),
        "test_labels_used": True, "test_speed_labels_used": False,
        "candidate_selection_on_test": False,
        "posthoc_caveat": "The 39 test flights were already used once for M7 historical models; this is informative revised-candidate evidence, not a fresh untouched-study claim.",
        "speed_model": "random forest fit on 211 train/validation speed labels; test speed labels excluded",
        "train_speed_oof_mae_m_s": float(np.mean(np.abs(train_speed_oof - speed_y))),
        "test_speed_predictions": {"min_m_s": float(test_speed.min()), "max_m_s": float(test_speed.max())},
        "f3_epochs": F3_EPOCHS, "f2_epochs": F2_EPOCHS,
        "results": rows, "paired_vs_frozen_m7_f2_ridge": paired,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(pd.DataFrame(rows)[["name", "mae_mah", "mean_underprediction_mah", "p95_underprediction_mah"]].to_string(index=False))


if __name__ == "__main__":
    main()
