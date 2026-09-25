#!/usr/bin/env python3
"""Validation-only check: is cruise-speed prediction the F3 bottleneck?

Builds F3 (cubic, learned speed) exactly as in the M13 sweep, then rebuilds it
with the measured middle-route RTL ground speed in place of the predicted speed
for both training and validation flights. The measured speed is only known
after the return, so this is a simulator-only oracle: it bounds how much a
perfect speed predictor could help. Uses the 172 training and 39 validation
flights; no test flight is read.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtl_charge.modeling import fit_dense_regressor, fit_ridge, load_train_validation  # noqa: E402
from rtl_charge.schema import load_feature_sets  # noqa: E402
from sweep_input_architectures import speed_predictions  # noqa: E402
from validate_controller_aware_features import SPEED_INPUTS, SPEED_TARGET, make_consistent_features  # noqa: E402


SPEED_LABELS = ROOT / "reports/m9/rtl_speed_train_validation.csv"
M13_RUNS = ROOT / "reports/m13_input_architecture_sweep/runs_summary.csv"
OUT = ROOT / "reports/m17_speed_oracle"
SEEDS = (20260924, 20260925, 20260926)
HEADWIND_BANDS = (2.0, 4.0)


def metrics(actual: np.ndarray, predicted: np.ndarray, headwind: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    under = actual - predicted
    result = {
        "mae_mah": float(np.mean(np.abs(error))),
        "rmse_mah": float(np.sqrt(np.mean(error**2))),
        "max_underprediction_mah": float(under.max()),
        "p95_underprediction_mah": float(np.quantile(np.maximum(under, 0.0), 0.95)),
    }
    for band in HEADWIND_BANDS:
        mask = headwind >= band
        result[f"headwind_ge_{band:g}_n"] = int(mask.sum())
        result[f"headwind_ge_{band:g}_mae_mah"] = float(np.mean(np.abs(error[mask])))
    return result


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing report: {OUT}")
    feature_sets = load_feature_sets(ROOT / "config/feature_sets.yaml")
    split = load_train_validation(ROOT / "data/processed/pipeline_dataset.csv", feature_sets["f2_physics_informed"])
    train, validation = split.train_x.copy(), split.validation_x.copy()
    train.index = pd.Index(split.train_run_ids, name="run_id")
    validation.index = pd.Index(split.validation_run_ids, name="run_id")
    labels = pd.read_csv(SPEED_LABELS).set_index("run_id", verify_integrity=True)
    if set(labels.index) != set(train.index) | set(validation.index):
        raise ValueError("speed labels must match exactly the train/validation flights")
    f1_columns = feature_sets["f1_vehicle_symptoms"]
    headwind = validation["headwind_m_s"].to_numpy(dtype=np.float64)

    predicted_train, predicted_validation = speed_predictions(train, validation, labels, SPEED_INPUTS)
    measured_train = labels.loc[train.index, SPEED_TARGET].to_numpy(dtype=np.float64)
    measured_validation = labels.loc[validation.index, SPEED_TARGET].to_numpy(dtype=np.float64)
    variants = {
        "f3_predicted_speed": (predicted_train, predicted_validation),
        "f3_measured_speed_oracle": (measured_train, measured_validation),
    }

    rows = []
    per_flight = pd.DataFrame({
        "run_id": validation.index, "actual_rtl_charge_mah": split.validation_y, "headwind_m_s": headwind,
        "predicted_speed_m_s": predicted_validation, "measured_speed_m_s": measured_validation,
        "lean_near_30deg_fraction": labels.loc[validation.index, "lean_near_30deg_fraction"].to_numpy(),
    })
    for variant, (speed_train, speed_validation) in variants.items():
        train_x = make_consistent_features(train, speed_train, f1_columns)
        validation_x = make_consistent_features(validation, speed_validation, f1_columns)
        ridge = fit_ridge(alpha=1.0).fit(train_x, split.train_y).predict(validation_x)
        rows.append({"variant": variant, "model": "ridge", "seed": None, **metrics(split.validation_y, ridge, headwind)})
        per_flight[f"{variant}__ridge"] = ridge
        neural = []
        for seed in SEEDS:
            model, epoch = fit_dense_regressor(
                train_x.to_numpy(dtype=np.float64), split.train_y,
                validation_x.to_numpy(dtype=np.float64), split.validation_y,
                hidden_layers=(1024,), seed=seed, loss="huber", alpha=0.001,
                learning_rate=0.001, max_epochs=600, patience=45,
            )
            values = model.predict(validation_x.to_numpy(dtype=np.float64))
            neural.append(values)
            rows.append({"variant": variant, "model": "neural_1024", "seed": seed, "selected_epoch": epoch,
                         **metrics(split.validation_y, values, headwind)})
        mean = np.mean(neural, axis=0)
        rows.append({"variant": variant, "model": "neural_1024_seed_mean", "seed": None,
                     **metrics(split.validation_y, mean, headwind)})
        per_flight[f"{variant}__neural_1024_seed_mean"] = mean

    # The predicted-speed variant must reproduce the saved M13 sweep.
    m13 = pd.read_csv(M13_RUNS)
    m13 = m13.loc[m13["variant"] == "f3_cubic_learned"]
    saved_ridge = float(m13.loc[m13["model"] == "ridge", "validation_mae_mah"].iloc[0])
    table = pd.DataFrame(rows)
    ours = float(table.loc[(table["variant"] == "f3_predicted_speed") & (table["model"] == "ridge"), "mae_mah"].iloc[0])
    if abs(ours - saved_ridge) > 1e-6:
        raise ValueError(f"predicted-speed F3 Ridge {ours} does not reproduce M13 {saved_ridge}")

    OUT.mkdir(parents=True)
    table.to_csv(OUT / "validation_metrics.csv", index=False)
    per_flight.to_csv(OUT / "validation_predictions.csv", index=False)
    summary = {
        "milestone": "M17_speed_oracle_validation",
        "train_flights": len(train), "validation_flights": len(validation), "test_flights_read": 0,
        "oracle": "measured middle-route RTL ground speed replaces predicted speed in all F3 speed-derived inputs, for training and validation rows",
        "caveat": "Measured speed is known only after the return; simulator-only upper bound. Neural epochs use validation early stopping as in M13, for both variants.",
        "speed_mae_validation_m_s": float(np.mean(np.abs(predicted_validation - measured_validation))),
        "m13_ridge_reproduced_mae_mah": saved_ridge,
        "results": rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(table.drop(columns=["selected_epoch"], errors="ignore").round(1).to_string(index=False))
    worst = per_flight.assign(
        pred_under=lambda f: f["actual_rtl_charge_mah"] - f["f3_predicted_speed__ridge"],
        oracle_under=lambda f: f["actual_rtl_charge_mah"] - f["f3_measured_speed_oracle__ridge"],
    ).sort_values("pred_under", ascending=False).head(6)
    print("\nLargest Ridge underpredictions (predicted-speed F3):")
    print(worst[["run_id", "headwind_m_s", "predicted_speed_m_s", "measured_speed_m_s", "lean_near_30deg_fraction",
                 "actual_rtl_charge_mah", "pred_under", "oracle_under"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
