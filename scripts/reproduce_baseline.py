#!/usr/bin/env python3
"""Fit a small teaching baseline and predict one independently simulated RTL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rtl_charge.modeling import fit_ridge
from rtl_charge.schema import load_feature_sets


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-csv", type=Path, default=ROOT / "data/examples/train_f0_sample.csv")
    parser.add_argument("--flight-row", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    features = load_feature_sets(ROOT / "config/feature_sets.yaml")["f0_route_and_wind"]
    training = pd.read_csv(args.training_csv)
    flight = json.loads(args.flight_row.read_text())
    missing = set(features).difference(training.columns) | set(features).difference(flight)
    if missing:
        raise ValueError(f"missing F0 features: {sorted(missing)}")
    if "rtl_charge_mah" not in training or len(training) < 2:
        raise ValueError("training sample must contain at least two labeled rows")
    x = training.loc[:, features].astype(float)
    y = training["rtl_charge_mah"].to_numpy(dtype=float)
    new_x = pd.DataFrame([{name: float(flight[name]) for name in features}])
    if not np.isfinite(x.to_numpy()).all() or not np.isfinite(y).all() or not np.isfinite(new_x.to_numpy()).all():
        raise ValueError("training and flight features must be finite")
    model = fit_ridge(alpha=1.0)
    model.fit(x, y)
    estimate = float(model.predict(new_x)[0])
    actual = float(flight["rtl_charge_mah"])
    report = {
        "purpose": "reproducibility demonstration, not the M7 final model",
        "model": "Ridge on 40 train-only F0 sample rows",
        "n_training_rows": len(training),
        "run_id": flight["run_id"],
        "predicted_rtl_charge_mah": estimate,
        "actual_rtl_charge_mah": actual,
        "absolute_error_mah": abs(estimate - actual),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
