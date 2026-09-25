#!/usr/bin/env python3
"""Validation-only check of a tilt-limit (physics-shaped) cruise-speed model.

At the 30-degree lean limit the vehicle flies at an approximately fixed maximum
airspeed that grows with vehicle mass (more weight gives more horizontal thrust
at the same lean). Mass is hidden, so the maximum airspeed is modelled as a
linear function of decision-window throttle, fitted only on training flights
that spent most of their middle route near the lean limit. Ground speed along
the route is then

    min(10, sqrt(V_max^2 - crosswind^2) - headwind)

which, unlike a tree ensemble, extrapolates to stronger headwinds than seen in
training. Training-row speeds are five-fold out-of-fold, as for the forest.
Uses the 172 training and 39 validation flights; no test flight is read.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtl_charge.modeling import fit_dense_regressor, fit_ridge, load_train_validation  # noqa: E402
from rtl_charge.schema import load_feature_sets  # noqa: E402
from speed_oracle_check import SEEDS, metrics  # noqa: E402
from sweep_input_architectures import speed_predictions  # noqa: E402
from validate_controller_aware_features import SPEED_INPUTS, SPEED_SETPOINT, SPEED_TARGET, make_consistent_features  # noqa: E402


SPEED_LABELS = ROOT / "reports/m9/rtl_speed_train_validation.csv"
OUT = ROOT / "reports/m18_tilt_limit_speed"
SATURATION_FRACTION = 0.5
FOLD_SEED = 20260923


class TiltLimitSpeed:
    """Maximum airspeed linear in throttle; ground speed capped at the setpoint."""

    def fit(self, frame: pd.DataFrame, measured_speed: np.ndarray, saturated: np.ndarray) -> "TiltLimitSpeed":
        air = observed_airspeed(frame, measured_speed)
        self.model = LinearRegression().fit(frame.loc[saturated, ["throttle_mean"]], air[saturated])
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        v_max = self.model.predict(frame.loc[:, ["throttle_mean"]])
        cross = frame["crosswind_abs_m_s"].to_numpy(dtype=np.float64)
        along_air = np.sqrt(np.maximum(v_max**2 - cross**2, 0.0))
        ground = along_air - frame["headwind_m_s"].to_numpy(dtype=np.float64)
        return np.clip(ground, 1.0, SPEED_SETPOINT)


def observed_airspeed(frame: pd.DataFrame, ground_speed: np.ndarray) -> np.ndarray:
    north = ground_speed * frame["route_bearing_cos"].to_numpy() - frame["ekf_wind_north_m_s"].to_numpy()
    east = ground_speed * frame["route_bearing_sin"].to_numpy() - frame["ekf_wind_east_m_s"].to_numpy()
    return np.hypot(north, east)


def speed_errors(predicted: np.ndarray, measured: np.ndarray, headwind: np.ndarray) -> dict[str, float]:
    high = headwind >= 2.0
    return {"speed_mae_m_s": float(np.mean(np.abs(predicted - measured))),
            "speed_mae_headwind_ge_2_m_s": float(np.mean(np.abs(predicted[high] - measured[high]))),
            "speed_max_overprediction_m_s": float(np.max(predicted - measured))}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing report: {OUT}")
    feature_sets = load_feature_sets(ROOT / "config/feature_sets.yaml")
    split = load_train_validation(ROOT / "data/processed/pipeline_dataset.csv", feature_sets["f2_physics_informed"])
    train, validation = split.train_x.copy(), split.validation_x.copy()
    train.index = pd.Index(split.train_run_ids, name="run_id")
    validation.index = pd.Index(split.validation_run_ids, name="run_id")
    labels = pd.read_csv(SPEED_LABELS).set_index("run_id", verify_integrity=True)
    f1_columns = feature_sets["f1_vehicle_symptoms"]
    measured_train = labels.loc[train.index, SPEED_TARGET].to_numpy(dtype=np.float64)
    measured_validation = labels.loc[validation.index, SPEED_TARGET].to_numpy(dtype=np.float64)
    saturated_train = labels.loc[train.index, "lean_near_30deg_fraction"].to_numpy() > SATURATION_FRACTION
    headwind = validation["headwind_m_s"].to_numpy(dtype=np.float64)

    tilt_train = np.empty(len(train))
    for fit_rows, predict_rows in KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED).split(train):
        fold = TiltLimitSpeed().fit(train.iloc[fit_rows], measured_train[fit_rows], saturated_train[fit_rows])
        tilt_train[predict_rows] = fold.predict(train.iloc[predict_rows])
    tilt = TiltLimitSpeed().fit(train, measured_train, saturated_train)
    tilt_validation = tilt.predict(validation)
    forest_train, forest_validation = speed_predictions(train, validation, labels, SPEED_INPUTS)

    speed_rows = []
    for name, values in (("fixed_10", np.full(len(validation), SPEED_SETPOINT)),
                         ("random_forest", forest_validation), ("tilt_limit", tilt_validation)):
        speed_rows.append({"speed_model": name, **speed_errors(values, measured_validation, headwind)})

    rows = []
    predictions = pd.DataFrame({"run_id": validation.index, "actual_rtl_charge_mah": split.validation_y,
                                "headwind_m_s": headwind, "measured_speed_m_s": measured_validation,
                                "forest_speed_m_s": forest_validation, "tilt_limit_speed_m_s": tilt_validation})
    for variant, (speed_train, speed_validation) in {
        "f3_forest_speed": (forest_train, forest_validation),
        "f3_tilt_limit_speed": (tilt_train, tilt_validation),
        "f3_measured_speed_oracle": (measured_train, measured_validation),
    }.items():
        train_x = make_consistent_features(train, speed_train, f1_columns)
        validation_x = make_consistent_features(validation, speed_validation, f1_columns)
        ridge = fit_ridge(alpha=1.0).fit(train_x, split.train_y).predict(validation_x)
        rows.append({"variant": variant, "model": "ridge", **metrics(split.validation_y, ridge, headwind)})
        predictions[f"{variant}__ridge"] = ridge
        neural = []
        for seed in SEEDS:
            model, _ = fit_dense_regressor(
                train_x.to_numpy(dtype=np.float64), split.train_y,
                validation_x.to_numpy(dtype=np.float64), split.validation_y,
                hidden_layers=(1024,), seed=seed, loss="huber", alpha=0.001,
                learning_rate=0.001, max_epochs=600, patience=45,
            )
            neural.append(model.predict(validation_x.to_numpy(dtype=np.float64)))
        mean = np.mean(neural, axis=0)
        rows.append({"variant": variant, "model": "neural_1024_seed_mean", **metrics(split.validation_y, mean, headwind)})
        predictions[f"{variant}__neural_1024_seed_mean"] = mean

    OUT.mkdir(parents=True)
    pd.DataFrame(speed_rows).to_csv(OUT / "validation_speed_metrics.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "validation_charge_metrics.csv", index=False)
    predictions.to_csv(OUT / "validation_predictions.csv", index=False)
    summary = {
        "milestone": "M18_tilt_limit_speed_validation",
        "train_flights": len(train), "validation_flights": len(validation), "test_flights_read": 0,
        "saturated_training_flights": int(saturated_train.sum()),
        "v_max_model": {"intercept_m_s": float(tilt.model.intercept_), "throttle_coefficient_m_s": float(tilt.model.coef_[0])},
        "speed_results": speed_rows, "charge_results": rows,
        "caveat": "Validation reused across M9-M17; exploratory. Neural epochs use validation early stopping as in M13.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"V_max = {tilt.model.intercept_:.2f} + {tilt.model.coef_[0]:.2f} x throttle  (fit on {saturated_train.sum()} saturated training flights)")
    print(pd.DataFrame(speed_rows).round(3).to_string(index=False))
    print()
    print(pd.DataFrame(rows).round(1).to_string(index=False))
    print()
    print(predictions.sort_values("headwind_m_s", ascending=False).head(6)[
        ["run_id", "headwind_m_s", "measured_speed_m_s", "forest_speed_m_s", "tilt_limit_speed_m_s"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
