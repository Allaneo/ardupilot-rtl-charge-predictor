#!/usr/bin/env python3
"""D034 / M23: rerun the M20 sweep for selected shapes with a 3,000-epoch cap.

Identical to M20 except the early-stopping cap, to check whether deeper
networks were undertrained at 600 epochs.
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

from rtl_charge.modeling import TARGET, fit_dense_regressor  # noqa: E402
from rtl_charge.schema import load_feature_sets  # noqa: E402
from cv_architecture_sweep import STOP_FRACTION, fold_features, load_training, paired, shape_name  # noqa: E402
from evaluate_confirmation import FOLD_SEED, NEURAL, SEEDS  # noqa: E402


M20 = ROOT / "reports/m20_cv_architecture_sweep/cv_predictions.csv"
OUT = ROOT / "reports/m23_depth_epoch_check"
MAX_EPOCHS = 3000
SHAPES = ((256,), (1024,), (64, 32), (128, 64), (64, 32, 16), (128, 64, 32), (128, 64, 32, 16),
          (16,) * 5, (16,) * 10)
WORKERS = 4


def fit_network(job):
    fold, hidden, seed, x_train, y_train, x_test = job
    fit_idx, stop_idx = train_test_split(np.arange(len(y_train)), test_size=STOP_FRACTION, random_state=FOLD_SEED)
    _, epoch = fit_dense_regressor(x_train[fit_idx], y_train[fit_idx], x_train[stop_idx], y_train[stop_idx],
                                   hidden_layers=hidden, seed=seed, max_epochs=MAX_EPOCHS, patience=45, **NEURAL)
    epoch = max(1, int(epoch))
    model, _ = fit_dense_regressor(x_train, y_train, x_train, y_train, hidden_layers=hidden, seed=seed,
                                   max_epochs=epoch, fixed_epochs=True, **NEURAL)
    return fold, hidden, seed, epoch, model.predict(x_test)


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing report: {OUT}")
    f1_columns = load_feature_sets(ROOT / "config/feature_sets.yaml")["f1_vehicle_symptoms"]
    frame, measured, saturated = load_training()
    y = frame[TARGET].to_numpy(dtype=np.float64)
    headwind = frame["headwind_m_s"].to_numpy(dtype=np.float64)
    m20 = pd.read_csv(M20)
    if not np.array_equal(m20["run_id"].to_numpy(), frame["run_id"].to_numpy()):
        raise ValueError("flight order differs from M20")

    jobs, test_rows_by_fold = [], {}
    for fold, (fit_rows, test_rows) in enumerate(KFold(n_splits=5, shuffle=True, random_state=FOLD_SEED).split(frame)):
        x_train, x_test = fold_features(frame, measured, saturated, fit_rows, test_rows, f1_columns)
        test_rows_by_fold[fold] = test_rows
        jobs += [(fold, hidden, seed, x_train, y[fit_rows], x_test) for hidden in SHAPES for seed in SEEDS]

    values = {(shape_name(h), s): np.empty(len(y)) for h in SHAPES for s in SEEDS}
    epochs: dict[str, list[int]] = {}
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for fold, hidden, seed, epoch, predicted in pool.map(fit_network, jobs):
            values[(shape_name(hidden), seed)][test_rows_by_fold[fold]] = predicted
            epochs.setdefault(shape_name(hidden), []).append(epoch)

    rng = np.random.default_rng(FOLD_SEED)
    strong = headwind >= 4.0
    rows = []
    per_flight = m20.loc[:, ["run_id", "actual_rtl_charge_mah", "headwind_m_s"]].copy()
    for hidden in SHAPES:
        name = shape_name(hidden)
        mean = np.mean([values[(name, s)] for s in SEEDS], axis=0)
        per_flight[f"net_{name}_cap3000"] = mean
        old = m20[f"net_{name}"].to_numpy()
        rows.append({"shape": name, "mae_cap600_mah": float(np.mean(np.abs(old - y))),
                     "mae_cap3000_mah": float(np.mean(np.abs(mean - y))),
                     "strong_headwind_mae_cap3000_mah": float(np.mean(np.abs(mean - y)[strong])),
                     "max_underprediction_cap3000_mah": float(np.max(y - mean)),
                     "median_epochs_cap3000": float(np.median(epochs[name])),
                     "max_epochs_cap3000": int(np.max(epochs[name])),
                     **{f"cap3000_minus_cap600_{k}": v for k, v in paired(mean - y, old - y, rng).items()}})
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True)
    table.to_csv(OUT / "depth_epoch_metrics.csv", index=False)
    per_flight.to_csv(OUT / "cv_predictions.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps({"milestone": "M23_depth_epoch_check", "protocol": "DECISIONS.md D034",
                                                  "max_epochs": MAX_EPOCHS, "results": rows}, indent=2) + "\n")
    print(table.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
