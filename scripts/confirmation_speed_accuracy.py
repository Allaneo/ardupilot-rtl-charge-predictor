#!/usr/bin/env python3
"""Post-scoring D029 check: cruise-speed accuracy on the confirmation flights.

Compares the speed predictions saved by evaluate_confirmation.py (random
forest and tilt-limit, both fitted on the 250 development flights) and a fixed
10 m/s assumption with the measured middle-route RTL ground speed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS = ROOT / "reports/m19_confirmation/confirmation_predictions.csv"
MEASURED = ROOT / "reports/m19_confirmation_speed/rtl_speed_confirmation.csv"
OUT = ROOT / "reports/m19_confirmation_speed/speed_accuracy.json"


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"Refusing to overwrite existing report: {OUT}")
    predictions = pd.read_csv(PREDICTIONS).set_index("run_id", verify_integrity=True)
    measured = pd.read_csv(MEASURED).set_index("run_id", verify_integrity=True)
    if set(predictions.index) != set(measured.index):
        raise ValueError("measured speeds must cover exactly the scored confirmation flights")
    frame = predictions.join(measured[["middle_route_ground_speed_m_s", "lean_near_30deg_fraction"]])
    frame["fixed_10_speed_m_s"] = 10.0
    actual = frame["middle_route_ground_speed_m_s"].to_numpy()
    headwind = frame["headwind_m_s"].to_numpy()
    rows = {}
    for name in ("fixed_10_speed_m_s", "forest_speed_m_s", "tilt_speed_m_s"):
        error = frame[name].to_numpy() - actual
        row = {"mae_m_s": float(np.mean(np.abs(error))), "max_overestimate_m_s": float(error.max())}
        for label, mask in (("group_h", frame["group"].to_numpy() == "h"), ("group_g", frame["group"].to_numpy() == "g"),
                            ("est_headwind_ge_3", headwind >= 3.0)):
            row[f"{label}_n"] = int(mask.sum())
            row[f"{label}_mae_m_s"] = float(np.mean(np.abs(error[mask])))
        rows[name] = row
    summary = {"flights": len(frame),
               "lean_limited_flights": int((frame["lean_near_30deg_fraction"] > 0.5).sum()),
               "measured_speed_range_m_s": [float(actual.min()), float(actual.max())],
               "speed_models": rows}
    OUT.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
