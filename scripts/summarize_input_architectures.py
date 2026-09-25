#!/usr/bin/env python3
"""Audit and summarize the validation-only input/architecture sweep."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rtl_charge.modeling import load_train_validation  # noqa: E402
from rtl_charge.schema import load_feature_sets  # noqa: E402

OUT = ROOT / "reports/m13_input_architecture_sweep"
PAIRS = (
    ("f0_original", "f0_fixed_time"),
    ("f0_fixed_time", "f0_learned_time"),
    ("f1_original", "f1_fixed_time"),
    ("f1_fixed_time", "f1_learned_speed_time"),
    ("f2_corrected_fixed", "f2_corrected_learned"),
    ("f3_cubic_fixed", "f3_cubic_learned"),
)


def main() -> None:
    manifest = json.loads((OUT / "manifest.json").read_text())
    sets = load_feature_sets(ROOT / "config/feature_sets.yaml")
    split = load_train_validation(ROOT / "data/processed/pipeline_dataset.csv", sets["f2_physics_informed"])
    if tuple(manifest["validation_run_ids"]) != split.validation_run_ids:
        raise ValueError("validation flight ordering changed")
    validation_y = split.validation_y
    headwind_mask = split.validation_x["headwind_m_s"].to_numpy(dtype=np.float64) >= 2.0
    files = sorted((OUT / "runs").glob("*.json"))
    expected = len(manifest["variants"]) * (2 + len(manifest["architectures"]) * len(manifest["seeds"]))
    if len(files) != expected:
        raise ValueError(f"expected {expected} checkpoints, found {len(files)}")
    rows = []
    for path in files:
        run = json.loads(path.read_text())
        if path.stem != run["run_id"] or run["variant"] not in manifest["variants"]:
            raise ValueError(f"invalid checkpoint identity: {path}")
        values = np.asarray(run["validation_predictions_mah"], dtype=np.float64)
        if len(values) != len(manifest["validation_run_ids"]) or not np.isfinite(values).all():
            raise ValueError(f"invalid validation predictions: {path}")
        checked = {
            "mae_mah": float(np.mean(np.abs(values - validation_y))),
            "rmse_mah": float(np.sqrt(np.mean(np.square(values - validation_y)))),
            "max_underprediction_mah": float(np.max(validation_y - values)),
            "high_headwind_mae_mah": float(np.mean(np.abs(values[headwind_mask] - validation_y[headwind_mask]))),
        }
        for key, expected_value in checked.items():
            if not np.isclose(run["validation"][key], expected_value, atol=1e-8, rtol=1e-10):
                raise ValueError(f"validation metric mismatch in {path}: {key}")
        rows.append({
            "run_id": run["run_id"],
            "variant": run["variant"],
            "model": run["model"],
            "architecture": "-".join(map(str, run["architecture"])) if run["architecture"] else "fixed",
            "seed": run["seed"],
            "parameters": run["parameter_count"],
            "selected_epoch": run["selected_epoch"],
            "train_mae_mah": run["train"]["mae_mah"],
            "validation_mae_mah": run["validation"]["mae_mah"],
            "validation_rmse_mah": run["validation"]["rmse_mah"],
            "validation_max_underprediction_mah": run["validation"]["max_underprediction_mah"],
            "validation_high_headwind_mae_mah": run["validation"]["high_headwind_mae_mah"],
        })
    frame = pd.DataFrame(rows)
    if not frame.run_id.is_unique:
        raise ValueError("duplicate run IDs")
    for variant in manifest["variants"]:
        subset = frame.loc[frame.variant == variant]
        if len(subset) != expected // len(manifest["variants"]):
            raise ValueError(f"missing runs for {variant}")
        neural = subset.loc[subset.model == "neural"]
        pairs = set(zip(neural.architecture, neural.seed))
        expected_pairs = {("-".join(map(str, shape)), seed)
                          for shape in manifest["architectures"] for seed in manifest["seeds"]}
        if pairs != expected_pairs:
            raise ValueError(f"neural catalog incomplete for {variant}")
        if set(subset.loc[subset.model != "neural", "model"]) != {"ridge", "xgboost"}:
            raise ValueError(f"reference models incomplete for {variant}")

    frame.to_csv(OUT / "runs_summary.csv", index=False)
    neural = frame.loc[frame.model == "neural"].copy()
    grouped = neural.groupby(["variant", "architecture"], sort=False).agg(
        parameters=("parameters", "first"),
        mean_train_mae_mah=("train_mae_mah", "mean"),
        mean_validation_mae_mah=("validation_mae_mah", "mean"),
        min_validation_mae_mah=("validation_mae_mah", "min"),
        max_validation_mae_mah=("validation_mae_mah", "max"),
        mean_high_headwind_mae_mah=("validation_high_headwind_mae_mah", "mean"),
        mean_max_underprediction_mah=("validation_max_underprediction_mah", "mean"),
    ).reset_index()
    grouped["seed_range_mah"] = grouped.max_validation_mae_mah - grouped.min_validation_mae_mah
    grouped["train_validation_gap_mah"] = grouped.mean_validation_mae_mah - grouped.mean_train_mae_mah
    grouped.to_csv(OUT / "architecture_summary.csv", index=False)

    pair_rows = []
    for reference, candidate in PAIRS:
        left = neural.loc[neural.variant == reference].set_index(["architecture", "seed"])
        right = neural.loc[neural.variant == candidate].set_index(["architecture", "seed"])
        difference = right.validation_mae_mah - left.validation_mae_mah
        pair_rows.append({
            "reference": reference,
            "candidate": candidate,
            "matched_neural_runs": len(difference),
            "improved_neural_runs": int((difference < 0).sum()),
            "mean_neural_mae_change_mah": float(difference.mean()),
            "median_neural_mae_change_mah": float(difference.median()),
            "ridge_mae_change_mah": float(frame.loc[(frame.variant == candidate) & (frame.model == "ridge"),
                                                     "validation_mae_mah"].iloc[0] -
                                          frame.loc[(frame.variant == reference) & (frame.model == "ridge"),
                                                    "validation_mae_mah"].iloc[0]),
            "xgboost_mae_change_mah": float(frame.loc[(frame.variant == candidate) & (frame.model == "xgboost"),
                                                       "validation_mae_mah"].iloc[0] -
                                            frame.loc[(frame.variant == reference) & (frame.model == "xgboost"),
                                                      "validation_mae_mah"].iloc[0]),
        })
    comparison = pd.DataFrame(pair_rows)
    comparison.to_csv(OUT / "paired_variant_summary.csv", index=False)
    best_shapes = grouped.sort_values(["variant", "mean_validation_mae_mah"]).groupby("variant").head(1)
    summary = {
        "completed_runs": len(frame),
        "neural_runs": len(neural),
        "reference_runs": len(frame) - len(neural),
        "validation_flights": len(manifest["validation_run_ids"]),
        "best_mean_neural_architecture_by_variant": best_shapes.to_dict(orient="records"),
        "paired_variants": pair_rows,
        "minimum_single_run_validation_mae_mah": float(neural.validation_mae_mah.min()),
        "maximum_single_run_validation_mae_mah": float(neural.validation_mae_mah.max()),
        "test_set_used": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
