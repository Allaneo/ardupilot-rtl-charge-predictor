#!/usr/bin/env python3
"""Export a small train-only F0 sample for the reproducibility example."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rtl_charge.schema import load_feature_sets


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "data/processed/pipeline_dataset.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data/examples/train_f0_sample.csv")
    parser.add_argument("--rows", type=int, default=40)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite training sample: {args.output}")
    features = load_feature_sets(ROOT / "config/feature_sets.yaml")["f0_route_and_wind"]
    frame = pd.read_csv(args.dataset, usecols=["split", "run_id", *features, "rtl_charge_mah"])
    train = frame.loc[frame["split"] == "train", ["run_id", *features, "rtl_charge_mah"]].head(args.rows)
    if len(train) != args.rows or train.isna().any().any() or train["run_id"].duplicated().any():
        raise ValueError("training sample is incomplete or invalid")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.output, index=False)
    print(f"Exported {len(train)} train-only F0 rows to {args.output}")


if __name__ == "__main__":
    main()
