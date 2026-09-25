#!/usr/bin/env python3
"""Generate a deterministic one-row-per-flight scenario manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import yaml

from rtl_charge.scenario import build_manifest_rows, generate_scenarios


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--prefix", default="pipeline")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "experiment.yaml")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "pipeline_scenarios.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite existing manifest: {args.output}")
    config = yaml.safe_load(args.config.read_text())
    count = args.count or int(config["dataset"]["pipeline_flights"])
    scenarios = generate_scenarios(
        count,
        seed=int(config["experiment"]["seed"]),
        ranges=config["scenario"],
        prefix=args.prefix,
    )
    rows = build_manifest_rows(
        scenarios,
        ardupilot_commit=str(config["simulation"]["ardupilot_commit"]),
        patch_hash=file_sha256(PROJECT_ROOT / "patches" / "ardupilot_centered_payload.patch"),
        parameter_file_hash=file_sha256(Path(args.config).parent / "base_parameters.parm"),
        observation_window_s=float(config["experiment"]["observation_window_s"]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].to_dict()))
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)
    split_counts = {name: sum(row.split == name for row in rows) for name in ("train", "validation", "test")}
    print(f"Wrote {len(rows)} deterministic scenarios to {args.output}")
    print(f"Split counts: {split_counts}")


if __name__ == "__main__":
    main()
