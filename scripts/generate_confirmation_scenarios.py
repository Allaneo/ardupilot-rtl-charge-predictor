#!/usr/bin/env python3
"""Generate the D029 confirmation manifest or the D031 headwind training manifest.

Group H oversamples strong return headwind: wind 4-6 m/s coming from the
return heading +/-30 degrees. Group G is a Latin hypercube over the full frozen
envelope. The defaults reproduce the 60-flight confirmation manifest (30 H +
30 G, labelled test). D031 training flights:

    generate_confirmation_scenarios.py --prefix train --seed 20261001 \
        --headwind-count 40 --general-count 0 --split train \
        --output data/manifests/headwind_training_scenarios.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import random
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rtl_charge.scenario import build_manifest_rows, generate_scenarios  # noqa: E402
from generate_scenarios import file_sha256  # noqa: E402


SEED = 20260930
GROUP_SIZE = 30
HEADWIND_SPEED_M_S = (4.0, 6.0)
HEADWIND_OFFSET_DEG = (-30.0, 30.0)
OUTPUT = PROJECT_ROOT / "data" / "manifests" / "confirmation_scenarios.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="confirm")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--headwind-count", type=int, default=GROUP_SIZE)
    parser.add_argument("--general-count", type=int, default=GROUP_SIZE)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing manifest: {args.output}")
    config = yaml.safe_load((PROJECT_ROOT / "config" / "experiment.yaml").read_text())
    ranges = config["scenario"]

    headwind = generate_scenarios(
        args.headwind_count, seed=args.seed, prefix=f"{args.prefix}-h",
        ranges={**ranges, "wind_speed_m_s": list(HEADWIND_SPEED_M_S), "wind_direction_deg": list(HEADWIND_OFFSET_DEG)},
    ) if args.headwind_count else []
    # The generated "direction" is an offset from the return heading (bearing + 180).
    headwind = [
        replace(item, wind_direction_deg=(item.bearing_deg + 180.0 + item.wind_direction_deg) % 360.0)
        for item in headwind
    ]
    general = generate_scenarios(
        args.general_count, seed=random.Random(args.seed).getrandbits(32), prefix=f"{args.prefix}-g", ranges=ranges,
    ) if args.general_count else []

    rows = build_manifest_rows(
        headwind + general,
        ardupilot_commit=str(config["simulation"]["ardupilot_commit"]),
        patch_hash=file_sha256(PROJECT_ROOT / "patches" / "ardupilot_centered_payload.patch"),
        parameter_file_hash=file_sha256(PROJECT_ROOT / "config" / "base_parameters.parm"),
        observation_window_s=float(config["experiment"]["observation_window_s"]),
    )
    rows = [replace(row, split=args.split) for row in rows]
    if len({row.scenario_fingerprint for row in rows}) != len(rows):
        raise ValueError("duplicate scenario fingerprints")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0].to_dict()))
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)
    print(f"Wrote {len(rows)} scenarios to {args.output}")


if __name__ == "__main__":
    main()
