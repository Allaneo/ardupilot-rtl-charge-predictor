#!/usr/bin/env python3
"""Parse one completed SITL run into a quality-checked processed row."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtl_charge.processing import build_flight_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = build_flight_row(args.run_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(
        f"Processed {args.run_dir.name}: {row['rtl_charge_mah']:.3f} mAh; "
        f"cross-check error {row['charge_crosscheck_error_mah']:.3f} mAh"
    )


if __name__ == "__main__":
    main()
