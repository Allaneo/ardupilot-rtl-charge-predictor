#!/usr/bin/env python3
"""Validate centred-payload behavior from matched SITL engineering flights."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from rtl_charge.validation import (
    PayloadRunMetrics,
    extract_payload_run_metrics,
    payload_validation_checks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-dir", type=Path, required=True)
    parser.add_argument("--patched-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    parser.add_argument("--ardupilot-commit", required=True)
    parser.add_argument("--patch", type=Path, required=True)
    return parser.parse_args()


def plot_runs(
    control: PayloadRunMetrics,
    patched_runs: Sequence[PayloadRunMetrics],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = sorted(patched_runs, key=lambda run: float(run.payload_mass_kg))
    payloads = [float(run.payload_mass_kg) for run in ordered]
    series = [
        ("Hover throttle", [run.hover_throttle_mean for run in ordered]),
        ("Hover current (A)", [run.hover_current_mean_a for run in ordered]),
        ("RTL charge (mAh)", [run.rtl_charge_mah for run in ordered]),
    ]
    control_values = [
        control.hover_throttle_mean,
        control.hover_current_mean_a,
        control.rtl_charge_mah,
    ]

    figure, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for axis, (label, values), control_value in zip(axes, series, control_values):
        axis.plot(payloads, values, marker="o", label="Patched SITL")
        axis.scatter([0.0], [control_value], marker="x", s=70, label="Unmodified")
        axis.set_xlabel("Centred payload (kg)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
    axes[0].legend(frameon=False)
    figure.suptitle("Centred-payload SITL validation")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    control = extract_payload_run_metrics(args.control_dir)
    patched = [extract_payload_run_metrics(path) for path in args.patched_dir]
    checks = payload_validation_checks(control, patched)
    patch_sha256 = hashlib.sha256(args.patch.read_bytes()).hexdigest()

    summary = {
        "ardupilot_commit": args.ardupilot_commit,
        "patch_path": str(args.patch),
        "patch_sha256": patch_sha256,
        "control": control.to_dict(),
        "patched_runs": [
            run.to_dict()
            for run in sorted(patched, key=lambda run: float(run.payload_mass_kg))
        ],
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    plot_runs(control, patched, args.plot)

    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
