#!/usr/bin/env python3
"""Summarize truth-versus-EKF wind estimates for engineering flights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtl_charge.validation import extract_wind_run_metrics, wind_validation_checks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--failed-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [extract_wind_run_metrics(run_dir) for run_dir in args.run_dir]
    checks = wind_validation_checks(runs)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 4))
    truth_north = [run.true_wind_north_m_s for run in runs]
    truth_east = [run.true_wind_east_m_s for run in runs]
    estimate_north = [run.estimated_wind_north_mean_m_s for run in runs]
    estimate_east = [run.estimated_wind_east_mean_m_s for run in runs]
    limits = truth_north + truth_east + estimate_north + estimate_east
    low, high = min(limits) - 0.5, max(limits) + 0.5
    for axis, truth, estimate, component in (
        (axes[0], truth_north, estimate_north, "North"),
        (axes[1], truth_east, estimate_east, "East"),
    ):
        axis.scatter(truth, estimate)
        axis.plot([low, high], [low, high], linestyle="--", color="black")
        axis.set_xlim(low, high)
        axis.set_ylim(low, high)
        axis.set_xlabel(f"True {component.lower()} wind (m/s)")
        axis.set_ylabel(f"EKF {component.lower()} wind (m/s)")
        axis.grid(alpha=0.3)
    figure.suptitle("EKF3 wind estimate at the RTL decision window")
    figure.tight_layout()
    args.plot.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.plot, dpi=180)
    plt.close(figure)

    summary = {
        "runs": [run.to_dict() for run in runs],
        "excluded_failed_runs": [
            {
                "run_name": run_dir.name,
                **json.loads((run_dir / "result.json").read_text()),
            }
            for run_dir in args.failed_run_dir
        ],
        "validated_wind_speed_range_m_s": [
            min(run.true_wind_speed_m_s for run in runs),
            max(run.true_wind_speed_m_s for run in runs),
        ],
        "mean_vector_error_m_s": sum(run.wind_vector_error_m_s for run in runs)
        / len(runs),
        "maximum_vector_error_m_s": max(run.wind_vector_error_m_s for run in runs),
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
