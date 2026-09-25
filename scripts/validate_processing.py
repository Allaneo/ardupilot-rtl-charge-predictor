#!/usr/bin/env python3
"""Run the feature/label builder over all locally usable acceptance flights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtl_charge.processing import ProcessingError, build_flight_row


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "processing_validation.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    accepted = []
    rejected_incomplete = []
    parser_failures = []
    for run_dir in sorted(path for path in args.raw_root.iterdir() if path.is_dir()):
        result_path = run_dir / "result.json"
        log_path = run_dir / "logs" / "00000001.BIN"
        if not result_path.exists() or not log_path.exists():
            continue
        result = json.loads(result_path.read_text())
        if result.get("wind_speed_m_s") is None or result.get("payload_mass_kg") is None:
            continue
        if not result.get("automatic_disarm_observed", False):
            rejected_incomplete.append(run_dir.name)
            try:
                build_flight_row(run_dir)
            except ProcessingError:
                pass
            else:
                parser_failures.append(
                    {"run_id": run_dir.name, "error": "incomplete run was accepted"}
                )
            continue
        try:
            row = build_flight_row(run_dir)
        except ProcessingError as error:
            parser_failures.append({"run_id": run_dir.name, "error": str(error)})
            continue
        accepted.append(
            {
                "run_id": run_dir.name,
                "rtl_charge_mah": row["rtl_charge_mah"],
                "integrated_charge_mah": row["rtl_charge_integrated_mah"],
                "runner_charge_mah": row["rtl_charge_runner_mah"],
                "crosscheck_error_mah": row["charge_crosscheck_error_mah"],
                "decision_battery_samples": row["decision_battery_sample_count"],
                "active_motor_count": row["active_motor_count"],
            }
        )

    checks = {
        "at_least_ten_completed_acceptance_flights": len(accepted) >= 10,
        "no_completed_flight_parser_failures": not parser_failures,
        "all_crosscheck_errors_below_2mah": bool(accepted)
        and max(item["crosscheck_error_mah"] for item in accepted) < 2.0,
        "all_windows_have_at_least_50_battery_samples": bool(accepted)
        and min(item["decision_battery_samples"] for item in accepted) >= 50,
        "all_runs_have_four_active_motors": bool(accepted)
        and all(item["active_motor_count"] == 4 for item in accepted),
        "incomplete_run_rejection_exercised": bool(rejected_incomplete),
    }
    report = {
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "accepted_count": len(accepted),
        "rejected_incomplete_runs": rejected_incomplete,
        "parser_failures": parser_failures,
        "maximum_crosscheck_error_mah": (
            max(item["crosscheck_error_mah"] for item in accepted) if accepted else None
        ),
        "runs": accepted,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_checks_passed"]:
        raise SystemExit("processing validation failed")


if __name__ == "__main__":
    main()
