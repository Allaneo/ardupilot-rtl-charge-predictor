#!/usr/bin/env python3
"""Run pending manifest rows with atomic, restartable state updates."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_failure(reason: str, returncode: int) -> str:
    text = reason.lower()
    if "parameter" in text and "confirmed" in text:
        category = "parameter_mismatch"
    elif "did not arm" in text:
        category = "arm_failure"
    elif "outbound point" in text:
        category = "navigation_timeout"
    elif "automatic disarm" in text:
        category = "automatic_disarm_timeout"
    elif "heartbeat" in text or "tcp endpoint" in text:
        category = "sitl_connection_failure"
    elif "battery" in text or "position" in text:
        category = "missing_telemetry"
    else:
        category = "flight_process_failure"
    detail = reason or f"process exit {returncode}"
    return f"{category}: {detail}"


def read_manifest(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        return list(reader.fieldnames or []), list(reader)


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "pipeline_scenarios.csv",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "validation", "test"),
        dest="splits",
    )
    parser.add_argument("--scenario-id", action="append", dest="scenario_ids")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--speedup", type=float, default=2.0)
    return parser.parse_args()


def run_flight_process(row: dict[str, str], instance: int, speedup: float) -> tuple[int, str]:
    run_dir = PROJECT_ROOT / "data" / "raw" / row["run_id"]
    command = [
        str(PROJECT_ROOT / ".venv" / "bin" / "python"),
        str(PROJECT_ROOT / "scripts" / "run_sitl_flight.py"),
        "--run-dir",
        str(run_dir),
        "--altitude-m",
        row["decision_altitude_m"],
        "--distance-m",
        row["distance_m"],
        "--bearing-deg",
        row["bearing_deg"],
        "--observation-window-s",
        row["observation_window_s"],
        "--payload-mass-kg",
        row["payload_mass_kg"],
        "--wind-speed-m-s",
        row["wind_speed_m_s"],
        "--wind-direction-deg",
        row["wind_direction_deg"],
        "--scenario-id",
        row["scenario_id"],
        "--run-id",
        row["run_id"],
        "--instance",
        str(instance),
        "--speedup",
        str(speedup),
    ]
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    result_path = run_dir / "result.json"
    failure_reason = "flight process exited without a result"
    if result_path.exists():
        result = json.loads(result_path.read_text())
        failure_reason = str(result.get("error", ""))
    return completed.returncode, failure_reason


def main() -> None:
    args = parse_args()
    if not 1 <= args.jobs <= 8:
        raise SystemExit("jobs must be between 1 and 8")
    if not 1.0 <= args.speedup <= 4.0:
        raise SystemExit("speedup must be between 1 and 4")
    manifest_path = args.manifest.resolve()
    fieldnames, rows = read_manifest(manifest_path)
    if not rows:
        raise SystemExit("manifest has no rows")
    expected_patch = sha256(PROJECT_ROOT / "patches" / "ardupilot_centered_payload.patch")
    expected_parameters = sha256(PROJECT_ROOT / "config" / "base_parameters.parm")
    reconciled = False
    for row in rows:
        if row["status"] != "running":
            continue
        result_path = PROJECT_ROOT / "data" / "raw" / row["run_id"] / "result.json"
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result.get("automatic_disarm_observed", False):
                row["status"] = "completed"
                row["failure_reason"] = ""
            else:
                row["status"] = "failed"
                row["failure_reason"] = str(result.get("error", "interrupted run"))
        else:
            row["status"] = "failed"
            row["failure_reason"] = "batch interrupted before a result was written"
        row["ended_unix_s"] = now_iso()
        reconciled = True
    if reconciled:
        write_manifest(manifest_path, fieldnames, rows)

    selected = [row for row in rows if row["status"] == "pending"]
    if args.splits:
        selected = [row for row in selected if row["split"] in args.splits]
    if args.scenario_ids:
        requested_ids = set(args.scenario_ids)
        selected = [row for row in selected if row["scenario_id"] in requested_ids]
    if args.limit is not None:
        selected = selected[: args.limit]

    queued = iter(enumerate(selected, start=1))
    available_instances = deque(range(args.jobs))
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        active = {}
        while True:
            while available_instances:
                try:
                    ordinal, row = next(queued)
                except StopIteration:
                    break
                if row["patch_hash"] != expected_patch or row["parameter_file_hash"] != expected_parameters:
                    raise SystemExit(f"provenance drift detected before {row['run_id']}")
                run_dir = PROJECT_ROOT / "data" / "raw" / row["run_id"]
                if run_dir.exists() and any(run_dir.iterdir()):
                    row["status"] = "failed"
                    row["failure_reason"] = "run_directory_conflict: run directory already exists and is non-empty"
                    row["ended_unix_s"] = now_iso()
                    write_manifest(manifest_path, fieldnames, rows)
                    continue
                instance = available_instances.popleft()
                row["status"] = "running"
                row["failure_reason"] = ""
                row["started_unix_s"] = now_iso()
                row["ended_unix_s"] = ""
                write_manifest(manifest_path, fieldnames, rows)
                future = executor.submit(run_flight_process, row, instance, args.speedup)
                active[future] = (row, instance)
                print(
                    f"[{ordinal}/{len(selected)}] Running {row['run_id']} on instance {instance}",
                    flush=True,
                )
            if not active:
                break
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                row, instance = active.pop(future)
                try:
                    returncode, failure_reason = future.result()
                except BaseException as error:
                    returncode, failure_reason = 1, f"orchestrator worker error: {error}"
                if returncode == 0:
                    row["status"] = "completed"
                    row["failure_reason"] = ""
                else:
                    row["status"] = "failed"
                    row["failure_reason"] = classify_failure(failure_reason, returncode)
                row["ended_unix_s"] = now_iso()
                write_manifest(manifest_path, fieldnames, rows)
                available_instances.append(instance)
                if returncode != 0:
                    print(
                        f"{row['run_id']} failed: {row['failure_reason']}",
                        file=sys.stderr,
                        flush=True,
                    )

    completed_count = sum(row["status"] == "completed" for row in rows)
    failed_count = sum(row["status"] == "failed" for row in rows)
    pending_count = sum(row["status"] == "pending" for row in rows)
    print(
        f"Manifest state: completed={completed_count}, failed={failed_count}, "
        f"pending={pending_count}",
        flush=True,
    )


if __name__ == "__main__":
    main()
