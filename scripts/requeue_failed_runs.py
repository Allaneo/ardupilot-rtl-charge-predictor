#!/usr/bin/env python3
"""Archive named failed attempts and safely requeue their manifest rows."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "pipeline_scenarios.csv",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "attempt_history.jsonl",
    )
    return parser.parse_args()


def write_manifest(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
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


def main() -> None:
    args = parse_args()
    with args.manifest.open(newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    by_run_id = {row["run_id"]: row for row in rows}
    requested = list(dict.fromkeys(args.run_ids))
    missing = [run_id for run_id in requested if run_id not in by_run_id]
    if missing:
        raise SystemExit(f"run IDs not found in manifest: {missing}")
    invalid = [run_id for run_id in requested if by_run_id[run_id]["status"] != "failed"]
    if invalid:
        raise SystemExit(f"only failed rows can be requeued: {invalid}")

    archive_root = PROJECT_ROOT / "data" / "raw" / "failed_attempts"
    archive_root.mkdir(parents=True, exist_ok=True)
    history_entries = []
    for run_id in requested:
        row = by_run_id[run_id]
        source_dir = PROJECT_ROOT / "data" / "raw" / run_id
        if not source_dir.is_dir():
            raise SystemExit(f"failed run directory is missing: {source_dir}")
        attempt_number = 1
        while True:
            archive_dir = archive_root / f"{run_id}-attempt-{attempt_number:03d}"
            if not archive_dir.exists():
                break
            attempt_number += 1
        result_path = source_dir / "result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        source_dir.replace(archive_dir)
        history_entries.append(
            {
                "archived_at": datetime.now(timezone.utc).isoformat(),
                "scenario_id": row["scenario_id"],
                "run_id": run_id,
                "attempt": attempt_number,
                "failure_reason": row["failure_reason"],
                "archived_path": str(archive_dir.relative_to(PROJECT_ROOT)),
                "result": result,
            }
        )
        row["status"] = "pending"
        row["failure_reason"] = ""
        row["started_unix_s"] = ""
        row["ended_unix_s"] = ""

    write_manifest(args.manifest, fieldnames, rows)
    args.history.parent.mkdir(parents=True, exist_ok=True)
    with args.history.open("a") as history:
        for entry in history_entries:
            history.write(json.dumps(entry, sort_keys=True) + "\n")
    print(f"Archived and requeued {len(history_entries)} failed runs")


if __name__ == "__main__":
    main()
