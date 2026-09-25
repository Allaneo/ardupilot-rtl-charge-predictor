#!/usr/bin/env python3
"""Build the one-row-per-flight dataset from completed manifest runs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile

import pandas as pd
import numpy as np

import rtl_charge.processing as processing
import rtl_charge.validation as validation
from rtl_charge.processing import ProcessingError, build_flight_row
from rtl_charge.schema import load_feature_sets


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def processor_hash() -> str:
    digest = hashlib.sha256()
    for module in (processing, validation):
        installed_source = Path(module.__file__).read_bytes()
        local_source = (PROJECT_ROOT / "src" / "rtl_charge" / f"{module.__name__.rsplit('.', 1)[-1]}.py").read_bytes()
        if installed_source != local_source:
            raise SystemExit("project package differs from src; reinstall it before building the dataset")
        digest.update(installed_source)
    return digest.hexdigest()


def cache_key(run_dir: Path, source_hash: str) -> dict[str, str | int]:
    return {
        "version": CACHE_VERSION,
        "processor_hash": source_hash,
        "result_sha256": sha256(run_dir / "result.json"),
        "log_sha256": sha256(run_dir / "logs" / "00000001.BIN"),
    }


def read_cache(path: Path, key: dict[str, str | int]) -> dict | None:
    try:
        cached = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(cached, dict):
        return None
    if cached.get("key") != key or not isinstance(cached.get("row"), dict):
        return None
    return cached["row"]


def write_cache(path: Path, key: dict[str, str | int], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump({"key": key, "row": row}, output, sort_keys=True)
            output.write("\n")
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
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "pipeline_dataset.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "data" / "manifests" / "pipeline_dataset_quality.json",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "intermediate" / "processed_rows",
    )
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.workers <= 4:
        raise SystemExit("workers must be between 1 and 4")
    with args.manifest.open(newline="") as source:
        manifest = list(csv.DictReader(source))
    completed = [item for item in manifest if item["status"] == "completed"]
    if not completed:
        raise SystemExit("manifest contains no completed flights")

    feature_sets = load_feature_sets(PROJECT_ROOT / "config" / "feature_sets.yaml")
    source_hash = processor_hash()
    raw_rows: dict[str, dict] = {}
    failures = []
    misses: list[tuple[dict[str, str], Path, Path, dict[str, str | int]]] = []
    cache_hits = 0
    for item in completed:
        run_dir = PROJECT_ROOT / "data" / "raw" / item["run_id"]
        try:
            key = cache_key(run_dir, source_hash)
        except OSError as error:
            failures.append({"run_id": item["run_id"], "error": str(error)})
            continue
        cache_path = args.cache_dir / f"{item['run_id']}.json"
        cached_row = read_cache(cache_path, key)
        if cached_row is not None:
            raw_rows[item["run_id"]] = cached_row
            cache_hits += 1
        else:
            misses.append((item, run_dir, cache_path, key))

    if misses:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(build_flight_row, run_dir): (item, cache_path, key)
                for item, run_dir, cache_path, key in misses
            }
            for future in as_completed(futures):
                item, cache_path, key = futures[future]
                try:
                    row = future.result()
                except ProcessingError as error:
                    failures.append({"run_id": item["run_id"], "error": str(error)})
                    continue
                write_cache(cache_path, key, row)
                raw_rows[item["run_id"]] = row

    rows = []
    for item in completed:
        if item["run_id"] not in raw_rows:
            continue
        row = dict(raw_rows[item["run_id"]])
        result = json.loads(
            (PROJECT_ROOT / "data" / "raw" / item["run_id"] / "result.json").read_text()
        )
        row.update(
            {
                "scenario_id": item["scenario_id"],
                "run_id": item["run_id"],
                "scenario_family_id": item["scenario_family_id"],
                "split": item["split"],
                "seed": int(item["seed"]),
                "ardupilot_commit": item["ardupilot_commit"],
                "patch_hash": item["patch_hash"],
                "parameter_file_hash": item["parameter_file_hash"],
                "scenario_fingerprint": item["scenario_fingerprint"],
                "raw_log_path": item["raw_log_path"],
                "requested_decision_altitude_m": float(item["decision_altitude_m"]),
                "sitl_speedup": float(result.get("sitl_speedup", 1.0)),
            }
        )
        rows.append(row)

    missing_features = {
        feature
        for features in feature_sets.values()
        for feature in features
        if rows and feature not in rows[0]
    }
    if missing_features:
        raise SystemExit(f"processed rows lack configured features: {sorted(missing_features)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(args.output, index=False)
    selected_features = sorted({feature for features in feature_sets.values() for feature in features})
    requested_distance = dataframe["requested_distance_m"]
    actual_distance = dataframe["distance_home_m"]
    requested_altitude = dataframe["requested_decision_altitude_m"]
    feature_values = dataframe[selected_features].apply(pd.to_numeric, errors="coerce")
    quality_checks = {
        "all_completed_runs_processed": len(rows) == len(completed) and not failures,
        "all_charge_crosschecks_passed": bool(rows) and bool(dataframe["charge_crosscheck_passed"].all()),
        "all_selected_features_finite": bool(rows) and bool(np.isfinite(feature_values.to_numpy()).all()),
        "all_decision_windows_have_at_least_50_battery_samples": bool(rows)
        and bool((dataframe["decision_battery_sample_count"] >= 50).all()),
        "all_actual_distances_within_5m_of_request": bool(rows)
        and bool(((actual_distance - requested_distance).abs() <= 5.0).all()),
        "all_actual_altitudes_within_2m_of_request": bool(rows)
        and bool(((dataframe["relative_altitude_m"] - requested_altitude).abs() <= 2.0).all()),
        "all_active_motor_counts_positive": bool(rows)
        and bool((dataframe["active_motor_count"] > 0).all()),
        "unique_run_ids": dataframe["run_id"].nunique() == len(dataframe),
    }
    try:
        output_label = str(args.output.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        output_label = str(args.output.resolve())
    quality = {
        "manifest_rows": len(manifest),
        "manifest_completed": len(completed),
        "processed_rows": len(rows),
        "cache_hits": cache_hits,
        "cache_misses": len(misses),
        "processing_failures": failures,
        "split_counts": dataframe["split"].value_counts().sort_index().to_dict() if rows else {},
        "checks": quality_checks,
        "all_checks_passed": all(quality_checks.values()),
        "maximum_charge_crosscheck_error_mah": (
            float(dataframe["charge_crosscheck_error_mah"].max()) if rows else None
        ),
        "feature_set_sizes": {name: len(features) for name, features in feature_sets.items()},
        "maximum_runner_charge_error_mah": (
            float((dataframe["rtl_charge_runner_mah"] - dataframe["rtl_charge_mah"]).abs().max())
            if rows
            else None
        ),
        "maximum_distance_error_m": (
            float((actual_distance - requested_distance).abs().max()) if rows else None
        ),
        "maximum_altitude_error_m": (
            float((dataframe["relative_altitude_m"] - requested_altitude).abs().max()) if rows else None
        ),
        "output": output_label,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(quality, indent=2, sort_keys=True) + "\n")
    print(f"Built {len(rows)} processed rows at {args.output}")
    print(f"Quality report: {args.report}")
    if failures:
        raise SystemExit(f"{len(failures)} completed flights failed processing")
    if not quality["all_checks_passed"]:
        raise SystemExit("processed dataset quality checks failed")


if __name__ == "__main__":
    main()
