#!/usr/bin/env python3
"""Audit scenario coverage, determinism, splitting and provenance."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from rtl_charge.scenario import Scenario, assign_split, generate_scenarios


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NUMERIC_SCENARIO_COLUMNS = [
    "distance_m",
    "bearing_deg",
    "decision_altitude_m",
    "wind_speed_m_s",
    "wind_direction_deg",
    "payload_mass_kg",
    "turbulence_m_s",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        default=PROJECT_ROOT / "data" / "manifests" / "pipeline_scenario_validation.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load((PROJECT_ROOT / "config" / "experiment.yaml").read_text())
    dataframe = pd.read_csv(args.manifest)
    with args.manifest.open(newline="") as source:
        raw_rows = list(csv.DictReader(source))
    expected = generate_scenarios(
        int(config["dataset"]["pipeline_flights"]),
        seed=int(config["experiment"]["seed"]),
        ranges=config["scenario"],
        prefix="pipeline",
    )
    expected_by_id = {scenario.scenario_id: scenario for scenario in expected}

    fingerprints_match = True
    deterministic_values_match = True
    for row in raw_rows:
        expected_scenario = expected_by_id.get(row["scenario_id"])
        if expected_scenario is None:
            deterministic_values_match = False
            fingerprints_match = False
            continue
        actual = Scenario(
            scenario_id=row["scenario_id"],
            seed=int(row["seed"]),
            **{name: float(row[name]) for name in NUMERIC_SCENARIO_COLUMNS},
        )
        fingerprints_match &= actual.fingerprint() == row["scenario_fingerprint"]
        deterministic_values_match &= actual == expected_scenario

    correlations = dataframe[NUMERIC_SCENARIO_COLUMNS[:-1]].corr().abs()
    for name in correlations.columns:
        correlations.loc[name, name] = 0.0
    max_correlation = float(correlations.max().max())
    expected_fractions = config["dataset"]["split_fraction"]
    observed_fractions = dataframe["split"].value_counts(normalize=True).to_dict()
    split_fraction_error = {
        name: abs(float(observed_fractions.get(name, 0.0)) - float(expected_fractions[name]))
        for name in expected_fractions
    }
    patch_hash = sha256(PROJECT_ROOT / "patches" / "ardupilot_centered_payload.patch")
    parameter_hash = sha256(PROJECT_ROOT / "config" / "base_parameters.parm")
    checks = {
        "expected_row_count": len(dataframe) == int(config["dataset"]["pipeline_flights"]),
        "unique_scenario_ids": dataframe["scenario_id"].nunique() == len(dataframe),
        "unique_run_ids": dataframe["run_id"].nunique() == len(dataframe),
        "unique_fingerprints": dataframe["scenario_fingerprint"].nunique() == len(dataframe),
        "fingerprints_match_rows": fingerprints_match,
        "generation_is_reproducible": deterministic_values_match,
        "family_splits_match_hash_assignment": all(
            row["split"] == assign_split(row["scenario_family_id"]) for row in raw_rows
        ),
        "all_bearing_quadrants_covered": set((dataframe["bearing_deg"] // 90).astype(int)) == {0, 1, 2, 3},
        "maximum_absolute_correlation_below_0_25": max_correlation < 0.25,
        "split_fractions_within_0_03": max(split_fraction_error.values()) <= 0.03,
        "patch_hash_current": set(dataframe["patch_hash"]) == {patch_hash},
        "parameter_hash_current": set(dataframe["parameter_file_hash"]) == {parameter_hash},
    }
    report = {
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "row_count": len(dataframe),
        "status_counts": dataframe["status"].value_counts().sort_index().to_dict(),
        "split_counts": dataframe["split"].value_counts().sort_index().to_dict(),
        "split_fraction_error": split_fraction_error,
        "bearing_quadrant_counts": (dataframe["bearing_deg"] // 90).astype(int).value_counts().sort_index().to_dict(),
        "maximum_absolute_pairwise_correlation": max_correlation,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["all_checks_passed"]:
        raise SystemExit("scenario manifest validation failed")


if __name__ == "__main__":
    main()
