#!/usr/bin/env python3
"""Measure RTL cruise speed from saved train/validation DataFlash logs only.

Post-RTL observations in this output are diagnostic or auxiliary training labels.
They must never enter a deployable feature vector at decision time.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pymavlink import mavutil


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/processed/pipeline_dataset.csv"
OUTPUT = ROOT / "reports/m9/rtl_speed_train_validation.csv"


def crossing_time(points: list[tuple[float, float]], threshold: float) -> float:
    """Interpolate the first inward crossing of a home-distance threshold."""
    for (t0, d0), (t1, d1) in zip(points, points[1:]):
        if d0 > threshold >= d1:
            return t0 + (threshold - d0) * (t1 - t0) / (d1 - d0)
    raise ValueError(f"no inward crossing of {threshold:.3f} m")


DEFAULT_SPLITS = frozenset({"train", "validation"})


def extract_one(record: tuple) -> dict[str, float | str]:
    """Record: run_id, split, raw log path, distance, altitude[, allowed splits]."""
    run_id, split, raw_path, distance, initial_altitude = record[:5]
    allowed_splits = record[5] if len(record) > 5 else DEFAULT_SPLITS
    if split not in allowed_splits:
        raise ValueError(f"refusing to inspect held-out split: {split}")
    log_path = ROOT / raw_path
    if not log_path.is_file():
        raise FileNotFoundError(log_path)

    log = mavutil.mavlink_connection(str(log_path))
    rtl_us = None
    disarm_found = False
    positions: list[tuple[float, float, float, float]] = []
    attitudes: list[tuple[float, float]] = []
    north_targets: list[tuple[float, float]] = []
    east_targets: list[tuple[float, float]] = []
    while True:
        message = log.recv_match()
        if message is None:
            break
        kind = message.get_type()
        if kind == "MODE" and rtl_us is None and int(message.ModeNum) == 6:
            rtl_us = int(message.TimeUS)
        elif kind == "ARM" and rtl_us is not None and int(message.ArmState) == 0:
            disarm_found = True
            break
        elif rtl_us is not None and kind == "XKF1" and int(message.C) == 0:
            north, east = float(message.PN), float(message.PE)
            radius = math.hypot(north, east)
            radial_speed = (
                -(north * float(message.VN) + east * float(message.VE)) / radius
                if radius > 1.0 else 0.0
            )
            positions.append((int(message.TimeUS) / 1e6, radius, radial_speed, -float(message.PD)))
        elif rtl_us is not None and kind == "ATT":
            roll, pitch = math.radians(float(message.Roll)), math.radians(float(message.Pitch))
            lean = math.degrees(math.acos(max(-1.0, min(1.0, math.cos(roll) * math.cos(pitch)))))
            attitudes.append((int(message.TimeUS) / 1e6, lean))
        elif rtl_us is not None and kind == "PIDN":
            north_targets.append((int(message.TimeUS) / 1e6, float(message.Tar)))
        elif rtl_us is not None and kind == "PIDE":
            east_targets.append((int(message.TimeUS) / 1e6, float(message.Tar)))

    if rtl_us is None or not disarm_found:
        raise ValueError(f"incomplete RTL interval: {run_id}")
    if len(positions) < 20:
        raise ValueError(f"too few RTL position samples: {run_id}")

    crossings = [(time, radius) for time, radius, _, _ in positions]
    start_s = crossing_time(crossings, 0.8 * distance)
    end_s = crossing_time(crossings, 0.2 * distance)
    if end_s <= start_s:
        raise ValueError(f"non-positive middle-route duration: {run_id}")
    cruise = [
        (speed, altitude) for time, _, speed, altitude in positions
        if start_s <= time <= end_s
    ]
    if len(cruise) < 20:
        raise ValueError(f"too few middle-route samples: {run_id}")
    lean = [angle for time, angle in attitudes if start_s <= time <= end_s]
    north = [value for time, value in north_targets if start_s <= time <= end_s]
    east = [value for time, value in east_targets if start_s <= time <= end_s]
    if not lean or not north or not east:
        raise ValueError(f"missing cruise attitude or velocity target: {run_id}")
    if len(north) != len(east):
        raise ValueError(f"unequal cruise target samples: {run_id}")

    target_magnitudes = np.hypot(north, east)
    actual_speed = 0.6 * distance / (end_s - start_s)
    return {
        "run_id": run_id,
        "split": split,
        "middle_route_duration_s": end_s - start_s,
        "middle_route_ground_speed_m_s": actual_speed,
        "median_radial_ground_speed_m_s": float(np.median([value[0] for value in cruise])),
        "median_target_ground_speed_m_s": float(np.median(target_magnitudes)),
        "median_lean_deg": float(np.median(lean)),
        "lean_near_30deg_fraction": float(np.mean(np.asarray(lean) >= 29.5)),
        "median_altitude_delta_m": float(np.median([value[1] - initial_altitude for value in cruise])),
        "n_middle_position_samples": len(cruise),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-flights", type=int, default=0, help="diagnostic subset only")
    parser.add_argument("--split", action="append", dest="splits", choices=("train", "validation", "test"),
                        help="splits to inspect (default: train and validation)")
    parser.add_argument("--expected-count", type=int, default=211)
    args = parser.parse_args()
    allowed_splits = frozenset(args.splits) if args.splits else DEFAULT_SPLITS
    frame = pd.read_csv(
        args.dataset,
        usecols=["run_id", "split", "raw_log_path", "distance_home_m", "relative_altitude_m"],
    )
    frame = frame.loc[frame.split.isin(allowed_splits)]
    if len(frame) != args.expected_count and not args.max_flights:
        raise ValueError(f"expected exactly {args.expected_count} flights, got {len(frame)}")
    if args.max_flights:
        frame = frame.head(args.max_flights)
    records = [
        (str(row.run_id), str(row.split), str(row.raw_log_path),
         float(row.distance_home_m), float(row.relative_altitude_m), allowed_splits)
        for row in frame.itertuples(index=False)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        result = list(executor.map(extract_one, records))
    output = pd.DataFrame(result)
    if output.run_id.duplicated().any() or not np.isfinite(output.select_dtypes(include="number")).all().all():
        raise ValueError("duplicate run IDs or non-finite diagnostic values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"Extracted {len(output)} RTL speed diagnostics to {args.output}")


if __name__ == "__main__":
    main()
