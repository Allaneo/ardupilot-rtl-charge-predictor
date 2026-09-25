#!/usr/bin/env python3
"""Execute one complete outbound-and-RTL flight against a running SITL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from rtl_charge.simulation import FlightError, SitlFlightClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--connection", default="tcp:127.0.0.1:5760")
    parser.add_argument("--altitude-m", type=float, default=20.0)
    parser.add_argument("--distance-m", type=float, default=100.0)
    parser.add_argument("--bearing-deg", type=float, default=0.0)
    parser.add_argument("--observation-window-s", type=float, default=10.0)
    parser.add_argument(
        "--payload-mass-kg",
        type=float,
        default=None,
        help="Set and verify SIM_PAYLOAD_MASS before flight (patched SITL only).",
    )
    parser.add_argument("--wind-speed-m-s", type=float, default=None)
    parser.add_argument("--wind-direction-deg", type=float, default=None)
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = SitlFlightClient(args.connection, progress=lambda text: print(text, flush=True))
    try:
        result = client.run_outbound_rtl(
            altitude_m=args.altitude_m,
            outbound_distance_m=args.distance_m,
            outbound_bearing_deg=args.bearing_deg,
            observation_window_s=args.observation_window_s,
            payload_mass_kg=args.payload_mass_kg,
            wind_speed_m_s=args.wind_speed_m_s,
            wind_direction_deg=args.wind_direction_deg,
            scenario_id=args.scenario_id,
            run_id=args.run_id,
        )
    except FlightError as error:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        failure = {
            "status": "failed",
            "error": str(error),
            "failed_unix_s": time.time(),
            "connection": args.connection,
            "altitude_m": args.altitude_m,
            "outbound_distance_m": args.distance_m,
            "outbound_bearing_deg": args.bearing_deg,
            "observation_window_s": args.observation_window_s,
            "payload_mass_kg": args.payload_mass_kg,
            "wind_speed_m_s": args.wind_speed_m_s,
            "wind_direction_deg": args.wind_direction_deg,
            "scenario_id": args.scenario_id,
            "run_id": args.run_id,
            "recent_status_text": client.status_text[-20:],
        }
        args.output.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        raise
    result.write_json(args.output)
    print(
        f"RTL complete: {result.rtl_charge_mah} mAh in "
        f"{result.rtl_duration_s:.1f} s; automatic disarm observed"
    )


if __name__ == "__main__":
    main()
