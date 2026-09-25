#!/usr/bin/env python3
"""Launch a fresh SITL instance, run one flight, and terminate it cleanly."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time

from rtl_charge.simulation import FlightError, SitlFlightClient


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARDUPILOT = PROJECT_ROOT / "external" / "ardupilot"
DEFAULT_BINARY = DEFAULT_ARDUPILOT / "build" / "sitl" / "bin" / "arducopter"
DEFAULT_PARAMETERS = PROJECT_ROOT / "config" / "base_parameters.parm"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--vehicle-binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--parameter-file", type=Path, default=DEFAULT_PARAMETERS)
    parser.add_argument("--altitude-m", type=float, default=30.0)
    parser.add_argument("--distance-m", type=float, default=200.0)
    parser.add_argument("--bearing-deg", type=float, default=0.0)
    parser.add_argument("--observation-window-s", type=float, default=10.0)
    parser.add_argument("--payload-mass-kg", type=float, required=True)
    parser.add_argument("--wind-speed-m-s", type=float, required=True)
    parser.add_argument("--wind-direction-deg", type=float, required=True)
    parser.add_argument("--scenario-id", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--instance", type=int, default=0)
    parser.add_argument("--speedup", type=float, default=2.0)
    parser.add_argument("--connection", default=None)
    return parser.parse_args()


def connect_client(connection: str, timeout_s: float = 30.0) -> SitlFlightClient:
    deadline = time.monotonic() + timeout_s
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            return SitlFlightClient(
                connection,
                progress=lambda text: print(text, flush=True),
            )
        except OSError as error:
            last_error = error
            time.sleep(0.5)
    raise FlightError(f"SITL TCP endpoint did not open: {last_error}")


def stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def main() -> None:
    args = parse_args()
    if args.instance < 0:
        raise SystemExit("instance must be non-negative")
    if not 1.0 <= args.speedup <= 4.0:
        raise SystemExit("speedup must be between 1 and 4")
    connection = args.connection or f"tcp:127.0.0.1:{5760 + 10 * args.instance}"
    run_dir = args.run_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    launch_log_path = run_dir / "sitl-launch.log"

    with tempfile.TemporaryDirectory(
        prefix="rtl-charge-sitl-",
        dir="/private/tmp" if os.uname().sysname == "Darwin" else None,
    ) as temporary_dir:
        staged_parameters = Path(temporary_dir) / "base_parameters.parm"
        shutil.copy2(args.parameter_file.resolve(), staged_parameters)
        command = [
            str(args.vehicle_binary.resolve()),
            "-w",
            "--model",
            "+",
            "--speedup",
            str(args.speedup),
            "--slave",
            "0",
            "--defaults",
            str(staged_parameters),
            "--sim-address=127.0.0.1",
            f"-I{args.instance}",
        ]

        with launch_log_path.open("w") as launch_log:
            process = subprocess.Popen(
                command,
                cwd=run_dir,
                stdout=launch_log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            client: SitlFlightClient | None = None
            try:
                client = connect_client(connection)
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
                    sitl_speedup=args.speedup,
                )
                result.write_json(result_path)
                print(
                    f"RTL complete: {result.rtl_charge_mah} mAh in "
                    f"{result.rtl_duration_s:.1f} wall s; automatic disarm observed"
                )
            except (FlightError, OSError, ValueError) as error:
                failure = {
                    "status": "failed",
                    "error": str(error),
                    "failed_unix_s": time.time(),
                    "connection": connection,
                    "altitude_m": args.altitude_m,
                    "outbound_distance_m": args.distance_m,
                    "outbound_bearing_deg": args.bearing_deg,
                    "observation_window_s": args.observation_window_s,
                    "payload_mass_kg": args.payload_mass_kg,
                    "wind_speed_m_s": args.wind_speed_m_s,
                    "wind_direction_deg": args.wind_direction_deg,
                    "scenario_id": args.scenario_id,
                    "run_id": args.run_id,
                    "sitl_speedup": args.speedup,
                    "recent_status_text": [] if client is None else client.status_text[-20:],
                }
                result_path.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
                raise
            finally:
                stop_process(process)


if __name__ == "__main__":
    main()
