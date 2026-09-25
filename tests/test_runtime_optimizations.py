from argparse import Namespace
import csv
from pathlib import Path
from threading import Event
from types import SimpleNamespace

from scripts import build_dataset, run_manifest
from rtl_charge.simulation import SitlFlightClient


def test_cache_invalidates_when_raw_log_changes(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "result.json").write_text('{"automatic_disarm_observed": true}')
    log_path = run_dir / "logs" / "00000001.BIN"
    log_path.write_bytes(b"first log")
    cache_path = tmp_path / "cache.json"
    first_key = build_dataset.cache_key(run_dir, "processor-v1")
    row = {"rtl_charge_mah": 123.4}

    build_dataset.write_cache(cache_path, first_key, row)
    assert build_dataset.read_cache(cache_path, first_key) == row

    log_path.write_bytes(b"changed log")
    assert build_dataset.read_cache(
        cache_path, build_dataset.cache_key(run_dir, "processor-v1")
    ) is None
    assert build_dataset.read_cache(cache_path, first_key | {"processor_hash": "v2"}) is None


def test_manifest_starts_next_flight_when_an_instance_finishes(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    rows = [
        {
            "scenario_id": f"scenario-{index}",
            "run_id": f"run-{index}",
            "status": "pending",
            "patch_hash": "expected",
            "parameter_file_hash": "expected",
            "failure_reason": "",
            "started_unix_s": "",
            "ended_unix_s": "",
        }
        for index in range(1, 4)
    ]
    fieldnames = list(rows[0])
    run_manifest.write_manifest(manifest_path, fieldnames, rows)
    third_started = Event()

    def fake_flight(row: dict[str, str], instance: int, speedup: float) -> tuple[int, str]:
        assert speedup == 2.0
        if row["run_id"] == "run-1":
            assert third_started.wait(2.0), "idle instance did not start the third flight"
        if row["run_id"] == "run-3":
            third_started.set()
        return 0, ""

    monkeypatch.setattr(run_manifest, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(run_manifest, "sha256", lambda path: "expected")
    monkeypatch.setattr(run_manifest, "run_flight_process", fake_flight)
    monkeypatch.setattr(
        run_manifest,
        "parse_args",
        lambda: Namespace(
            manifest=manifest_path,
            limit=None,
            splits=None,
            scenario_ids=None,
            jobs=2,
            speedup=2.0,
        ),
    )

    run_manifest.main()

    with manifest_path.open(newline="") as source:
        finished = list(csv.DictReader(source))
    assert third_started.is_set()
    assert [row["status"] for row in finished] == ["completed"] * 3


def test_observation_window_uses_simulated_clock() -> None:
    client = object.__new__(SitlFlightClient)
    client.progress = None
    client.latest = {"GLOBAL_POSITION_INT": SimpleNamespace(time_boot_ms=1_000)}
    messages = iter(
        SimpleNamespace(get_type=lambda: "GLOBAL_POSITION_INT", time_boot_ms=boot_ms)
        for boot_ms in (3_000, 7_000, 11_000)
    )
    seen = []

    def receive(*, timeout_s: float):
        message = next(messages)
        seen.append(message.time_boot_ms)
        return message

    client._receive = receive
    client.observe(10.0)

    assert seen == [3_000, 7_000, 11_000]
