from rtl_charge.scenario import (
    Scenario,
    assign_split,
    build_manifest_rows,
    generate_scenarios,
)


def test_scenario_fingerprint_is_stable() -> None:
    scenario = Scenario(
        scenario_id="engineering-001",
        seed=42,
        distance_m=500.0,
        bearing_deg=90.0,
        decision_altitude_m=60.0,
        wind_speed_m_s=4.0,
        wind_direction_deg=270.0,
        payload_mass_kg=0.5,
    )

    assert scenario.fingerprint() == scenario.fingerprint()
    assert len(scenario.fingerprint()) == 64


def test_split_assignment_is_stable_for_scenario_family() -> None:
    assert assign_split("family-17") == assign_split("family-17")


def test_scenario_generation_is_deterministic_and_stratified() -> None:
    ranges = {
        "distance_m": [200.0, 1500.0],
        "bearing_deg": [0.0, 360.0],
        "decision_altitude_m": [30.0, 120.0],
        "wind_speed_m_s": [0.0, 6.0],
        "wind_direction_deg": [0.0, 360.0],
        "payload_mass_kg": [0.0, 1.0],
        "turbulence_m_s": [0.0, 0.0],
    }
    first = generate_scenarios(8, seed=1234, ranges=ranges)
    second = generate_scenarios(8, seed=1234, ranges=ranges)

    assert first == second
    assert {int(item.bearing_deg // 90) for item in first} == {0, 1, 2, 3}
    assert all(200.0 <= item.distance_m <= 1500.0 for item in first)
    assert all(0.0 <= item.payload_mass_kg <= 1.0 for item in first)


def test_manifest_preserves_scenario_provenance() -> None:
    scenario = Scenario(
        scenario_id="pipeline-0001",
        seed=42,
        distance_m=500.0,
        bearing_deg=90.0,
        decision_altitude_m=60.0,
        wind_speed_m_s=4.0,
        wind_direction_deg=270.0,
        payload_mass_kg=0.5,
    )
    row = build_manifest_rows(
        [scenario],
        ardupilot_commit="abc",
        patch_hash="def",
        parameter_file_hash="ghi",
    )[0]

    assert row.status == "pending"
    assert row.scenario_fingerprint == scenario.fingerprint()
    assert row.raw_log_path.endswith("pipeline-0001-run-001/logs/00000001.BIN")
