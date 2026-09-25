from pathlib import Path
from types import SimpleNamespace

import pytest

from rtl_charge.processing import ProcessingError, _motor_features, build_flight_row


def _motor_message(count: int, pwm: int = 1500) -> SimpleNamespace:
    values = {f"C{index}": pwm if index <= count else 0 for index in range(1, 15)}
    return SimpleNamespace(**values)


@pytest.mark.parametrize("motor_count", [4, 6])
def test_motor_features_are_fixed_length_for_different_topologies(motor_count: int) -> None:
    features = _motor_features(
        [_motor_message(motor_count)],
        pwm_min=1000.0,
        pwm_max=2000.0,
    )

    assert set(features) == {
        "active_motor_count",
        "motor_output_mean",
        "motor_output_std",
        "motor_output_max",
        "motor_saturation_fraction",
    }
    assert features["active_motor_count"] == motor_count
    assert features["motor_output_mean"] == 0.5


def test_real_engineering_log_passes_charge_crosscheck() -> None:
    run_dir = Path("data/raw/m3-wind-5ms-000deg-p1-001")
    if not run_dir.exists():
        pytest.skip("local engineering log is not present")

    row = build_flight_row(run_dir)

    assert row["charge_crosscheck_passed"] is True
    assert row["charge_crosscheck_error_mah"] < 2.0
    assert row["active_motor_count"] == 4
    assert row["decision_battery_sample_count"] >= 50
    assert abs(row["rtl_charge_mah"] - 731.091) < 0.01


def test_incomplete_engineering_flight_is_rejected() -> None:
    run_dir = Path("data/raw/m3-wind-8ms-270deg-p0.5-001")
    if not run_dir.exists():
        pytest.skip("local failed engineering run is not present")

    with pytest.raises(ProcessingError, match="automatic disarm"):
        build_flight_row(run_dir)
