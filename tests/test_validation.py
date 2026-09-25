import pytest

from rtl_charge.validation import (
    PayloadRunMetrics,
    WindRunMetrics,
    circular_error_deg,
    payload_validation_checks,
    wind_validation_checks,
    wind_vector_from_direction,
)


def make_run(
    name: str,
    payload: float | None,
    charge: int,
    current: float,
    throttle: float,
) -> PayloadRunMetrics:
    return PayloadRunMetrics(
        run_name=name,
        payload_mass_kg=payload,
        rtl_charge_mah=charge,
        rtl_duration_s=53.0,
        automatic_disarm_observed=True,
        rtl_start_boot_s=80.0,
        window_duration_s=10.0,
        current_sample_count=100,
        hover_current_mean_a=current,
        hover_current_std_a=0.2,
        hover_throttle_mean=throttle,
        motor_pwm_mean=1600.0,
    )


def test_payload_validation_accepts_equivalent_zero_and_monotonic_payload() -> None:
    control = make_run("control", None, 413, 28.90, 0.3722)
    patched = [
        make_run("zero", 0.0, 412, 28.90, 0.3723),
        make_run("half", 0.5, 482, 33.85, 0.4341),
        make_run("one", 1.0, 553, 38.85, 0.4970),
    ]

    assert all(payload_validation_checks(control, patched).values())


def test_payload_validation_rejects_non_monotonic_charge() -> None:
    control = make_run("control", None, 413, 28.90, 0.3722)
    patched = [
        make_run("zero", 0.0, 412, 28.90, 0.3723),
        make_run("half", 0.5, 400, 33.85, 0.4341),
    ]

    checks = payload_validation_checks(control, patched)
    assert not checks["rtl_charge_increases_with_payload"]


def test_wind_direction_conversion_uses_direction_from_convention() -> None:
    north, east = wind_vector_from_direction(5.0, 0.0)
    assert north == pytest.approx(-5.0)
    assert east == pytest.approx(0.0, abs=1e-12)
    assert circular_error_deg(359.0, 1.0) == pytest.approx(2.0)


def test_wind_validation_accepts_finite_accurate_complete_run() -> None:
    run = WindRunMetrics(
        run_name="north-five",
        payload_mass_kg=0.0,
        true_wind_speed_m_s=5.0,
        true_wind_direction_deg=0.0,
        true_wind_north_m_s=-5.0,
        true_wind_east_m_s=0.0,
        estimated_wind_north_mean_m_s=-5.5,
        estimated_wind_east_mean_m_s=-0.1,
        estimated_wind_north_std_m_s=0.05,
        estimated_wind_east_std_m_s=0.04,
        estimated_wind_speed_m_s=5.5,
        estimated_wind_direction_deg=359.0,
        wind_vector_error_m_s=0.51,
        wind_direction_error_deg=1.0,
        sample_count=100,
        rtl_start_boot_s=110.0,
        window_duration_s=10.0,
        automatic_disarm_observed=True,
    )

    assert all(wind_validation_checks([run]).values())
