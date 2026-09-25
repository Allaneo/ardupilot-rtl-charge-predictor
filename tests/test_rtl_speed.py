"""Tests for middle-route crossing interpolation."""

import pytest

from scripts.analyze_rtl_speed import crossing_time, extract_one


def test_crossing_time_interpolates_first_inward_crossing():
    assert crossing_time([(0.0, 100.0), (2.0, 80.0), (4.0, 40.0)], 60.0) == pytest.approx(3.0)


def test_crossing_time_rejects_missing_crossing():
    with pytest.raises(ValueError, match="no inward crossing"):
        crossing_time([(0.0, 100.0), (2.0, 80.0)], 60.0)


def test_extractor_refuses_test_split_before_opening_a_log():
    with pytest.raises(ValueError, match="held-out split"):
        extract_one(("test-flight", "test", "missing.BIN", 500.0, 60.0))
