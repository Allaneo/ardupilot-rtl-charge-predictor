"""Feature allowlists and leakage guards."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml


PRIVILEGED_COLUMNS = frozenset(
    {
        "payload_mass_kg",
        "true_wind_north_m_s",
        "true_wind_east_m_s",
        "sitl_speedup",
    }
)

TARGET_COLUMNS = frozenset(
    {
        "rtl_charge_mah",
        "rtl_energy_wh",
        "rtl_duration_s",
    }
)

POST_DECISION_DIAGNOSTIC_COLUMNS = frozenset(
    {
        "disarm_boot_s",
        "rtl_charge_integrated_mah",
        "rtl_charge_runner_mah",
        "charge_crosscheck_error_mah",
        "charge_crosscheck_tolerance_mah",
        "charge_crosscheck_passed",
    }
)


def assert_deployable_features(feature_names: Iterable[str]) -> tuple[str, ...]:
    """Return a normalized feature tuple or fail on known leakage columns."""

    normalized = tuple(feature_names)
    duplicates = {name for name in normalized if normalized.count(name) > 1}
    if duplicates:
        raise ValueError(f"duplicate feature names: {sorted(duplicates)}")

    forbidden = (
        PRIVILEGED_COLUMNS | TARGET_COLUMNS | POST_DECISION_DIAGNOSTIC_COLUMNS
    ).intersection(normalized)
    if forbidden:
        raise ValueError(f"non-deployable columns selected: {sorted(forbidden)}")
    return normalized


def load_feature_sets(path: Path) -> dict[str, tuple[str, ...]]:
    """Load, resolve and leakage-check all named model feature sets."""

    document = yaml.safe_load(path.read_text())
    resolved: dict[str, tuple[str, ...]] = {}

    def resolve(name: str, visiting: frozenset[str] = frozenset()) -> tuple[str, ...]:
        if name in resolved:
            return resolved[name]
        if name in visiting:
            raise ValueError(f"cyclic feature-set inheritance at {name}")
        definition = document[name]
        if isinstance(definition, list):
            own = definition
            parent: tuple[str, ...] = ()
        else:
            own = definition.get("features", [])
            parent_name = definition.get("extends")
            parent = () if parent_name is None else resolve(parent_name, visiting | {name})
        resolved[name] = assert_deployable_features((*parent, *own))
        return resolved[name]

    for feature_set_name in document:
        if feature_set_name.startswith("f"):
            resolve(feature_set_name)
    return resolved
