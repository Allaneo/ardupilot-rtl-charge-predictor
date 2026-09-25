"""Scenario definitions and deterministic split assignment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import random
from typing import Literal, Mapping, Sequence


SplitName = Literal["train", "validation", "test"]

SCENARIO_VARIABLES = (
    "distance_m",
    "bearing_deg",
    "decision_altitude_m",
    "wind_speed_m_s",
    "wind_direction_deg",
    "payload_mass_kg",
    "turbulence_m_s",
)


@dataclass(frozen=True)
class Scenario:
    """Simulator truth for one complete outbound-and-RTL flight."""

    scenario_id: str
    seed: int
    distance_m: float
    bearing_deg: float
    decision_altitude_m: float
    wind_speed_m_s: float
    wind_direction_deg: float
    payload_mass_kg: float
    turbulence_m_s: float = 0.0

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def fingerprint(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ScenarioManifestRow:
    """One pre-assigned flight request and its immutable provenance."""

    scenario_id: str
    run_id: str
    scenario_family_id: str
    seed: int
    split: SplitName
    status: str
    failure_reason: str
    ardupilot_commit: str
    patch_hash: str
    parameter_file_hash: str
    scenario_fingerprint: str
    distance_m: float
    bearing_deg: float
    decision_altitude_m: float
    wind_speed_m_s: float
    wind_direction_deg: float
    payload_mass_kg: float
    turbulence_m_s: float
    observation_window_s: float
    started_unix_s: str
    ended_unix_s: str
    raw_log_path: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_range(name: str, limits: Sequence[float]) -> tuple[float, float]:
    if len(limits) != 2:
        raise ValueError(f"{name} must contain exactly two bounds")
    lower, upper = float(limits[0]), float(limits[1])
    if lower > upper:
        raise ValueError(f"{name} lower bound exceeds upper bound")
    return lower, upper


def _latin_hypercube_column(
    count: int,
    lower: float,
    upper: float,
    rng: random.Random,
) -> list[float]:
    if lower == upper:
        return [lower] * count
    values = [lower + (upper - lower) * ((index + rng.random()) / count) for index in range(count)]
    rng.shuffle(values)
    return values


def generate_scenarios(
    count: int,
    *,
    seed: int,
    ranges: Mapping[str, Sequence[float]],
    prefix: str = "pipeline",
) -> list[Scenario]:
    """Generate deterministic, independently stratified scenario dimensions."""

    if count <= 0:
        raise ValueError("count must be positive")
    missing = set(SCENARIO_VARIABLES).difference(ranges)
    if missing:
        raise ValueError(f"missing scenario ranges: {sorted(missing)}")

    rng = random.Random(seed)
    columns: dict[str, list[float]] = {}
    for name in SCENARIO_VARIABLES:
        lower, upper = _validate_range(name, ranges[name])
        columns[name] = _latin_hypercube_column(count, lower, upper, rng)

    scenarios = []
    for index in range(count):
        scenarios.append(
            Scenario(
                scenario_id=f"{prefix}-{index + 1:04d}",
                seed=rng.getrandbits(63),
                distance_m=columns["distance_m"][index],
                bearing_deg=columns["bearing_deg"][index],
                decision_altitude_m=columns["decision_altitude_m"][index],
                wind_speed_m_s=columns["wind_speed_m_s"][index],
                wind_direction_deg=columns["wind_direction_deg"][index],
                payload_mass_kg=columns["payload_mass_kg"][index],
                turbulence_m_s=columns["turbulence_m_s"][index],
            )
        )
    return scenarios


def build_manifest_rows(
    scenarios: Sequence[Scenario],
    *,
    ardupilot_commit: str,
    patch_hash: str,
    parameter_file_hash: str,
    raw_root: str = "data/raw",
    observation_window_s: float = 10.0,
) -> list[ScenarioManifestRow]:
    """Create pending manifest rows with stable family-level split assignment."""

    rows = []
    for scenario in scenarios:
        family_id = f"{scenario.scenario_id}-family"
        run_id = f"{scenario.scenario_id}-run-001"
        rows.append(
            ScenarioManifestRow(
                scenario_id=scenario.scenario_id,
                run_id=run_id,
                scenario_family_id=family_id,
                seed=scenario.seed,
                split=assign_split(family_id),
                status="pending",
                failure_reason="",
                ardupilot_commit=ardupilot_commit,
                patch_hash=patch_hash,
                parameter_file_hash=parameter_file_hash,
                scenario_fingerprint=scenario.fingerprint(),
                distance_m=scenario.distance_m,
                bearing_deg=scenario.bearing_deg,
                decision_altitude_m=scenario.decision_altitude_m,
                wind_speed_m_s=scenario.wind_speed_m_s,
                wind_direction_deg=scenario.wind_direction_deg,
                payload_mass_kg=scenario.payload_mass_kg,
                turbulence_m_s=scenario.turbulence_m_s,
                observation_window_s=observation_window_s,
                started_unix_s="",
                ended_unix_s="",
                raw_log_path=f"{raw_root}/{run_id}/logs/00000001.BIN",
            )
        )
    return rows


def assign_split(
    scenario_family_id: str,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> SplitName:
    """Assign an entire scenario family to a stable dataset split."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train and validation fractions must leave room for test")

    digest = hashlib.sha256(scenario_family_id.encode("utf-8")).digest()
    unit_interval = int.from_bytes(digest[:8], "big") / 2**64
    if unit_interval < train_fraction:
        return "train"
    if unit_interval < train_fraction + validation_fraction:
        return "validation"
    return "test"
