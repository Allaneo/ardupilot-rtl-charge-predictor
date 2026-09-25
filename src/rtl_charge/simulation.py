"""MAVLink control for one outbound-and-RTL SITL flight."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional

from pymavlink import mavutil


@dataclass(frozen=True)
class FlightResult:
    """Minimal verified result from one complete RTL experiment."""

    connection: str
    altitude_m: float
    outbound_distance_m: float
    observation_window_s: float
    payload_mass_kg: Optional[float]
    wind_speed_m_s: Optional[float]
    wind_direction_deg: Optional[float]
    rtl_started_unix_s: float
    disarmed_unix_s: float
    rtl_duration_s: float
    charge_at_rtl_mah: int
    charge_at_disarm_mah: int
    rtl_charge_mah: int
    final_relative_altitude_m: float
    automatic_disarm_observed: bool
    outbound_bearing_deg: float = 0.0
    scenario_id: Optional[str] = None
    run_id: Optional[str] = None
    sitl_speedup: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def write_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")


class FlightError(RuntimeError):
    """Raised when the simulated flight fails a required gate."""


class SitlFlightClient:
    """Small synchronous controller for a single ArduCopter SITL instance."""

    _POSITION_ONLY_TYPE_MASK = 0b110111111000

    def __init__(
        self,
        connection: str,
        *,
        source_system: int = 250,
        progress: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.connection_string = connection
        self.master = mavutil.mavlink_connection(
            connection,
            source_system=source_system,
            autoreconnect=True,
        )
        self.latest: Dict[str, Any] = {}
        self.status_text: List[str] = []
        self.progress = progress

    def _report(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def connect(self, timeout_s: float = 30.0) -> None:
        heartbeat = self.master.wait_heartbeat(timeout=timeout_s)
        if heartbeat is None:
            raise FlightError(f"no heartbeat received from {self.connection_string}")
        self.latest[heartbeat.get_type()] = heartbeat
        self._request_message_intervals()
        self._report("MAVLink heartbeat received")

    def _request_message_intervals(self) -> None:
        message_rates_hz = {
            mavutil.mavlink.MAVLINK_MSG_ID_HEARTBEAT: 2,
            mavutil.mavlink.MAVLINK_MSG_ID_GLOBAL_POSITION_INT: 10,
            mavutil.mavlink.MAVLINK_MSG_ID_LOCAL_POSITION_NED: 10,
            mavutil.mavlink.MAVLINK_MSG_ID_BATTERY_STATUS: 5,
            mavutil.mavlink.MAVLINK_MSG_ID_ATTITUDE: 10,
            mavutil.mavlink.MAVLINK_MSG_ID_SYS_STATUS: 2,
            mavutil.mavlink.MAVLINK_MSG_ID_SERVO_OUTPUT_RAW: 10,
        }
        for message_id, rate_hz in message_rates_hz.items():
            interval_us = int(1_000_000 / rate_hz)
            self.master.mav.command_long_send(
                self.master.target_system,
                self.master.target_component,
                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL,
                0,
                message_id,
                interval_us,
                0,
                0,
                0,
                0,
                0,
            )

    def _receive(self, *, timeout_s: float = 1.0) -> Optional[Any]:
        message = self.master.recv_match(blocking=True, timeout=timeout_s)
        if message is not None and message.get_type() != "BAD_DATA":
            self.latest[message.get_type()] = message
            if message.get_type() == "STATUSTEXT":
                self.status_text.append(str(message.text))
                self.status_text = self.status_text[-20:]
        return message

    def _wait_for_type(self, message_type: str, timeout_s: float) -> Any:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            message = self._receive(timeout_s=min(1.0, deadline - time.monotonic()))
            if message is not None and message.get_type() == message_type:
                return message
        raise FlightError(f"timed out waiting for {message_type}")

    def wait_until_ready(self, timeout_s: float = 90.0) -> None:
        """Wait for usable position and a battery-consumption reading."""

        deadline = time.monotonic() + timeout_s
        position_seen = False
        battery_seen = False
        while time.monotonic() < deadline:
            message = self._receive(timeout_s=1.0)
            if message is None:
                continue
            if message.get_type() == "GLOBAL_POSITION_INT":
                position_seen = int(message.lat) != 0 and int(message.lon) != 0
            elif message.get_type() == "BATTERY_STATUS":
                battery_seen = int(message.current_consumed) >= 0
            if position_seen and battery_seen:
                self._report("Position and battery telemetry ready")
                return
        raise FlightError(
            "SITL did not provide both a global position and battery charge reading"
        )

    @staticmethod
    def _parameter_name(message: Any) -> str:
        raw_name = message.param_id
        if isinstance(raw_name, bytes):
            return raw_name.decode("ascii").rstrip("\x00")
        return str(raw_name).rstrip("\x00")

    def set_parameter(
        self,
        name: str,
        value: float,
        *,
        timeout_s: float = 15.0,
    ) -> float:
        """Set one float parameter and require a matching value response."""

        if len(name) > 16:
            raise ValueError("MAVLink parameter names cannot exceed 16 characters")
        deadline = time.monotonic() + timeout_s
        next_request = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_request:
                self.master.mav.param_set_send(
                    self.master.target_system,
                    self.master.target_component,
                    name.encode("ascii"),
                    float(value),
                    mavutil.mavlink.MAV_PARAM_TYPE_REAL32,
                )
                next_request = now + 1.0
            message = self._receive(timeout_s=min(1.0, deadline - time.monotonic()))
            if message is None or message.get_type() != "PARAM_VALUE":
                continue
            if self._parameter_name(message) != name:
                continue
            confirmed = float(message.param_value)
            if math.isclose(confirmed, value, rel_tol=1e-5, abs_tol=1e-5):
                self._report(f"Parameter {name} confirmed at {confirmed:g}")
                return confirmed
        raise FlightError(f"parameter {name} was not confirmed at {value:g}")

    def set_mode(self, mode_name: str, timeout_s: float = 20.0) -> None:
        mapping = self.master.mode_mapping()
        if mode_name not in mapping:
            raise FlightError(f"mode {mode_name!r} is unavailable")
        requested_mode = mapping[mode_name]
        self.master.mav.set_mode_send(
            self.master.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            requested_mode,
        )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            message = self._receive(timeout_s=1.0)
            if (
                message is not None
                and message.get_type() == "HEARTBEAT"
                and int(message.custom_mode) == requested_mode
            ):
                self._report(f"Mode changed to {mode_name}")
                return
        raise FlightError(f"vehicle did not enter {mode_name} mode")

    def arm(self, timeout_s: float = 60.0) -> None:
        deadline = time.monotonic() + timeout_s
        next_request = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_request:
                self.master.arducopter_arm()
                next_request = now + 5.0
            message = self._receive(timeout_s=1.0)
            if message is not None and message.get_type() == "HEARTBEAT":
                if message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
                    self._report("Vehicle armed")
                    return
        detail = "; ".join(self.status_text[-5:]) or "no STATUSTEXT received"
        raise FlightError(f"vehicle did not arm: {detail}")

    def takeoff(self, altitude_m: float, timeout_s: float = 90.0) -> None:
        self.master.mav.command_long_send(
            self.master.target_system,
            self.master.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0,
            0,
            0,
            0,
            math.nan,
            0,
            0,
            altitude_m,
        )
        self._wait_for_relative_altitude(
            minimum_m=altitude_m * 0.90,
            timeout_s=timeout_s,
        )
        self._report(f"Takeoff altitude reached ({altitude_m:.1f} m commanded)")

    def _wait_for_relative_altitude(self, minimum_m: float, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            message = self._receive(timeout_s=1.0)
            if message is not None and message.get_type() == "GLOBAL_POSITION_INT":
                if message.relative_alt / 1000.0 >= minimum_m:
                    return
        raise FlightError(f"vehicle did not reach {minimum_m:.1f} m relative altitude")

    def fly_to_bearing(
        self,
        distance_m: float,
        bearing_deg: float,
        altitude_m: float,
        timeout_s: float | None = None,
    ) -> None:
        """Fly to a local-NED point at the requested compass bearing."""

        if distance_m <= 0:
            raise ValueError("distance_m must be positive")
        if not 0.0 <= bearing_deg <= 360.0:
            raise ValueError("bearing_deg must be between 0 and 360")
        bearing_rad = math.radians(bearing_deg)
        north_m = distance_m * math.cos(bearing_rad)
        east_m = distance_m * math.sin(bearing_rad)
        if timeout_s is None:
            timeout_s = max(120.0, distance_m / 5.0 + 90.0)
        deadline = time.monotonic() + timeout_s
        next_setpoint = 0.0
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_setpoint:
                self.master.mav.set_position_target_local_ned_send(
                    0,
                    self.master.target_system,
                    self.master.target_component,
                    mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                    self._POSITION_ONLY_TYPE_MASK,
                    north_m,
                    east_m,
                    -altitude_m,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
                next_setpoint = now + 1.0
            message = self._receive(timeout_s=0.25)
            if message is not None and message.get_type() == "LOCAL_POSITION_NED":
                horizontal_error = math.hypot(
                    message.x - north_m,
                    message.y - east_m,
                )
                if horizontal_error <= 5.0:
                    self._report(
                        "Outbound point reached "
                        f"({distance_m:.1f} m at {bearing_deg:.1f} deg commanded)"
                    )
                    return
        raise FlightError(
            "vehicle did not reach the "
            f"{distance_m:.1f} m, {bearing_deg:.1f} deg outbound point"
        )

    def observe(self, duration_s: float) -> None:
        self._report(f"Collecting {duration_s:.1f} s pre-RTL observation window")
        position = self.latest.get("GLOBAL_POSITION_INT")
        if position is None:
            raise FlightError("no position clock before observation window")
        start_boot_ms = int(position.time_boot_ms)
        target_boot_ms = start_boot_ms + int(duration_s * 1000)
        deadline = time.monotonic() + max(30.0, duration_s * 2.0)
        while time.monotonic() < deadline:
            message = self._receive(timeout_s=min(0.5, deadline - time.monotonic()))
            if (
                message is not None
                and message.get_type() == "GLOBAL_POSITION_INT"
                and int(message.time_boot_ms) >= target_boot_ms
            ):
                return
        raise FlightError("observation window did not span the requested simulated time")

    def battery_charge_mah(self, timeout_s: float = 5.0) -> int:
        message = self._wait_for_type("BATTERY_STATUS", timeout_s)
        charge = int(message.current_consumed)
        if charge < 0:
            raise FlightError("BATTERY_STATUS.current_consumed is unavailable")
        return charge

    def rtl_and_wait_for_disarm(self, timeout_s: float = 600.0) -> tuple[float, float]:
        self.set_mode("RTL")
        rtl_started = time.time()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            message = self._receive(timeout_s=1.0)
            if message is None or message.get_type() != "HEARTBEAT":
                continue
            armed = bool(
                message.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
            )
            if not armed:
                self._report("Automatic disarm observed after RTL")
                return rtl_started, time.time()
        raise FlightError("RTL did not end in automatic disarm")

    def relative_altitude_m(self, timeout_s: float = 5.0) -> float:
        message = self._wait_for_type("GLOBAL_POSITION_INT", timeout_s)
        return float(message.relative_alt) / 1000.0

    def run_outbound_rtl(
        self,
        *,
        altitude_m: float,
        outbound_distance_m: float,
        outbound_bearing_deg: float = 0.0,
        observation_window_s: float,
        payload_mass_kg: Optional[float] = None,
        wind_speed_m_s: Optional[float] = None,
        wind_direction_deg: Optional[float] = None,
        scenario_id: Optional[str] = None,
        run_id: Optional[str] = None,
        sitl_speedup: float = 1.0,
    ) -> FlightResult:
        self.connect()
        if payload_mass_kg is not None:
            if payload_mass_kg < 0:
                raise ValueError("payload_mass_kg cannot be negative")
            self.set_parameter("SIM_PAYLOAD_MASS", payload_mass_kg)
        if wind_speed_m_s is not None:
            if wind_speed_m_s < 0:
                raise ValueError("wind_speed_m_s cannot be negative")
            if wind_direction_deg is None:
                raise ValueError("wind_direction_deg is required when wind speed is set")
            if not 0 <= wind_direction_deg <= 360:
                raise ValueError("wind_direction_deg must be between 0 and 360")
            self.set_parameter("SIM_WIND_DIR", wind_direction_deg)
            self.set_parameter("SIM_WIND_SPD", wind_speed_m_s)
        self.wait_until_ready()
        self.set_mode("GUIDED")
        self.arm()
        self.takeoff(altitude_m)
        self.fly_to_bearing(
            outbound_distance_m,
            outbound_bearing_deg,
            altitude_m,
        )
        self.observe(observation_window_s)
        charge_at_rtl = self.battery_charge_mah()
        self._report(f"Decision point frozen at {charge_at_rtl} mAh consumed")
        rtl_started, disarmed = self.rtl_and_wait_for_disarm()
        charge_at_disarm = self.battery_charge_mah()
        final_altitude = self.relative_altitude_m()
        return FlightResult(
            connection=self.connection_string,
            altitude_m=altitude_m,
            outbound_distance_m=outbound_distance_m,
            outbound_bearing_deg=outbound_bearing_deg,
            observation_window_s=observation_window_s,
            payload_mass_kg=payload_mass_kg,
            wind_speed_m_s=wind_speed_m_s,
            wind_direction_deg=wind_direction_deg,
            rtl_started_unix_s=rtl_started,
            disarmed_unix_s=disarmed,
            rtl_duration_s=disarmed - rtl_started,
            charge_at_rtl_mah=charge_at_rtl,
            charge_at_disarm_mah=charge_at_disarm,
            rtl_charge_mah=charge_at_disarm - charge_at_rtl,
            final_relative_altitude_m=final_altitude,
            automatic_disarm_observed=True,
            scenario_id=scenario_id,
            run_id=run_id,
            sitl_speedup=sitl_speedup,
        )
