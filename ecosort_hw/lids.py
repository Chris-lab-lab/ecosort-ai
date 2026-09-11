"""Safe three-lid servo control for the EcoSort prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import logging
import math
from pathlib import Path
import threading
import time
from typing import Callable, Iterable, Protocol

from .pca9685 import PCA9685, PCA9685Config


LOG = logging.getLogger(__name__)


class LidName(str, Enum):
    PLASTIC = "plastic"
    GENERAL = "general"
    # Backward-compatible programmatic alias. Text input "paper" is handled
    # in _coerce_name so the same middle mechanism works with either label.
    PAPER = "general"
    METAL = "metal"


@dataclass(frozen=True)
class ServoConfig:
    """Mechanical calibration for one lid servo."""

    name: LidName
    channel: int
    # These stay away from common SG90 end stops. Mount the disconnected horn
    # at the commanded closed position, then tune each mechanism as documented.
    closed_angle: float = 45.0
    open_angle: float = 120.0
    min_pulse_us: float = 500.0
    max_pulse_us: float = 2500.0
    min_angle: float = 0.0
    max_angle: float = 180.0
    closed_pulse_us: float | None = None
    open_pulse_us: float | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.channel <= 15:
            raise ValueError("servo channel must be between 0 and 15")
        if self.min_angle >= self.max_angle:
            raise ValueError("min_angle must be less than max_angle")
        for label, angle in (
            ("closed_angle", self.closed_angle),
            ("open_angle", self.open_angle),
        ):
            if not self.min_angle <= angle <= self.max_angle:
                raise ValueError(f"{label} is outside the calibrated angle range")
        if self.min_pulse_us <= 0 or self.min_pulse_us >= self.max_pulse_us:
            raise ValueError("servo pulse range is invalid")
        for label, pulse in (
            ("closed_pulse_us", self.closed_pulse_us),
            ("open_pulse_us", self.open_pulse_us),
        ):
            if pulse is not None and not 900.0 <= pulse <= 2100.0:
                raise ValueError(f"{label} must be between 900 and 2100 microseconds")
        span = self.max_angle - self.min_angle
        for label, angle, override in (
            ("closed", self.closed_angle, self.closed_pulse_us),
            ("open", self.open_angle, self.open_pulse_us),
        ):
            calculated = self.min_pulse_us + (
                (angle - self.min_angle) / span
            ) * (self.max_pulse_us - self.min_pulse_us)
            commanded = calculated if override is None else override
            if not 900.0 <= commanded <= 2100.0:
                raise ValueError(
                    f"{label} command resolves to {commanded:.0f} us; keep initial calibration "
                    "between 900 and 2100 us"
                )


DEFAULT_LIDS = (
    ServoConfig(LidName.PLASTIC, channel=0),
    ServoConfig(LidName.GENERAL, channel=1),
    ServoConfig(LidName.METAL, channel=2),
)


@dataclass(frozen=True)
class LidControllerConfig:
    """Settings shared by the complete three-lid mechanism."""

    lids: tuple[ServoConfig, ...] = field(default_factory=lambda: DEFAULT_LIDS)
    dwell_seconds: float = 3.0
    movement_seconds: float = 0.45
    release_after_move: bool = True
    dry_run: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.dwell_seconds) or self.dwell_seconds < 0:
            raise ValueError("dwell_seconds must be finite and non-negative")
        if not math.isfinite(self.movement_seconds) or self.movement_seconds < 0:
            raise ValueError("movement_seconds must be finite and non-negative")
        names = [lid.name for lid in self.lids]
        channels = [lid.channel for lid in self.lids]
        if set(names) != set(LidName) or len(names) != len(LidName):
            raise ValueError("configure exactly one plastic, middle/general and metal lid")
        if len(channels) != len(set(channels)):
            raise ValueError("each lid must use a different PCA9685 channel")


def load_lid_config(path: str | Path, *, dry_run: bool) -> LidControllerConfig:
    """Load per-servo channels and calibration from a small JSON file."""

    source = Path(path)
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read hardware configuration {source}: {exc}") from exc
    if not isinstance(document, dict) or not isinstance(document.get("lids"), dict):
        raise ValueError("hardware configuration must contain a 'lids' object")

    allowed_top = {"lids", "movement_seconds", "release_after_move"}
    unknown_top = set(document) - allowed_top
    if unknown_top:
        raise ValueError(f"unknown hardware configuration fields: {sorted(unknown_top)}")

    normalized: dict[LidName, dict] = {}
    for raw_name, values in document["lids"].items():
        lid_name = LidController._coerce_name(raw_name)
        if lid_name in normalized:
            raise ValueError(f"duplicate middle/lid configuration for {raw_name!r}")
        if not isinstance(values, dict):
            raise ValueError(f"configuration for {raw_name} must be an object")
        normalized[lid_name] = values
    if set(normalized) != set(LidName):
        raise ValueError("configure plastic, general (or paper), and metal exactly once")

    allowed_servo = {
        "channel",
        "closed_angle",
        "open_angle",
        "min_pulse_us",
        "max_pulse_us",
        "min_angle",
        "max_angle",
        "closed_pulse_us",
        "open_pulse_us",
    }
    lids: list[ServoConfig] = []
    for name in LidName:
        values = normalized[name]
        unknown = set(values) - allowed_servo
        if unknown:
            raise ValueError(f"unknown fields for {name.value}: {sorted(unknown)}")
        if "channel" not in values:
            raise ValueError(f"{name.value} configuration requires a channel")
        lids.append(ServoConfig(name=name, **values))

    movement_seconds = document.get("movement_seconds", 0.45)
    release_after_move = document.get("release_after_move", True)
    if not isinstance(release_after_move, bool):
        raise ValueError("release_after_move must be true or false")
    return LidControllerConfig(
        lids=tuple(lids),
        movement_seconds=movement_seconds,
        release_after_move=release_after_move,
        dry_run=dry_run,
    )


class PWMController(Protocol):
    def initialize(self) -> None: ...

    def set_pulse_us(self, channel: int, pulse_us: float) -> None: ...

    def disable_channel(self, channel: int) -> None: ...

    def close(self) -> None: ...


class LidController:
    """Coordinate three servos while allowing at most one open lid.

    ``open_for()`` blocks through the dwell and closure. ``open_timed()`` uses
    an independent timer so a stalled camera/UI cannot postpone closure;
    applications must call ``raise_pending_error()`` to surface timer failures.
    ``open_lid()`` leaves closure to the caller. The context manager closes all
    lids during cleanup.
    """

    def __init__(
        self,
        config: LidControllerConfig | None = None,
        *,
        pwm: PWMController | None = None,
        pca_config: PCA9685Config | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or LidControllerConfig()
        self._pwm = pwm
        self._pca_config = pca_config or PCA9685Config()
        self._sleep = sleep
        self._lids = {lid.name: lid for lid in self.config.lids}
        self._active_lid: LidName | None = None
        self._started = False
        self._closed = False
        self._lock = threading.RLock()
        self._close_timer: threading.Timer | None = None
        self._timer_generation = 0
        self._pending_error: Exception | None = None
        self.command_log: list[tuple[str, LidName, float]] = []

    @classmethod
    def from_defaults(
        cls,
        bus: int = 1,
        address: int = 0x40,
        dry_run: bool = True,
    ) -> "LidController":
        """Build the standard plastic/paper/metal controller used by the app.

        Channels 0, 1 and 2 are assigned to plastic, middle/general and metal. Dry-run
        mode is the default so this constructor is safe on a development PC.
        """

        return cls(
            LidControllerConfig(dry_run=dry_run),
            pca_config=PCA9685Config(bus_number=bus, address=address),
        )

    @classmethod
    def from_config_file(
        cls,
        path: str | Path,
        *,
        bus: int,
        address: int = 0x40,
        dry_run: bool = True,
    ) -> "LidController":
        return cls(
            load_lid_config(path, dry_run=dry_run),
            pca_config=PCA9685Config(bus_number=bus, address=address),
        )

    @property
    def active_lid(self) -> LidName | None:
        with self._lock:
            return self._active_lid

    def raise_pending_error(self) -> None:
        """Report the first timer failure until this controller is replaced."""

        with self._lock:
            if self._pending_error is not None:
                raise RuntimeError(
                    "Timed lid closure failed; check hardware and restart the controller"
                ) from self._pending_error

    def start(self) -> None:
        """Initialize the PWM board and move every lid to its closed angle."""

        with self._lock:
            if self._closed:
                raise RuntimeError("lid controller is closed")
            if self._started:
                return
            if not self.config.dry_run:
                if self._pwm is None:
                    self._pwm = PCA9685(self._pca_config)
                self._pwm.initialize()
            self._started = True
            try:
                self.close_all()
            except BaseException:
                if self._pwm is not None and not self.config.dry_run:
                    try:
                        self._pwm.close()
                    except Exception:
                        LOG.exception("failed to close PWM controller after startup error")
                self._started = False
                raise

    def open_lid(self, name: LidName | str) -> None:
        """Close all lids, then open exactly one selected lid."""

        lid_name = self._coerce_name(name)
        with self._lock:
            self._require_started()
            self.raise_pending_error()
            self._cancel_close_timer()
            # Always issue close commands, even if our remembered state says
            # none is open.  This makes startup and recovery fail-safe.
            for other_name, other in self._lids.items():
                if other_name != lid_name:
                    self._move(other, other.closed_angle, "close", release=True)
            selected = self._lids[lid_name]
            # Keep PWM applied while open so gravity cannot make the lid drift
            # shut during the dwell interval.
            self._move(selected, selected.open_angle, "open", release=False)
            self._active_lid = lid_name

    def close_lid(self, name: LidName | str) -> None:
        lid_name = self._coerce_name(name)
        with self._lock:
            self._require_started()
            lid = self._lids[lid_name]
            self._move(lid, lid.closed_angle, "close", release=True)
            if self._active_lid == lid_name:
                self._active_lid = None
                self._cancel_close_timer()

    def close_all(self) -> None:
        with self._lock:
            self._require_started()
            self._cancel_close_timer()
            for lid in self._lids.values():
                self._move(lid, lid.closed_angle, "close", release=True)
            self._active_lid = None

    def open_for(self, name: LidName | str, dwell_seconds: float | None = None) -> None:
        """Open one lid and guarantee closure after a bounded dwell."""

        dwell = self.config.dwell_seconds if dwell_seconds is None else dwell_seconds
        if not math.isfinite(dwell) or dwell < 0:
            raise ValueError("dwell_seconds must be finite and non-negative")
        lid_name = self._coerce_name(name)
        try:
            self.open_lid(lid_name)
            self._sleep(dwell)
        except BaseException:
            # Preserve the original error if the recovery attempt also fails.
            try:
                self.close_all()
            except Exception:
                LOG.exception("failed to command all lids closed during recovery")
            raise
        else:
            self.close_all()

    def open_timed(self, name: LidName | str, dwell_seconds: float | None = None) -> None:
        """Open a lid and independently command closure after the given dwell.

        The dwell starts after the opening movement. Timer failures are logged,
        retained by ``raise_pending_error()``, and prevent subsequent opens.
        """

        dwell = self.config.dwell_seconds if dwell_seconds is None else dwell_seconds
        if not math.isfinite(dwell) or dwell < 0:
            raise ValueError("dwell_seconds must be finite and non-negative")
        if dwell > threading.TIMEOUT_MAX:
            raise ValueError("dwell_seconds exceeds the timer's maximum supported duration")
        lid_name = self._coerce_name(name)
        with self._lock:
            self._require_started()
            self.raise_pending_error()
            try:
                self.open_lid(lid_name)
                timer = threading.Timer(
                    dwell, self._close_when_due, args=(self._timer_generation,)
                )
                timer.daemon = True
                self._close_timer = timer
                timer.start()
            except BaseException:
                self._cancel_close_timer()
                try:
                    self.close_all()
                except Exception:
                    LOG.exception("failed to command all lids closed after timer setup error")
                raise

    def _cancel_close_timer(self) -> None:
        """Invalidate callbacks that may already be waiting for our lock."""

        self._timer_generation += 1
        if self._close_timer is not None:
            self._close_timer.cancel()
            self._close_timer = None

    def _close_when_due(self, generation: int) -> None:
        with self._lock:
            if self._closed or not self._started or generation != self._timer_generation:
                return
            self._close_timer = None
            try:
                self.close_all()
            except Exception as exc:
                if self._pending_error is None:
                    self._pending_error = exc
                LOG.exception("timed lid closure failed; physical lid position is unknown")

    def open_temporarily(self, label: str, seconds: float | None = None) -> None:
        """Application-friendly alias for bounded one-lid operation."""

        self.open_for(label, dwell_seconds=seconds)

    def self_test(self, order: Iterable[LidName | str] = tuple(LidName)) -> None:
        """Open and close each lid in sequence."""

        for name in order:
            self.open_for(name)

    def shutdown(self) -> None:
        """Close all lids and release the hardware connection."""

        with self._lock:
            if self._closed:
                return
            self._cancel_close_timer()
            try:
                if self._started:
                    self.close_all()
            finally:
                try:
                    if self._pwm is not None and not self.config.dry_run and self._started:
                        self._pwm.close()
                finally:
                    self._started = False
                    self._closed = True

    def _move(self, servo: ServoConfig, angle: float, action: str, *, release: bool) -> None:
        override = servo.open_pulse_us if action == "open" else servo.closed_pulse_us
        pulse = override if override is not None else self.angle_to_pulse_us(servo, angle)
        self.command_log.append((action, servo.name, angle))
        LOG.info(
            "%s lid=%s channel=%d angle=%.1f pulse=%.0fus%s",
            action,
            servo.name.value,
            servo.channel,
            angle,
            pulse,
            " [dry run]" if self.config.dry_run else "",
        )
        if self.config.dry_run:
            return
        assert self._pwm is not None
        self._pwm.set_pulse_us(servo.channel, pulse)
        if self.config.movement_seconds > 0:
            self._sleep(self.config.movement_seconds)
        if release and self.config.release_after_move:
            self._pwm.disable_channel(servo.channel)

    @staticmethod
    def angle_to_pulse_us(servo: ServoConfig, angle: float) -> float:
        if not servo.min_angle <= angle <= servo.max_angle:
            raise ValueError("angle is outside the calibrated range")
        fraction = (angle - servo.min_angle) / (servo.max_angle - servo.min_angle)
        return servo.min_pulse_us + fraction * (
            servo.max_pulse_us - servo.min_pulse_us
        )

    @staticmethod
    def _coerce_name(name: LidName | str) -> LidName:
        if isinstance(name, LidName):
            return name
        try:
            normalized = name.strip().lower()
            if normalized in {"paper", "middle"}:
                normalized = "general"
            return LidName(normalized)
        except (AttributeError, ValueError) as exc:
            valid = ", ".join(lid.value for lid in LidName)
            raise ValueError(f"unknown lid {name!r}; choose {valid}") from exc

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("call start() or use 'with LidController(...)'")

    def __enter__(self) -> "LidController":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.shutdown()
