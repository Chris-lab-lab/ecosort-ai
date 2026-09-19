"""Small, board-friendly digital sensors used by the EcoSort safety gate."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class DigitalInputSensor:
    """Read an active-high or active-low sensor from a Linux value file.

    ``value_path`` can point at an exported sysfs GPIO value or another BSP file
    that returns one of: 0/1, low/high, false/true, or off/on.  The class never
    configures a GPIO and therefore cannot accidentally change pin direction.
    """

    value_path: Path
    active_low: bool = False

    def __init__(self, value_path: str | Path, *, active_low: bool = False) -> None:
        object.__setattr__(self, "value_path", Path(value_path))
        object.__setattr__(self, "active_low", bool(active_low))

    def read(self) -> bool:
        raw = self.value_path.read_text(encoding="ascii").strip().lower()
        if raw in {"1", "high", "true", "on"}:
            active = True
        elif raw in {"0", "low", "false", "off"}:
            active = False
        else:
            raise ValueError(
                f"Digital sensor {self.value_path} returned {raw!r}; expected a digital 0 or 1"
            )
        return not active if self.active_low else active


class DigitalMetalSensor(DigitalInputSensor):
    """Semantic name for the digital input used by the metal safety gate."""


@dataclass(frozen=True)
class NumericSensor:
    """Read a finite value, such as grams from a BSP or IIO value file."""

    value_path: Path

    def __init__(self, value_path: str | Path) -> None:
        object.__setattr__(self, "value_path", Path(value_path))

    def read(self) -> float:
        raw = self.value_path.read_text(encoding="ascii").strip()
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"Numeric sensor {self.value_path} returned {raw!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"Numeric sensor {self.value_path} returned a non-finite value")
        return value


class VL53L0XDistanceSensor:
    """Read one VL53L0X on a selected Linux ``/dev/i2c-*`` bus.

    Imports are intentionally lazy: camera/model development and unit tests do
    not need the CircuitPython packages installed. ``adafruit-extended-bus``
    selects a Linux bus number without relying on Raspberry Pi pin mappings.
    """

    def __init__(
        self,
        bus_number: int,
        *,
        address: int = 0x29,
        io_timeout_seconds: float = 1.0,
        i2c_factory: Callable[[int], Any] | None = None,
        sensor_factory: Callable[..., Any] | None = None,
    ) -> None:
        if bus_number < 0:
            raise ValueError("bus_number must be zero or greater")
        if not 0x08 <= address <= 0x77:
            raise ValueError("address must be a valid 7-bit I2C device address")
        if io_timeout_seconds <= 0:
            raise ValueError("io_timeout_seconds must be positive")

        if i2c_factory is None or sensor_factory is None:
            try:
                import adafruit_vl53l0x
                from adafruit_extended_bus import ExtendedI2C
            except ImportError as exc:
                raise RuntimeError(
                    "VL53L0X support requires requirements-board.txt; install "
                    "adafruit-circuitpython-vl53l0x and adafruit-extended-bus"
                ) from exc
            i2c_factory = i2c_factory or ExtendedI2C
            sensor_factory = sensor_factory or adafruit_vl53l0x.VL53L0X

        self._i2c = i2c_factory(bus_number)
        try:
            self._sensor = sensor_factory(
                self._i2c,
                address=address,
                io_timeout_s=io_timeout_seconds,
            )
        except Exception:
            self.close()
            raise

    def read_mm(self) -> float:
        distance = float(self._sensor.range)
        if not math.isfinite(distance):
            raise ValueError("VL53L0X returned a non-finite distance")
        return distance

    def close(self) -> None:
        i2c = getattr(self, "_i2c", None)
        self._i2c = None
        if i2c is not None:
            deinit = getattr(i2c, "deinit", None)
            if callable(deinit):
                deinit()

    def __enter__(self) -> "VL53L0XDistanceSensor":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
