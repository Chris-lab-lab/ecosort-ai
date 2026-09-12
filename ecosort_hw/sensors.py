"""Small, board-friendly digital sensors used by the EcoSort safety gate."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path


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
