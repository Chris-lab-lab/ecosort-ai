"""Small, board-friendly digital sensors used by the EcoSort safety gate."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DigitalMetalSensor:
    """Read an active-high or active-low metal detector from a Linux value file.

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
                f"Metal sensor {self.value_path} returned {raw!r}; expected a digital 0 or 1"
            )
        return not active if self.active_low else active
