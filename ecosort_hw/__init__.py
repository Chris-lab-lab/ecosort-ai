"""Hardware control helpers for the EcoSort three-lid prototype.

The package intentionally has no third-party runtime dependency.  On Linux,
``LinuxI2CBus`` talks directly to ``/dev/i2c-*``.  Tests and demos can inject
another object implementing the small ``RegisterBus`` protocol.
"""

from .lids import (
    DEFAULT_LIDS,
    LidController,
    LidControllerConfig,
    LidName,
    ServoConfig,
    load_lid_config,
)
from .pca9685 import LinuxI2CBus, PCA9685, PCA9685Config, RegisterBus
from .sensors import DigitalMetalSensor

__all__ = [
    "DEFAULT_LIDS",
    "DigitalMetalSensor",
    "LidController",
    "LidControllerConfig",
    "LidName",
    "LinuxI2CBus",
    "load_lid_config",
    "PCA9685",
    "PCA9685Config",
    "RegisterBus",
    "ServoConfig",
]
