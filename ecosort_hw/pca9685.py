"""Small PCA9685 driver using the Linux ``i2c-dev`` interface.

Only register reads/writes needed by the PCA9685 are implemented.  Keeping the
bus behind a protocol makes the code easy to test without a board attached.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import threading
import time
from typing import Callable, Protocol, runtime_checkable

try:  # Linux-only module; keeping import optional lets fake-bus tests run anywhere.
    import fcntl
except ImportError:  # pragma: no cover - exercised only on non-Linux hosts
    fcntl = None  # type: ignore[assignment]


@runtime_checkable
class RegisterBus(Protocol):
    """Minimal byte-register bus used by :class:`PCA9685`."""

    def read_register(self, address: int, register: int) -> int: ...

    def write_register(self, address: int, register: int, value: int) -> None: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class PCA9685Config:
    """PCA9685 connection and timing settings."""

    bus_number: int = 1
    address: int = 0x40
    frequency_hz: float = 50.0
    oscillator_hz: float = 25_000_000.0

    def __post_init__(self) -> None:
        if self.bus_number < 0:
            raise ValueError("bus_number must be zero or greater")
        if not 0 <= self.address <= 0x7F:
            raise ValueError("address must be a 7-bit I2C address")
        if self.oscillator_hz <= 0:
            raise ValueError("oscillator_hz must be positive")
        _prescale_for(self.frequency_hz, self.oscillator_hz)


def _prescale_for(frequency_hz: float, oscillator_hz: float) -> int:
    """Return a realizable PCA9685 prescale instead of silently clamping."""

    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    prescale_value = oscillator_hz / (4096.0 * frequency_hz) - 1.0
    prescale = int(math.floor(prescale_value + 0.5))
    if not 3 <= prescale <= 255:
        minimum = oscillator_hz / (4096.0 * 256.0)
        maximum = oscillator_hz / (4096.0 * 4.0)
        raise ValueError(
            f"frequency_hz is not realizable; use approximately {minimum:.1f} to {maximum:.1f} Hz"
        )
    return prescale


class LinuxI2CBus:
    """Byte-register bus backed by a Linux ``/dev/i2c-N`` device."""

    _I2C_SLAVE = 0x0703

    def __init__(self, bus_number: int = 1) -> None:
        if bus_number < 0:
            raise ValueError("bus_number must be zero or greater")
        if fcntl is None:
            raise OSError("LinuxI2CBus requires Linux and the i2c-dev kernel interface")
        self.path = f"/dev/i2c-{bus_number}"
        self._fd = os.open(self.path, os.O_RDWR)
        self._selected_address: int | None = None
        self._lock = threading.RLock()

    def _select(self, address: int) -> None:
        if self._fd is None:
            raise RuntimeError("I2C bus is closed")
        if not 0 <= address <= 0x7F:
            raise ValueError("address must be a 7-bit I2C address")
        if address != self._selected_address:
            assert fcntl is not None
            fcntl.ioctl(self._fd, self._I2C_SLAVE, address)
            self._selected_address = address

    def read_register(self, address: int, register: int) -> int:
        self._validate_byte("register", register)
        with self._lock:
            self._select(address)
            assert self._fd is not None
            os.write(self._fd, bytes((register,)))
            data = os.read(self._fd, 1)
            if len(data) != 1:
                raise OSError(f"short I2C read from {self.path}")
            return data[0]

    def write_register(self, address: int, register: int, value: int) -> None:
        self._validate_byte("register", register)
        self._validate_byte("value", value)
        with self._lock:
            self._select(address)
            assert self._fd is not None
            written = os.write(self._fd, bytes((register, value)))
            if written != 2:
                raise OSError(f"short I2C write to {self.path}")

    @staticmethod
    def _validate_byte(name: str, value: int) -> None:
        if not 0 <= value <= 0xFF:
            raise ValueError(f"{name} must fit in one byte")

    def close(self) -> None:
        with self._lock:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
                self._selected_address = None

    def __enter__(self) -> "LinuxI2CBus":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class PCA9685:
    """Dependency-free controller for the 16-channel PCA9685 PWM chip."""

    MODE1 = 0x00
    MODE2 = 0x01
    LED0_ON_L = 0x06
    PRESCALE = 0xFE

    _MODE1_RESTART = 0x80
    _MODE1_AUTO_INCREMENT = 0x20
    _MODE1_SLEEP = 0x10
    _MODE2_OUTDRV = 0x04
    _FULL_OFF = 0x10

    def __init__(
        self,
        config: PCA9685Config | None = None,
        *,
        bus: RegisterBus | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config or PCA9685Config()
        self._bus = bus
        self._owns_bus = bus is None
        self._sleep = sleep
        self._initialized = False
        self._closed = False
        self._frequency_hz = self.config.frequency_hz

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def frequency_hz(self) -> float:
        return self._frequency_hz

    def initialize(self) -> None:
        """Open the bus and configure the PWM frequency."""

        if self._closed:
            raise RuntimeError("PCA9685 controller is closed")
        if self._initialized:
            return
        if self._bus is None:
            self._bus = LinuxI2CBus(self.config.bus_number)

        self._write(self.MODE2, self._MODE2_OUTDRV)
        self._write(self.MODE1, self._MODE1_AUTO_INCREMENT)
        self.set_frequency(self.config.frequency_hz)
        self._initialized = True

    def set_frequency(self, frequency_hz: float) -> None:
        prescale = _prescale_for(frequency_hz, self.config.oscillator_hz)
        old_mode = self._read(self.MODE1)
        sleep_mode = (old_mode & ~self._MODE1_RESTART) | self._MODE1_SLEEP
        self._write(self.MODE1, sleep_mode)
        self._write(self.PRESCALE, prescale)
        self._write(self.MODE1, old_mode)
        self._sleep(0.005)
        self._write(
            self.MODE1,
            old_mode | self._MODE1_RESTART | self._MODE1_AUTO_INCREMENT,
        )
        self._frequency_hz = frequency_hz

    def set_pwm(self, channel: int, on_count: int, off_count: int) -> None:
        """Set raw 12-bit on/off counts for one PWM channel."""

        self._require_ready()
        self._validate_channel(channel)
        self._validate_count("on_count", on_count)
        self._validate_count("off_count", off_count)
        base = self.LED0_ON_L + 4 * channel
        self._write(base, on_count & 0xFF)
        self._write(base + 1, (on_count >> 8) & 0x0F)
        self._write(base + 2, off_count & 0xFF)
        self._write(base + 3, (off_count >> 8) & 0x0F)

    def set_pulse_us(self, channel: int, pulse_us: float) -> None:
        """Set a positive servo pulse width in microseconds."""

        if pulse_us <= 0:
            raise ValueError("pulse_us must be positive")
        period_us = 1_000_000.0 / self.frequency_hz
        if pulse_us >= period_us:
            raise ValueError("pulse_us must be shorter than the PWM period")
        count = int(round(pulse_us * 4096.0 / period_us))
        self.set_pwm(channel, 0, max(1, min(4095, count)))

    def disable_channel(self, channel: int) -> None:
        """Stop pulses on a channel after the servo reaches its position."""

        self._require_ready()
        self._validate_channel(channel)
        base = self.LED0_ON_L + 4 * channel
        self._write(base, 0)
        self._write(base + 1, 0)
        self._write(base + 2, 0)
        self._write(base + 3, self._FULL_OFF)

    def close(self) -> None:
        if self._closed:
            return
        if self._owns_bus and self._bus is not None:
            self._bus.close()
        self._closed = True
        self._initialized = False

    def _read(self, register: int) -> int:
        if self._bus is None:
            raise RuntimeError("I2C bus is not open")
        return self._bus.read_register(self.config.address, register)

    def _write(self, register: int, value: int) -> None:
        if self._bus is None:
            raise RuntimeError("I2C bus is not open")
        self._bus.write_register(self.config.address, register, value)

    def _require_ready(self) -> None:
        if not self._initialized:
            raise RuntimeError("call initialize() before controlling PWM")

    @staticmethod
    def _validate_channel(channel: int) -> None:
        if not 0 <= channel <= 15:
            raise ValueError("channel must be between 0 and 15")

    @staticmethod
    def _validate_count(name: str, count: int) -> None:
        if not 0 <= count <= 4095:
            raise ValueError(f"{name} must be between 0 and 4095")

    def __enter__(self) -> "PCA9685":
        self.initialize()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
