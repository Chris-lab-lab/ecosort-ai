"""Background bridge from one physical depth sensor into the edge state."""

from __future__ import annotations

from collections import deque
from statistics import median
from threading import Event, Thread
from typing import Protocol

from .state import EdgeStateStore


class DistanceSensor(Protocol):
    def read_mm(self) -> float: ...

    def close(self) -> None: ...


class DepthSensorMonitor:
    def __init__(
        self,
        sensor: DistanceSensor,
        state: EdgeStateStore,
        *,
        poll_seconds: float = 1.0,
        median_samples: int = 5,
        failure_limit: int = 3,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if median_samples < 1:
            raise ValueError("median_samples must be positive")
        if failure_limit < 1:
            raise ValueError("failure_limit must be positive")
        self._sensor = sensor
        self._state = state
        self._poll_seconds = poll_seconds
        self._failure_limit = failure_limit
        self._samples: deque[float] = deque(maxlen=median_samples)
        self._failures = 0
        self._stop = Event()
        self._thread: Thread | None = None

    def sample_once(self) -> bool:
        try:
            distance_mm = float(self._sensor.read_mm())
            if not 10 <= distance_mm <= 4000:
                raise ValueError(f"out-of-range VL53L0X reading: {distance_mm:g} mm")
            self._samples.append(distance_mm)
            self._state.update_distance_mm(float(median(self._samples)))
            self._failures = 0
            return True
        except Exception as exc:
            self._failures += 1
            if self._failures >= self._failure_limit:
                self._state.mark_sensor_offline(str(exc))
            return False

    def start(self) -> "DepthSensorMonitor":
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = Thread(target=self._run, name="ecosort-depth", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self._poll_seconds + 0.5))
            self._thread = None
        self._sensor.close()

    def __enter__(self) -> "DepthSensorMonitor":
        return self.start()

    def __exit__(self, *_: object) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.sample_once()
            self._stop.wait(self._poll_seconds)
