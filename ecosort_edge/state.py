"""Thread-safe state shared by the AI loop, depth sensor, and HTTP API."""

from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import math
from threading import RLock
from typing import Any
from uuid import uuid4


BIN_CATEGORIES = ("plastic", "metal", "general")
FILL_STATES = ("empty", "half-full", "full")
ROUTE_TO_CATEGORY = {
    "plastic": "plastic",
    "metal": "metal",
    "general": "general",
    "paper": "general",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00", "Z"
    )


def calculate_fill_percentage(empty_depth_cm: float, distance_cm: float) -> int:
    if not math.isfinite(empty_depth_cm) or empty_depth_cm <= 0:
        raise ValueError("empty depth must be a positive finite number")
    if not math.isfinite(distance_cm) or distance_cm < 0:
        raise ValueError("distance must be a non-negative finite number")
    raw = ((empty_depth_cm - distance_cm) / empty_depth_cm) * 100.0
    return round(min(100.0, max(0.0, raw)))


def fill_state_for_percentage(fill_percentage: float) -> str:
    """Normalize continuous sensor data to the app's three-state contract."""
    if not math.isfinite(fill_percentage):
        raise ValueError("fill percentage must be finite")
    if fill_percentage >= 75:
        return "full"
    if fill_percentage <= 5:
        return "empty"
    return "half-full"


class EdgeStateStore:
    """Own the latest device snapshot without coupling it to the HTTP server."""

    def __init__(
        self,
        *,
        monitored_bin: str = "plastic",
        empty_depth_cm: float = 30.0,
        max_events: int = 100,
    ) -> None:
        if monitored_bin not in BIN_CATEGORIES:
            raise ValueError(f"unknown monitored bin: {monitored_bin}")
        if not math.isfinite(empty_depth_cm) or empty_depth_cm <= 0:
            raise ValueError("empty depth must be a positive finite number")
        if max_events < 1:
            raise ValueError("max_events must be positive")

        self.monitored_bin = monitored_bin
        self._lock = RLock()
        self._started_at = datetime.now(timezone.utc)
        self._sensor_error: str | None = None
        timestamp = utc_now()
        self._bins: dict[str, dict[str, Any]] = {
            category: {
                "id": category,
                "distance_cm": round(empty_depth_cm, 1),
                "empty_depth_cm": round(empty_depth_cm, 1),
                "fill_state": "empty",
                "fill_source": None,
                "fill_confidence": None,
                "fill_online": False,
                "sensor_online": False,
                "last_updated": timestamp,
                "last_emptied": timestamp,
                "item_count_today": 0,
            }
            for category in BIN_CATEGORIES
        }
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)

    def update_distance_mm(
        self,
        distance_mm: float,
        *,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        if not math.isfinite(distance_mm) or distance_mm <= 0:
            raise ValueError("distance must be a positive finite number")
        distance_cm = distance_mm / 10.0
        with self._lock:
            bin_state = self._bins[self.monitored_bin]
            fill_percentage = calculate_fill_percentage(
                float(bin_state["empty_depth_cm"]), distance_cm
            )
            bin_state.update(
                {
                    "distance_cm": round(distance_cm, 1),
                    "fill_state": fill_state_for_percentage(fill_percentage),
                    "fill_source": "depth_sensor",
                    "fill_confidence": None,
                    "fill_online": True,
                    "sensor_online": True,
                    "last_updated": timestamp or utc_now(),
                }
            )
            self._sensor_error = None
            return dict(bin_state)

    def mark_sensor_offline(self, reason: str) -> dict[str, Any]:
        with self._lock:
            bin_state = self._bins[self.monitored_bin]
            bin_state["sensor_online"] = False
            if bin_state["fill_source"] == "depth_sensor":
                bin_state["fill_online"] = False
            bin_state["last_updated"] = utc_now()
            self._sensor_error = reason.strip() or "depth sensor unavailable"
            return dict(bin_state)

    def update_fill_state(
        self,
        category: str,
        fill_state: str,
        *,
        source: str = "webcam",
        confidence: float | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        if category not in BIN_CATEGORIES:
            raise ValueError(f"unknown bin category: {category}")
        normalized = fill_state.strip().lower()
        if normalized not in FILL_STATES:
            raise ValueError(f"unknown fill state: {fill_state}")
        normalized_source = source.strip().lower()
        if not normalized_source or len(normalized_source) > 40:
            raise ValueError("source must be between 1 and 40 characters")
        if confidence is not None and (
            not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0
        ):
            raise ValueError("confidence must be between 0 and 1")

        with self._lock:
            bin_state = self._bins[category]
            bin_state.update(
                {
                    "fill_state": normalized,
                    "fill_source": normalized_source,
                    "fill_confidence": (
                        round(confidence, 6) if confidence is not None else None
                    ),
                    "fill_online": True,
                    "last_updated": timestamp or utc_now(),
                }
            )
            return dict(bin_state)

    def record_disposal(
        self,
        *,
        route: str,
        confidence: float,
        detected_object: str | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        category = ROUTE_TO_CATEGORY.get(route.strip().lower())
        if category is None:
            raise ValueError(f"route is not a supported bin category: {route}")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        occurred_at = timestamp or utc_now()
        object_name = (detected_object or "").strip() or f"{category.title()} item"
        event = {
            "id": f"event-{uuid4().hex}",
            "object": object_name,
            "category": category,
            "confidence": round(confidence, 6),
            "timestamp": occurred_at,
        }
        with self._lock:
            self._events.appendleft(event)
            self._bins[category]["item_count_today"] += 1
            return dict(event)

    def mark_emptied(self, category: str) -> tuple[dict[str, Any], dict[str, Any]]:
        if category not in BIN_CATEGORIES:
            raise ValueError(f"unknown bin category: {category}")
        timestamp = utc_now()
        with self._lock:
            bin_state = self._bins[category]
            bin_state.update(
                {
                    "distance_cm": float(bin_state["empty_depth_cm"]),
                    "fill_state": "empty",
                    "fill_source": "manual",
                    "fill_confidence": None,
                    "fill_online": True,
                    "last_updated": timestamp,
                    "last_emptied": timestamp,
                }
            )
            event = {
                "id": f"event-{uuid4().hex}",
                "object": "Bin emptied by staff",
                "category": category,
                "timestamp": timestamp,
                "kind": "maintenance",
            }
            self._events.appendleft(event)
            return dict(bin_state), dict(event)

    def status_payload(self) -> dict[str, Any]:
        with self._lock:
            monitored = self._bins[self.monitored_bin]
            uptime = (datetime.now(timezone.utc) - self._started_at).total_seconds()
            return {
                "device": "FRDM-i.MX93",
                "online": True,
                "timestamp": utc_now(),
                "uptime_seconds": round(max(0.0, uptime), 1),
                "monitored_bin": self.monitored_bin,
                "depth_sensor_online": bool(monitored["sensor_online"]),
                "sensor_error": self._sensor_error,
            }

    def bins_payload(self) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            return {"bins": [dict(self._bins[name]) for name in BIN_CATEGORIES]}

    def events_payload(
        self,
        *,
        limit: int = 50,
        since: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        with self._lock:
            events = list(self._events)
        if since:
            events = [event for event in events if event["timestamp"] > since]
        return {"events": [dict(event) for event in events[: max(1, min(limit, 100))]]}
