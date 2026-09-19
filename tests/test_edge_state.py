from __future__ import annotations

import unittest

from ecosort_edge.monitor import DepthSensorMonitor
from ecosort_edge.state import EdgeStateStore, calculate_fill_percentage


class FakeDistanceSensor:
    def __init__(self, readings: list[float | Exception]) -> None:
        self.readings = iter(readings)
        self.closed = False

    def read_mm(self) -> float:
        value = next(self.readings)
        if isinstance(value, Exception):
            raise value
        return value

    def close(self) -> None:
        self.closed = True


class EdgeStateTests(unittest.TestCase):
    def test_fill_calculation_clamps(self) -> None:
        self.assertEqual(calculate_fill_percentage(30, 7.5), 75)
        self.assertEqual(calculate_fill_percentage(30, 40), 0)
        self.assertEqual(calculate_fill_percentage(30, 0), 100)

    def test_depth_monitor_filters_readings_and_updates_selected_bin(self) -> None:
        state = EdgeStateStore(monitored_bin="general", empty_depth_cm=30)
        sensor = FakeDistanceSensor([100, 120, 110])
        monitor = DepthSensorMonitor(sensor, state, median_samples=3)
        self.assertTrue(monitor.sample_once())
        self.assertTrue(monitor.sample_once())
        self.assertTrue(monitor.sample_once())

        bins = {item["id"]: item for item in state.bins_payload()["bins"]}
        self.assertEqual(bins["general"]["distance_cm"], 11.0)
        self.assertEqual(bins["general"]["fill_percent"], 63)
        self.assertTrue(bins["general"]["sensor_online"])
        self.assertFalse(bins["plastic"]["sensor_online"])

    def test_repeated_sensor_failures_mark_it_offline(self) -> None:
        state = EdgeStateStore()
        state.update_distance_mm(100)
        sensor = FakeDistanceSensor([OSError("no response")] * 3)
        monitor = DepthSensorMonitor(sensor, state, failure_limit=3)
        for _ in range(3):
            self.assertFalse(monitor.sample_once())
        self.assertFalse(state.status_payload()["depth_sensor_online"])
        self.assertEqual(state.status_payload()["sensor_error"], "no response")

    def test_paper_route_is_published_as_general(self) -> None:
        state = EdgeStateStore()
        event = state.record_disposal(
            route="paper",
            confidence=0.91,
            detected_object="Paper cup",
        )
        self.assertEqual(event["category"], "general")
        self.assertEqual(state.events_payload()["events"][0]["object"], "Paper cup")


if __name__ == "__main__":
    unittest.main()
