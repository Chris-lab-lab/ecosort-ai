from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ecosort_hw.sensors import DigitalMetalSensor


class DigitalMetalSensorTests(unittest.TestCase):
    def test_active_high_sensor(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            value = Path(folder) / "value"
            value.write_text("1\n", encoding="ascii")
            self.assertTrue(DigitalMetalSensor(value).read())
            value.write_text("0\n", encoding="ascii")
            self.assertFalse(DigitalMetalSensor(value).read())

    def test_active_low_sensor(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            value = Path(folder) / "value"
            value.write_text("0", encoding="ascii")
            self.assertTrue(DigitalMetalSensor(value, active_low=True).read())

    def test_invalid_sensor_value_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            value = Path(folder) / "value"
            value.write_text("maybe", encoding="ascii")
            with self.assertRaisesRegex(ValueError, "expected a digital"):
                DigitalMetalSensor(value).read()


if __name__ == "__main__":
    unittest.main()
