from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ecosort_hw.sensors import (
    DigitalInputSensor,
    DigitalMetalSensor,
    NumericSensor,
    VL53L0XDistanceSensor,
)


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

    def test_generic_digital_input(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            value = Path(folder) / "presence"
            value.write_text("on\n", encoding="ascii")
            self.assertTrue(DigitalInputSensor(value).read())

    def test_numeric_sensor(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            value = Path(folder) / "weight"
            value.write_text("124.5\n", encoding="ascii")
            self.assertEqual(NumericSensor(value).read(), 124.5)

    def test_vl53l0x_uses_selected_linux_bus(self) -> None:
        created: dict[str, object] = {}

        class FakeI2C:
            def __init__(self, bus_number: int) -> None:
                created["bus"] = bus_number
                self.closed = False

            def deinit(self) -> None:
                self.closed = True

        class FakeVL53L0X:
            range = 173

            def __init__(self, i2c: object, **options: object) -> None:
                created["i2c"] = i2c
                created["options"] = options

        sensor = VL53L0XDistanceSensor(
            4,
            address=0x29,
            i2c_factory=FakeI2C,
            sensor_factory=FakeVL53L0X,
        )
        self.assertEqual(sensor.read_mm(), 173.0)
        self.assertEqual(created["bus"], 4)
        self.assertEqual(created["options"], {"address": 0x29, "io_timeout_s": 1.0})
        bus = created["i2c"]
        sensor.close()
        self.assertTrue(bus.closed)


if __name__ == "__main__":
    unittest.main()
