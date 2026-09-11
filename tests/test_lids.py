"""Hardware-free tests for the PCA9685 and three-lid safety logic."""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from ecosort_hw.lids import LidController, LidControllerConfig, LidName, load_lid_config
from ecosort_hw.pca9685 import PCA9685, PCA9685Config


class FakeRegisterBus:
    def __init__(self) -> None:
        self.registers: dict[tuple[int, int], int] = {}
        self.writes: list[tuple[int, int, int]] = []
        self.closed = False

    def read_register(self, address: int, register: int) -> int:
        return self.registers.get((address, register), 0)

    def write_register(self, address: int, register: int, value: int) -> None:
        self.registers[(address, register)] = value
        self.writes.append((address, register, value))

    def close(self) -> None:
        self.closed = True


class FakePWM:
    def __init__(self) -> None:
        self.events: list[tuple[str, int, float | None]] = []
        self.initialized = False
        self.closed = False

    def initialize(self) -> None:
        self.initialized = True

    def set_pulse_us(self, channel: int, pulse_us: float) -> None:
        self.events.append(("pulse", channel, pulse_us))

    def disable_channel(self, channel: int) -> None:
        self.events.append(("disable", channel, None))

    def close(self) -> None:
        self.closed = True


class PCA9685Tests(unittest.TestCase):
    def test_unachievable_pwm_frequency_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not realizable"):
            PCA9685Config(frequency_hz=1)

    def test_initialization_programs_50_hz_prescale(self) -> None:
        bus = FakeRegisterBus()
        driver = PCA9685(PCA9685Config(), bus=bus, sleep=lambda _: None)

        driver.initialize()

        self.assertTrue(driver.initialized)
        self.assertEqual(bus.registers[(0x40, PCA9685.PRESCALE)], 121)

    def test_servo_pulse_writes_expected_channel_registers(self) -> None:
        bus = FakeRegisterBus()
        driver = PCA9685(PCA9685Config(), bus=bus, sleep=lambda _: None)
        driver.initialize()
        bus.writes.clear()

        driver.set_pulse_us(2, 1500)

        base = PCA9685.LED0_ON_L + 4 * 2
        self.assertEqual(
            bus.writes,
            [
                (0x40, base, 0),
                (0x40, base + 1, 0),
                (0x40, base + 2, 51),
                (0x40, base + 3, 1),
            ],
        )

    def test_pulse_conversion_uses_updated_frequency(self) -> None:
        bus = FakeRegisterBus()
        driver = PCA9685(PCA9685Config(), bus=bus, sleep=lambda _: None)
        driver.initialize()
        driver.set_frequency(100)
        bus.writes.clear()

        driver.set_pulse_us(0, 1000)

        self.assertEqual(driver.frequency_hz, 100)
        # 1000 us is 10% of a 100 Hz period: round(4096 * .1) = 410.
        self.assertEqual(bus.writes[-2][2], 154)
        self.assertEqual(bus.writes[-1][2], 1)


class LidControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pwm = FakePWM()
        self.delays: list[float] = []
        self.config = LidControllerConfig(
            dry_run=False,
            dwell_seconds=2.5,
            movement_seconds=0,
            release_after_move=False,
        )
        self.controller = LidController(
            self.config,
            pwm=self.pwm,
            sleep=self.delays.append,
        )
        self.controller.start()
        self.pwm.events.clear()
        self.delays.clear()

    def tearDown(self) -> None:
        self.controller.shutdown()

    def test_open_closes_other_lids_before_selected_lid(self) -> None:
        self.controller.open_lid(LidName.METAL)

        self.assertEqual([event[1] for event in self.pwm.events], [0, 1, 2])
        metal_pulse = self.pwm.events[-1][2]
        closed_pulse = self.pwm.events[0][2]
        assert metal_pulse is not None and closed_pulse is not None
        self.assertGreater(metal_pulse, closed_pulse)
        self.assertEqual(self.controller.active_lid, LidName.METAL)

    def test_opening_a_second_lid_closes_first_before_opening_second(self) -> None:
        self.controller.open_lid("plastic")
        self.pwm.events.clear()

        self.controller.open_lid("paper")

        self.assertEqual([event[1] for event in self.pwm.events], [0, 2, 1])
        self.assertEqual(self.controller.active_lid, LidName.PAPER)

    def test_paper_label_is_an_alias_for_general_middle_lid(self) -> None:
        self.controller.open_lid("paper")

        self.assertEqual(self.controller.active_lid, LidName.GENERAL)
        self.assertEqual(self.pwm.events[-1][1], 1)

    def test_open_for_waits_then_closes_selected_lid(self) -> None:
        self.controller.open_for("plastic")

        self.assertEqual(self.delays, [2.5])
        self.assertIsNone(self.controller.active_lid)
        self.assertEqual(
            [event[1] for event in self.controller.command_log[-3:]],
            [LidName.PLASTIC, LidName.GENERAL, LidName.METAL],
        )

    def test_timed_open_closes_without_main_thread_progress(self) -> None:
        closed = threading.Event()
        close_all = self.controller.close_all

        def observed_close() -> None:
            close_all()
            closed.set()

        with patch.object(self.controller, "close_all", side_effect=observed_close):
            self.controller.open_timed("metal", dwell_seconds=0.03)
            # Model a blocked camera/UI: no main-thread controller work occurs.
            self.assertTrue(closed.wait(timeout=2), "independent timer did not close lids")

        self.assertIsNone(self.controller.active_lid)
        self.controller.raise_pending_error()
        self.assertEqual([event[1] for event in self.pwm.events[-3:]], [0, 1, 2])

    def test_context_cleanup_cancels_timer_and_prevents_later_writes(self) -> None:
        pwm = FakePWM()
        with LidController(self.config, pwm=pwm, sleep=lambda _: None) as controller:
            controller.open_timed("plastic", dwell_seconds=30)
            timer = controller._close_timer
            generation = controller._timer_generation
            self.assertIsNotNone(timer)

        assert timer is not None
        timer.join(timeout=2)
        self.assertFalse(timer.is_alive())
        self.assertTrue(pwm.closed)
        self.assertIsNone(controller.active_lid)
        final_events = list(pwm.events)
        # A callback already queued behind shutdown's lock must also be inert.
        controller._close_when_due(generation)
        self.assertEqual(pwm.events, final_events)

    def test_stale_timer_cannot_close_a_newer_open(self) -> None:
        self.controller.open_timed("plastic", dwell_seconds=30)
        old_timer = self.controller._close_timer
        old_generation = self.controller._timer_generation
        self.controller.open_timed("metal", dwell_seconds=30)

        self.controller._close_when_due(old_generation)

        self.assertEqual(self.controller.active_lid, LidName.METAL)
        assert old_timer is not None
        old_timer.join(timeout=2)
        self.assertFalse(old_timer.is_alive())

    def test_async_close_error_is_reported_and_blocks_new_opens(self) -> None:
        failed = threading.Event()
        write_pulse = self.pwm.set_pulse_us
        original_error = OSError("I2C close failed")

        def fail_in_timer(channel: int, pulse_us: float) -> None:
            if threading.current_thread() is not threading.main_thread():
                failed.set()
                raise original_error
            write_pulse(channel, pulse_us)

        with self.assertLogs("ecosort_hw.lids", level="ERROR"):
            with patch.object(self.pwm, "set_pulse_us", side_effect=fail_in_timer):
                self.controller.open_timed("metal", dwell_seconds=0.03)
                self.assertTrue(failed.wait(timeout=2), "timer did not attempt closure")
                with self.assertRaisesRegex(RuntimeError, "Timed lid closure failed") as raised:
                    self.controller.raise_pending_error()

        self.assertIs(raised.exception.__cause__, original_error)
        self.assertEqual(self.controller.active_lid, LidName.METAL)
        event_count = len(self.pwm.events)
        with self.assertRaisesRegex(RuntimeError, "Timed lid closure failed"):
            self.controller.open_timed("plastic", dwell_seconds=0)
        self.assertEqual(len(self.pwm.events), event_count)

    def test_invalid_dwell_is_rejected_before_any_movement(self) -> None:
        for dwell in (-1, math.nan, math.inf, -math.inf):
            for operation in (self.controller.open_for, self.controller.open_timed):
                with self.subTest(dwell=dwell, operation=operation.__name__):
                    with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                        operation("plastic", dwell_seconds=dwell)
        self.assertEqual(self.pwm.events, [])

    def test_timed_open_rejects_wait_overflow_before_movement(self) -> None:
        with self.assertRaisesRegex(ValueError, "maximum supported duration"):
            self.controller.open_timed("plastic", dwell_seconds=threading.TIMEOUT_MAX * 2)
        self.assertEqual(self.pwm.events, [])

    def test_invalid_config_timings_are_rejected(self) -> None:
        for field in ("dwell_seconds", "movement_seconds"):
            for duration in (-1, math.nan, math.inf, -math.inf):
                with self.subTest(field=field, duration=duration):
                    with self.assertRaisesRegex(ValueError, "finite and non-negative"):
                        LidControllerConfig(**{field: duration})

    def test_open_lid_keeps_pwm_enabled_until_close(self) -> None:
        pwm = FakePWM()
        config = LidControllerConfig(
            dry_run=False,
            movement_seconds=0,
            release_after_move=True,
        )

        with LidController(config, pwm=pwm, sleep=lambda _: None) as controller:
            pwm.events.clear()
            controller.open_lid("general")
            self.assertEqual(pwm.events[-1][0], "pulse")
            self.assertEqual(pwm.events[-1][1], 1)
            controller.close_lid("general")
            self.assertEqual(pwm.events[-1], ("disable", 1, None))

    def test_application_convenience_interface(self) -> None:
        controller = LidController.from_defaults(bus=7, address=0x41, dry_run=True)
        controller._sleep = self.delays.append

        with controller:
            controller.open_temporarily("metal", seconds=0.25)

        self.assertEqual(self.delays, [0.25])
        self.assertEqual(controller._pca_config.bus_number, 7)
        self.assertEqual(controller._pca_config.address, 0x41)
        self.assertIsNone(controller.active_lid)

    def test_dry_run_never_touches_supplied_pwm(self) -> None:
        pwm = FakePWM()
        config = LidControllerConfig(dry_run=True, movement_seconds=0)

        with LidController(config, pwm=pwm, sleep=lambda _: None) as controller:
            controller.open_for("paper", dwell_seconds=0)

        self.assertFalse(pwm.initialized)
        self.assertEqual(pwm.events, [])
        self.assertFalse(pwm.closed)

    def test_context_manager_closes_all_lids_and_hardware(self) -> None:
        pwm = FakePWM()
        config = LidControllerConfig(
            dry_run=False,
            movement_seconds=0,
            release_after_move=False,
        )

        with LidController(config, pwm=pwm, sleep=lambda _: None) as controller:
            controller.open_lid("metal")
        final_channels = [event[1] for event in pwm.events[-3:]]

        self.assertEqual(final_channels, [0, 1, 2])
        self.assertTrue(pwm.closed)

    def test_json_config_supports_individual_and_reversed_servo_pulses(self) -> None:
        document = {
            "lids": {
                "plastic": {"channel": 0, "closed_pulse_us": 1000, "open_pulse_us": 1800},
                "general": {"channel": 1, "closed_pulse_us": 1050, "open_pulse_us": 1750},
                "metal": {"channel": 2, "closed_pulse_us": 1850, "open_pulse_us": 950},
            },
            "movement_seconds": 0.3,
            "release_after_move": True,
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "hardware.json"
            path.write_text(json.dumps(document), encoding="utf-8")

            config = load_lid_config(path, dry_run=True)

        self.assertEqual(config.lids[1].open_pulse_us, 1750)
        self.assertEqual(config.lids[2].closed_pulse_us, 1850)
        self.assertEqual(config.lids[2].open_pulse_us, 950)
        self.assertEqual(config.movement_seconds, 0.3)


if __name__ == "__main__":
    unittest.main()
