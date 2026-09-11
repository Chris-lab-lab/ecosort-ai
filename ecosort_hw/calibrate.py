"""Move one disconnected servo with a raw pulse for safe calibration."""

from __future__ import annotations

import argparse
import time

from .pca9685 import PCA9685, PCA9685Config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calibrate one SG90 channel without commanding the other lids."
    )
    parser.add_argument("--live", action="store_true", help="actually access I2C (default: preview only)")
    parser.add_argument("--bus", type=int, help="required with --live; find it with i2cdetect -l")
    parser.add_argument("--address", type=lambda value: int(value, 0), default=0x40)
    parser.add_argument("--channel", type=int, required=True, choices=range(16))
    parser.add_argument("--pulse-us", type=float, default=1500.0)
    parser.add_argument("--seconds", type=float, default=0.5)
    args = parser.parse_args(argv)

    if not 900 <= args.pulse_us <= 2100:
        parser.error("start calibration between 900 and 2100 microseconds")
    if not 0 <= args.seconds <= 3:
        parser.error("--seconds must be between 0 and 3")

    print(
        f"Channel {args.channel}: {args.pulse_us:.0f} us for {args.seconds:.2f} s "
        f"({'LIVE' if args.live else 'DRY RUN'})"
    )
    if not args.live:
        print("No hardware changed. Add --live --bus BUS only after removing the horn/linkage.")
        return 0
    if args.bus is None:
        parser.error("--live requires --bus")

    driver = PCA9685(PCA9685Config(bus_number=args.bus, address=args.address))
    try:
        driver.initialize()
        driver.set_pulse_us(args.channel, args.pulse_us)
        time.sleep(args.seconds)
    finally:
        if driver.initialized:
            driver.disable_channel(args.channel)
        driver.close()
    print("Pulse disabled on the selected channel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
