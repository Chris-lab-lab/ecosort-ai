"""Command-line self-test for the EcoSort three-lid controller."""

from __future__ import annotations

import argparse
import logging

from .lids import LidController, LidControllerConfig, LidName
from .pca9685 import PCA9685Config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Safely open and close the EcoSort prototype lids.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="print movements without accessing I2C (default)",
    )
    mode.add_argument(
        "--live",
        action="store_true",
        help="control the physical PCA9685 and servos",
    )
    parser.add_argument(
        "--lid",
        choices=[lid.value for lid in LidName] + ["paper", "middle", "all"],
        default="all",
        help="lid to test (default: all, one at a time)",
    )
    parser.add_argument("--dwell", type=float, default=1.0, help="open time in seconds")
    parser.add_argument(
        "--bus",
        type=int,
        help="Linux I2C bus number; required with --live (find it with i2cdetect -l)",
    )
    parser.add_argument(
        "--address",
        type=lambda value: int(value, 0),
        default=0x40,
        help="PCA9685 I2C address (default: 0x40)",
    )
    parser.add_argument(
        "--hardware-config",
        help="JSON file with per-lid channels and calibrated pulses",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.dwell < 0:
        raise SystemExit("--dwell cannot be negative")
    if args.live and args.bus is None:
        raise SystemExit("--live requires --bus; first identify the header bus with i2cdetect -l")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dry_run = not args.live
    pca_config = PCA9685Config(
        bus_number=args.bus if args.bus is not None else 1,
        address=args.address,
    )

    print("EcoSort three-lid self-test")
    print(f"Mode: {'SIMULATION' if dry_run else 'LIVE HARDWARE'}")
    if not dry_run:
        print("Keep hands clear; lids will move one at a time.")

    order = tuple(LidName) if args.lid == "all" else (args.lid,)
    controller = (
        LidController.from_config_file(
            args.hardware_config,
            bus=pca_config.bus_number,
            address=pca_config.address,
            dry_run=dry_run,
        )
        if args.hardware_config
        else LidController(
            LidControllerConfig(dry_run=dry_run, dwell_seconds=args.dwell),
            pca_config=pca_config,
        )
    )
    with controller:
        for name in order:
            controller.open_for(name, dwell_seconds=args.dwell)

    print("Self-test complete; all lids commanded closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
