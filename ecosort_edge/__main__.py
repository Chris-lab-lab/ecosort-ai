"""Run the EcoSort API without the camera loop for wiring and app tests."""

from __future__ import annotations

import argparse
from threading import Event

from ecosort_hw.sensors import VL53L0XDistanceSensor

from .monitor import DepthSensorMonitor
from .server import EdgeApiServer
from .state import BIN_CATEGORIES, EdgeStateStore


def main() -> int:
    parser = argparse.ArgumentParser(description="EcoSort edge telemetry API")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--depth-i2c-bus", type=int)
    parser.add_argument("--depth-i2c-address", type=lambda value: int(value, 0), default=0x29)
    parser.add_argument("--depth-bin", choices=BIN_CATEGORIES, default="plastic")
    parser.add_argument("--empty-depth-cm", type=float, default=30.0)
    parser.add_argument("--depth-poll-seconds", type=float, default=1.0)
    args = parser.parse_args()

    state = EdgeStateStore(
        monitored_bin=args.depth_bin,
        empty_depth_cm=args.empty_depth_cm,
    )
    server = EdgeApiServer(state, host=args.host, port=args.port).start()
    monitor = None
    try:
        if args.depth_i2c_bus is not None:
            sensor = VL53L0XDistanceSensor(
                args.depth_i2c_bus,
                address=args.depth_i2c_address,
            )
            monitor = DepthSensorMonitor(
                sensor,
                state,
                poll_seconds=args.depth_poll_seconds,
            ).start()
        print(f"EcoSort API listening on http://{server.host}:{server.port}")
        print("Press Ctrl+C to stop.")
        Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        if monitor is not None:
            monitor.stop()
        server.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
