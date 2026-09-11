from __future__ import annotations

import glob
import importlib.util
import os


def yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def main() -> None:
    print("EcoSort i.MX93 board check")
    print("--------------------------")
    for module in ("numpy", "cv2", "tflite_runtime", "ai_edge_litert"):
        print(f"Python {module:16s}: {yes_no(importlib.util.find_spec(module) is not None)}")
    print(f"Ethos-U device       : {glob.glob('/dev/ethosu*') or 'not found'}")
    delegates = sorted(set(glob.glob("/usr/lib*/libethosu_delegate.so*")))
    print(f"Ethos-U delegate     : {delegates or 'not found'}")
    print(f"Cameras              : {glob.glob('/dev/video*') or 'not found'}")
    print(f"I2C controllers      : {glob.glob('/dev/i2c-*') or 'not found'}")
    print(f"Wayland display      : {os.getenv('WAYLAND_DISPLAY', 'not set')}")
    print("\nUse `i2cdetect -l` to map the physical P12 I2C header to a bus number.")
    print(
        "After wiring only logic power/SDA/SCL/GND, probe only the expected address "
        "with `i2cdetect -y BUS 0x40 0x40`."
    )


if __name__ == "__main__":
    main()
