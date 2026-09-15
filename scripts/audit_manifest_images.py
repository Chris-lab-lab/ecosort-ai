"""Report image formats that TensorFlow's native decoder cannot read."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

SUPPORTED = {"JPEG", "PNG", "GIF", "BMP"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-data", type=Path, required=True)
    args = parser.parse_args()

    counts: Counter[str] = Counter()
    unsupported: list[tuple[str, str]] = []
    with args.manifest.open(encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("status") != "accepted":
                continue
            source = str(record["source"])
            with Image.open(args.source_data / source) as image:
                image_format = str(image.format)
            counts[image_format] += 1
            if image_format not in SUPPORTED:
                unsupported.append((source, image_format))

    print(f"Image formats: {dict(counts)}")
    print(f"Unsupported images: {len(unsupported)}")
    for source, image_format in unsupported:
        print(f"{image_format}: {source}")


if __name__ == "__main__":
    main()
