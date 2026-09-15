"""Map dataset_new's disposal subclasses onto EcoSort model labels.

This rewrites metadata only. RGB source images and teacher masks are unchanged.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

FIXED_MAPPING = {
    ("recyclable - others", "glass"): "other",
    ("recyclable - others", "metals"): "metal",
    ("recyclable - others", "plastics"): "plastic",
    ("recyclable - paper", "cardboard"): "general",
    ("recyclable - paper", "paper"): "general",
    ("regular trash", "tissue"): "general",
}


def mapped_label(source: str, plastic_bags_route: str) -> str:
    parts = Path(source.replace("\\", "/")).parts
    if len(parts) < 3:
        raise ValueError(f"Expected top-level group/subclass/image: {source!r}")
    key = (parts[0].casefold(), parts[1].casefold())
    if key == ("regular trash", "plastic bags"):
        return plastic_bags_route
    if key not in FIXED_MAPPING:
        raise ValueError(f"No approved EcoSort mapping for {source!r}")
    return FIXED_MAPPING[key]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create four-class training metadata for dataset_new"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("dataset_segmented_new_raw/segmentation_manifest.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset_segmented_new_raw/segmentation_manifest_mapped.jsonl"),
    )
    parser.add_argument(
        "--plastic-bags-route",
        choices=("general", "plastic"),
        required=True,
        help="explicit project rule for Regular Trash / Plastic Bags",
    )
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    output = args.output.resolve()
    if not manifest.is_file():
        raise SystemExit(f"Mask manifest does not exist: {manifest}")
    if output.exists():
        raise SystemExit(f"Output already exists; choose a new path: {output}")
    if output.parent != manifest.parent:
        raise SystemExit("Mapped manifest must sit beside the original manifest")
    report_path = manifest.parent / "segmentation_report.json"
    if not report_path.is_file():
        raise SystemExit(
            "GPU mask preparation has not completed; segmentation_report.json "
            "is missing"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    expected_records = sum(int(count) for count in report["counts"].values())
    with manifest.open("r", encoding="utf-8") as stream:
        actual_records = sum(bool(line.strip()) for line in stream)
    if actual_records != expected_records:
        raise SystemExit(
            f"Incomplete mask manifest: {actual_records} records, "
            f"report expects {expected_records}"
        )

    counts: Counter[tuple[str, str]] = Counter()
    with (
        manifest.open("r", encoding="utf-8") as source_stream,
        output.open("x", encoding="utf-8") as mapped_stream,
    ):
        for line_number, line in enumerate(source_stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                label = mapped_label(str(record["source"]), args.plastic_bags_route)
            except (KeyError, ValueError) as exc:
                raise SystemExit(f"Manifest line {line_number}: {exc}") from exc
            record["source_label"] = record.get("label")
            record["label"] = label
            record["supported"] = (
                label != "other" and record.get("status") == "accepted"
            )
            counts[(label, str(record.get("status", "unknown")))] += 1
            mapped_stream.write(json.dumps(record) + "\n")

    print(f"Mapped manifest: {output}")
    for (label, status), count in sorted(counts.items()):
        print(f"{label:8s} {status:22s} {count:6d}")


if __name__ == "__main__":
    main()
