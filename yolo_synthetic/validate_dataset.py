"""Independently check synthetic YOLO annotations against saved instance masks."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys

import numpy as np
from PIL import Image


CLASS_NAMES = ("plastic_bottle", "metal_can", "wrapper")
SPLITS = ("train", "val", "test")


def _check_yaml(dataset: Path, errors: list[str]) -> None:
    """Check the deliberately small portable data.yaml without a YAML dependency."""
    yaml_path = dataset / "data.yaml"
    if not yaml_path.is_file():
        errors.append("Missing data.yaml")
        return
    lines = yaml_path.read_text(encoding="utf-8-sig").splitlines()
    top_level: dict[str, str] = {}
    name_lines: list[str] = []
    in_names = False
    for original in lines:
        line = original.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line[0].isspace():
            match = re.fullmatch(r"([a-zA-Z_]+):\s*(.*)", line)
            if not match:
                errors.append(f"Unsupported data.yaml line: {line!r}")
                continue
            key, value = match.groups()
            if key in top_level:
                errors.append(f"Duplicate data.yaml key: {key}")
            top_level[key] = value.strip().strip("\"'")
            in_names = key == "names"
        elif in_names:
            name_lines.append(line.strip())
    if "path" in top_level:
        errors.append("data.yaml must omit path so the dataset remains portable")
    for split in SPLITS:
        if top_level.get(split) not in (f"images/{split}", f"images/{split}/"):
            errors.append(f"data.yaml {split} must be images/{split}")
    if "nc" in top_level and top_level["nc"] != "3":
        errors.append("data.yaml nc must equal 3")
    parsed_names: dict[int, str] = {}
    inline_names = top_level.get("names", "")
    if inline_names.startswith("[") and inline_names.endswith("]"):
        parsed_names = {
            i: part.strip().strip("\"'")
            for i, part in enumerate(inline_names[1:-1].split(","))
        }
    else:
        for line in name_lines:
            match = re.fullmatch(r"(\d+):\s*(.+)", line)
            if match:
                index, name = match.groups()
                parsed_names[int(index)] = name.strip().strip("\"'")
            elif line.startswith("- "):
                parsed_names[len(parsed_names)] = line[2:].strip().strip("\"'")
            else:
                errors.append(f"Unsupported data.yaml names line: {line!r}")
    if parsed_names != dict(enumerate(CLASS_NAMES)):
        errors.append(f"data.yaml names must be {dict(enumerate(CLASS_NAMES))}")


def _manifest_path(
    dataset: Path, row: dict, field: str, directory: str, suffix: str,
    split: str, prefix: str, errors: list[str], referenced: dict[str, set[str]],
) -> Path | None:
    value = row.get(field)
    if not isinstance(value, str):
        errors.append(f"{prefix}: {field} must be a relative POSIX path")
        return None
    relative = PurePosixPath(value)
    if (
        relative.is_absolute() or ".." in relative.parts or "\\" in value
        or ":" in value or len(relative.parts) != 3
        or relative.parts[:2] != (directory, split) or relative.suffix != suffix
    ):
        errors.append(f"{prefix}: invalid {field} path {value!r}")
        return None
    canonical = relative.as_posix()
    if canonical in referenced[field]:
        errors.append(f"{prefix}: duplicate manifest {field} {canonical}")
    referenced[field].add(canonical)
    path = dataset.joinpath(*relative.parts)
    if not path.is_file():
        errors.append(f"{prefix}: missing {field} file {canonical}")
        return None
    return path


def validate_dataset(dataset: Path | str) -> dict:
    """Return a JSON-serializable report; all validation failures are collected."""
    dataset = Path(dataset).resolve()
    errors: list[str] = []
    counts = {
        split: {"images": 0, "negative_images": 0, "objects": 0,
                "class_instances": {name: 0 for name in CLASS_NAMES}}
        for split in SPLITS
    }
    referenced = {field: set() for field in ("image", "label", "mask")}
    seen_seeds: dict[int, str] = {}
    seen_hashes: dict[str, tuple[str, str]] = {}
    _check_yaml(dataset, errors)
    manifest = dataset / "manifest.jsonl"
    rows: list[tuple[int, dict]] = []
    if not manifest.is_file():
        errors.append("Missing manifest.jsonl")
    else:
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                errors.append(f"manifest line {line_number}: blank record")
                continue
            try:
                row = json.loads(line)
            except (ValueError, TypeError) as exc:
                errors.append(f"manifest line {line_number}: invalid JSON ({exc})")
                continue
            if not isinstance(row, dict):
                errors.append(f"manifest line {line_number}: record must be an object")
                continue
            rows.append((line_number, row))
    for line_number, row in rows:
        prefix = f"manifest line {line_number} ({row.get('image', '?')})"
        split = row.get("split")
        if split not in SPLITS:
            errors.append(f"{prefix}: unknown split {split!r}")
            continue
        counts[split]["images"] += 1
        seed = row.get("seed")
        if type(seed) is not int:
            errors.append(f"{prefix}: seed must be an integer")
        elif seed in seen_seeds:
            errors.append(f"{prefix}: duplicate seed {seed}, first used by {seen_seeds[seed]}")
        else:
            seen_seeds[seed] = str(row.get("image"))
        image_path = _manifest_path(dataset, row, "image", "images", ".jpg", split, prefix, errors, referenced)
        label_path = _manifest_path(dataset, row, "label", "labels", ".txt", split, prefix, errors, referenced)
        mask_path = _manifest_path(dataset, row, "mask", "masks", ".png", split, prefix, errors, referenced)
        existing_paths = [path for path in (image_path, label_path, mask_path) if path]
        if len({path.stem for path in existing_paths}) > 1:
            errors.append(f"{prefix}: image, label and mask stems must match")
        width, height = row.get("width"), row.get("height")
        valid_size = type(width) is int and type(height) is int and width > 0 and height > 0
        if not valid_size:
            errors.append(f"{prefix}: width and height must be positive integers")
        if image_path:
            try:
                with Image.open(image_path) as image:
                    image.load()
                    if image.format != "JPEG":
                        errors.append(f"{prefix}: image must be a JPEG")
                    if valid_size and image.size != (width, height):
                        errors.append(f"{prefix}: image resolution differs from manifest")
                    # Hash decoded pixels, catching identical images with different JPEG metadata.
                    rgb = image.convert("RGB")
                    digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
                    if digest in seen_hashes:
                        previous_split, previous_image = seen_hashes[digest]
                        if previous_split != split:
                            errors.append(f"{prefix}: duplicate image across splits, also {previous_image}")
                    else:
                        seen_hashes[digest] = (split, str(row["image"]))
            except (OSError, ValueError) as exc:
                errors.append(f"{prefix}: cannot decode image ({exc})")
        mask = None
        if mask_path:
            try:
                with Image.open(mask_path) as mask_image:
                    if mask_image.format != "PNG":
                        errors.append(f"{prefix}: mask must be a PNG")
                    mask = np.asarray(mask_image)
                    if valid_size and mask_image.size != (width, height):
                        errors.append(f"{prefix}: mask resolution differs from manifest")
                    if mask.ndim != 2 or mask.dtype != np.uint8:
                        errors.append(f"{prefix}: instance mask must be single-channel uint8")
                        mask = None
            except (OSError, ValueError) as exc:
                errors.append(f"{prefix}: cannot decode mask ({exc})")
        objects = row.get("objects")
        if not isinstance(objects, list) or not all(isinstance(obj, dict) for obj in objects):
            errors.append(f"{prefix}: objects must be a list of objects")
            continue
        counts[split]["objects"] += len(objects)
        if not objects:
            counts[split]["negative_images"] += 1
        instance_ids = [obj.get("instance_id") for obj in objects]
        if any(type(instance_id) is not int for instance_id in instance_ids) or instance_ids != list(range(1, len(objects) + 1)):
            errors.append(f"{prefix}: instance IDs must be 1..N in object order")
        if mask is not None:
            actual_ids = set(np.unique(mask).tolist()) - {0}
            if actual_ids != set(range(1, len(objects) + 1)):
                errors.append(f"{prefix}: mask instance IDs differ from manifest objects")
        label_lines: list[str] = []
        if label_path:
            try:
                label_lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            except (OSError, UnicodeError) as exc:
                errors.append(f"{prefix}: cannot read labels ({exc})")
            if len(label_lines) != len(objects):
                errors.append(f"{prefix}: label row count {len(label_lines)} differs from object count {len(objects)}")
        for index, obj in enumerate(objects):
            object_prefix = f"{prefix}, object {index + 1}"
            class_id = obj.get("class_id")
            if type(class_id) is not int or class_id not in range(len(CLASS_NAMES)):
                errors.append(f"{object_prefix}: invalid manifest class_id")
            else:
                counts[split]["class_instances"][CLASS_NAMES[class_id]] += 1
            box = obj.get("bbox_xyxy")
            valid_box = isinstance(box, list) and len(box) == 4 and all(type(v) is int for v in box)
            if not valid_box:
                errors.append(f"{object_prefix}: bbox_xyxy must contain four integer pixel bounds")
            elif valid_size:
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    errors.append(f"{object_prefix}: manifest bbox is outside the image or empty")
            if mask is not None:
                ys, xs = np.where(mask == index + 1)
                if not len(xs):
                    errors.append(f"{object_prefix}: instance has no visible pixels")
                elif valid_box:
                    tight = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                    if box != tight:
                        errors.append(f"{object_prefix}: bbox {box} is not tight around visible mask {tight}")
            if index >= len(label_lines):
                continue
            columns = label_lines[index].split()
            if len(columns) != 5:
                errors.append(f"{object_prefix}: YOLO label must have exactly 5 columns")
                continue
            if not re.fullmatch(r"[0-2]", columns[0]):
                errors.append(f"{object_prefix}: YOLO class must be integer 0, 1 or 2")
            elif int(columns[0]) != class_id:
                errors.append(f"{object_prefix}: label class differs from manifest")
            try:
                xc, yc, bw, bh = map(float, columns[1:])
            except ValueError:
                errors.append(f"{object_prefix}: YOLO coordinates must be numbers")
                continue
            if not all(math.isfinite(value) and 0 <= value <= 1 for value in (xc, yc, bw, bh)) or bw <= 0 or bh <= 0:
                errors.append(f"{object_prefix}: YOLO coordinates must be finite and normalized, with positive size")
                continue
            if valid_size:
                projected = [(xc - bw / 2) * width, (yc - bh / 2) * height,
                             (xc + bw / 2) * width, (yc + bh / 2) * height]
                if projected[0] < -1 or projected[1] < -1 or projected[2] > width + 1 or projected[3] > height + 1:
                    errors.append(f"{object_prefix}: YOLO bounds extend outside the image")
                if valid_box and any(abs(actual - expected) > 1 for actual, expected in zip(projected, box)):
                    errors.append(f"{object_prefix}: YOLO bbox differs from manifest by more than 1 pixel")
    for field, directory in (("image", "images"), ("label", "labels"), ("mask", "masks")):
        disk_paths = {
            path.relative_to(dataset).as_posix()
            for path in (dataset / directory).rglob("*") if path.is_file()
        }
        for orphan in sorted(disk_paths - referenced[field]):
            errors.append(f"Orphan {field} file: {orphan}")
    for split, split_counts in counts.items():
        if not split_counts["images"]:
            errors.append(f"Split {split} has no images")
        for class_name, count in split_counts["class_instances"].items():
            if not count:
                errors.append(f"Split {split} has no {class_name} instances")
    return {
        "dataset": str(dataset), "valid": not errors,
        "images": sum(count["images"] for count in counts.values()),
        "objects": sum(count["objects"] for count in counts.values()),
        "classes": dict(enumerate(CLASS_NAMES)), "splits": counts,
        "unique_seeds": len(seen_seeds), "unique_image_hashes": len(seen_hashes),
        "error_count": len(errors), "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset directory containing data.yaml")
    parser.add_argument("--report", type=Path, help="Also write the JSON report to this file")
    args = parser.parse_args()
    report = validate_dataset(args.dataset)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    sys.exit(main())
