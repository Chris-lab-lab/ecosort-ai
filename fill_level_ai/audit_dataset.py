from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision import transforms


CLASSES = ("empty", "half-full", "full")
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit original trash-bin images and flag manual-review candidates.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--near-duplicate-distance", type=int, default=6)
    parser.add_argument("--label-warning-confidence", type=float, default=0.75)
    return parser.parse_args()


def dhash(image: Image.Image, size: int = 16) -> int:
    gray = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.int16)
    bits = (values[:, 1:] > values[:, :-1]).flatten()
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def blur_score(image: Image.Image) -> float:
    gray = np.asarray(image.convert("L").resize((320, 320)), dtype=np.float32)
    laplacian = -4 * gray
    laplacian[1:, :] += gray[:-1, :]
    laplacian[:-1, :] += gray[1:, :]
    laplacian[:, 1:] += gray[:, :-1]
    laplacian[:, :-1] += gray[:, 1:]
    return float(laplacian[2:-2, 2:-2].var())


def make_contact_sheets(records: list[dict], output_dir: Path) -> None:
    cell_width, cell_height = 240, 280
    columns, rows = 4, 4
    for class_name in CLASSES:
        group = [record for record in records if record["label"] == class_name]
        for page_index in range(math.ceil(len(group) / (columns * rows))):
            page = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
            draw = ImageDraw.Draw(page)
            page_records = group[page_index * columns * rows : (page_index + 1) * columns * rows]
            for index, record in enumerate(page_records):
                with Image.open(record["absolute_path"]) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                image.thumbnail((cell_width - 12, cell_height - 58), Image.Resampling.LANCZOS)
                x = (index % columns) * cell_width
                y = (index // columns) * cell_height
                page.paste(image, (x + (cell_width - image.width) // 2, y + 4))
                predicted = record["prediction"]
                confidence = record["confidence"]
                color = "#b00020" if predicted != class_name else "#145a32"
                draw.text((x + 6, y + cell_height - 48), f"{record['split']} | {record['name'][:22]}", fill="black")
                draw.text((x + 6, y + cell_height - 28), f"pred {predicted} {confidence:.2f}", fill=color)
            destination = output_dir / "contact_sheets" / f"{class_name}_{page_index + 1:02d}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            page.save(destination, quality=90)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize(255),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    model = torch.jit.load(str(args.model), map_location="cpu").eval()

    records = []
    corrupt = []
    for split in ("train", "validation", "test"):
        for class_name in CLASSES:
            folder = args.data_dir / split / class_name
            for path in sorted(folder.glob("original_*")):
                if path.suffix.lower() not in SUFFIXES:
                    continue
                try:
                    with Image.open(path) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB")
                    width, height = image.size
                    array = np.asarray(image.convert("L"), dtype=np.float32)
                    tensor = transform(image).unsqueeze(0)
                    with torch.inference_mode():
                        probabilities = torch.softmax(model(tensor), dim=1)[0]
                    predicted_index = int(probabilities.argmax())
                    confidence = float(probabilities[predicted_index])
                    record = {
                        "path": path.relative_to(args.data_dir).as_posix(),
                        "absolute_path": str(path.resolve()),
                        "split": split,
                        "label": class_name,
                        "name": path.name,
                        "width": width,
                        "height": height,
                        "mean_luminance": float(array.mean()),
                        "luminance_std": float(array.std()),
                        "blur_score": blur_score(image),
                        "sha256_pixels": hashlib.sha256(image.tobytes()).hexdigest(),
                        "dhash": f"{dhash(image):064x}",
                        "prediction": CLASSES[predicted_index],
                        "confidence": confidence,
                        "probabilities": {name: float(probabilities[index]) for index, name in enumerate(CLASSES)},
                        "flags": [],
                    }
                    if min(width, height) < 256:
                        record["flags"].append("low_resolution")
                    if record["blur_score"] < 18:
                        record["flags"].append("blurry")
                    if record["mean_luminance"] < 28:
                        record["flags"].append("too_dark")
                    if record["mean_luminance"] > 238:
                        record["flags"].append("too_bright")
                    if record["luminance_std"] < 14:
                        record["flags"].append("low_contrast")
                    if record["prediction"] != class_name and confidence >= args.label_warning_confidence:
                        record["flags"].append("possible_label_error")
                    if confidence < 0.50:
                        record["flags"].append("ambiguous_model_prediction")
                    records.append(record)
                except Exception as exc:
                    corrupt.append({"path": str(path), "error": str(exc)})

    exact_groups = defaultdict(list)
    for index, record in enumerate(records):
        exact_groups[record["sha256_pixels"]].append(index)
    exact_duplicates = [indices for indices in exact_groups.values() if len(indices) > 1]
    for indices in exact_duplicates:
        for index in indices:
            records[index]["flags"].append("exact_duplicate")

    near_duplicates = []
    hashes = [int(record["dhash"], 16) for record in records]
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            distance = (hashes[left] ^ hashes[right]).bit_count()
            if distance <= args.near_duplicate_distance:
                across_splits = records[left]["split"] != records[right]["split"]
                pair = {
                    "left": records[left]["path"],
                    "right": records[right]["path"],
                    "distance": distance,
                    "across_splits": across_splits,
                }
                near_duplicates.append(pair)
                if across_splits:
                    records[left]["flags"].append("near_duplicate_across_splits")
                    records[right]["flags"].append("near_duplicate_across_splits")

    for record in records:
        record["flags"] = sorted(set(record["flags"]))

    quality_values = {
        "blur_score": [record["blur_score"] for record in records],
        "mean_luminance": [record["mean_luminance"] for record in records],
        "luminance_std": [record["luminance_std"] for record in records],
    }
    quantiles = {
        name: {str(q): float(np.quantile(values, q)) for q in (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)}
        for name, values in quality_values.items()
    }
    report = {
        "total_original_images": len(records),
        "by_class": dict(Counter(record["label"] for record in records)),
        "by_split": dict(Counter(record["split"] for record in records)),
        "by_flag": dict(Counter(flag for record in records for flag in record["flags"])),
        "corrupt": corrupt,
        "exact_duplicate_groups": [[records[index]["path"] for index in group] for group in exact_duplicates],
        "near_duplicate_pairs": near_duplicates,
        "near_duplicate_pairs_across_splits": sum(pair["across_splits"] for pair in near_duplicates),
        "quality_quantiles": quantiles,
        "manual_review_candidates": [record["path"] for record in records if record["flags"]],
        "records": records,
    }
    (args.output_dir / "dataset_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "manual_review_candidates.txt").write_text(
        "\n".join(report["manual_review_candidates"]), encoding="utf-8"
    )
    make_contact_sheets(records, args.output_dir)
    print(
        json.dumps(
            {key: report[key] for key in ("total_original_images", "by_class", "by_split", "by_flag", "near_duplicate_pairs_across_splits")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
