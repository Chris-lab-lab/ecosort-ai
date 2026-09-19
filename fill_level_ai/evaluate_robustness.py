from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision import transforms

from inference import CLASS_NAMES, load_model, predict_pil
from preprocessing import median_color, to_normalized_tensor
from train import compute_metrics


CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
OLD_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and robust models under controlled camera variations.")
    parser.add_argument("--test-dir", type=Path, required=True)
    parser.add_argument("--baseline-model", type=Path, required=True)
    parser.add_argument("--improved-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.58)
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def old_preprocess(image: Image.Image) -> torch.Tensor:
    return OLD_TRANSFORM(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0)


def predict_old(model, image: Image.Image) -> dict:
    tensor = old_preprocess(image)
    started = time.perf_counter()
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    elapsed_ms = (time.perf_counter() - started) * 1000
    index = int(probabilities.argmax())
    return {"prediction": CLASS_NAMES[index], "raw_prediction": CLASS_NAMES[index], "confidence": float(probabilities[index]), "inference_ms": elapsed_ms, "warnings": []}


def zoom_crop(image: Image.Image, zoom: float) -> Image.Image:
    width, height = image.size
    resized = image.resize((round(width * zoom), round(height * zoom)), Image.Resampling.LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def scaled_canvas(image: Image.Image, scale: float, rng: random.Random, offset: bool) -> Image.Image:
    width, height = image.size
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    color = tuple(min(255, max(0, value + rng.randint(-35, 35))) for value in median_color(image))
    canvas = Image.new("RGB", image.size, color)
    max_x, max_y = width - resized.width, height - resized.height
    x = rng.randint(0, max_x) if offset else max_x // 2
    y = rng.randint(0, max_y) if offset else max_y // 2
    canvas.paste(resized, (x, y))
    return canvas


def transform(image: Image.Image, condition: str, rng: random.Random) -> Image.Image:
    image = image.convert("RGB")
    if condition == "medium":
        return image
    if condition == "close":
        return zoom_crop(image, 1.16)
    if condition == "far":
        return scaled_canvas(image, 0.55, rng, False)
    if condition == "angle":
        return image.rotate(rng.choice((-10, -7, 7, 10)), Image.Resampling.BICUBIC, fillcolor=median_color(image))
    if condition == "lighting":
        result = ImageEnhance.Brightness(image).enhance(rng.choice((0.58, 1.42)))
        result = ImageEnhance.Contrast(result).enhance(rng.choice((0.74, 1.28)))
        array = np.asarray(result, dtype=np.float32)
        temperature = rng.choice((-0.18, 0.18))
        array[..., 0] *= 1 + temperature
        array[..., 2] *= 1 - temperature
        return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")
    if condition == "background_position":
        return scaled_canvas(image, 0.68, rng, True)
    if condition == "low_resolution":
        return image.resize((96, 96), Image.Resampling.BILINEAR).resize(image.size, Image.Resampling.BILINEAR)
    if condition == "blur_noise":
        result = image.filter(ImageFilter.GaussianBlur(1.35))
        array = np.asarray(result, dtype=np.float32)
        generator = np.random.default_rng(rng.randrange(2**32))
        return Image.fromarray(np.clip(array + generator.normal(0, 8.0, array.shape), 0, 255).astype(np.uint8), "RGB")
    raise ValueError(condition)


def summarize(rows: list[dict], model_name: str) -> dict:
    selected = [row for row in rows if row["model"] == model_name]
    by_condition = {}
    for condition in sorted({row["condition"] for row in selected}):
        group = [row for row in selected if row["condition"] == condition]
        by_condition[condition] = {
            "images": len(group),
            "accuracy": sum(row["correct"] for row in group) / len(group),
            "raw_accuracy": sum(row["raw_correct"] for row in group) / len(group),
            "consistency_with_medium": sum(row["consistent"] for row in group) / len(group),
            "uncertain": sum(row["prediction"] == "Uncertain" for row in group),
            "mean_confidence": sum(row["confidence"] for row in group) / len(group),
            "mean_inference_ms": sum(row["inference_ms"] for row in group) / len(group),
        }
    medium = [row for row in selected if row["condition"] == "medium"]
    targets = [CLASS_TO_INDEX[row["actual"]] for row in medium]
    predictions = [CLASS_TO_INDEX[row["raw_prediction"]] for row in medium]
    metrics = compute_metrics(targets, predictions)
    supports = [metrics["per_class"][name]["support"] for name in CLASS_NAMES]
    total = max(1, sum(supports))
    metrics["weighted_f1"] = sum(metrics["per_class"][name]["f1"] * metrics["per_class"][name]["support"] for name in CLASS_NAMES) / total
    transformed = [row for row in selected if row["condition"] != "medium"]
    return {
        "independent_test": metrics,
        "conditions": by_condition,
        "transformed_overall": {
            "accuracy": sum(row["correct"] for row in transformed) / len(transformed),
            "raw_accuracy": sum(row["raw_correct"] for row in transformed) / len(transformed),
            "consistency": sum(row["consistent"] for row in transformed) / len(transformed),
            "uncertain": sum(row["prediction"] == "Uncertain" for row in transformed),
            "images": len(transformed),
        },
        "average_inference_ms": sum(row["inference_ms"] for row in selected) / len(selected),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = args.output_dir / "examples"
    examples_dir.mkdir(exist_ok=True)
    baseline = load_model(args.baseline_model)
    improved = load_model(args.improved_model)
    conditions = ("medium", "close", "far", "angle", "lighting", "background_position", "low_resolution", "blur_noise")
    samples = []
    for class_name in CLASS_NAMES:
        for path in sorted((args.test_dir / class_name).glob("original_*")):
            samples.append((path, class_name))

    rows = []
    medium_predictions = {"baseline": {}, "improved": {}}
    saved_examples = Counter()
    for path, actual in samples:
        with Image.open(path) as opened:
            original = ImageOps.exif_transpose(opened).convert("RGB")
        stable_seed = int(hashlib.sha256(path.name.encode()).hexdigest()[:8], 16) + args.seed
        for condition in conditions:
            variant = transform(original, condition, random.Random(stable_seed + conditions.index(condition) * 997))
            for model_name, predictor in (
                ("baseline", lambda value: predict_old(baseline, value)),
                ("improved", lambda value: predict_pil(improved, value, args.confidence_threshold)),
            ):
                result = predictor(variant)
                if condition == "medium":
                    medium_predictions[model_name][str(path)] = result["raw_prediction"]
                reference = medium_predictions[model_name].get(str(path), result["raw_prediction"])
                row = {
                    "model": model_name,
                    "path": str(path.resolve()),
                    "actual": actual,
                    "condition": condition,
                    "prediction": result["prediction"],
                    "raw_prediction": result["raw_prediction"],
                    "confidence": result["confidence"],
                    "correct": result["prediction"] == actual,
                    "raw_correct": result["raw_prediction"] == actual,
                    "consistent": result["raw_prediction"] == reference,
                    "inference_ms": result["inference_ms"],
                    "warnings": " | ".join(result.get("warnings", [])),
                }
                rows.append(row)
                if model_name == "improved":
                    category = None
                    if row["prediction"] == "Uncertain":
                        category = "uncertain"
                    elif not row["raw_correct"]:
                        category = "incorrect"
                    elif not row["consistent"]:
                        category = "changed"
                    elif condition == "medium":
                        category = "correct"
                    if category and saved_examples[category] < 4:
                        destination = examples_dir / f"{category}_{saved_examples[category] + 1}_{condition}_{actual}.jpg"
                        variant.save(destination, quality=90)
                        saved_examples[category] += 1

    with (args.output_dir / "robustness_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "test_images": len(samples),
        "conditions": list(conditions),
        "confidence_threshold": args.confidence_threshold,
        "baseline": summarize(rows, "baseline"),
        "improved": summarize(rows, "improved"),
        "model_sizes_bytes": {"baseline": args.baseline_model.stat().st_size, "improved": args.improved_model.stat().st_size},
        "examples_saved": dict(saved_examples),
        "limitations": [
            "Controlled transformations simulate distance and capture changes but are not a substitute for a new physical recording session.",
            "The current source data still contains strong correlations between bin appearance and class.",
        ],
    }
    (args.output_dir / "robustness_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
