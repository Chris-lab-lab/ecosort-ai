from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps

from preprocessing import make_inference_views, to_normalized_tensor


CLASS_NAMES = ("empty", "half-full", "full")
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def default_model_path() -> Path:
    model_dir = Path(__file__).with_name("model")
    candidates = (
        model_dir / "waste_bin_robust_v2.torchscript.pt",
        model_dir / "waste_bin_custom.torchscript.pt",
        model_dir / "waste_bin_fill_level_mobilenet_v3_small.torchscript.pt",
    )
    return next((path for path in candidates if path.exists()), candidates[0])


def load_model(path: Path):
    return torch.jit.load(str(path), map_location="cpu").eval()


def predict_pil(model, image: Image.Image, confidence_threshold: float = 0.58) -> dict:
    views, quality = make_inference_views(image)
    batch = torch.stack([to_normalized_tensor(view) for view in views])
    started = time.perf_counter()
    with torch.inference_mode():
        view_probabilities = torch.softmax(model(batch), dim=1)
    elapsed_ms = (time.perf_counter() - started) * 1000
    probabilities = view_probabilities.mean(dim=0)
    index = int(probabilities.argmax())
    confidence = float(probabilities[index])
    view_predictions = view_probabilities.argmax(dim=1)
    agreement = float((view_predictions == index).float().mean())
    entropy = float(-(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum() / math.log(len(CLASS_NAMES)))
    return {
        "prediction": CLASS_NAMES[index],
        "raw_prediction": CLASS_NAMES[index],
        "confidence": confidence,
        "probabilities": {name: float(probabilities[position]) for position, name in enumerate(CLASS_NAMES)},
        "view_agreement": agreement,
        "normalized_entropy": entropy,
        "quality": {
            "blur_score": quality.blur_score,
            "mean_luminance": quality.mean_luminance,
            "contrast": quality.contrast,
            "roi_area_ratio": quality.roi_area_ratio,
            "active_edge_fraction": quality.active_edge_fraction,
        },
        "warnings": list(quality.warnings),
        "inference_ms": elapsed_ms,
    }


def predict_path(model, path: Path, confidence_threshold: float) -> dict:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    result = predict_pil(model, image, confidence_threshold)
    result["path"] = str(path.resolve())
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust trash-bin fill inference for one image or a folder.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", type=Path, default=default_model_path())
    parser.add_argument("--output", type=Path, help="Optional JSON (single image) or CSV (folder) output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_model(args.model)
    if args.input.is_file():
        result = predict_path(model, args.input, 0.0)
        payload = json.dumps({"prediction": result["prediction"]}, indent=2)
        print(payload)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(payload, encoding="utf-8")
        return
    if not args.input.is_dir():
        raise FileNotFoundError(args.input)
    paths = sorted(path for path in args.input.rglob("*") if path.is_file() and path.suffix.lower() in SUFFIXES)
    rows = [predict_path(model, path, 0.0) for path in paths]
    output = args.output or Path("folder_predictions.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["path", "prediction"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in rows:
            writer.writerow({key: result[key] for key in fieldnames})
    summary = {
        "images": len(rows),
        "predictions": {name: sum(result["prediction"] == name for result in rows) for name in CLASS_NAMES},
        "output": str(output.resolve()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
