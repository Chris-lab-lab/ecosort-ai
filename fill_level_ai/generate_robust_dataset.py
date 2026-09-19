from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps


CLASSES = ("empty", "half-full", "full")
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_range(value: str) -> tuple[float, float]:
    left, right = value.split(",", 1)
    return float(left), float(right)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate conservative robustness augmentations for trash-bin fill classification.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manual-review-list", type=Path)
    parser.add_argument("--target-per-class", type=int, default=600)
    parser.add_argument("--augmentation-probability", type=float, default=0.95)
    parser.add_argument("--zoom-range", type=parse_range, default=(0.55, 1.12))
    parser.add_argument("--rotation-range", type=parse_range, default=(-9.0, 9.0))
    parser.add_argument("--perspective-intensity", type=float, default=0.055)
    parser.add_argument("--brightness-range", type=parse_range, default=(0.67, 1.34))
    parser.add_argument("--contrast-range", type=parse_range, default=(0.72, 1.30))
    parser.add_argument("--saturation-range", type=parse_range, default=(0.65, 1.30))
    parser.add_argument("--blur-intensity", type=float, default=1.25)
    parser.add_argument("--noise-intensity", type=float, default=8.0)
    parser.add_argument("--translation", type=float, default=0.13)
    parser.add_argument("--output-resolution", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def median_color(image: Image.Image) -> tuple[int, int, int]:
    sample = np.asarray(image.resize((32, 32)), dtype=np.uint8).reshape(-1, 3)
    return tuple(int(value) for value in np.median(sample, axis=0))


def cover_resize(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    width, height = size
    scale = max(width / image.width, height / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def background_for(image: Image.Image, rng: random.Random) -> Image.Image:
    width, height = image.size
    if rng.random() < 0.70:
        background = cover_resize(image, (width, height)).filter(ImageFilter.GaussianBlur(radius=max(width, height) / 35))
        background = ImageEnhance.Brightness(background).enhance(rng.uniform(0.65, 1.05))
        background = ImageEnhance.Color(background).enhance(rng.uniform(0.25, 0.80))
        return background
    base = np.array(median_color(image), dtype=np.int16)
    tint = np.clip(base + np.array([rng.randint(-35, 35) for _ in range(3)]), 0, 255)
    background = Image.new("RGB", (width, height), tuple(int(value) for value in tint))
    noise = rng.randint(4, 16)
    array = np.asarray(background, dtype=np.int16)
    generator = np.random.default_rng(rng.randrange(2**32))
    array = np.clip(array + generator.normal(0, noise, array.shape), 0, 255).astype(np.uint8)
    return Image.fromarray(array, "RGB").filter(ImageFilter.GaussianBlur(radius=2.0))


def distance_position(image: Image.Image, rng: random.Random, zoom_range: tuple[float, float], translation: float) -> tuple[Image.Image, dict]:
    width, height = image.size
    zoom = rng.uniform(*zoom_range)
    resized = image.resize((max(32, round(width * zoom)), max(32, round(height * zoom))), Image.Resampling.LANCZOS)
    if zoom <= 1.0:
        canvas = background_for(image, rng)
        max_x = max(0, width - resized.width)
        max_y = max(0, height - resized.height)
        center_x = max_x // 2
        center_y = max_y // 2
        x = int(np.clip(center_x + rng.uniform(-translation, translation) * width, 0, max_x))
        y = int(np.clip(center_y + rng.uniform(-translation, translation) * height, 0, max_y))
        canvas.paste(resized, (x, y))
        return canvas, {"zoom": zoom, "offset_x": x, "offset_y": y, "distance": "far" if zoom < 0.72 else "medium"}
    max_x = resized.width - width
    max_y = resized.height - height
    x = int(np.clip(max_x / 2 + rng.uniform(-0.03, 0.03) * width, 0, max_x))
    y = int(np.clip(max_y / 2 + rng.uniform(-0.03, 0.03) * height, 0, max_y))
    return resized.crop((x, y, x + width, y + height)), {"zoom": zoom, "offset_x": -x, "offset_y": -y, "distance": "close"}


def perspective(image: Image.Image, rng: random.Random, intensity: float) -> Image.Image:
    width, height = image.size
    dx, dy = width * intensity, height * intensity
    quad = (
        rng.uniform(0, dx), rng.uniform(0, dy),
        width - rng.uniform(0, dx), rng.uniform(0, dy),
        width - rng.uniform(0, dx), height - rng.uniform(0, dy),
        rng.uniform(0, dx), height - rng.uniform(0, dy),
    )
    return image.transform(image.size, Image.Transform.QUAD, quad, Image.Resampling.BICUBIC, fillcolor=median_color(image))


def lighting(image: Image.Image, rng: random.Random, args: argparse.Namespace) -> tuple[Image.Image, dict]:
    brightness = rng.uniform(*args.brightness_range)
    contrast = rng.uniform(*args.contrast_range)
    saturation = rng.uniform(*args.saturation_range)
    result = ImageEnhance.Brightness(image).enhance(brightness)
    result = ImageEnhance.Contrast(result).enhance(contrast)
    result = ImageEnhance.Color(result).enhance(saturation)

    # Color-temperature shift, bounded to keep bin/trash colors recognizable.
    temperature = rng.uniform(-0.16, 0.16)
    array = np.asarray(result, dtype=np.float32)
    array[..., 0] *= 1.0 + temperature
    array[..., 2] *= 1.0 - temperature
    result = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), "RGB")

    if rng.random() < 0.35:
        mask = Image.new("L", result.size, 0)
        draw = ImageDraw.Draw(mask)
        width, height = result.size
        side = rng.choice(("left", "right", "top", "bottom"))
        if side in ("left", "right"):
            x = rng.uniform(0.18, 0.46) * width
            points = [(0, 0), (x, 0), (x * rng.uniform(0.6, 1.4), height), (0, height)]
            if side == "right":
                points = [(width - px, py) for px, py in points]
        else:
            y = rng.uniform(0.18, 0.42) * height
            points = [(0, 0), (width, 0), (width, y * rng.uniform(0.6, 1.4)), (0, y)]
            if side == "bottom":
                points = [(px, height - py) for px, py in points]
        draw.polygon(points, fill=rng.randint(35, 95))
        mask = mask.filter(ImageFilter.GaussianBlur(radius=max(width, height) / 18))
        result = Image.composite(ImageEnhance.Brightness(result).enhance(rng.uniform(0.45, 0.76)), result, mask)

    if rng.random() < 0.16:
        reflection = Image.new("RGBA", result.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(reflection)
        width, height = result.size
        x = rng.randint(-width // 4, width)
        draw.ellipse((x, -height // 2, x + width // 2, height * 3 // 2), fill=(255, 255, 255, rng.randint(12, 35)))
        reflection = reflection.filter(ImageFilter.GaussianBlur(radius=max(width, height) / 20))
        result = Image.alpha_composite(result.convert("RGBA"), reflection).convert("RGB")
    return result, {"brightness": brightness, "contrast": contrast, "saturation": saturation, "temperature": temperature}


def degrade(image: Image.Image, rng: random.Random, args: argparse.Namespace) -> tuple[Image.Image, dict]:
    blur = 0.0
    if rng.random() < 0.30:
        blur = rng.uniform(0.15, args.blur_intensity)
        image = image.filter(ImageFilter.GaussianBlur(blur))
    if rng.random() < 0.18:
        factor = rng.uniform(0.35, 0.72)
        small = image.resize((max(48, round(image.width * factor)), max(48, round(image.height * factor))), Image.Resampling.BILINEAR)
        image = small.resize(image.size, Image.Resampling.BILINEAR)
    noise_sigma = 0.0
    if rng.random() < 0.32:
        noise_sigma = rng.uniform(1.5, args.noise_intensity)
        generator = np.random.default_rng(rng.randrange(2**32))
        array = np.asarray(image, dtype=np.float32)
        image = Image.fromarray(np.clip(array + generator.normal(0, noise_sigma, array.shape), 0, 255).astype(np.uint8), "RGB")
    jpeg_quality = 95
    if rng.random() < 0.38:
        jpeg_quality = rng.randint(45, 88)
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=jpeg_quality)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            image = decoded.convert("RGB")
    return image, {"blur": blur, "noise_sigma": noise_sigma, "jpeg_quality": jpeg_quality}


def edge_obstruction(image: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() >= 0.12:
        return image
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    width, height = image.size
    edge = rng.choice(("left", "right", "top", "bottom"))
    color = tuple(rng.randint(35, 220) for _ in range(3)) + (rng.randint(90, 180),)
    if edge == "left":
        box = (0, rng.randint(0, height // 2), rng.randint(width // 20, width // 10), rng.randint(height // 2, height))
    elif edge == "right":
        box = (width - rng.randint(width // 20, width // 10), rng.randint(0, height // 2), width, rng.randint(height // 2, height))
    elif edge == "top":
        box = (rng.randint(0, width // 2), 0, rng.randint(width // 2, width), rng.randint(height // 20, height // 10))
    else:
        box = (rng.randint(0, width // 2), height - rng.randint(height // 20, height // 10), rng.randint(width // 2, width), height)
    draw.rectangle(box, fill=color)
    return overlay


def output_resize(image: Image.Image, max_side: int) -> Image.Image:
    if max(image.size) != max_side:
        scale = max_side / max(image.size)
        image = image.resize((max(32, round(image.width * scale)), max(32, round(image.height * scale))), Image.Resampling.LANCZOS)
    return image


def augment(image: Image.Image, rng: random.Random, args: argparse.Namespace) -> tuple[Image.Image, dict]:
    result, metadata = distance_position(image, rng, args.zoom_range, args.translation)
    if rng.random() <= args.augmentation_probability:
        angle = rng.uniform(*args.rotation_range)
        result = result.rotate(angle, Image.Resampling.BICUBIC, expand=False, fillcolor=median_color(result))
        metadata["rotation"] = angle
        if rng.random() < 0.55:
            result = perspective(result, rng, args.perspective_intensity)
            metadata["perspective"] = True
        else:
            metadata["perspective"] = False
        if rng.random() < 0.50:
            result = ImageOps.mirror(result)
            metadata["horizontal_flip"] = True
        else:
            metadata["horizontal_flip"] = False
        result, light_metadata = lighting(result, rng, args)
        metadata.update(light_metadata)
        result = edge_obstruction(result, rng)
        result, degradation_metadata = degrade(result, rng, args)
        metadata.update(degradation_metadata)
    result = output_resize(result, args.output_resolution)
    return result, metadata


def read_review_list(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {
        line.strip().replace("\\", "/")
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def main() -> None:
    args = parse_args()
    review_paths = read_review_list(args.manual_review_list)
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)
    rng = random.Random(args.seed)
    manifest = []
    counts = Counter()

    training_sources: dict[str, list[Path]] = {class_name: [] for class_name in CLASSES}
    for split in ("train", "validation", "test"):
        for class_name in CLASSES:
            source_folder = args.source_dir / split / class_name
            for source in sorted(source_folder.glob("original_*")):
                relative = source.relative_to(args.source_dir).as_posix()
                if relative in review_paths:
                    destination = args.output_dir / "manual-review" / class_name / f"{split}_{source.name}"
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    manifest.append({"split": "manual-review", "class": class_name, "kind": "original", "path": destination.relative_to(args.output_dir).as_posix(), "source": relative, "metadata": "strict-fill-definition ambiguity"})
                    counts[("manual-review", class_name, "original")] += 1
                    continue
                if split == "train":
                    destination = args.output_dir / "train" / "original" / class_name / source.name
                    training_sources[class_name].append(source)
                else:
                    destination = args.output_dir / split / class_name / source.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                manifest.append({"split": split, "class": class_name, "kind": "original", "path": destination.relative_to(args.output_dir).as_posix(), "source": relative, "metadata": ""})
                counts[(split, class_name, "original")] += 1

    for class_name in CLASSES:
        sources = training_sources[class_name]
        if not sources:
            raise RuntimeError(f"No usable training sources for {class_name}")
        synthetic_needed = max(0, args.target_per_class - len(sources))
        class_rng = random.Random(args.seed + CLASSES.index(class_name) * 100_003)
        for index in range(1, synthetic_needed + 1):
            source = sources[(index - 1) % len(sources)]
            with Image.open(source) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
            generated, metadata = augment(image, class_rng, args)
            destination = args.output_dir / "train" / "synthetic" / class_name / f"synthetic_{index:04d}.jpg"
            destination.parent.mkdir(parents=True, exist_ok=True)
            generated.save(destination, "JPEG", quality=91, optimize=True)
            manifest.append({"split": "train", "class": class_name, "kind": "synthetic", "path": destination.relative_to(args.output_dir).as_posix(), "source": source.relative_to(args.source_dir).as_posix(), "metadata": json.dumps(metadata, separators=(",", ":"))})
            counts[("train", class_name, "synthetic")] += 1

    with (args.output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "class", "kind", "path", "source", "metadata"])
        writer.writeheader()
        writer.writerows(manifest)

    report = {
        "class_definitions": {"empty": "exactly 0%", "half-full": "1-74%", "full": "75-100%"},
        "seed": args.seed,
        "configuration": {
            "target_per_class": args.target_per_class,
            "augmentation_probability": args.augmentation_probability,
            "zoom_range": args.zoom_range,
            "rotation_range": args.rotation_range,
            "perspective_intensity": args.perspective_intensity,
            "brightness_range": args.brightness_range,
            "contrast_range": args.contrast_range,
            "saturation_range": args.saturation_range,
            "blur_intensity": args.blur_intensity,
            "noise_intensity": args.noise_intensity,
            "translation": args.translation,
            "output_resolution": args.output_resolution,
        },
        "counts": [
            {"split": split, "class": class_name, "kind": kind, "count": count}
            for (split, class_name, kind), count in sorted(counts.items())
        ],
        "manual_review_paths": sorted(review_paths),
        "leakage_policy": "Synthetic images are generated only from training originals and written only to train/synthetic.",
    }
    (args.output_dir / "generation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
