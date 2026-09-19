from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


COLORS = {"train": "#2471A3", "validation": "#C0392B", "baseline": "#7F8C8D", "improved": "#1E8449"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create training and evaluation graphics without optional plotting dependencies.")
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--robustness-report", type=Path, required=True)
    parser.add_argument("--examples-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def line_chart(report: dict, destination: Path) -> None:
    history = report["history"]
    image = Image.new("RGB", (1100, 650), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 90, 70, 1040, 560
    draw.rectangle((left, top, right, bottom), outline="#333333", width=2)
    for tick in range(6):
        value = tick / 5
        y = bottom - value * (bottom - top)
        draw.line((left, y, right, y), fill="#E5E7E9", width=1)
        draw.text((35, y - 8), f"{value:.1f}", fill="#333333")
    draw.text((left, 24), "Robust v2 training accuracy", fill="#111111")
    for series_name, key in (("train", "train"), ("validation", "validation")):
        points = []
        for index, record in enumerate(history):
            x = left + index * (right - left) / max(1, len(history) - 1)
            y = bottom - record[key]["accuracy"] * (bottom - top)
            points.append((x, y))
        if len(points) > 1:
            draw.line(points, fill=COLORS[series_name], width=4)
        for point in points:
            draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=COLORS[series_name])
        draw.text((right - 180, 25 + (0 if series_name == "train" else 24)), series_name, fill=COLORS[series_name])
    for index, record in enumerate(history):
        x = left + index * (right - left) / max(1, len(history) - 1)
        draw.text((x - 4, bottom + 12), str(record["epoch"]), fill="#333333")
    draw.text((500, 605), "Epoch", fill="#333333")
    image.save(destination)


def confusion_matrix(report: dict, destination: Path) -> None:
    matrix = report["test_metrics"]["confusion_matrix"]
    labels = report["test_metrics"]["confusion_matrix_labels"]
    image = Image.new("RGB", (760, 700), "white")
    draw = ImageDraw.Draw(image)
    draw.text((170, 30), "Improved-model confusion matrix", fill="#111111")
    origin_x, origin_y, cell = 190, 130, 145
    maximum = max(max(row) for row in matrix)
    for row in range(3):
        draw.text((40, origin_y + row * cell + 55), f"Actual {labels[row]}", fill="#222222")
        draw.text((origin_x + row * cell + 35, 92), f"Pred {labels[row]}", fill="#222222")
        for column in range(3):
            value = matrix[row][column]
            intensity = int(235 - 150 * value / max(1, maximum))
            fill = (intensity, 245, intensity) if row == column else (255, intensity, intensity)
            x0, y0 = origin_x + column * cell, origin_y + row * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=fill, outline="white", width=4)
            draw.text((x0 + 65, y0 + 60), str(value), fill="#111111")
    image.save(destination)


def robustness_chart(report: dict, destination: Path) -> None:
    conditions = list(report["conditions"])
    image = Image.new("RGB", (1250, 720), "white")
    draw = ImageDraw.Draw(image)
    left, top, right, bottom = 110, 80, 1180, 600
    draw.text((left, 25), "Accuracy by controlled capture condition", fill="#111111")
    draw.rectangle((left, top, right, bottom), outline="#333333", width=2)
    for tick in range(6):
        value = tick / 5
        y = bottom - value * (bottom - top)
        draw.line((left, y, right, y), fill="#E5E7E9")
        draw.text((55, y - 8), f"{value:.1f}", fill="#333333")
    group_width = (right - left) / len(conditions)
    bar_width = group_width * 0.32
    for index, condition in enumerate(conditions):
        center = left + (index + 0.5) * group_width
        for offset, model_name in ((-bar_width, "baseline"), (0, "improved")):
            value = report[model_name]["conditions"][condition]["accuracy"]
            x0 = center + offset
            y0 = bottom - value * (bottom - top)
            draw.rectangle((x0, y0, x0 + bar_width, bottom), fill=COLORS[model_name])
        label = condition.replace("background_position", "background").replace("low_resolution", "low-res").replace("blur_noise", "blur/noise")
        draw.text((center - group_width * 0.40, bottom + 14), label, fill="#333333")
    draw.rectangle((left, 642, left + 25, 667), fill=COLORS["baseline"])
    draw.text((left + 34, 644), "baseline", fill="#333333")
    draw.rectangle((left + 160, 642, left + 185, 667), fill=COLORS["improved"])
    draw.text((left + 194, 644), "improved (Uncertain counts as incorrect)", fill="#333333")
    image.save(destination)


def example_montage(examples_dir: Path, destination: Path) -> None:
    paths = sorted(examples_dir.glob("*.jpg"))
    if not paths:
        return
    thumb_size = (280, 220)
    columns = 4
    rows = (len(paths) + columns - 1) // columns
    image = Image.new("RGB", (columns * 300, rows * 270), "white")
    draw = ImageDraw.Draw(image)
    for index, path in enumerate(paths):
        with Image.open(path) as opened:
            thumb = ImageOps.contain(opened.convert("RGB"), thumb_size)
        x, y = (index % columns) * 300, (index // columns) * 270
        image.paste(thumb, (x + (300 - thumb.width) // 2, y + 4))
        draw.text((x + 8, y + 232), path.stem[:42], fill="#222222")
    image.save(destination, quality=92)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training = json.loads(args.training_report.read_text(encoding="utf-8"))
    robustness = json.loads(args.robustness_report.read_text(encoding="utf-8"))
    line_chart(training, args.output_dir / "training_history.png")
    confusion_matrix(training, args.output_dir / "confusion_matrix.png")
    robustness_chart(robustness, args.output_dir / "robustness_comparison.png")
    example_montage(args.examples_dir, args.output_dir / "prediction_examples.jpg")
    print(str(args.output_dir.resolve()))


if __name__ == "__main__":
    main()
