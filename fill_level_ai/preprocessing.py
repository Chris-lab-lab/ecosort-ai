from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageOps


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class QualityReport:
    blur_score: float
    mean_luminance: float
    contrast: float
    roi_area_ratio: float
    active_edge_fraction: float
    warnings: list[str]
    severe: bool


def median_color(image: Image.Image) -> tuple[int, int, int]:
    sample = np.asarray(image.convert("RGB").resize((32, 32)), dtype=np.uint8).reshape(-1, 3)
    return tuple(int(value) for value in np.median(sample, axis=0))


def letterbox(image: Image.Image, size: int = 224, scale: float = 1.0) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    available = max(32, round(size * min(max(scale, 0.35), 1.0)))
    ratio = min(available / image.width, available / image.height)
    resized = image.resize((max(1, round(image.width * ratio)), max(1, round(image.height * ratio))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), median_color(image))
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return canvas


def _analysis_arrays(image: Image.Image, size: int = 192):
    rgb = np.asarray(image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
    gray = rgb.mean(axis=2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    gradient = np.sqrt(gx * gx + gy * gy)
    return rgb, gray, gradient


def locate_bin_roi(image: Image.Image) -> tuple[tuple[int, int, int, int], float, float]:
    """Heuristic bin ROI proposal; conservative expansion keeps the complete interior in view."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    _, gray, gradient = _analysis_arrays(image)
    threshold = max(8.0, float(np.quantile(gradient, 0.72)))
    active = gradient >= threshold
    active_fraction = float(active.mean())
    ys, xs = np.where(active)
    width, height = image.size
    if len(xs) < 24:
        return (0, 0, width, height), 1.0, active_fraction

    x0, x1 = float(xs.min()) / 192, float(xs.max() + 1) / 192
    y0, y1 = float(ys.min()) / 192, float(ys.max() + 1) / 192
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    roi_width = max(0.62, (x1 - x0) * 1.45)
    roi_height = max(0.62, (y1 - y0) * 1.45)
    x0 = max(0.0, center_x - roi_width / 2)
    x1 = min(1.0, center_x + roi_width / 2)
    y0 = max(0.0, center_y - roi_height / 2)
    y1 = min(1.0, center_y + roi_height / 2)
    box = (round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height))
    area_ratio = ((box[2] - box[0]) * (box[3] - box[1])) / max(1, width * height)
    if area_ratio < 0.32:
        return (0, 0, width, height), 1.0, active_fraction
    return box, float(area_ratio), active_fraction


def quality_report(image: Image.Image) -> QualityReport:
    image = ImageOps.exif_transpose(image).convert("RGB")
    _, gray, gradient = _analysis_arrays(image, 224)
    laplacian = -4 * gray
    laplacian[1:, :] += gray[:-1, :]
    laplacian[:-1, :] += gray[1:, :]
    laplacian[:, 1:] += gray[:, :-1]
    laplacian[:, :-1] += gray[:, 1:]
    blur = float(laplacian[2:-2, 2:-2].var())
    mean = float(gray.mean())
    contrast = float(gray.std())
    _, area_ratio, active_fraction = locate_bin_roi(image)
    warnings = []
    severe = False
    if min(image.size) < 120:
        warnings.append("very low resolution")
        severe = True
    elif min(image.size) < 256:
        warnings.append("low resolution")
    if blur < 2.0:
        warnings.append("image is severely blurred")
        severe = True
    elif blur < 7.0:
        warnings.append("image is blurry")
    if mean < 16:
        warnings.append("image is too dark")
        severe = True
    elif mean < 30:
        warnings.append("image is dark")
    if mean > 247:
        warnings.append("image is overexposed")
        severe = True
    elif mean > 235:
        warnings.append("image is very bright")
    if contrast < 8:
        warnings.append("image has insufficient contrast")
        severe = True
    if active_fraction < 0.025:
        warnings.append("bin or trash structure is not clearly visible")
        severe = True
    return QualityReport(blur, mean, contrast, area_ratio, active_fraction, warnings, severe)


def make_inference_views(image: Image.Image, size: int = 224) -> tuple[list[Image.Image], QualityReport]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    report = quality_report(image)
    box, area_ratio, _ = locate_bin_roi(image)
    roi = image.crop(box)
    views = [letterbox(image, size=size), letterbox(image, size=size, scale=0.78)]
    if area_ratio < 0.97:
        views.append(letterbox(roi, size=size))
    else:
        width, height = image.size
        margin_x, margin_y = round(width * 0.055), round(height * 0.055)
        views.append(letterbox(image.crop((margin_x, margin_y, width - margin_x, height - margin_y)), size=size))
    return views, report


def to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.transpose(array, (2, 0, 1))).float()
