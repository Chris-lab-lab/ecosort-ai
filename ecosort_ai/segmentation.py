"""Pure NumPy/Pillow helpers for offline segmentation-teacher preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class MaskDecision:
    status: str
    mask: np.ndarray | None
    bbox_xyxy: tuple[int, int, int, int] | None
    candidate_count: int
    distinct_object_count: int


def _mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return intersection / union if union else 0.0


def _containment(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int(np.logical_and(left, right).sum())
    smaller = min(int(left.sum()), int(right.sum()))
    return intersection / smaller if smaller else 0.0


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    rows, columns = np.nonzero(mask)
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()) + 1,
        int(rows.max()) + 1,
    )


def select_primary_mask(
    raw_masks: list[dict[str, Any]],
    image_shape: tuple[int, int] | tuple[int, int, int],
    *,
    min_area_ratio: float = 0.04,
    max_area_ratio: float = 0.85,
    multiple_object_ratio: float = 0.30,
    require_center: bool = True,
    reject_multiple_objects: bool = True,
) -> MaskDecision:
    """Select a central object and reject scenes with a second distinct object.

    EfficientViT-SAM's automatic generator can return nested masks for an
    object and its parts. High containment is therefore treated as one object,
    while a large, disjoint mask is treated as another object.
    """

    height, width = int(image_shape[0]), int(image_shape[1])
    image_area = height * width
    if image_area <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0.0 < min_area_ratio < max_area_ratio <= 1.0:
        raise ValueError("mask area limits are invalid")

    center_top, center_bottom = int(height * 0.35), max(1, int(height * 0.65))
    center_left, center_right = int(width * 0.35), max(1, int(width * 0.65))
    candidates: list[tuple[float, bool, np.ndarray]] = []
    for item in raw_masks:
        mask = np.asarray(item.get("segmentation"), dtype=bool)
        if mask.shape != (height, width):
            continue
        area_ratio = float(mask.sum()) / image_area
        if not min_area_ratio <= area_ratio <= max_area_ratio:
            continue
        center_hit = bool(
            mask[center_top:center_bottom, center_left:center_right].any()
        )
        stability = float(item.get("stability_score", 0.0))
        predicted_iou = float(item.get("predicted_iou", 0.0))
        central_bonus = 2.0 if center_hit else 0.0
        size_score = 1.0 - abs(area_ratio - 0.30)
        candidates.append(
            (central_bonus + size_score + stability + predicted_iou, center_hit, mask)
        )

    if not candidates:
        return MaskDecision("no_object", None, None, 0, 0)
    candidates.sort(key=lambda item: item[0], reverse=True)
    central_candidates = [item for item in candidates if item[1]]
    if require_center and not central_candidates:
        return MaskDecision("no_object", None, None, len(candidates), 0)
    primary = (central_candidates[0] if require_center else candidates[0])[2]
    if not reject_multiple_objects:
        return MaskDecision("accepted", primary, _bbox(primary), len(candidates), 1)

    primary_area = int(primary.sum())
    distinct = [primary]
    for _, _, candidate in candidates:
        if candidate is primary:
            continue
        if int(candidate.sum()) < primary_area * multiple_object_ratio:
            continue
        if (
            _containment(primary, candidate) >= 0.80
            or _mask_iou(primary, candidate) >= 0.55
        ):
            continue
        if any(_containment(existing, candidate) >= 0.80 for existing in distinct):
            continue
        distinct.append(candidate)

    if len(distinct) > 1:
        return MaskDecision(
            "multiple_objects", None, None, len(candidates), len(distinct)
        )
    return MaskDecision("accepted", primary, _bbox(primary), len(candidates), 1)


def synthetic_background(
    size: int, rng: np.random.Generator, variant: int
) -> np.ndarray:
    """Create class-independent neutral, gradient, or textured RGB backgrounds."""

    if size < 1:
        raise ValueError("background size must be positive")
    if variant % 3 == 0:
        level = int(rng.integers(45, 215))
        return np.full((size, size, 3), level, dtype=np.uint8)
    if variant % 3 == 1:
        top = rng.integers(30, 225, size=3)
        bottom = rng.integers(30, 225, size=3)
        mix = np.linspace(0.0, 1.0, size, dtype=np.float32)[:, None, None]
        return (
            np.clip(top * (1.0 - mix) + bottom * mix, 0, 255)
            .astype(np.uint8)
            .repeat(size, axis=1)
        )
    base = rng.integers(35, 220, size=3)
    noise = rng.normal(0.0, 24.0, size=(size, size, 3))
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def composite_masked_object(
    rgb: np.ndarray,
    mask: np.ndarray,
    *,
    output_size: int,
    rng: np.random.Generator,
    variant: int,
    padding: float = 0.08,
) -> np.ndarray:
    """Crop a segmented object and place it on a synthetic background."""

    if rgb.ndim != 3 or rgb.shape[2] != 3 or mask.shape != rgb.shape[:2]:
        raise ValueError("RGB image and mask shapes do not match")
    left, top, right, bottom = _bbox(mask)
    pad_x = int((right - left) * padding)
    pad_y = int((bottom - top) * padding)
    left, top = max(0, left - pad_x), max(0, top - pad_y)
    right, bottom = min(rgb.shape[1], right + pad_x), min(rgb.shape[0], bottom + pad_y)
    object_crop = Image.fromarray(rgb[top:bottom, left:right])
    alpha_crop = Image.fromarray((mask[top:bottom, left:right] * 255).astype(np.uint8))

    scale = float(rng.uniform(0.62, 0.88))
    target_long_side = max(1, int(output_size * scale))
    resize_scale = target_long_side / max(object_crop.size)
    target_size = (
        max(1, int(object_crop.width * resize_scale)),
        max(1, int(object_crop.height * resize_scale)),
    )
    object_crop = object_crop.resize(target_size, Image.Resampling.LANCZOS)
    alpha_crop = alpha_crop.resize(target_size, Image.Resampling.LANCZOS)

    background = synthetic_background(output_size, rng, variant)
    canvas = Image.fromarray(background)
    max_x = max(0, output_size - target_size[0])
    max_y = max(0, output_size - target_size[1])
    center_x, center_y = max_x // 2, max_y // 2
    jitter_x = int(rng.integers(-max(1, output_size // 14), max(2, output_size // 14)))
    jitter_y = int(rng.integers(-max(1, output_size // 14), max(2, output_size // 14)))
    position = (
        min(max(center_x + jitter_x, 0), max_x),
        min(max(center_y + jitter_y, 0), max_y),
    )
    canvas.paste(object_crop, position, alpha_crop)
    return np.asarray(canvas, dtype=np.uint8)


def make_background_only(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Remove the object while retaining the rest of the scene for a bias audit."""

    if rgb.ndim != 3 or rgb.shape[2] != 3 or mask.shape != rgb.shape[:2]:
        raise ValueError("RGB image and mask shapes do not match")
    background_pixels = rgb[~mask]
    fill = (
        np.median(background_pixels, axis=0)
        if len(background_pixels)
        else np.array([127] * 3)
    )
    result = rgb.copy()
    result[mask] = np.asarray(fill, dtype=np.uint8)
    return result
