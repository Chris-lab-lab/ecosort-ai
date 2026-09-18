"""Lightweight mask cleanup suitable for the i.MX93 live preview.

The neural network intentionally emits a small 56x56 probability mask.  Once
that mask is enlarged for display, isolated errors and jagged edges become very
visible.  These helpers clean the display mask using ordinary OpenCV operations;
they do not add another neural network or require a GPU.
"""

from __future__ import annotations

import numpy as np


def smooth_mask_probability(
    current: np.ndarray,
    previous: np.ndarray | None,
    *,
    current_weight: float = 0.72,
) -> np.ndarray:
    """Blend the current probability map with the preceding camera frame."""

    probability = np.asarray(current, dtype=np.float32)
    if previous is None or previous.shape != probability.shape:
        return probability.copy()
    weight = float(np.clip(current_weight, 0.0, 1.0))
    return probability * weight + np.asarray(previous, dtype=np.float32) * (
        1.0 - weight
    )


def refine_binary_mask(
    cv2: object,
    probability: np.ndarray,
    *,
    threshold: float,
    minimum_area_ratio: float = 0.004,
    center_ratio: float = 0.42,
) -> np.ndarray:
    """Return one clean foreground object, preferring the presentation center.

    The cleanup smooths pixel-scale noise, closes small cracks, removes tiny
    islands, and keeps the connected component that overlaps the center most.
    If no component reaches the center, the largest plausible component wins.
    """

    values = np.asarray(probability, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a 2-D mask probability map, got {values.shape}")
    height, width = values.shape
    if height == 0 or width == 0:
        return np.zeros((height, width), dtype=bool)

    # Blur probabilities before thresholding so upscaled 56x56 cells do not
    # produce staircase contours. Kernel sizes stay small for board-side CPU.
    short_side = min(height, width)
    blur_size = max(3, round(short_side * 0.012))
    if blur_size % 2 == 0:
        blur_size += 1
    blurred = cv2.GaussianBlur(values, (blur_size, blur_size), 0)
    binary = (blurred >= threshold).astype(np.uint8)

    close_size = max(3, round(short_side * 0.018))
    if close_size % 2 == 0:
        close_size += 1
    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_size, close_size)
    )
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return np.zeros((height, width), dtype=bool)

    minimum_area = max(12, round(height * width * minimum_area_ratio))
    center_width = max(1, round(width * center_ratio))
    center_height = max(1, round(height * center_ratio))
    center_left = (width - center_width) // 2
    center_top = (height - center_height) // 2
    center_labels = labels[
        center_top : center_top + center_height,
        center_left : center_left + center_width,
    ]

    candidates: list[tuple[int, int, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        center_overlap = int(np.count_nonzero(center_labels == label))
        candidates.append((center_overlap, area, label))
    if not candidates:
        return np.zeros((height, width), dtype=bool)

    # Center overlap is the main presentation-area cue; area breaks ties.
    _, _, selected_label = max(candidates)
    selected = (labels == selected_label).astype(np.uint8)

    # Close small internal gaps without filling legitimate large openings such
    # as a cup handle.
    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, close_kernel)
    return selected.astype(bool)
