from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def calibrate_validity_threshold(
    probabilities: Iterable[float],
    supported_targets: Iterable[int | bool],
) -> tuple[float, float]:
    """Choose the threshold with the best supported/reject balanced accuracy."""

    scores = np.asarray(tuple(probabilities), dtype=np.float64)
    targets = np.asarray(tuple(supported_targets), dtype=np.int8)
    if scores.ndim != 1 or targets.ndim != 1 or len(scores) != len(targets) or not len(scores):
        raise ValueError("probabilities and targets must be non-empty one-dimensional sequences")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("validity probabilities must be finite and between 0 and 1")
    if np.any((targets != 0) & (targets != 1)):
        raise ValueError("supported targets must contain only 0 and 1")
    if len(np.unique(targets)) != 2:
        raise ValueError("validity calibration requires both supported and rejected examples")

    unique = np.unique(scores)
    candidates = np.unique(
        np.concatenate(([0.0], (unique[:-1] + unique[1:]) / 2.0, unique, [1.0]))
    )
    positives = targets == 1
    negatives = ~positives
    best_threshold = 0.5
    best_score = -1.0
    for threshold in candidates:
        predicted = scores >= threshold
        sensitivity = float(np.mean(predicted[positives]))
        specificity = float(np.mean(~predicted[negatives]))
        balanced = (sensitivity + specificity) / 2.0
        # Prefer a more conservative (higher) threshold when scores tie.
        if balanced > best_score or (balanced == best_score and threshold > best_threshold):
            best_threshold = float(threshold)
            best_score = balanced
    return best_threshold, best_score


def calibrate_class_confidence_thresholds(
    probabilities: Iterable[Iterable[float]],
    targets: Iterable[int],
    *,
    minimum_precision: float = 0.90,
    floor: float = 0.50,
) -> tuple[list[float], list[dict[str, float | int]]]:
    """Find a conservative confidence boundary for each predicted class.

    Targets outside ``0..class_count-1`` are treated as unknown/incorrect. If
    no threshold reaches the requested precision, that route is disabled with
    a threshold of 1.0 and the calibration report makes the reason visible.
    """

    scores = np.asarray(tuple(tuple(row) for row in probabilities), dtype=np.float64)
    expected = np.asarray(tuple(targets), dtype=np.int64)
    if scores.ndim != 2 or not scores.shape[0] or scores.shape[1] < 2:
        raise ValueError("probabilities must be a non-empty two-dimensional matrix")
    if expected.shape != (scores.shape[0],):
        raise ValueError("targets length must match the probability rows")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("class probabilities must be finite and between 0 and 1")
    if not 0.0 < minimum_precision <= 1.0 or not 0.0 <= floor <= 1.0:
        raise ValueError("minimum precision and floor must be in 0..1")

    winners = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    thresholds: list[float] = []
    reports: list[dict[str, float | int]] = []
    for class_index in range(scores.shape[1]):
        predicted_here = winners == class_index
        candidates = np.unique(
            np.concatenate(([floor], confidences[predicted_here & (confidences >= floor)], [1.0]))
        )
        chosen = 1.0
        chosen_precision = 0.0
        chosen_count = 0
        for threshold in candidates:
            accepted = predicted_here & (confidences >= threshold)
            count = int(accepted.sum())
            if not count:
                continue
            precision = float(np.mean(expected[accepted] == class_index))
            if precision >= minimum_precision:
                chosen = float(threshold)
                chosen_precision = precision
                chosen_count = count
                break
            if threshold == 1.0:
                chosen_precision = precision
                chosen_count = count
        thresholds.append(chosen)
        reports.append(
            {
                "threshold": chosen,
                "precision": chosen_precision,
                "accepted_validation_samples": chosen_count,
            }
        )
    return thresholds, reports


def confusion_metrics(confusion: Iterable[Iterable[int]]) -> dict[str, object]:
    """Return per-class and macro metrics from a square confusion matrix."""

    matrix = np.asarray(tuple(tuple(row) for row in confusion), dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("confusion matrix must be non-empty and square")
    if np.any(matrix < 0):
        raise ValueError("confusion matrix counts cannot be negative")

    per_class: list[dict[str, float]] = []
    for index in range(matrix.shape[0]):
        true_positive = int(matrix[index, index])
        actual = int(matrix[index, :].sum())
        predicted = int(matrix[:, index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / actual if actual else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"precision": precision, "recall": recall, "f1": f1})

    total = int(matrix.sum())
    return {
        "accuracy": float(np.trace(matrix) / total) if total else 0.0,
        "macro_precision": float(np.mean([item["precision"] for item in per_class])),
        "macro_recall": float(np.mean([item["recall"] for item in per_class])),
        "macro_f1": float(np.mean([item["f1"] for item in per_class])),
        "per_class": per_class,
    }
