from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math


# The middle physical lid can be labelled either PAPER or GENERAL.  Both names
# route to PCA9685 channel 1; a trained model must contain only one of them.
ROUTABLE_CLASSES = frozenset({"plastic", "paper", "general", "metal"})


@dataclass(frozen=True)
class Decision:
    """Result of applying safety rules to model probabilities."""

    label: str
    confidence: float
    margin: float
    route: str | None
    reason: str

    @property
    def accepted(self) -> bool:
        return self.route is not None


def decide_route(
    probabilities: dict[str, float],
    *,
    confidence_threshold: float = 0.75,
    margin_threshold: float = 0.15,
    supported_probability: float | None = None,
    validity_threshold: float = 0.5,
    prototype_distance: float | None = None,
    prototype_threshold: float | None = None,
    class_confidence_thresholds: Mapping[str, float] | None = None,
    metal_detected: bool | None = None,
    object_present: bool | None = None,
    hand_present: bool | None = None,
    weight_value: float | None = None,
    weight_range: tuple[float, float] | None = None,
) -> Decision:
    """Choose a bin conservatively.

    `other` is deliberately not routable. A metal sensor disagreement also
    keeps every lid closed so the user can confirm the item.
    """
    if not probabilities:
        return Decision("unknown", 0.0, 0.0, None, "model returned no scores")

    for raw_label, score in probabilities.items():
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            return Decision(
                str(raw_label).strip().lower() or "unknown",
                0.0,
                0.0,
                None,
                "model returned an invalid score",
            )

    ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    label, confidence = ordered[0]
    runner_up = ordered[1][1] if len(ordered) > 1 else 0.0
    margin = confidence - runner_up
    label = label.strip().lower()

    if label not in ROUTABLE_CLASSES:
        return Decision(label, confidence, margin, None, "unknown/other item")
    if object_present is False:
        return Decision(label, confidence, margin, None, "presence sensor sees no item")
    if hand_present is True:
        return Decision(label, confidence, margin, None, "hand detected; remove hand before opening a lid")
    if weight_value is not None or weight_range is not None:
        if weight_value is None or weight_range is None:
            return Decision(label, confidence, margin, None, "incomplete weight-sensor limits")
        lower, upper = weight_range
        if (
            not math.isfinite(weight_value)
            or not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower > upper
        ):
            return Decision(label, confidence, margin, None, "invalid weight-sensor reading")
        if not lower <= weight_value <= upper:
            return Decision(label, confidence, margin, None, "item weight is outside safe limits")
    if supported_probability is not None:
        if not math.isfinite(validity_threshold) or not 0.0 <= validity_threshold <= 1.0:
            return Decision(label, confidence, margin, None, "invalid validity threshold")
        if not math.isfinite(supported_probability) or not 0.0 <= supported_probability <= 1.0:
            return Decision(label, confidence, margin, None, "validity head returned an invalid score")
        if supported_probability < validity_threshold:
            return Decision(label, confidence, margin, None, "validity head rejected the item")
    if prototype_distance is not None or prototype_threshold is not None:
        if prototype_distance is None or prototype_threshold is None:
            return Decision(label, confidence, margin, None, "incomplete feature-distance metadata")
        if (
            not math.isfinite(prototype_distance)
            or not math.isfinite(prototype_threshold)
            or prototype_distance < 0.0
            or prototype_threshold < 0.0
        ):
            return Decision(label, confidence, margin, None, "invalid feature-distance score")
        if prototype_distance > prototype_threshold:
            return Decision(label, confidence, margin, None, "outside known-class feature space")
    required_confidence = confidence_threshold
    if class_confidence_thresholds is not None and label in class_confidence_thresholds:
        class_threshold = float(class_confidence_thresholds[label])
        if not math.isfinite(class_threshold) or not 0.0 <= class_threshold <= 1.0:
            return Decision(label, confidence, margin, None, "invalid per-class threshold")
        required_confidence = max(required_confidence, class_threshold)
    if confidence < required_confidence:
        return Decision(
            label,
            confidence,
            margin,
            None,
            f"confidence below {label} threshold ({required_confidence:.0%})",
        )
    if margin < margin_threshold:
        return Decision(label, confidence, margin, None, "top classes are too close")
    if metal_detected is True and label != "metal":
        return Decision(label, confidence, margin, None, "metal sensor disagrees with camera")
    if metal_detected is False and label == "metal":
        return Decision(label, confidence, margin, None, "camera says metal but sensor does not")

    return Decision(label, confidence, margin, label, "accepted")
