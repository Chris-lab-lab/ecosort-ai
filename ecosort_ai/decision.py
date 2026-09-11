from __future__ import annotations

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
    metal_detected: bool | None = None,
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
    if confidence < confidence_threshold:
        return Decision(label, confidence, margin, None, "confidence below threshold")
    if margin < margin_threshold:
        return Decision(label, confidence, margin, None, "top classes are too close")
    if metal_detected is True and label != "metal":
        return Decision(label, confidence, margin, None, "metal sensor disagrees with camera")
    if metal_detected is False and label == "metal":
        return Decision(label, confidence, margin, None, "camera says metal but sensor does not")

    return Decision(label, confidence, margin, label, "accepted")
