"""Measure how well a classifier predicts after the segmented object is removed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from ecosort_ai.decision import decide_route
from ecosort_ai.model import WasteClassifier
from ecosort_ai.segmentation import make_background_only


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit material predictions made from background-only images"
    )
    parser.add_argument("--data", type=Path, default=Path("dataset_enhanced"))
    parser.add_argument("--segmented", type=Path, default=Path("dataset_segmented"))
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("teacher_outputs/background_bias.json")
    )
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--margin", type=float, default=0.15)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--examples-dir",
        type=Path,
        help="optionally save the most suspicious routed background-only inputs",
    )
    return parser.parse_args()


def main() -> None:
    args = arguments()
    data = args.data.resolve()
    segmented = args.segmented.resolve()
    manifest_path = segmented / "segmentation_manifest.jsonl"
    if not manifest_path.is_file():
        raise SystemExit(f"Segmentation manifest does not exist: {manifest_path}")
    classifier = WasteClassifier(
        args.model,
        args.labels,
        delegate=None,
        metadata_path=args.metadata,
    )
    validity_threshold = classifier.recommended_validity_threshold
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [record for record in records if record.get("status") == "accepted"]
    if args.limit is not None:
        records = records[: args.limit]

    correct = 0
    routed = 0
    confidence_sum = 0.0
    per_class: dict[str, dict[str, int]] = {}
    suspicious: list[tuple[float, str, np.ndarray]] = []
    for record in records:
        label = str(record["label"]).lower()
        source_path = data / str(record["source"])
        mask_path = segmented / str(record["mask"])
        rgb = np.asarray(Image.open(source_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L")) >= 128
        background = make_background_only(rgb, mask)
        prediction = classifier.predict_rgb(background)
        winner, confidence = prediction.top
        distance = (
            prediction.prototype_distances.get(winner)
            if prediction.prototype_distances is not None
            else None
        )
        prototype_threshold = (
            prediction.prototype_thresholds.get(winner)
            if prediction.prototype_thresholds is not None
            else None
        )
        decision = decide_route(
            prediction.scores,
            confidence_threshold=args.threshold,
            margin_threshold=args.margin,
            supported_probability=prediction.supported_probability,
            validity_threshold=validity_threshold,
            prototype_distance=distance,
            prototype_threshold=prototype_threshold,
            class_confidence_thresholds=classifier.material_confidence_thresholds,
        )
        is_correct = winner == label
        correct += int(is_correct)
        routed += int(decision.accepted)
        confidence_sum += confidence
        stats = per_class.setdefault(label, {"samples": 0, "correct": 0, "routed": 0})
        stats["samples"] += 1
        stats["correct"] += int(is_correct)
        stats["routed"] += int(decision.accepted)
        if decision.accepted:
            suspicious.append((confidence, str(record["source"]), background))

    sample_count = len(records)
    if not sample_count:
        raise SystemExit("The manifest has no accepted segmented images to audit")
    chance_accuracy = 1.0 / len(classifier.labels)
    result = {
        "samples": sample_count,
        "class_count": len(classifier.labels),
        "chance_accuracy": chance_accuracy,
        "background_only_material_accuracy": correct / sample_count,
        "background_only_mean_confidence": confidence_sum / sample_count,
        "background_only_routed_rate": routed / sample_count,
        "per_class": {
            label: {
                **stats,
                "accuracy": stats["correct"] / stats["samples"],
                "routed_rate": stats["routed"] / stats["samples"],
            }
            for label, stats in per_class.items()
        },
        "interpretation": (
            "WARNING: background-only accuracy is substantially above chance; inspect data bias"
            if correct / sample_count > chance_accuracy + 0.15
            else "Background-only accuracy is near the expected chance region"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.examples_dir is not None:
        args.examples_dir.mkdir(parents=True, exist_ok=True)
        for rank, (confidence, source, image) in enumerate(
            sorted(suspicious, key=lambda item: item[0], reverse=True)[:25], start=1
        ):
            safe_name = Path(source).name.replace(" ", "_")
            Image.fromarray(image).save(
                args.examples_dir / f"{rank:02d}_{confidence:.3f}_{safe_name}.jpg"
            )
    print(json.dumps(result, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
