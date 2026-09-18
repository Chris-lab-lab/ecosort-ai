"""Compare deployable EcoSort artifacts before changing the NXP backbone."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from ecosort_ai.model import WasteClassifier


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare TFLite size, latency, and saved validation safety metrics"
    )
    parser.add_argument(
        "artifacts", nargs="+", type=Path, help="two or more artifact folders"
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--delegate", default="none", help="none, auto, or delegate .so path"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if len(args.artifacts) < 2 or args.runs < 1 or args.warmup < 0:
        parser.error("provide at least two artifact folders; runs must be positive")
    return args


def main() -> None:
    args = arguments()
    delegate = None if args.delegate.lower() == "none" else args.delegate
    rows = []
    for folder in args.artifacts:
        folder = folder.resolve()
        model_path = folder / "waste_classifier_int8.tflite"
        labels_path = folder / "labels.txt"
        summary_path = folder / "training_summary.json"
        classifier = WasteClassifier(model_path, labels_path, delegate=delegate)
        sample = np.full((classifier.height, classifier.width, 3), 127, dtype=np.uint8)
        for _ in range(args.warmup):
            classifier.predict_rgb(sample)
        timings = []
        for _ in range(args.runs):
            started = time.perf_counter()
            classifier.predict_rgb(sample)
            timings.append((time.perf_counter() - started) * 1000.0)
        summary = (
            json.loads(summary_path.read_text(encoding="utf-8"))
            if summary_path.is_file()
            else {}
        )
        metrics = summary.get("quantized_metrics", {})
        rows.append(
            {
                "artifact": str(folder),
                "model_bytes": model_path.stat().st_size,
                "runtime": classifier.runtime_name,
                "delegate": classifier.delegate_path or "CPU",
                "latency_ms_median": statistics.median(timings),
                "latency_ms_p95": float(np.percentile(timings, 95)),
                "macro_f1": metrics.get("macro_f1"),
                "unknown_false_acceptance_rate": summary.get(
                    "unknown_false_acceptance_rate"
                ),
                "wrong_lid_activation_rate": summary.get("wrong_lid_activation_rate"),
            }
        )
    result = {"runs": args.runs, "models": rows}
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
