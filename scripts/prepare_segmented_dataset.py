"""Build a background-randomized classifier dataset with EfficientViT-SAM.

This is an offline laptop tool. EfficientViT-SAM never becomes a dependency of
the NXP runtime or the exported MobileNet TFLite model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from ecosort_ai.efficientvit_compat import install_triton_rms_norm_fallback
from ecosort_ai.segmentation import composite_masked_object, select_primary_mask

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare masked EcoSort crops with an offline EfficientViT-SAM teacher"
    )
    parser.add_argument("--data", type=Path, default=Path("dataset_enhanced"))
    parser.add_argument("--output", type=Path, default=Path("dataset_segmented"))
    parser.add_argument(
        "--efficientvit-repo",
        type=Path,
        default=Path("external/efficientvit"),
        help="clone of https://github.com/mit-han-lab/efficientvit",
    )
    parser.add_argument(
        "--model",
        default="efficientvit-sam-l0",
        choices=(
            "efficientvit-sam-l0",
            "efficientvit-sam-l1",
            "efficientvit-sam-l2",
            "efficientvit-sam-xl0",
            "efficientvit-sam-xl1",
        ),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--variants", type=int, default=3)
    parser.add_argument("--output-size", type=int, default=512)
    parser.add_argument("--min-area-ratio", type=float, default=0.04)
    parser.add_argument("--max-area-ratio", type=float, default=0.85)
    parser.add_argument("--multiple-object-ratio", type=float, default=0.30)
    parser.add_argument("--points-per-side", type=int, default=32)
    parser.add_argument(
        "--points-per-batch",
        type=int,
        default=64,
        help="automatic-mask prompt batch size; larger can be faster but uses more VRAM",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("auto", "center"),
        default="auto",
        help="center is much faster for datasets containing one centered object",
    )
    parser.add_argument(
        "--pred-iou-threshold",
        type=float,
        default=0.70,
        help="minimum SAM predicted mask quality (L0-friendly default: 0.70)",
    )
    parser.add_argument(
        "--stability-threshold",
        type=float,
        default=0.85,
        help="minimum SAM mask stability (L0-friendly default: 0.85)",
    )
    parser.add_argument(
        "--detections-jsonl",
        type=Path,
        help="optional boxes created by yolo_world_teacher.py",
    )
    parser.add_argument("--detection-threshold", type=float, default=0.25)
    parser.add_argument("--other-label", default="other")
    parser.add_argument(
        "--segment-other",
        action="store_true",
        help=(
            "also create object masks for reject/other images; required by "
            "train_full_model.py when those masks are available"
        ),
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--mask-only",
        action="store_true",
        help="save teacher masks without classifier composite JPEGs",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="keep completed outputs and skip source paths already in the manifest",
    )
    args = parser.parse_args()
    if (
        args.variants < 1
        or args.output_size < 64
        or args.points_per_side < 4
        or args.points_per_batch < 1
    ):
        parser.error(
            "variants and points-per-batch must be positive, output size >= 64, "
            "and points-per-side >= 4"
        )
    if not 0.0 < args.min_area_ratio < args.max_area_ratio <= 1.0:
        parser.error("mask area limits are invalid")
    if not 0.0 < args.multiple_object_ratio <= 1.0:
        parser.error("--multiple-object-ratio must be between 0 and 1")
    if not 0.0 <= args.detection_threshold <= 1.0:
        parser.error("--detection-threshold must be between 0 and 1")
    if not 0.0 <= args.pred_iou_threshold <= 1.0:
        parser.error("--pred-iou-threshold must be between 0 and 1")
    if not 0.0 <= args.stability_threshold <= 1.0:
        parser.error("--stability-threshold must be between 0 and 1")
    return args


def _load_detections(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {}
    records: dict[str, list[dict[str, Any]]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            image = str(record.get("image", "")).replace("\\", "/")
            if not image:
                raise ValueError(f"Detection record {line_number} has no image path")
            records[image] = list(record.get("detections", []))
    return records


class EfficientViTSamTeacher:
    def __init__(self, args: argparse.Namespace) -> None:
        repository = args.efficientvit_repo.resolve()
        if not (repository / "efficientvit" / "sam_model_zoo.py").is_file():
            raise RuntimeError(
                f"EfficientViT was not found at {repository}. Clone the official repository "
                "and install its requirements as described in OFFLINE_TEACHERS.md."
            )
        sys.path.insert(0, str(repository))
        try:
            import torch

            using_triton_fallback = install_triton_rms_norm_fallback(torch)
            from efficientvit.models.efficientvit.sam import (
                EfficientViTSamAutomaticMaskGenerator,
                EfficientViTSamPredictor,
            )
            from efficientvit.sam_model_zoo import create_efficientvit_sam_model
        except ImportError as exc:
            raise RuntimeError(
                "EfficientViT-SAM dependencies are missing. Use the separate teacher environment "
                "described in OFFLINE_TEACHERS.md."
            ) from exc

        device = args.device
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if using_triton_fallback:
            print(
                "Triton is unavailable; using the PyTorch RMSNorm inference fallback."
            )
        checkpoint = args.checkpoint
        if checkpoint is None:
            suffix = args.model.removeprefix("efficientvit-sam-").replace("-", "_")
            checkpoint = (
                repository
                / "assets"
                / "checkpoints"
                / "efficientvit_sam"
                / f"efficientvit_sam_{suffix}.pt"
            )
        checkpoint = checkpoint.resolve()
        if not checkpoint.is_file():
            raise RuntimeError(
                f"Checkpoint not found: {checkpoint}. Download the official {args.model} "
                "checkpoint before running this script."
            )

        model = (
            create_efficientvit_sam_model(
                name=args.model,
                pretrained=True,
                weight_url=str(checkpoint),
            )
            .to(device)
            .eval()
        )
        self.device = device
        self.prompt_mode = args.prompt_mode
        self.generator = EfficientViTSamAutomaticMaskGenerator(
            model,
            points_per_side=args.points_per_side,
            points_per_batch=args.points_per_batch,
            pred_iou_thresh=args.pred_iou_threshold,
            stability_score_thresh=args.stability_threshold,
            # OpenCV connected-component cleanup duplicates every full-resolution
            # mask and can fail on unusually large source images. The downstream
            # area/center filter already rejects tiny irrelevant proposals.
            min_mask_region_area=0,
        )
        self.predictor = EfficientViTSamPredictor(model)

    def masks(
        self, rgb: np.ndarray, box_xyxy: list[float] | None
    ) -> list[dict[str, Any]]:
        if box_xyxy is None and self.prompt_mode == "auto":
            return list(self.generator.generate(rgb))
        self.predictor.set_image(rgb)
        if box_xyxy is not None:
            masks, scores, _ = self.predictor.predict(
                box=np.asarray(box_xyxy, dtype=np.float32),
                multimask_output=True,
            )
        else:
            height, width = rgb.shape[:2]
            masks, scores, _ = self.predictor.predict(
                point_coords=np.asarray(
                    [[width / 2.0, height / 2.0]], dtype=np.float32
                ),
                point_labels=np.ones(1, dtype=np.int32),
                multimask_output=True,
            )
        return [
            {
                "segmentation": mask,
                "predicted_iou": float(score),
                "stability_score": 1.0,
            }
            for mask, score in zip(masks, scores, strict=True)
        ]


def _relative_key(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _stem_for(relative: str) -> str:
    digest = hashlib.sha1(relative.encode("utf-8"), usedforsecurity=False).hexdigest()[
        :12
    ]
    return f"{Path(relative).stem}_{digest}"


def _review_copy(source: Path, output: Path, reason: str, label: str, stem: str) -> str:
    destination = output / "review" / reason / label / f"{stem}{source.suffix.lower()}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination.relative_to(output).as_posix()


def main() -> None:
    args = arguments()
    data = args.data.resolve()
    output = args.output.resolve()
    if not data.is_dir():
        raise SystemExit(f"Dataset folder does not exist: {data}")
    if output == data or output.is_relative_to(data):
        raise SystemExit("The segmented output must not be inside the source dataset")
    manifest_path = output / "segmentation_manifest.jsonl"
    if manifest_path.exists() and not args.resume:
        raise SystemExit(
            f"{manifest_path} already exists; use --resume or choose a new output"
        )

    completed: set[str] = set()
    previous_counts: dict[str, int] = {}
    if args.resume and manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    previous = json.loads(line)
                    completed.add(str(previous["source"]))
                    status = str(previous.get("status", "error"))
                    previous_counts[status] = previous_counts.get(status, 0) + 1

    detections = _load_detections(args.detections_jsonl)
    teacher = EfficientViTSamTeacher(args)
    print(f"EfficientViT-SAM device: {teacher.device}")
    training_root = output / "data"
    sources = sorted(
        path
        for path in data.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if args.limit is not None:
        sources = sources[: args.limit]
    output.mkdir(parents=True, exist_ok=True)
    counts = {
        "accepted": 0,
        "negative_passthrough": 0,
        "no_object": 0,
        "multiple_objects": 0,
        "error": 0,
    }
    for status, count in previous_counts.items():
        counts[status] = counts.get(status, 0) + count

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for index, source in enumerate(sources, start=1):
            relative = _relative_key(source, data)
            if relative in completed:
                continue
            label = source.relative_to(data).parts[0].lower()
            stem = _stem_for(relative)
            record: dict[str, Any] = {"source": relative, "label": label}
            try:
                if label == args.other_label.lower() and not args.segment_other:
                    destination = (
                        training_root / label / f"{stem}{source.suffix.lower()}"
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    record.update(
                        status="negative_passthrough",
                        supported=False,
                        object_present=None,
                        outputs=[destination.relative_to(output).as_posix()],
                    )
                else:
                    with Image.open(source) as image:
                        rgb = np.asarray(image.convert("RGB"))
                    detected = [
                        item
                        for item in detections.get(relative, [])
                        if float(item.get("score", 0.0)) >= args.detection_threshold
                    ]
                    if len(detected) > 1:
                        record.update(
                            status="multiple_objects",
                            supported=False,
                            object_present=True,
                            detector_objects=len(detected),
                            review=_review_copy(
                                source, output, "multiple_objects", label, stem
                            ),
                        )
                    else:
                        box = list(detected[0]["xyxy"]) if detected else None
                        masks = teacher.masks(rgb, box)
                        decision = select_primary_mask(
                            masks,
                            rgb.shape,
                            min_area_ratio=args.min_area_ratio,
                            max_area_ratio=args.max_area_ratio,
                            multiple_object_ratio=args.multiple_object_ratio,
                            # Automatic SAM masks are overlapping alternatives and
                            # object parts, not detected instances. Multiple-object
                            # rejection is reliable only when detector boxes exist;
                            # more than one accepted box is handled above.
                            reject_multiple_objects=False,
                        )
                        if decision.status != "accepted":
                            record.update(
                                status=decision.status,
                                supported=False,
                                object_present=decision.status == "multiple_objects",
                                mask_candidates=decision.candidate_count,
                                distinct_objects=decision.distinct_object_count,
                                review=_review_copy(
                                    source, output, decision.status, label, stem
                                ),
                            )
                        else:
                            mask_path = output / "masks" / label / f"{stem}.png"
                            mask_path.parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(
                                (decision.mask * 255).astype(np.uint8)
                            ).save(mask_path)
                            generated: list[str] = []
                            if not args.mask_only:
                                for variant in range(args.variants):
                                    seed = args.seed + int(stem[-8:], 16) + variant
                                    prepared = composite_masked_object(
                                        rgb,
                                        decision.mask,
                                        output_size=args.output_size,
                                        rng=np.random.default_rng(seed),
                                        variant=variant,
                                    )
                                    destination = (
                                        training_root
                                        / label
                                        / f"{stem}_seg{variant}.jpg"
                                    )
                                    destination.parent.mkdir(
                                        parents=True, exist_ok=True
                                    )
                                    Image.fromarray(prepared).save(
                                        destination, quality=94
                                    )
                                    generated.append(
                                        destination.relative_to(output).as_posix()
                                    )
                            record.update(
                                status="accepted",
                                supported=label != args.other_label.lower(),
                                object_present=True,
                                mask=mask_path.relative_to(output).as_posix(),
                                mask_confidence=decision.mask_confidence,
                                bbox_xyxy=list(decision.bbox_xyxy),
                                mask_candidates=decision.candidate_count,
                                distinct_objects=decision.distinct_object_count,
                                detector_objects=len(detected),
                                outputs=generated,
                            )
            # A corrupt or abnormally large image must not terminate an overnight
            # dataset run. KeyboardInterrupt is a BaseException and remains safe.
            except Exception as exc:  # noqa: BLE001
                record.update(status="error", supported=False, error=str(exc))
            counts[record["status"]] = counts.get(record["status"], 0) + 1
            manifest.write(json.dumps(record) + "\n")
            manifest.flush()
            if index % 25 == 0 or index == len(sources):
                print(f"Processed {index}/{len(sources)}: {counts}")

    report = {
        "source": str(data),
        "output": str(output),
        "training_data": str(training_root),
        "model": args.model,
        "device": teacher.device,
        "prompt_mode": args.prompt_mode,
        "mask_only": args.mask_only,
        "variants": args.variants,
        "counts": counts,
        "note": "Review rejected masks before training; source images were not modified.",
    }
    (output / "segmentation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
