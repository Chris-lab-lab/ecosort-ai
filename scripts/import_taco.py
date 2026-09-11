"""Download reviewed TACO images and build an EcoSort classification dataset.

TACO stores full scenes with COCO annotations. This importer downloads the
640-pixel Flickr versions, crops each reviewed annotation, maps it to the
EcoSort routing taxonomy, and adds conservative background/unsupported crops
as ``other``. The original dataset is hard-linked into the output when
possible, so the source files remain untouched and use almost no extra space.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import threading
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError
import requests


CLASS_MAP = {
    # Metal
    "Aluminium foil": "metal",
    "Food Can": "metal",
    "Aerosol": "metal",
    "Drink can": "metal",
    "Metal bottle cap": "metal",
    "Metal lid": "metal",
    "Pop tab": "metal",
    "Scrap metal": "metal",
    # Plastic
    "Other plastic bottle": "plastic",
    "Clear plastic bottle": "plastic",
    "Plastic bottle cap": "plastic",
    "Other plastic cup": "plastic",
    "Disposable plastic cup": "plastic",
    "Foam cup": "plastic",
    "Plastic lid": "plastic",
    "Other plastic": "plastic",
    "Plastic film": "plastic",
    "Six pack rings": "plastic",
    "Garbage bag": "plastic",
    "Other plastic wrapper": "plastic",
    "Single-use carrier bag": "plastic",
    "Polypropylene bag": "plastic",
    "Spread tub": "plastic",
    "Tupperware": "plastic",
    "Disposable food container": "plastic",
    "Foam food container": "plastic",
    "Other plastic container": "plastic",
    "Plastic glooves": "plastic",  # Spelling used by the official taxonomy.
    "Plastic utensils": "plastic",
    "Plastic straw": "plastic",
    "Styrofoam piece": "plastic",
    # General routing class
    "Toilet tube": "general",
    "Other carton": "general",
    "Egg carton": "general",
    "Drink carton": "general",
    "Corrugated carton": "general",
    "Meal carton": "general",
    "Pizza box": "general",
    "Paper cup": "general",
    "Food waste": "general",
    "Magazine paper": "general",
    "Tissues": "general",
    "Wrapping paper": "general",
    "Normal paper": "general",
    "Paper bag": "general",
    "Plastified paper bag": "general",
    "Crisp packet": "general",
    "Paper straw": "general",
    "Cigarette": "general",
    # Unsupported, unsafe, or visually/materially ambiguous for three bins.
    "Battery": "other",
    "Aluminium blister pack": "other",
    "Carded blister pack": "other",
    "Glass bottle": "other",
    "Broken glass": "other",
    "Glass cup": "other",
    "Glass jar": "other",
    "Rope & strings": "other",
    "Shoe": "other",
    "Squeezable tube": "other",
    # This label gives no reliable material information; omit it instead of
    # teaching the model a potentially contradictory target.
    "Unlabeled litter": None,
}

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
IMAGE_MIRROR = (
    "https://huggingface.co/datasets/RandyHuynh5815/"
    "TACO-Waste-Recognition/resolve/main"
)
ANNOTATIONS_URL = (
    "https://raw.githubusercontent.com/pedropro/TACO/"
    "master/data/annotations.json"
)
_thread_state = threading.local()


@dataclass(frozen=True)
class DownloadResult:
    image_id: int
    path: Path | None
    error: str | None


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert reviewed TACO COCO annotations into EcoSort image classes"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("external/TACO/data/annotations.json"),
    )
    parser.add_argument("--base-data", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("dataset_enhanced"))
    parser.add_argument(
        "--cache", type=Path, default=Path("external/TACO/cache_images")
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--padding", type=float, default=0.15)
    parser.add_argument("--min-side", type=int, default=28)
    parser.add_argument(
        "--max-images",
        type=int,
        help="process only the first N TACO images (useful for a pipeline test)",
    )
    return parser.parse_args()


def _session() -> requests.Session:
    session = getattr(_thread_state, "session", None)
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = "EcoSort-TACO-import/1.0"
        _thread_state.session = session
    return session


def _safe_relative_file_name(raw_name: str) -> Path:
    candidate = Path(raw_name.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Unsafe TACO file name: {raw_name!r}")
    return candidate


def _valid_cached_image(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError):
        return False


def _ensure_annotations(path: Path) -> None:
    """Download the official reviewed annotations on a fresh project clone."""

    if path.is_file():
        return
    print(f"Downloading reviewed TACO annotations to {path}...")
    try:
        response = requests.get(
            ANNOTATIONS_URL,
            headers={"User-Agent": "EcoSort-TACO-import/1.0"},
            timeout=(10, 60),
        )
        response.raise_for_status()
        payload = response.content
        decoded = json.loads(payload)
        if not all(key in decoded for key in ("images", "annotations", "categories")):
            raise ValueError("downloaded JSON is not a TACO COCO annotation file")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    except (OSError, ValueError, requests.RequestException) as exc:
        raise SystemExit(f"Could not download reviewed TACO annotations: {exc}") from exc


def _download_image(record: dict[str, Any], cache: Path) -> DownloadResult:
    image_id = int(record["id"])
    relative = _safe_relative_file_name(str(record["file_name"]))
    destination = cache / relative
    if _valid_cached_image(destination):
        return DownloadResult(image_id, destination, None)

    # Prefer TACO's smaller official Flickr rendition. Some OpenLitterMap/S3
    # records have no resized URL and are very slow from their original host;
    # a public dataset mirror is tried before that original URL.
    mirror_url = f"{IMAGE_MIRROR}/{relative.as_posix()}?download=true"
    urls = [record.get("flickr_640_url"), mirror_url, record.get("flickr_url")]
    errors: list[str] = []
    for url in dict.fromkeys(url for url in urls if url):
        try:
            response = _session().get(url, timeout=(10, 45))
            response.raise_for_status()
            payload = response.content
            with Image.open(BytesIO(payload)) as image:
                image.verify()
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(payload)
            os.replace(temporary, destination)
            return DownloadResult(image_id, destination, None)
        except (OSError, requests.RequestException, UnidentifiedImageError) as exc:
            errors.append(f"{url}: {exc}")
    return DownloadResult(image_id, None, " | ".join(errors) or "no download URL")


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _link_or_copy(source: Path, destination: Path) -> str:
    if destination.exists():
        return "existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "linked"
    except OSError:
        shutil.copy2(source, destination)
        return "copied"


def _copy_base_dataset(base: Path, output: Path) -> Counter[str]:
    result: Counter[str] = Counter()
    for class_name in ("general", "metal", "plastic", "other"):
        source_dir = base / class_name
        destination_dir = output / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)
        if not source_dir.is_dir():
            continue
        for source in source_dir.rglob("*"):
            if not source.is_file() or source.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue
            relative = source.relative_to(source_dir)
            result[_link_or_copy(source, destination_dir / relative)] += 1
    return result


def _scaled_box(
    annotation: dict[str, Any],
    image_record: dict[str, Any],
    actual_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in annotation["bbox"])
    metadata_width = float(image_record["width"])
    metadata_height = float(image_record["height"])
    scale_x = actual_size[0] / metadata_width
    scale_y = actual_size[1] / metadata_height
    return x * scale_x, y * scale_y, width * scale_x, height * scale_y


def _padded_crop_box(
    box: tuple[float, float, float, float],
    image_size: tuple[int, int],
    padding: float,
) -> tuple[int, int, int, int]:
    x, y, width, height = box
    pad_x = width * padding
    pad_y = height * padding
    left = max(0, round(x - pad_x))
    top = max(0, round(y - pad_y))
    right = min(image_size[0], round(x + width + pad_x))
    bottom = min(image_size[1], round(y + height + pad_y))
    return left, top, right, bottom


def _intersection_area(
    a: tuple[int, int, int, int], b: tuple[float, float, float, float]
) -> float:
    bx, by, bw, bh = b
    left = max(float(a[0]), bx)
    top = max(float(a[1]), by)
    right = min(float(a[2]), bx + bw)
    bottom = min(float(a[3]), by + bh)
    return max(0.0, right - left) * max(0.0, bottom - top)


def _background_box(
    image_size: tuple[int, int], boxes: list[tuple[float, float, float, float]]
) -> tuple[int, int, int, int] | None:
    width, height = image_size
    side = max(32, round(min(width, height) * 0.38))
    candidates = [
        (0, 0, side, side),
        (width - side, 0, width, side),
        (0, height - side, side, height),
        (width - side, height - side, width, height),
        ((width - side) // 2, (height - side) // 2,
         (width + side) // 2, (height + side) // 2),
    ]
    candidate = min(candidates, key=lambda item: sum(_intersection_area(item, box) for box in boxes))
    overlap = sum(_intersection_area(candidate, box) for box in boxes)
    return candidate if overlap / float(side * side) <= 0.02 else None


def main() -> None:
    args = arguments()
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if not 0.0 <= args.padding <= 1.0:
        raise SystemExit("--padding must be between 0 and 1")
    if args.min_side < 1:
        raise SystemExit("--min-side must be positive")

    annotations_path = args.annotations.resolve()
    _ensure_annotations(annotations_path)
    payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): str(item["name"]) for item in payload["categories"]}
    unknown_categories = sorted(set(categories.values()) - set(CLASS_MAP))
    missing_categories = sorted(set(CLASS_MAP) - set(categories.values()))
    if unknown_categories or missing_categories:
        raise SystemExit(
            "TACO taxonomy does not match the reviewed mapping. "
            f"Unknown={unknown_categories}; missing={missing_categories}"
        )

    images = list(payload["images"])
    if args.max_images is not None:
        images = images[: args.max_images]
    image_ids = {int(item["id"]) for item in images}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload["annotations"]:
        if int(annotation["image_id"]) in image_ids:
            annotations_by_image[int(annotation["image_id"])].append(annotation)

    output = args.output.resolve()
    cache = args.cache.resolve()
    base = args.base_data.resolve()
    base_actions = _copy_base_dataset(base, output)
    print(f"Base dataset prepared: {dict(base_actions)}")

    downloads: dict[int, DownloadResult] = {}
    print(f"Downloading/verifying {len(images)} reviewed TACO images with {args.workers} workers...")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_download_image, image, cache): int(image["id"])
            for image in images
        }
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            downloads[result.image_id] = result
            completed += 1
            if completed % 100 == 0 or completed == len(images):
                failures = sum(item.error is not None for item in downloads.values())
                print(f"  {completed}/{len(images)} images checked; failures={failures}")

    imported: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    manifest: list[dict[str, Any]] = []
    failed_downloads: list[dict[str, Any]] = []

    for image_record in images:
        image_id = int(image_record["id"])
        download = downloads[image_id]
        if download.path is None:
            failed_downloads.append({"image_id": image_id, "error": download.error})
            continue
        try:
            with Image.open(download.path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            failed_downloads.append({"image_id": image_id, "error": str(exc)})
            continue

        scaled_boxes = [
            _scaled_box(annotation, image_record, image.size)
            for annotation in annotations_by_image[image_id]
        ]
        for annotation, scaled in zip(
            annotations_by_image[image_id], scaled_boxes, strict=True
        ):
            category = categories[int(annotation["category_id"])]
            target = CLASS_MAP[category]
            if target is None:
                skipped["unmapped_material"] += 1
                continue
            crop_box = _padded_crop_box(scaled, image.size, args.padding)
            if crop_box[2] - crop_box[0] < args.min_side or crop_box[3] - crop_box[1] < args.min_side:
                skipped["crop_too_small"] += 1
                continue
            name = f"taco_{image_id:05d}_{int(annotation['id']):06d}_{_slug(category)}.jpg"
            destination = output / target / name
            if not destination.exists():
                image.crop(crop_box).save(destination, "JPEG", quality=92, optimize=True)
            imported[target] += 1
            manifest.append(
                {
                    "file": str(destination.relative_to(output)).replace("\\", "/"),
                    "target": target,
                    "taco_category": category,
                    "image_id": image_id,
                    "annotation_id": int(annotation["id"]),
                    "source_url": image_record.get("flickr_url"),
                    "license_id": image_record.get("license"),
                }
            )

        background = _background_box(image.size, scaled_boxes)
        if background is not None:
            destination = output / "other" / f"taco_{image_id:05d}_background.jpg"
            if not destination.exists():
                image.crop(background).save(destination, "JPEG", quality=90, optimize=True)
            imported["other"] += 1
            manifest.append(
                {
                    "file": str(destination.relative_to(output)).replace("\\", "/"),
                    "target": "other",
                    "taco_category": "Background",
                    "image_id": image_id,
                    "annotation_id": None,
                    "source_url": image_record.get("flickr_url"),
                    "license_id": image_record.get("license"),
                }
            )
        else:
            skipped["no_clean_background"] += 1

    report = {
        "source": "TACO reviewed annotations",
        "source_repository": "https://github.com/pedropro/TACO",
        "image_mirror": IMAGE_MIRROR,
        "annotations": str(annotations_path),
        "base_dataset": str(base),
        "output_dataset": str(output),
        "processed_taco_images": len(images),
        "download_failures": failed_downloads,
        "imported_crops": dict(sorted(imported.items())),
        "skipped": dict(sorted(skipped.items())),
        "class_map": CLASS_MAP,
        "license_note": (
            "TACO annotations are CC BY 4.0; image license IDs and source URLs are "
            "preserved in taco_import_manifest.json. Check attribution requirements "
            "before redistributing images."
        ),
    }
    (output / "taco_import_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (output / "taco_import_manifest.jsonl").open("w", encoding="utf-8") as stream:
        for row in manifest:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    final_counts = {
        class_name: sum(
            path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            for path in (output / class_name).rglob("*")
        )
        for class_name in ("general", "metal", "other", "plastic")
    }
    print(f"Imported TACO crops: {dict(imported)}")
    print(f"Skipped: {dict(skipped)}")
    print(f"Download failures: {len(failed_downloads)}")
    print(f"Enhanced dataset counts: {final_counts}")
    print(f"Report: {output / 'taco_import_report.json'}")


if __name__ == "__main__":
    main()
