"""Train a small YOLO detector on the synthetic trash dataset.

Run in the separate .venv-yolo environment; this does not use TensorFlow or lids.
"""

from __future__ import annotations

import argparse
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parent / "datasets" / "trash_yolo_synthetic_v1" / "data.yaml"


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train a YOLO11 trash detector; default is a 3-epoch CPU smoke test."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Dataset YAML path")
    parser.add_argument("--model", default="yolo11n.pt", help="Model weights or model YAML")
    parser.add_argument("--epochs", type=positive_int, default=3)
    parser.add_argument("--imgsz", type=positive_int, default=640)
    parser.add_argument("--batch", type=positive_int, default=4)
    parser.add_argument("--device", default="cpu", help="cpu, or 0 for a configured CUDA GPU")
    parser.add_argument("--name", default="trash_smoke", help="Run folder name inside yolo_synthetic/runs")
    args = parser.parse_args(argv)

    data = args.data.expanduser().resolve()
    if not data.is_file():
        parser.error(f"Dataset YAML does not exist: {data}")
    if args.name in {".", ".."} or not args.name or any(char in args.name for char in '/\\:'):
        parser.error("--name must be a folder name without path separators")
    if args.imgsz % 32:
        parser.error("--imgsz must be a multiple of 32, such as 320 or 640")

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is unavailable. Follow yolo_synthetic/README.md to create "
            "the separate .venv-yolo environment and install requirements-yolo.txt. "
            f"Import detail: {exc}"
        ) from exc

    print(f"Dataset: {data}")
    print("Synthetic test only: its scores do not measure real camera performance.")
    model = YOLO(args.model)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=0,
        project=str(HERE / "runs"),
        name=args.name,
        exist_ok=False,
        seed=42,
        cache=False,
        plots=True,
    )
    run_dir = Path(model.trainer.save_dir).resolve()
    print(f"Training output: {run_dir}")
    print(f"Trained weights: {run_dir / 'weights' / 'best.pt'}")
    print("Use these trained weights for the test/camera commands in README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
