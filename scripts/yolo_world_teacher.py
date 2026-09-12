"""Run the official YOLO-World implementation as an offline review teacher."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_PROMPTS = (
    "trash,litter,bottle,can,wrapper,paper,cardboard,plastic object,metal object,"
    "glass,battery,electronic waste,food,person,hand"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export YOLO-World teacher boxes as JSONL"
    )
    parser.add_argument("--data", type=Path, default=Path("dataset_enhanced"))
    parser.add_argument(
        "--yolo-world-repo", type=Path, default=Path("external/YOLO-World")
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("teacher_outputs/yolo_world.jsonl")
    )
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0 or args.topk < 1:
        parser.error("threshold must be in 0..1 and topk must be positive")
    return args


def main() -> None:
    args = arguments()
    repository = args.yolo_world_repo.resolve()
    if not repository.is_dir():
        raise SystemExit(
            f"YOLO-World was not found at {repository}; follow OFFLINE_TEACHERS.md first"
        )
    sys.path.insert(0, str(repository))
    try:
        import cv2
        import torch
        from mmdet.apis import init_detector
        from mmdet.utils import get_test_pipeline_cfg
        from mmengine.config import Config
        from mmengine.dataset import Compose
    except ImportError as exc:
        raise SystemExit(
            "YOLO-World dependencies are missing; install its official environment first: "
            f"{exc}"
        ) from exc

    data = args.data.resolve()
    if not data.is_dir():
        raise SystemExit(f"Dataset folder does not exist: {data}")
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    cfg = Config.fromfile(str(config_path))
    cfg.load_from = str(checkpoint_path)
    model = init_detector(cfg, checkpoint=str(checkpoint_path), device=args.device)
    pipeline_config = get_test_pipeline_cfg(cfg=cfg)
    pipeline_config[0].type = "mmdet.LoadImageFromNDArray"
    pipeline = Compose(pipeline_config)
    prompt_names = [
        prompt.strip() for prompt in args.prompts.split(",") if prompt.strip()
    ]
    texts = [[prompt] for prompt in prompt_names] + [[" "]]
    model.reparameterize(texts)

    sources = sorted(
        path
        for path in data.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if args.limit is not None:
        sources = sources[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for index, source in enumerate(sources, start=1):
            bgr = cv2.imread(str(source))
            if bgr is None:
                continue
            rgb = bgr[:, :, ::-1]
            data_info = pipeline({"img": rgb, "img_id": index, "texts": texts})
            batch = {
                "inputs": data_info["inputs"].unsqueeze(0),
                "data_samples": [data_info["data_samples"]],
            }
            with torch.no_grad():
                instances = model.test_step(batch)[0].pred_instances
            instances = instances[instances.scores.float() >= args.threshold]
            if len(instances.scores) > args.topk:
                instances = instances[instances.scores.float().topk(args.topk)[1]]
            values = instances.cpu().numpy()
            detections = []
            for box, label_index, score in zip(
                values["bboxes"], values["labels"], values["scores"], strict=True
            ):
                label_index = int(label_index)
                if label_index >= len(prompt_names):
                    continue
                detections.append(
                    {
                        "xyxy": [float(value) for value in box],
                        "label": prompt_names[label_index],
                        "score": float(score),
                    }
                )
            stream.write(
                json.dumps(
                    {
                        "image": source.relative_to(data).as_posix(),
                        "detections": detections,
                    }
                )
                + "\n"
            )
            if index % 25 == 0 or index == len(sources):
                print(f"YOLO-World: {index}/{len(sources)}")
    print(f"Teacher detections: {args.output}")


if __name__ == "__main__":
    main()
