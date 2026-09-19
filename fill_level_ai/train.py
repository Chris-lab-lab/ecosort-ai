from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageFile, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


ImageFile.LOAD_TRUNCATED_IMAGES = True
CLASS_NAMES = ["empty", "half-full", "full"]
FOLDER_TO_INDEX = {"empty": 0, "halffull": 1, "full": 2}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class WasteBinDataset(Dataset):
    def __init__(self, samples: list[dict], transform) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample["path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = self.transform(image)
        return image, sample["label"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a waste-bin fill-level classifier.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=4)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_samples(data_dir: Path) -> list[dict]:
    samples: list[dict] = []
    for source in ("simulated", "real"):
        for folder_name, label in FOLDER_TO_INDEX.items():
            folder = data_dir / source / folder_name
            if not folder.is_dir():
                raise FileNotFoundError(f"Expected dataset folder was not found: {folder}")
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                    samples.append({"path": path, "label": label, "source": source})
    if not samples:
        raise RuntimeError(f"No images found under {data_dir}")
    return samples


def split_samples(samples: list[dict], seed: int):
    """Stratify by class and source; reserve a real-only deployment test set."""
    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []
    test: list[dict] = []

    for label in range(len(CLASS_NAMES)):
        simulated = [s for s in samples if s["label"] == label and s["source"] == "simulated"]
        real = [s for s in samples if s["label"] == label and s["source"] == "real"]
        rng.shuffle(simulated)
        rng.shuffle(real)

        simulated_val_count = max(1, round(0.10 * len(simulated)))
        val.extend(simulated[:simulated_val_count])
        train.extend(simulated[simulated_val_count:])

        real_test_count = max(1, round(0.20 * len(real)))
        real_val_count = max(1, round(0.20 * len(real)))
        test.extend(real[:real_test_count])
        val.extend(real[real_test_count : real_test_count + real_val_count])
        train.extend(real[real_test_count + real_val_count :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def split_summary(split: list[dict]) -> dict:
    by_class = Counter(CLASS_NAMES[s["label"]] for s in split)
    by_source = Counter(s["source"] for s in split)
    return {
        "total": len(split),
        "by_class": dict(sorted(by_class.items())),
        "by_source": dict(sorted(by_source.items())),
    }


def write_split_manifest(path: Path, splits: dict[str, list[dict]], data_dir: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "source", "label", "path"])
        writer.writeheader()
        for split_name, split in splits.items():
            for sample in split:
                writer.writerow(
                    {
                        "split": split_name,
                        "source": sample["source"],
                        "label": CLASS_NAMES[sample["label"]],
                        "path": sample["path"].relative_to(data_dir).as_posix(),
                    }
                )


def make_transforms(image_size: int):
    weights = MobileNet_V3_Small_Weights.DEFAULT
    normalize = transforms.Normalize(mean=weights.transforms().mean, std=weights.transforms().std)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.68, 1.0), ratio=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.03),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, eval_transform


def build_model() -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(CLASS_NAMES))
    return model


def compute_metrics(targets: list[int], predictions: list[int]) -> dict:
    matrix = [[0 for _ in CLASS_NAMES] for _ in CLASS_NAMES]
    for target, prediction in zip(targets, predictions):
        matrix[target][prediction] += 1

    per_class = {}
    f1_values = []
    for index, name in enumerate(CLASS_NAMES):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(len(CLASS_NAMES))) - tp
        fn = sum(matrix[index]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(matrix[index])}

    correct = sum(matrix[i][i] for i in range(len(CLASS_NAMES)))
    total = max(1, len(targets))
    return {
        "accuracy": correct / total,
        "macro_f1": sum(f1_values) / len(f1_values),
        "per_class": per_class,
        "confusion_matrix": matrix,
        "confusion_matrix_labels": CLASS_NAMES,
    }


def run_epoch(model, loader, criterion, device, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    running_loss = 0.0
    targets: list[int] = []
    predictions: list[int] = []

    context = torch.enable_grad() if is_training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if is_training:
                loss.backward()
                optimizer.step()
            running_loss += loss.item() * images.size(0)
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())

    metrics = compute_metrics(targets, predictions)
    metrics["loss"] = running_loss / max(1, len(loader.dataset))
    return metrics, targets, predictions


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu_count = os.cpu_count() or 1
    torch.set_num_threads(max(1, min(cpu_count - 1, 12)))

    samples = discover_samples(args.data_dir)
    train_samples, val_samples, test_samples = split_samples(samples, args.seed)
    splits = {"train": train_samples, "validation": val_samples, "test": test_samples}
    write_split_manifest(args.output_dir / "dataset_splits.csv", splits, args.data_dir)
    split_info = {name: split_summary(split) for name, split in splits.items()}
    print(json.dumps(split_info, indent=2))
    print(f"Device: {device}; CPU threads: {torch.get_num_threads()}")

    train_transform, eval_transform = make_transforms(args.image_size)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        WasteBinDataset(train_samples, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        generator=generator,
    )
    val_loader = DataLoader(
        WasteBinDataset(val_samples, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        WasteBinDataset(test_samples, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )

    model = build_model().to(device)
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    counts = Counter(s["label"] for s in train_samples)
    class_weights = torch.tensor(
        [len(train_samples) / (len(CLASS_NAMES) * counts[i]) for i in range(len(CLASS_NAMES))],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4, weight_decay=1e-4)

    best_state = None
    best_score = -1.0
    epochs_without_improvement = 0
    history = []
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_epochs + 1:
            for block in list(model.features.children())[-4:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.classifier.parameters(), "lr": 1e-4},
                    {
                        "params": [p for p in model.features.parameters() if p.requires_grad],
                        "lr": 4e-5,
                    },
                ],
                weight_decay=1e-4,
            )
            print("Unfroze the last four MobileNet feature blocks for fine-tuning.")

        epoch_started = time.time()
        train_metrics, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_metrics, _, _ = run_epoch(model, val_loader, criterion, device)
        record = {
            "epoch": epoch,
            "seconds": time.time() - epoch_started,
            "train": train_metrics,
            "validation": val_metrics,
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_metrics['loss']:.4f}, acc {train_metrics['accuracy']:.3f} | "
            f"val loss {val_metrics['loss']:.4f}, acc {val_metrics['accuracy']:.3f}, "
            f"macro-F1 {val_metrics['macro_f1']:.3f} | {record['seconds']:.1f}s"
        )

        if val_metrics["macro_f1"] > best_score + 1e-4:
            best_score = val_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch >= args.freeze_epochs + 3 and epochs_without_improvement >= args.patience:
            print(f"Early stopping after {epoch} epochs.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    test_metrics, test_targets, test_predictions = run_epoch(model, test_loader, criterion, device)

    checkpoint = {
        "architecture": "mobilenet_v3_small",
        "class_names": CLASS_NAMES,
        "image_size": args.image_size,
        "state_dict": model.cpu().state_dict(),
    }
    torch.save(checkpoint, args.output_dir / "waste_bin_fill_level_mobilenet_v3_small.pth")

    model.eval()
    example = torch.zeros(1, 3, args.image_size, args.image_size)
    traced = torch.jit.trace(model, example)
    traced.save(str(args.output_dir / "waste_bin_fill_level_mobilenet_v3_small.torchscript.pt"))

    with (args.output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "actual", "predicted", "correct"])
        writer.writeheader()
        for sample, target, prediction in zip(test_samples, test_targets, test_predictions):
            writer.writerow(
                {
                    "path": sample["path"].relative_to(args.data_dir).as_posix(),
                    "actual": CLASS_NAMES[target],
                    "predicted": CLASS_NAMES[prediction],
                    "correct": target == prediction,
                }
            )

    report = {
        "model": "MobileNetV3-Small pretrained on ImageNet",
        "device_used": str(device),
        "seed": args.seed,
        "epochs_completed": len(history),
        "elapsed_seconds": time.time() - started,
        "dataset": split_info,
        "best_validation_macro_f1": best_score,
        "test_real_images_only": True,
        "test_metrics": test_metrics,
        "history": history,
    }
    (args.output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output_dir / "labels.json").write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")
    print("\nFinal real-image test metrics:")
    print(json.dumps(test_metrics, indent=2))
    print(f"Saved artifacts to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
