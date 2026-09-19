from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageEnhance, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_small

from preprocessing import letterbox, to_normalized_tensor
from train import CLASS_NAMES, compute_metrics


CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a calibrated, ordinal-aware robust trash-bin classifier.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--ordinal-weight", type=float, default=0.55)
    return parser.parse_args()


def discover(data_dir: Path, split: str) -> list[dict]:
    samples = []
    if split == "train":
        for kind in ("original", "synthetic"):
            for class_name, label in CLASS_TO_INDEX.items():
                folder = data_dir / split / kind / class_name
                for path in sorted(folder.rglob("*")):
                    if path.is_file() and path.suffix.lower() in SUFFIXES:
                        samples.append({"path": path, "label": label, "kind": kind})
    else:
        for class_name, label in CLASS_TO_INDEX.items():
            folder = data_dir / split / class_name
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix.lower() in SUFFIXES:
                    samples.append({"path": path, "label": label, "kind": "original"})
    if not samples:
        raise RuntimeError(f"No images found for split {split}")
    return samples


class RobustDataset(Dataset):
    def __init__(self, samples: list[dict], image_size: int, training: bool, seed: int) -> None:
        self.samples = samples
        self.image_size = image_size
        self.training = training
        self.seed = seed

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        with Image.open(sample["path"]) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        if self.training:
            rng = random.Random(self.seed + index + random.randint(0, 2**20))
            if rng.random() < 0.5:
                image = ImageOps.mirror(image)
            if rng.random() < 0.35:
                image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.88, 1.12))
                image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.88, 1.12))
            if rng.random() < 0.18:
                image = ImageOps.grayscale(image).convert("RGB")
            scale = rng.uniform(0.90, 1.0)
        else:
            scale = 1.0
        tensor = to_normalized_tensor(letterbox(image, self.image_size, scale=scale))
        return tensor, sample["label"]


class OrdinalAwareLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, ordinal_weight: float) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.04)
        self.ordinal_weight = ordinal_weight
        indices = torch.arange(len(CLASS_NAMES), dtype=torch.float32)
        self.register_buffer("indices", indices)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cross_entropy = self.cross_entropy(logits, labels)
        probabilities = torch.softmax(logits, dim=1)
        distances = (self.indices.unsqueeze(0) - labels.float().unsqueeze(1)).abs()
        ordinal = (probabilities * distances.pow(1.6)).sum(dim=1).mean()
        return cross_entropy + self.ordinal_weight * ordinal


def extended_metrics(targets: list[int], predictions: list[int]) -> dict:
    metrics = compute_metrics(targets, predictions)
    supports = [metrics["per_class"][name]["support"] for name in CLASS_NAMES]
    total = max(1, sum(supports))
    metrics["weighted_f1"] = sum(
        metrics["per_class"][name]["f1"] * metrics["per_class"][name]["support"] for name in CLASS_NAMES
    ) / total
    metrics["mean_absolute_class_error"] = sum(abs(a - b) for a, b in zip(targets, predictions)) / max(1, len(targets))
    metrics["severe_error_rate"] = sum(abs(a - b) == 2 for a, b in zip(targets, predictions)) / max(1, len(targets))
    return metrics


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    targets, predictions, logits_all = [], [], []
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
            loss_total += float(loss.detach()) * images.size(0)
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
            logits_all.append(logits.detach().cpu())
    metrics = extended_metrics(targets, predictions)
    metrics["loss"] = loss_total / max(1, len(loader.dataset))
    return metrics, targets, predictions, torch.cat(logits_all)


def calibrate_temperature(logits: torch.Tensor, targets: list[int]) -> tuple[float, float]:
    labels = torch.tensor(targets, dtype=torch.long)
    best_temperature, best_nll = 1.0, math.inf
    for step in range(251):
        temperature = 0.50 + step * 0.01
        nll = float(nn.functional.cross_entropy(logits / temperature, labels))
        if nll < best_nll:
            best_temperature, best_nll = temperature, nll
    return best_temperature, best_nll


class TemperatureWrapper(nn.Module):
    def __init__(self, model: nn.Module, temperature: float) -> None:
        super().__init__()
        self.model = model
        self.register_buffer("temperature", torch.tensor(float(temperature)))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.model(images) / self.temperature


def summarize(samples: list[dict]) -> dict:
    return {
        "total": len(samples),
        "by_class": dict(Counter(CLASS_NAMES[sample["label"]] for sample in samples)),
        "by_kind": dict(Counter(sample["kind"] for sample in samples)),
    }


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min((os.cpu_count() or 1) - 1, 12)))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_samples = discover(args.data_dir, "train")
    validation_samples = discover(args.data_dir, "validation")
    test_samples = discover(args.data_dir, "test")
    split_info = {"train": summarize(train_samples), "validation": summarize(validation_samples), "test": summarize(test_samples)}
    print(json.dumps(split_info, indent=2))

    train_loader = DataLoader(RobustDataset(train_samples, args.image_size, True, args.seed), batch_size=args.batch_size, shuffle=True, num_workers=0, generator=torch.Generator().manual_seed(args.seed))
    validation_loader = DataLoader(RobustDataset(validation_samples, args.image_size, False, args.seed), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(RobustDataset(test_samples, args.image_size, False, args.seed), batch_size=args.batch_size, shuffle=False, num_workers=0)

    checkpoint = torch.load(args.base_model, map_location="cpu", weights_only=False)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(CLASS_NAMES))
    model.load_state_dict(checkpoint["state_dict"])
    if isinstance(model.classifier[2], nn.Dropout):
        model.classifier[2].p = 0.35
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for block in list(model.features.children())[-5:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    model = model.to(device)

    counts = Counter(sample["label"] for sample in train_samples)
    weights = torch.tensor([len(train_samples) / (len(CLASS_NAMES) * counts[index]) for index in range(len(CLASS_NAMES))], device=device)
    criterion = OrdinalAwareLoss(weights, args.ordinal_weight)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.classifier.parameters(), "lr": 1.4e-4},
            {"params": [p for p in model.features.parameters() if p.requires_grad], "lr": 3.5e-5},
        ],
        weight_decay=3e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(2, args.epochs), eta_min=3e-6)

    best_state, best_metrics, best_score = None, None, -1.0
    history, stale = [], 0
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics, _, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        validation_metrics, validation_targets, _, validation_logits = run_epoch(model, validation_loader, criterion, device)
        score = validation_metrics["macro_f1"] - 0.15 * validation_metrics["severe_error_rate"]
        history.append({"epoch": epoch, "learning_rates": [group["lr"] for group in optimizer.param_groups], "train": train_metrics, "validation": validation_metrics})
        print(f"Epoch {epoch:02d}/{args.epochs} | train {train_metrics['accuracy']:.3f} | val {validation_metrics['accuracy']:.3f}, macro-F1 {validation_metrics['macro_f1']:.3f}, severe {validation_metrics['severe_error_rate']:.3f}")
        if score > best_score + 1e-4:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = validation_metrics
            stale = 0
        else:
            stale += 1
        scheduler.step()
        if epoch >= 7 and stale >= args.patience:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    model.load_state_dict(best_state)
    validation_metrics, validation_targets, _, validation_logits = run_epoch(model, validation_loader, criterion, device)
    temperature, calibrated_nll = calibrate_temperature(validation_logits, validation_targets)
    test_metrics, test_targets, test_predictions, test_logits = run_epoch(model, test_loader, criterion, device)
    calibrated_predictions = (test_logits / temperature).argmax(1).tolist()
    calibrated_test_metrics = extended_metrics(test_targets, calibrated_predictions)

    model = model.cpu().eval()
    checkpoint_out = {
        "architecture": "mobilenet_v3_small",
        "class_names": CLASS_NAMES,
        "class_definitions": {"empty": "0%", "half-full": "1-74%", "full": "75-100%"},
        "image_size": args.image_size,
        "temperature": temperature,
        "state_dict": model.state_dict(),
        "ordinal_weight": args.ordinal_weight,
    }
    checkpoint_path = args.output_dir / "waste_bin_robust_v2.pth"
    script_path = args.output_dir / "waste_bin_robust_v2.torchscript.pt"
    torch.save(checkpoint_out, checkpoint_path)
    wrapper = TemperatureWrapper(model, temperature).eval()
    traced = torch.jit.trace(wrapper, torch.zeros(1, 3, args.image_size, args.image_size))
    traced.save(str(script_path))

    with (args.output_dir / "robust_v2_test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "actual", "predicted", "correct"])
        writer.writeheader()
        for sample, target, prediction in zip(test_samples, test_targets, calibrated_predictions):
            writer.writerow({"path": sample["path"].relative_to(args.data_dir).as_posix(), "actual": CLASS_NAMES[target], "predicted": CLASS_NAMES[prediction], "correct": target == prediction})

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "model": "MobileNetV3-Small with ordinal-aware loss and temperature calibration",
        "device": str(device),
        "parameter_count": parameter_count,
        "seed": args.seed,
        "epochs_completed": len(history),
        "elapsed_seconds": time.time() - started,
        "dataset": split_info,
        "best_validation_metrics": best_metrics,
        "temperature": temperature,
        "calibrated_validation_nll": calibrated_nll,
        "test_metrics_uncalibrated": test_metrics,
        "test_metrics": calibrated_test_metrics,
        "history": history,
    }
    (args.output_dir / "robust_v2_training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"temperature": temperature, "test_metrics": calibrated_test_metrics}, indent=2))
    print(f"Saved {script_path.resolve()}")


if __name__ == "__main__":
    main()
