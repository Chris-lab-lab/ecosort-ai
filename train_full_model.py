"""Train the experimental one-model EcoSort segmentation and classification network.

This file is deliberately separate from train.py. It consumes masks produced by
the offline EfficientViT-SAM teacher and exports one INT8 TFLite model with:

* a foreground object mask
* class probabilities (materials or the three dataset_new disposal routes)
* a supported/reject safety probability for material taxonomy only
* a compact feature embedding for future prototype rejection
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parent
os.environ.setdefault("KERAS_HOME", str(PROJECT / ".cache" / "keras"))

MATERIAL_ORDER = ("general", "paper", "metal", "plastic")


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path | None
    label: str
    supported: bool
    mask_weight: float


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an INT8 mask + class + optional validity MobileNetV2 model"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="segmentation_manifest.jsonl produced by prepare_segmented_dataset.py",
    )
    parser.add_argument(
        "--source-data",
        type=Path,
        help="original dataset root; normally read from segmentation_report.json",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--fine-tune-epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument(
        "--taxonomy", choices=("material", "disposal"), default="material",
        help="disposal predicts the three top-level dataset_new folders",
    )
    parser.add_argument(
        "--skip-unsupported-images", action="store_true",
        help="skip images whose actual format is unsupported by TensorFlow",
    )
    parser.add_argument("--validation-split", type=float, default=0.20)
    parser.add_argument("--mask-loss-weight", type=float, default=1.0)
    parser.add_argument("--validity-loss-weight", type=float, default=0.5)
    parser.add_argument("--eos-weight", type=float, default=0.15)
    parser.add_argument("--material-threshold", type=float, default=0.70)
    parser.add_argument("--validity-threshold", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--backbone-weights",
        choices=("imagenet", "none"),
        default="imagenet",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="train on generated shapes to validate build, INT8 export, and inference",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace only the explicitly selected output folder",
    )
    args = parser.parse_args()

    if not args.smoke_test and args.manifest is None:
        parser.error("--manifest is required unless --smoke-test is used")
    args.output = args.output or Path(
        "artifacts_full_smoke" if args.smoke_test else "artifacts_full"
    )
    args.epochs = args.epochs if args.epochs is not None else (1 if args.smoke_test else 15)
    args.fine_tune_epochs = (
        args.fine_tune_epochs
        if args.fine_tune_epochs is not None
        else (0 if args.smoke_test else 5)
    )
    args.batch_size = (
        args.batch_size if args.batch_size is not None else (4 if args.smoke_test else 32)
    )
    if args.image_size < 96 or args.image_size % 32:
        parser.error("--image-size must be at least 96 and divisible by 32")
    if args.embedding_size < 8 or args.batch_size < 1:
        parser.error("embedding size and batch size must be positive")
    if args.epochs < 1 or args.fine_tune_epochs < 0:
        parser.error("epochs must be positive and fine-tune epochs cannot be negative")
    for name in (
        "validation_split",
        "material_threshold",
        "validity_threshold",
    ):
        value = float(getattr(args, name))
        if not 0.0 < value < 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    return args


def _resolve(path_text: str, root: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else root / path


def load_samples(
    manifest_path: Path,
    source_data: Path | None,
    taxonomy: str = "material",
    skip_unsupported_images: bool = False,
) -> tuple[list[Sample], list[str]]:
    manifest_path = manifest_path.resolve()
    segmented_root = manifest_path.parent
    report_path = segmented_root / "segmentation_report.json"
    if source_data is None and report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        source_data = Path(report["source"])
    if source_data is None:
        raise SystemExit(
            "--source-data is required when segmentation_report.json has no source path"
        )
    source_root = source_data.resolve()
    if not source_root.is_dir():
        raise SystemExit(f"Original source dataset does not exist: {source_root}")

    samples: list[Sample] = []
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            label = str(record.get("label", "")).strip().lower()
            status = str(record.get("status", ""))
            if status not in {"accepted", "negative_passthrough"}:
                continue
            source = str(record["source"])
            if taxonomy == "disposal":
                group = Path(source.replace("\\", "/")).parts[0].casefold()
                allowed = {
                    "recyclable - others",
                    "recyclable - paper",
                    "regular trash",
                }
                if group not in allowed:
                    raise SystemExit(f"Manifest line {line_number} has unknown route: {source}")
                label = group
            elif not label:
                continue
            image = _resolve(source, source_root)
            mask_text = record.get("mask")
            mask = _resolve(str(mask_text), segmented_root) if mask_text else None
            if not image.is_file():
                raise SystemExit(
                    f"Manifest line {line_number} image does not exist: {image}"
                )
            if mask is not None and not mask.is_file():
                raise SystemExit(
                    f"Manifest line {line_number} mask does not exist: {mask}"
                )
            if skip_unsupported_images:
                with Image.open(image) as source_image:
                    actual_format = source_image.format
                if actual_format not in {"JPEG", "PNG", "GIF", "BMP"}:
                    print(f"Skipping unsupported {actual_format} image: {image}")
                    continue
            supported = bool(record.get("supported", label != "other"))
            if taxonomy == "disposal":
                supported = True
            elif label == "other":
                supported = False
            confidence = float(record.get("mask_confidence", 1.0))
            samples.append(
                Sample(
                    image=image,
                    mask=mask,
                    label=label,
                    supported=supported,
                    mask_weight=max(0.0, min(confidence, 1.0)) if mask else 0.0,
                )
            )

    labels = sorted({sample.label for sample in samples})
    if taxonomy == "disposal":
        expected = ["recyclable - others", "recyclable - paper", "regular trash"]
        if labels != expected:
            raise SystemExit(f"Disposal-route training needs all three groups; found {labels}")
        return samples, expected
    middle = [label for label in ("general", "paper") if label in labels]
    if len(middle) != 1 or any(label not in labels for label in ("metal", "plastic", "other")):
        counts = {label: sum(sample.label == label for sample in samples) for label in labels}
        raise SystemExit(
            "Full-model training needs exactly one of general/paper plus metal, plastic, "
            f"and other. Usable manifest counts: {counts}"
        )
    material_labels = [middle[0], "metal", "plastic"]
    return samples, material_labels


def split_samples(
    samples: list[Sample], validation_split: float, seed: int
) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(seed)
    grouped: dict[str, list[Sample]] = {}
    for sample in samples:
        grouped.setdefault(sample.label, []).append(sample)
    training: list[Sample] = []
    validation: list[Sample] = []
    for label, values in sorted(grouped.items()):
        rng.shuffle(values)
        if len(values) < 2:
            raise SystemExit(
                f"Class {label!r} needs at least two usable manifest records; found {len(values)}"
            )
        validation_count = max(1, round(len(values) * validation_split))
        validation_count = min(validation_count, len(values) - 1)
        validation.extend(values[:validation_count])
        training.extend(values[validation_count:])
    rng.shuffle(training)
    rng.shuffle(validation)
    return training, validation


def make_smoke_manifest(folder: Path) -> Path:
    rng = np.random.default_rng(14)
    source = folder / "source"
    masks = folder / "masks"
    manifest = folder / "segmentation_manifest.jsonl"
    colors = {
        "general": (205, 145, 65),
        "metal": (155, 175, 190),
        "plastic": (60, 150, 225),
        "other": (180, 65, 165),
    }
    records = []
    for label_index, (label, color) in enumerate(colors.items()):
        for index in range(6):
            pixels = rng.integers(20, 90, size=(128, 128, 3), dtype=np.uint8)
            yy, xx = np.ogrid[:128, :128]
            center_x = 55 + (index % 3) * 8
            center_y = 58 + (index % 2) * 10
            if label_index % 2:
                mask = (
                    (xx >= center_x - 25)
                    & (xx <= center_x + 25)
                    & (yy >= center_y - 34)
                    & (yy <= center_y + 34)
                )
            else:
                mask = ((xx - center_x) ** 2) / 30**2 + (
                    (yy - center_y) ** 2
                ) / 38**2 <= 1
            noise = rng.integers(-18, 19, size=(int(mask.sum()), 3))
            pixels[mask] = np.clip(np.asarray(color) + noise, 0, 255)
            image_path = source / label / f"{index}.png"
            mask_path = masks / label / f"{index}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pixels).save(image_path)
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
            records.append(
                {
                    "source": str(image_path.resolve()),
                    "label": label,
                    "status": "accepted",
                    "supported": label != "other",
                    "mask": str(mask_path.resolve()),
                    "mask_confidence": 0.95,
                }
            )
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return manifest


def make_dataset(
    tf: Any,
    samples: list[Sample],
    material_labels: list[str],
    *,
    image_size: int,
    batch_size: int,
    augment: bool,
    seed: int,
    material_weights: np.ndarray,
    validity_weights: np.ndarray,
) -> Any:
    label_to_index = {label: index for index, label in enumerate(material_labels)}
    image_paths = [str(sample.image) for sample in samples]
    mask_paths = [str(sample.mask) if sample.mask else "" for sample in samples]
    material_indices = [
        label_to_index.get(sample.label, 0) for sample in samples
    ]
    material_known_values = [
        float(sample.label in label_to_index) for sample in samples
    ]
    supported_values = [float(sample.supported) for sample in samples]
    mask_weights = [sample.mask_weight for sample in samples]
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            image_paths,
            mask_paths,
            material_indices,
            material_known_values,
            supported_values,
            mask_weights,
        )
    )
    if augment:
        dataset = dataset.shuffle(
            len(samples), seed=seed, reshuffle_each_iteration=True
        )

    material_weight_tensor = tf.constant(material_weights, dtype=tf.float32)
    validity_weight_tensor = tf.constant(validity_weights, dtype=tf.float32)
    material_count = len(material_labels)

    def decode(
        image_path: Any,
        mask_path: Any,
        material_index: Any,
        material_known: Any,
        supported: Any,
        mask_weight: Any,
    ) -> tuple[Any, Any, Any]:
        image = tf.io.decode_image(
            tf.io.read_file(image_path), channels=3, expand_animations=False
        )
        image.set_shape([None, None, 3])
        image = tf.image.resize(
            tf.cast(image, tf.float32),
            [image_size, image_size],
            antialias=True,
        )

        def read_mask() -> Any:
            value = tf.io.decode_image(
                tf.io.read_file(mask_path), channels=1, expand_animations=False
            )
            value.set_shape([None, None, 1])
            return tf.image.resize(
                tf.cast(value, tf.float32) / 255.0,
                [image_size // 4, image_size // 4],
                method="nearest",
            )

        mask = tf.cond(
            tf.strings.length(mask_path) > 0,
            read_mask,
            lambda: tf.zeros(
                [image_size // 4, image_size // 4, 1], dtype=tf.float32
            ),
        )
        if augment:
            flip = tf.random.uniform(()) > 0.5
            image = tf.cond(flip, lambda: tf.image.flip_left_right(image), lambda: image)
            mask = tf.cond(flip, lambda: tf.image.flip_left_right(mask), lambda: mask)
            image = tf.image.random_brightness(image, max_delta=22.0)
            image = tf.image.random_contrast(image, lower=0.82, upper=1.18)
            image = tf.clip_by_value(image, 0.0, 255.0)

        material_index = tf.cast(material_index, tf.int32)
        material_known = tf.cast(material_known, tf.float32)
        supported = tf.cast(supported, tf.float32)
        eos = tf.ones([material_count], dtype=tf.float32) / material_count
        targets = {
            "material": material_index,
            "eos_material": eos,
            "validity": supported[None],
            "object_mask": mask,
        }
        weights = {
            "material": material_known * tf.gather(
                material_weight_tensor, material_index
            ),
            "eos_material": (1.0 - material_known) * validity_weight_tensor[0],
            "validity": tf.gather(
                validity_weight_tensor, tf.cast(supported, tf.int32)
            ),
            "object_mask": tf.cast(mask_weight, tf.float32),
        }
        return image, targets, weights

    return (
        dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )


def build_models(
    tf: Any,
    *,
    image_size: int,
    embedding_size: int,
    material_count: int,
    weights: str | None,
) -> tuple[Any, Any, Any]:
    base = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        alpha=0.35,
        include_top=False,
        weights=weights,
    )
    base.trainable = False
    skip_extractor = tf.keras.Model(
        base.input,
        [base.get_layer("block_3_expand_relu").output, base.output],
        name="mobilenet_features",
    )

    inputs = tf.keras.Input((image_size, image_size, 3), name="rgb_0_255")
    normalized = tf.keras.layers.Rescaling(
        1 / 127.5, offset=-1, name="normalize"
    )(inputs)
    mask_features, final_features = skip_extractor(normalized, training=False)
    mask_features = tf.keras.layers.DepthwiseConv2D(
        3, padding="same", use_bias=False, name="mask_depthwise"
    )(mask_features)
    mask_features = tf.keras.layers.BatchNormalization(name="mask_bn")(
        mask_features
    )
    mask_features = tf.keras.layers.ReLU(max_value=6.0, name="mask_relu")(
        mask_features
    )
    mask_features = tf.keras.layers.Conv2D(
        24, 1, activation="relu", name="mask_projection"
    )(mask_features)
    object_mask = tf.keras.layers.Conv2D(
        1, 1, activation="sigmoid", name="object_mask"
    )(mask_features)

    pooled = tf.keras.layers.GlobalAveragePooling2D(
        name="global_features"
    )(final_features)
    embedding = tf.keras.layers.Dense(
        embedding_size, activation="relu", name="embedding"
    )(pooled)
    dropped = tf.keras.layers.Dropout(0.2, name="classification_dropout")(
        embedding
    )
    material = tf.keras.layers.Dense(
        material_count, activation="softmax", name="material"
    )(dropped)
    eos_material = tf.keras.layers.Activation(
        "linear", name="eos_material"
    )(material)
    validity = tf.keras.layers.Dense(
        1, activation="sigmoid", name="validity"
    )(dropped)

    training_model = tf.keras.Model(
        inputs,
        {
            "material": material,
            "eos_material": eos_material,
            "validity": validity,
            "object_mask": object_mask,
        },
        name="ecosort_full_training",
    )
    deployment_model = tf.keras.Model(
        inputs,
        {
            "material": material,
            "validity": validity,
            "object_mask": object_mask,
            "embedding": embedding,
        },
        name="ecosort_full_int8",
    )
    return training_model, deployment_model, base


def mask_loss(tf: Any) -> Any:
    def loss(y_true: Any, y_pred: Any) -> Any:
        epsilon = tf.keras.backend.epsilon()
        binary = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        binary = tf.reduce_mean(binary, axis=[1, 2, 3])
        intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        denominator = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3])
        dice = 1.0 - (2.0 * intersection + epsilon) / (
            denominator + epsilon
        )
        true_edges = tf.image.sobel_edges(y_true)
        predicted_edges = tf.image.sobel_edges(y_pred)
        boundary = tf.reduce_mean(
            tf.abs(true_edges - predicted_edges), axis=[1, 2, 3, 4]
        )
        return binary + dice + 0.20 * boundary

    return loss


def class_weights(
    samples: list[Sample], material_labels: list[str], taxonomy: str = "material"
) -> tuple[np.ndarray, np.ndarray]:
    material_counts = np.asarray(
        [
            sum(sample.label == label for sample in samples)
            for label in material_labels
        ],
        dtype=np.float32,
    )
    supported_count = float(sum(sample.supported for sample in samples))
    reject_count = float(len(samples) - supported_count)
    if taxonomy == "disposal":
        if np.any(material_counts == 0):
            raise SystemExit("Training split must contain every disposal route")
        material = material_counts.sum() / (len(material_labels) * material_counts)
        return material, np.ones(2, dtype=np.float32)
    if np.any(material_counts == 0) or supported_count == 0 or reject_count == 0:
        raise SystemExit("Training split must contain every material and reject examples")
    material = material_counts.sum() / (len(material_labels) * material_counts)
    validity = np.asarray(
        [
            len(samples) / (2.0 * reject_count),
            len(samples) / (2.0 * supported_count),
        ],
        dtype=np.float32,
    )
    return material, validity


def main() -> None:
    args = arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        if not args.overwrite:
            raise SystemExit(
                f"Output folder is not empty: {output}. Use --overwrite or choose a new folder."
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    temporary: Any = None
    if args.smoke_test:
        temporary = tempfile.TemporaryDirectory(prefix="ecosort-full-smoke-")
        smoke_root = Path(temporary.name)
        manifest = make_smoke_manifest(smoke_root)
        source_data = smoke_root
    else:
        manifest = args.manifest
        source_data = args.source_data

    samples, material_labels = load_samples(
        manifest,
        source_data,
        taxonomy=args.taxonomy,
        skip_unsupported_images=args.skip_unsupported_images,
    )
    training_samples, validation_samples = split_samples(
        samples, args.validation_split, args.seed
    )
    counts = {
        label: sum(sample.label == label for sample in samples)
        for label in sorted({sample.label for sample in samples})
    }
    masks_by_label = {
        label: sum(sample.label == label and sample.mask is not None for sample in samples)
        for label in counts
    }
    print(f"Usable samples: {counts}")
    print(f"Samples with teacher masks: {masks_by_label}")
    print(
        f"Split: {len(training_samples)} training / "
        f"{len(validation_samples)} validation"
    )

    import tensorflow as tf

    devices = tf.config.list_physical_devices("GPU")
    print(
        "TensorFlow compute devices: "
        + (", ".join(device.name for device in devices) if devices else "CPU only")
    )
    material_balance, validity_balance = class_weights(
        training_samples, material_labels, args.taxonomy
    )
    training = make_dataset(
        tf,
        training_samples,
        material_labels,
        image_size=args.image_size,
        batch_size=args.batch_size,
        augment=True,
        seed=args.seed,
        material_weights=material_balance,
        validity_weights=validity_balance,
    )
    validation = make_dataset(
        tf,
        validation_samples,
        material_labels,
        image_size=args.image_size,
        batch_size=args.batch_size,
        augment=False,
        seed=args.seed,
        material_weights=material_balance,
        validity_weights=validity_balance,
    )

    backbone_weights = (
        None
        if args.smoke_test or args.backbone_weights == "none"
        else args.backbone_weights
    )
    model, deployment_model, base = build_models(
        tf,
        image_size=args.image_size,
        embedding_size=args.embedding_size,
        material_count=len(material_labels),
        weights=backbone_weights,
    )

    def compile_model(learning_rate: float) -> None:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate),
            loss={
                "material": tf.keras.losses.SparseCategoricalCrossentropy(),
                "eos_material": tf.keras.losses.CategoricalCrossentropy(),
                "validity": tf.keras.losses.BinaryCrossentropy(),
                "object_mask": mask_loss(tf),
            },
            loss_weights={
                "material": 1.0,
                "eos_material": args.eos_weight if args.taxonomy == "material" else 0.0,
                "validity": args.validity_loss_weight if args.taxonomy == "material" else 0.0,
                "object_mask": args.mask_loss_weight,
            },
            weighted_metrics={
                "material": [
                    tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
                ],
                "validity": [
                    tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                    tf.keras.metrics.AUC(name="auc"),
                ],
            },
        )

    compile_model(1e-3)
    best_weights = output / "best.weights.h5"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            best_weights,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
        ),
    ]
    first = model.fit(
        training,
        validation_data=validation,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    histories = {
        name: list(values) for name, values in first.history.items()
    }

    if args.fine_tune_epochs:
        base.trainable = True
        for layer in base.layers[:-20]:
            layer.trainable = False
        compile_model(1e-5)
        start = len(first.history["loss"])
        second = model.fit(
            training,
            validation_data=validation,
            initial_epoch=start,
            epochs=start + args.fine_tune_epochs,
            callbacks=callbacks,
            verbose=2,
        )
        for name, values in second.history.items():
            histories.setdefault(name, []).extend(values)

    model.load_weights(best_weights)
    evaluation = model.evaluate(validation, verbose=0, return_dict=True)
    (output / "training_history.json").write_text(
        json.dumps(histories, indent=2), encoding="utf-8"
    )
    (output / "labels.txt").write_text(
        "\n".join(material_labels) + "\n", encoding="utf-8"
    )
    metadata = {
        "model_type": "ecosort_full_multitask_v1",
        "taxonomy": args.taxonomy,
        "architecture": "MobileNetV2-0.35 shared encoder",
        "outputs": ["object_mask", "material", "validity", "embedding"],
        "material_labels": material_labels,
        "input_size": args.image_size,
        "mask_size": args.image_size // 4,
        "material_threshold": args.material_threshold,
        "validity_threshold": args.validity_threshold,
        "smoke_test_only": bool(args.smoke_test),
        "validity_trained": args.taxonomy == "material",
        "routing_note": (
            "Disposal-route model is a visual preview only: no reject class or "
            "trained validity gate; never actuate a lid from it."
            if args.taxonomy == "disposal"
            else "Never actuate from smoke-test weights. A production lid may open only "
            "when validity and material thresholds both pass."
        ),
    }
    (output / "full_model.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    saved_model = output / "saved_model"
    deployment_model.export(saved_model)

    def representative_data() -> Any:
        for images, _, _ in training.unbatch().batch(1).take(100):
            yield [tf.cast(images, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    tflite = converter.convert()
    model_path = output / "full_waste_model_int8.tflite"
    model_path.write_bytes(tflite)

    from ecosort_ai.full_model import FullWasteModel

    runtime = FullWasteModel(
        model_path, output / "labels.txt", delegate=None
    )
    with Image.open(validation_samples[0].image) as image:
        prediction = runtime.predict_rgb(np.asarray(image.convert("RGB")))
    if prediction.object_mask.shape != (
        args.image_size // 4,
        args.image_size // 4,
    ):
        raise RuntimeError(
            f"Unexpected exported mask shape: {prediction.object_mask.shape}"
        )
    if not np.isfinite(prediction.object_mask).all():
        raise RuntimeError("Exported model produced a non-finite mask")

    summary = {
        **metadata,
        "manifest": str(manifest.resolve()),
        "sample_counts": counts,
        "mask_counts": masks_by_label,
        "training_samples": len(training_samples),
        "validation_samples": len(validation_samples),
        "evaluation": {
            name: float(value) for name, value in evaluation.items()
        },
        "int8_model_bytes": len(tflite),
        "int8_smoke_prediction": {
            "scores": prediction.scores,
            "supported_probability": prediction.supported_probability,
            "mask_min": float(prediction.object_mask.min()),
            "mask_max": float(prediction.object_mask.max()),
        },
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"INT8 model: {model_path}")
    if args.smoke_test:
        print(
            "PASS: synthetic architecture test only. These weights must not control a lid."
        )
    if temporary is not None:
        temporary.cleanup()


if __name__ == "__main__":
    main()
