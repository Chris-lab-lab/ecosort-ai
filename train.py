from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from ecosort_ai.evaluation import (
    calibrate_class_confidence_thresholds,
    calibrate_validity_threshold,
    confusion_metrics,
)


REQUIRED_CLASSES = {"metal", "plastic"}
MIDDLE_CLASSES = {"paper", "general"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and fully quantize the open-set EcoSort image classifier"
    )
    parser.add_argument("--data", default="data", help="folder containing one subfolder per class")
    parser.add_argument(
        "--validation-data",
        help="optional independently collected folder with the same class subfolders",
    )
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=5)
    parser.add_argument(
        "--synthetic-reject-ratio",
        type=float,
        default=0.25,
        help="extra blurred/dark/empty/CutMix reject examples per training batch",
    )
    parser.add_argument(
        "--eos-weight",
        type=float,
        default=0.15,
        help="weight for entropic open-set loss on unknown examples",
    )
    parser.add_argument(
        "--deployment-confidence",
        type=float,
        default=0.75,
        help="global confidence floor used during exported-model evaluation",
    )
    parser.add_argument(
        "--deployment-margin",
        type=float,
        default=0.15,
        help="winner/runner-up margin used during exported-model evaluation",
    )
    parser.add_argument(
        "--target-route-precision",
        type=float,
        default=0.90,
        help="minimum validation precision requested for each material route",
    )
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument(
        "--prototype-samples-per-class",
        type=int,
        default=500,
        help="maximum training embeddings used for each material prototype",
    )
    parser.add_argument(
        "--prototype-percentile",
        type=float,
        default=97.5,
        help="known-class cosine-distance percentile used as the reject boundary",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--allow-no-other",
        action="store_true",
        help=(
            "allow a legacy three-class plastic/metal/general-or-paper model; suitable for "
            "classification and dry runs, but not live lid actuation"
        ),
    )
    parser.add_argument(
        "--legacy-single-head",
        action="store_true",
        help="train the older four-class softmax instead of the safer V2 open-set model",
    )
    parser.add_argument(
        "--no-class-balance",
        action="store_true",
        help="disable inverse-frequency sample weights",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="stop instead of silently training on the CPU when no TensorFlow GPU is visible",
    )
    args = parser.parse_args()
    if args.image_size < 32:
        parser.error("--image-size must be at least 32")
    if args.batch_size < 1 or args.epochs < 1 or args.fine_tune_epochs < 0:
        parser.error("batch size and epochs must be positive; fine-tune epochs may be zero")
    if args.embedding_size < 4:
        parser.error("--embedding-size must be at least 4")
    if not 0.0 <= args.synthetic_reject_ratio <= 1.0 or args.eos_weight < 0.0:
        parser.error("synthetic reject ratio must be in 0..1 and EOS weight cannot be negative")
    if not 0.0 <= args.deployment_confidence <= 1.0:
        parser.error("--deployment-confidence must be between 0 and 1")
    if not 0.0 <= args.deployment_margin <= 1.0:
        parser.error("--deployment-margin must be between 0 and 1")
    if not 0.0 < args.target_route_precision <= 1.0:
        parser.error("--target-route-precision must be greater than 0 and at most 1")
    if args.prototype_samples_per_class < 1:
        parser.error("--prototype-samples-per-class must be at least 1")
    if not 50.0 <= args.prototype_percentile <= 100.0:
        parser.error("--prototype-percentile must be between 50 and 100")
    return args


def _class_counts(data_dir: Path, class_names: list[str]) -> dict[str, int]:
    return {
        name: sum(
            path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            for path in (data_dir / name).rglob("*")
        )
        for name in class_names
    }


def _quantize_input(sample: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    if np.issubdtype(detail["dtype"], np.integer):
        scale, zero = detail["quantization"]
        if not scale:
            raise RuntimeError("converted integer input has no quantization scale")
        limits = np.iinfo(detail["dtype"])
        sample = np.clip(np.rint(sample / scale + zero), limits.min, limits.max)
    return sample.astype(detail["dtype"])


def _dequantize_output(value: np.ndarray, detail: dict[str, Any]) -> np.ndarray:
    if np.issubdtype(detail["dtype"], np.integer):
        scale, zero = detail["quantization"]
        return (value.astype(np.float32) - zero) * scale
    return value.astype(np.float32)


def _normalise_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def _compute_prototypes(
    embedding_model: Any,
    dataset: Any,
    class_names: list[str],
    material_names: list[str],
    *,
    samples_per_class: int,
    percentile: float,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    collected: dict[str, list[np.ndarray]] = {name: [] for name in material_names}
    for images, labels in dataset:
        embeddings = _normalise_rows(
            np.asarray(embedding_model(images, training=False), dtype=np.float32)
        )
        for embedding, original_index in zip(embeddings, labels.numpy(), strict=True):
            label = class_names[int(original_index)]
            if label in collected and len(collected[label]) < samples_per_class:
                collected[label].append(embedding)
        if all(len(rows) >= samples_per_class for rows in collected.values()):
            break

    prototypes: dict[str, list[float]] = {}
    thresholds: dict[str, float] = {}
    for label, rows in collected.items():
        if not rows:
            raise RuntimeError(f"No training embeddings were collected for {label}")
        matrix = np.stack(rows)
        prototype = np.mean(matrix, axis=0)
        prototype /= max(float(np.linalg.norm(prototype)), 1e-8)
        distances = np.maximum(0.0, 1.0 - matrix @ prototype)
        # Quantization allowance prevents a zero-width boundary on tiny datasets.
        boundary = min(2.0, float(np.percentile(distances, percentile)) + 0.02)
        prototypes[label] = [float(value) for value in prototype]
        thresholds[label] = boundary
    return prototypes, thresholds


def _open_set_prediction(
    material_scores: np.ndarray,
    supported_probability: float,
    embedding: np.ndarray,
    material_names: list[str],
    validity_threshold: float,
    prototypes: dict[str, list[float]],
    prototype_thresholds: dict[str, float],
    confidence_thresholds: dict[str, float],
    margin_threshold: float,
) -> str:
    order = np.argsort(material_scores)[::-1]
    label = material_names[int(order[0])]
    confidence = float(material_scores[order[0]])
    runner_up = float(material_scores[order[1]])
    normalized = embedding / max(float(np.linalg.norm(embedding)), 1e-8)
    distance = max(0.0, 1.0 - float(np.dot(normalized, prototypes[label])))
    if (
        supported_probability < validity_threshold
        or distance > prototype_thresholds[label]
        or confidence < confidence_thresholds[label]
        or confidence - runner_up < margin_threshold
    ):
        return "other"
    return label


def main() -> None:
    args = arguments()
    os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parent / ".cache" / "keras"))
    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    gpu_devices = tf.config.list_physical_devices("GPU")
    if args.require_gpu and not gpu_devices:
        raise SystemExit(
            "--require-gpu was specified, but TensorFlow cannot see a GPU. "
            "Native Windows TensorFlow runs on the CPU; use a supported Linux/WSL2 GPU setup."
        )
    for device in gpu_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            pass
    print(
        "TensorFlow compute devices: "
        + (", ".join(device.name for device in gpu_devices) if gpu_devices else "CPU only")
    )

    data_dir = Path(args.data)
    if args.validation_data:
        validation_dir = Path(args.validation_data).resolve()
        training_dir = data_dir.resolve()
        if (
            validation_dir == training_dir
            or validation_dir.is_relative_to(training_dir)
            or training_dir.is_relative_to(validation_dir)
        ):
            raise SystemExit("Use separate, non-overlapping training and validation folders.")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    loader_options = {
        "image_size": (args.image_size, args.image_size),
        "batch_size": args.batch_size,
        "label_mode": "int",
    }
    if args.validation_data:
        train_raw = tf.keras.utils.image_dataset_from_directory(
            data_dir, shuffle=True, seed=args.seed, **loader_options
        )
        validation_raw = tf.keras.utils.image_dataset_from_directory(
            Path(args.validation_data),
            class_names=train_raw.class_names,
            shuffle=False,
            **loader_options,
        )
        validation_source = str(Path(args.validation_data))
    else:
        train_raw, validation_raw = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="both",
            seed=args.seed,
            **loader_options,
        )
        validation_source = "20% file-level split from training data"

    class_names = [name.lower() for name in train_raw.class_names]
    found = set(class_names)
    middle = found & MIDDLE_CLASSES
    expected = REQUIRED_CLASSES | middle
    valid_four_class = found == expected | {"other"}
    valid_three_class = args.allow_no_other and found == expected
    if len(middle) != 1 or not (valid_four_class or valid_three_class):
        raise SystemExit(
            "Expected plastic, metal, other, and exactly one middle class (paper or general); "
            f"found {class_names}. For classification-only training without an other class, "
            "pass --allow-no-other. Do not train paper and general in the same model."
        )

    has_other_class = "other" in found
    open_set_v2 = has_other_class and not args.legacy_single_head
    material_names = [name for name in class_names if name != "other"] if open_set_v2 else class_names
    (output / "labels.txt").write_text("\n".join(material_names) + "\n", encoding="utf-8")
    if not has_other_class:
        print("WARNING: no 'other' class; this model is classification/dry-run only.")
    elif open_set_v2:
        print("EcoSort V2: material head + supported-item head + prototype rejection")
    else:
        print("Legacy mode: one four-class softmax output")

    class_counts = _class_counts(data_dir, class_names)
    if any(count == 0 for count in class_counts.values()):
        raise SystemExit(f"Every class needs at least one supported image; found {class_counts}")
    total_images = sum(class_counts.values())
    class_weights = {
        index: total_images / (len(class_names) * class_counts[name])
        for index, name in enumerate(class_names)
    }
    print(f"Images per class: {class_counts}")
    material_weight_summary: dict[str, float] | None = None
    validity_weight_summary: dict[str, float] | None = None

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.15),
            tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255)),
        ],
        name="training_augmentation",
    )
    autotune = tf.data.AUTOTUNE

    base = tf.keras.applications.MobileNetV2(
        input_shape=(args.image_size, args.image_size, 3),
        alpha=0.35,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    inputs = tf.keras.Input((args.image_size, args.image_size, 3), name="rgb_0_255")
    features = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="normalize")(inputs)
    features = base(features, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D(name="global_features")(features)

    if open_set_v2:
        embedding = tf.keras.layers.Dense(
            args.embedding_size, activation="relu", name="embedding_features"
        )(features)
        dropped = tf.keras.layers.Dropout(0.2)(embedding)
        material_output = tf.keras.layers.Dense(
            len(material_names), activation="softmax", name="material"
        )(dropped)
        # The duplicate training-only output applies EOS loss to reject examples
        # while the normal material loss/metric ignores them. It is removed from
        # the deployment model below.
        eos_output = tf.keras.layers.Activation("linear", name="eos_material")(
            material_output
        )
        validity_output = tf.keras.layers.Dense(1, activation="sigmoid", name="validity")(dropped)
        model = tf.keras.Model(
            inputs,
            {
                "material": material_output,
                "eos_material": eos_output,
                "validity": validity_output,
            },
            name="ecosort_v2_mobilenetv2",
        )

        original_to_material = np.zeros(len(class_names), dtype=np.int32)
        for original_index, name in enumerate(class_names):
            if name in material_names:
                original_to_material[original_index] = material_names.index(name)
        material_lookup = tf.constant(original_to_material)
        other_index = class_names.index("other")
        known_count = total_images - class_counts["other"]
        material_weights = np.zeros(len(class_names), dtype=np.float32)
        for original_index, name in enumerate(class_names):
            if name in material_names:
                material_weights[original_index] = known_count / (
                    len(material_names) * class_counts[name]
                )
        synthetic_reject_count = int(round(total_images * args.synthetic_reject_ratio))
        effective_other_count = class_counts["other"] + synthetic_reject_count
        effective_total = known_count + effective_other_count
        validity_weights = np.ones(2, dtype=np.float32)
        if not args.no_class_balance:
            validity_weights = np.asarray(
                [
                    effective_total / (2 * effective_other_count),
                    effective_total / (2 * known_count),
                ],
                dtype=np.float32,
            )
        if not args.no_class_balance:
            material_weight_summary = {
                name: float(material_weights[class_names.index(name)])
                for name in material_names
            }
            validity_weight_summary = {
                "reject": float(validity_weights[0]),
                "supported": float(validity_weights[1]),
            }
        material_weight_lookup = tf.constant(material_weights)
        validity_weight_lookup = tf.constant(validity_weights)
        synthetic_per_batch = int(round(args.batch_size * args.synthetic_reject_ratio))

        def create_synthetic_rejects(images: Any) -> Any:
            count = tf.minimum(tf.shape(images)[0], synthetic_per_batch)
            source = images[:count]
            partner = tf.reverse(source, axis=[0])
            midpoint = tf.shape(source)[2] // 2
            cutmix = tf.concat([source[:, :, :midpoint, :], partner[:, :, midpoint:, :]], axis=2)
            blurred = tf.nn.avg_pool2d(source, ksize=31, strides=1, padding="SAME")
            dark = tf.clip_by_value(source * 0.08, 0.0, 255.0)
            empty = tf.ones_like(source) * tf.reduce_mean(source, axis=[1, 2], keepdims=True)
            modes = tf.math.mod(tf.range(count), 4)
            result = tf.where((modes == 0)[:, None, None, None], cutmix, blurred)
            result = tf.where((modes == 2)[:, None, None, None], dark, result)
            return tf.where((modes == 3)[:, None, None, None], empty, result)

        def prepare(images: Any, labels: Any, *, augment: bool) -> tuple[Any, Any, Any]:
            images = augmentation(images, training=True) if augment else images
            if augment and synthetic_per_batch:
                synthetic = create_synthetic_rejects(images)
                images = tf.concat([images, synthetic], axis=0)
                labels = tf.concat(
                    [labels, tf.fill([tf.shape(synthetic)[0]], tf.cast(other_index, labels.dtype))],
                    axis=0,
                )
            supported = tf.cast(tf.not_equal(labels, other_index), tf.float32)
            eos_targets = tf.ones(
                [tf.shape(labels)[0], len(material_names)], dtype=tf.float32
            ) / len(material_names)
            targets = {
                "material": tf.gather(material_lookup, labels),
                "eos_material": eos_targets,
                "validity": supported[:, None],
            }
            material_sample_weight = (
                supported
                if args.no_class_balance
                else tf.gather(material_weight_lookup, labels)
            )
            sample_weights = {
                "material": material_sample_weight,
                "eos_material": (1.0 - supported)
                * tf.gather(validity_weight_lookup, tf.zeros_like(labels)),
                "validity": tf.gather(validity_weight_lookup, tf.cast(supported, tf.int32)),
            }
            return images, targets, sample_weights

        training = train_raw.map(
            lambda images, labels: prepare(images, labels, augment=True),
            num_parallel_calls=autotune,
        ).prefetch(autotune)
        validation = validation_raw.map(
            lambda images, labels: prepare(images, labels, augment=False),
            num_parallel_calls=autotune,
        ).prefetch(autotune)

        def compile_model(learning_rate: float) -> None:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss={
                    "material": tf.keras.losses.SparseCategoricalCrossentropy(),
                    "eos_material": tf.keras.losses.CategoricalCrossentropy(),
                    "validity": tf.keras.losses.BinaryCrossentropy(),
                },
                loss_weights={
                    "material": 1.0,
                    "eos_material": args.eos_weight,
                    "validity": 0.5,
                },
                weighted_metrics={
                    "material": [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
                    "validity": [
                        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                        tf.keras.metrics.AUC(name="auc"),
                    ],
                },
            )

        compile_model(1e-3)
        fit_class_weights = None
    else:
        dropped = tf.keras.layers.Dropout(0.2)(features)
        category_output = tf.keras.layers.Dense(
            len(class_names), activation="softmax", name="category"
        )(dropped)
        model = tf.keras.Model(inputs, category_output, name="ecosort_mobilenetv2")
        training = train_raw.map(
            lambda images, labels: (augmentation(images, training=True), labels),
            num_parallel_calls=autotune,
        ).prefetch(autotune)
        validation = validation_raw.prefetch(autotune)

        def compile_model(learning_rate: float) -> None:
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=["accuracy"],
            )

        compile_model(1e-3)
        fit_class_weights = None if args.no_class_balance else class_weights
        if fit_class_weights is not None:
            material_weight_summary = {
                class_names[index]: float(weight) for index, weight in class_weights.items()
            }

    if args.no_class_balance:
        print("Class balancing: disabled")
    else:
        print(f"Material class weights: {material_weight_summary}")
        if validity_weight_summary is not None:
            print(f"Validity class weights: {validity_weight_summary}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(
            output / "best.keras", monitor="val_loss", save_best_only=True
        ),
    ]
    history_a = model.fit(
        training,
        validation_data=validation,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=fit_class_weights,
        verbose=2,
    )

    history_b: dict[str, list[float]] = {}
    if args.fine_tune_epochs:
        base.trainable = True
        for layer in base.layers[:-20]:
            layer.trainable = False
        compile_model(1e-5)
        start = len(history_a.history["loss"])
        fitted = model.fit(
            training,
            validation_data=validation,
            initial_epoch=start,
            epochs=start + args.fine_tune_epochs,
            callbacks=callbacks,
            class_weight=fit_class_weights,
            verbose=2,
        )
        history_b = fitted.history

    history = {name: list(values) for name, values in history_a.history.items()}
    for name, values in history_b.items():
        history.setdefault(name, []).extend(values)
    (output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    model = tf.keras.models.load_model(output / "best.keras")
    evaluation = model.evaluate(validation, verbose=0, return_dict=True)
    validity_threshold: float | None = None
    validity_balanced_accuracy: float | None = None
    prototypes: dict[str, list[float]] = {}
    prototype_thresholds: dict[str, float] = {}
    material_confidence_thresholds = {
        name: args.deployment_confidence for name in material_names
    }
    material_threshold_reports: dict[str, dict[str, float | int]] = {}
    embedding_model: Any | None = None

    if open_set_v2:
        validation_validity: list[float] = []
        validation_targets: list[int] = []
        validation_material: list[list[float]] = []
        validation_material_targets: list[int] = []
        for images, original_labels in validation_raw:
            outputs = model(images, training=False)
            validation_validity.extend(np.asarray(outputs["validity"]).reshape(-1).tolist())
            validation_targets.extend(
                (original_labels.numpy() != class_names.index("other")).astype(np.int8).tolist()
            )
            validation_material.extend(np.asarray(outputs["material"]).tolist())
            validation_material_targets.extend(
                [
                    material_names.index(class_names[int(index)])
                    if class_names[int(index)] in material_names
                    else -1
                    for index in original_labels.numpy()
                ]
            )
        validity_threshold, validity_balanced_accuracy = calibrate_validity_threshold(
            validation_validity, validation_targets
        )
        calibrated_thresholds, threshold_reports = calibrate_class_confidence_thresholds(
            validation_material,
            validation_material_targets,
            minimum_precision=args.target_route_precision,
            floor=args.deployment_confidence,
        )
        material_confidence_thresholds = dict(
            zip(material_names, calibrated_thresholds, strict=True)
        )
        material_threshold_reports = dict(zip(material_names, threshold_reports, strict=True))
        embedding_model = tf.keras.Model(
            model.input, model.get_layer("embedding_features").output
        )
        prototypes, prototype_thresholds = _compute_prototypes(
            embedding_model,
            train_raw,
            class_names,
            material_names,
            samples_per_class=args.prototype_samples_per_class,
            percentile=args.prototype_percentile,
        )
        normalized_embedding = tf.keras.layers.UnitNormalization(axis=-1, name="embedding")(
            model.get_layer("embedding_features").output
        )
        deployment_model = tf.keras.Model(
            model.input,
            [
                model.get_layer("material").output,
                model.get_layer("validity").output,
                normalized_embedding,
            ],
            name="ecosort_v2_deployment",
        )
        open_set_metadata = {
            "schema_version": 1,
            "model_type": "ecosort_v2_open_set",
            "material_labels": material_names,
            "validity_threshold": validity_threshold,
            "validity_balanced_accuracy": validity_balanced_accuracy,
            "prototype_metric": "cosine_distance",
            "prototype_percentile": args.prototype_percentile,
            "prototypes": prototypes,
            "prototype_thresholds": prototype_thresholds,
            "material_confidence_thresholds": material_confidence_thresholds,
            "material_threshold_calibration": material_threshold_reports,
            "target_route_precision": args.target_route_precision,
            "deployment_margin": args.deployment_margin,
            "training": {
                "eos_weight": args.eos_weight,
                "synthetic_reject_ratio": args.synthetic_reject_ratio,
            },
        }
        (output / "open_set.json").write_text(
            json.dumps(open_set_metadata, indent=2), encoding="utf-8"
        )
    else:
        deployment_model = model
        stale_metadata = output / "open_set.json"
        if stale_metadata.exists():
            stale_metadata.unlink()

    def evaluate_float_model() -> tuple[float, list[list[int]]]:
        confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        for images, labels in validation_raw:
            outputs = model(images, training=False)
            if open_set_v2:
                material_batch = np.asarray(outputs["material"])
                validity_batch = np.asarray(outputs["validity"]).reshape(-1)
                embedding_batch = np.asarray(embedding_model(images, training=False))
                predicted_labels = [
                    _open_set_prediction(
                        material_scores,
                        float(supported_probability),
                        embedding_value,
                        material_names,
                        float(validity_threshold),
                        prototypes,
                        prototype_thresholds,
                        material_confidence_thresholds,
                        args.deployment_margin,
                    )
                    for material_scores, supported_probability, embedding_value in zip(
                        material_batch, validity_batch, embedding_batch, strict=True
                    )
                ]
                predicted_indices = [class_names.index(label) for label in predicted_labels]
            else:
                predicted_indices = np.argmax(np.asarray(outputs), axis=1).tolist()
            for expected_index, predicted_index in zip(
                labels.numpy(), predicted_indices, strict=True
            ):
                confusion[int(expected_index), int(predicted_index)] += 1
        total = int(confusion.sum())
        if not total:
            raise RuntimeError("validation dataset is empty")
        return float(np.trace(confusion) / total), confusion.tolist()

    float_accuracy, float_confusion_matrix = evaluate_float_model()

    deployment_model.export(output / "saved_model")

    def representative_data():
        for images, _ in train_raw.unbatch().batch(1).take(150):
            yield [tf.cast(images, tf.float32)]

    converter = tf.lite.TFLiteConverter.from_saved_model(str(output / "saved_model"))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    quantized = converter.convert()
    (output / "waste_classifier_int8.tflite").write_bytes(quantized)

    def evaluate_tflite(model_bytes: bytes) -> tuple[float, list[list[int]]]:
        interpreter = tf.lite.Interpreter(model_content=model_bytes, num_threads=2)
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()
        if open_set_v2:
            material_detail = next(
                detail
                for detail in output_details
                if int(np.prod(detail["shape"][1:])) == len(material_names)
            )
            validity_detail = next(
                detail
                for detail in output_details
                if int(np.prod(detail["shape"][1:])) == 1
            )
            embedding_detail = next(
                detail
                for detail in output_details
                if int(np.prod(detail["shape"][1:])) == args.embedding_size
            )
        else:
            material_detail = output_details[0]
            validity_detail = None
            embedding_detail = None

        confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        for images, labels in validation_raw:
            for image, expected_index in zip(images.numpy(), labels.numpy(), strict=True):
                sample = _quantize_input(image[None, ...].astype(np.float32), input_detail)
                interpreter.set_tensor(input_detail["index"], sample)
                interpreter.invoke()
                material_scores = _dequantize_output(
                    interpreter.get_tensor(material_detail["index"])[0], material_detail
                ).reshape(-1)
                if open_set_v2:
                    supported_probability = float(
                        _dequantize_output(
                            interpreter.get_tensor(validity_detail["index"])[0], validity_detail
                        ).reshape(-1)[0]
                    )
                    embedding_value = _dequantize_output(
                        interpreter.get_tensor(embedding_detail["index"])[0], embedding_detail
                    ).reshape(-1)
                    predicted_label = _open_set_prediction(
                        material_scores,
                        supported_probability,
                        embedding_value,
                        material_names,
                        float(validity_threshold),
                        prototypes,
                        prototype_thresholds,
                        material_confidence_thresholds,
                        args.deployment_margin,
                    )
                    predicted_index = class_names.index(predicted_label)
                else:
                    predicted_index = int(np.argmax(material_scores))
                confusion[int(expected_index), predicted_index] += 1

        total = int(confusion.sum())
        if not total:
            raise RuntimeError("validation dataset is empty")
        return float(np.trace(confusion) / total), confusion.tolist()

    quantized_accuracy, confusion_matrix = evaluate_tflite(quantized)
    metrics = confusion_metrics(confusion_matrix)
    metrics["per_class"] = {
        label: values for label, values in zip(class_names, metrics["per_class"], strict=True)
    }
    other_index = class_names.index("other") if has_other_class else None
    other_total = sum(confusion_matrix[other_index]) if other_index is not None else 0
    unknown_false_acceptance_rate = (
        (other_total - confusion_matrix[other_index][other_index]) / other_total
        if other_index is not None and other_total
        else None
    )
    wrong_lid_activations = sum(
        count
        for actual, row in enumerate(confusion_matrix)
        for predicted, count in enumerate(row)
        if actual != predicted and (other_index is None or predicted != other_index)
    )

    summary = {
        "model_type": "ecosort_v2_open_set" if open_set_v2 else "legacy_single_head",
        "classes": class_names,
        "material_classes": material_names,
        "middle_class": next(iter(middle)),
        "validation_accuracy": float_accuracy,
        "validation_loss": float(evaluation["loss"]),
        "training_model_evaluation": {
            name: float(value) for name, value in evaluation.items()
        },
        "validation_confusion_matrix": {
            "row_and_column_labels": class_names,
            "rows_are_actual_columns_are_predicted": float_confusion_matrix,
        },
        "quantized_validation_accuracy": quantized_accuracy,
        "quantized_metrics": metrics,
        "unknown_false_acceptance_rate": unknown_false_acceptance_rate,
        "wrong_lid_activation_rate": wrong_lid_activations / max(sum(map(sum, confusion_matrix)), 1),
        "validity_threshold": validity_threshold,
        "validity_balanced_accuracy": validity_balanced_accuracy,
        "material_confidence_thresholds": material_confidence_thresholds,
        "material_threshold_calibration": material_threshold_reports,
        "deployment_margin": args.deployment_margin,
        "eos_weight": args.eos_weight if open_set_v2 else None,
        "synthetic_reject_ratio": args.synthetic_reject_ratio if open_set_v2 else None,
        "quantized_confusion_matrix": {
            "row_and_column_labels": class_names,
            "rows_are_actual_columns_are_predicted": confusion_matrix,
        },
        "input_size": args.image_size,
        "input_range": "uint8 RGB 0..255",
        "validation_source": validation_source,
        "training_image_counts": class_counts,
        "class_weights": material_weight_summary,
        "validity_class_weights": validity_weight_summary,
        "has_other_class": has_other_class,
        "routing_note": (
            "validity and prototype rejection never open a lid"
            if open_set_v2
            else (
                "other never opens a lid"
                if has_other_class
                else "classification/dry-run only; live actuation requires unknown rejection"
            )
        ),
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Model: {output / 'waste_classifier_int8.tflite'}")


if __name__ == "__main__":
    main()
