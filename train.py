from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


REQUIRED_CLASSES = {"metal", "plastic"}
MIDDLE_CLASSES = {"paper", "general"}


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train and quantize the EcoSort image classifier")
    p.add_argument("--data", default="data", help="Folder containing one subfolder per class")
    p.add_argument(
        "--validation-data",
        help="optional independently collected folder with the same class subfolders",
    )
    p.add_argument("--output", default="artifacts")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--fine-tune-epochs", type=int, default=5)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--allow-no-other",
        action="store_true",
        help=(
            "allow a three-class plastic/metal/general-or-paper model; suitable for "
            "classification and dry runs, but not live lid actuation"
        ),
    )
    p.add_argument(
        "--no-class-balance",
        action="store_true",
        help="disable inverse-frequency class weights",
    )
    p.add_argument(
        "--require-gpu",
        action="store_true",
        help="stop instead of silently training on the CPU when no TensorFlow GPU is visible",
    )
    return p.parse_args()


def main() -> None:
    args = arguments()
    # Keep downloaded backbone weights inside this project by default.
    os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parent / ".cache" / "keras"))
    import tensorflow as tf

    tf.keras.utils.set_random_seed(args.seed)
    gpu_devices = tf.config.list_physical_devices("GPU")
    if args.require_gpu and not gpu_devices:
        raise SystemExit(
            "--require-gpu was specified, but TensorFlow cannot see a GPU. "
            "On Windows, use WSL2 or the supplied Docker GPU launcher."
        )
    for device in gpu_devices:
        try:
            tf.config.experimental.set_memory_growth(device, True)
        except RuntimeError:
            # TensorFlow raises when device initialization has already happened.
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
            data_dir,
            shuffle=True,
            seed=args.seed,
            **loader_options,
        )
        validation = tf.keras.utils.image_dataset_from_directory(
            Path(args.validation_data),
            class_names=train_raw.class_names,
            shuffle=False,
            **loader_options,
        )
        validation_source = str(Path(args.validation_data))
    else:
        # One call with subset="both" guarantees complementary partitions.
        # For credible contest metrics, prefer --validation-data so near-identical
        # frames from one capture session cannot leak into both partitions.
        train_raw, validation = tf.keras.utils.image_dataset_from_directory(
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
    if not has_other_class:
        print(
            "WARNING: training without an 'other' class. Use this model only for "
            "classification or dry-run testing; it cannot safely reject every unsupported item."
        )
    Path(output / "labels.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    class_counts = {
        name: sum(
            path.is_file() and path.suffix.lower() in image_extensions
            for path in (data_dir / name).rglob("*")
        )
        for name in class_names
    }
    if any(count == 0 for count in class_counts.values()):
        raise SystemExit(f"Every class needs at least one supported image; found {class_counts}")
    total_images = sum(class_counts.values())
    class_weights = {
        index: total_images / (len(class_names) * class_counts[name])
        for index, name in enumerate(class_names)
    }
    print(f"Images per class: {class_counts}")
    if args.no_class_balance:
        print("Class balancing: disabled")
    else:
        readable_weights = {
            class_names[index]: round(weight, 4) for index, weight in class_weights.items()
        }
        print(f"Class weights: {readable_weights}")

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.15),
        ],
        name="training_augmentation",
    )
    autotune = tf.data.AUTOTUNE
    training = train_raw.map(
        lambda images, labels: (augmentation(images, training=True), labels),
        num_parallel_calls=autotune,
    ).prefetch(autotune)
    validation = validation.prefetch(autotune)

    base = tf.keras.applications.MobileNetV2(
        input_shape=(args.image_size, args.image_size, 3),
        alpha=0.35,
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    inputs = tf.keras.Input((args.image_size, args.image_size, 3), name="rgb_0_255")
    x = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="normalize")(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax", name="category")(x)
    model = tf.keras.Model(inputs, outputs, name="ecosort_mobilenetv2")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(output / "best.keras", save_best_only=True),
    ]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    fit_class_weights = None if args.no_class_balance else class_weights
    history_a = model.fit(
        training,
        validation_data=validation,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=fit_class_weights,
        verbose=2,
    )

    # Short fine-tune: unfreeze only the final backbone layers.
    base.trainable = True
    for layer in base.layers[:-20]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    start = len(history_a.history["loss"])
    history_b = model.fit(
        training,
        validation_data=validation,
        initial_epoch=start,
        epochs=start + args.fine_tune_epochs,
        callbacks=callbacks,
        class_weight=fit_class_weights,
        verbose=2,
    )

    history = {name: list(values) for name, values in history_a.history.items()}
    for name, values in history_b.history.items():
        history.setdefault(name, []).extend(values)
    (output / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )

    # ModelCheckpoint keeps the best validation result across both fit calls.
    # Reload it so a fine-tuning regression is never the model we convert.
    model = tf.keras.models.load_model(output / "best.keras")
    loss, accuracy = model.evaluate(validation, verbose=0)
    model.export(output / "saved_model")

    # Fully integer model: required starting point for Ethos-U/Vela.
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
        """Smoke-test the exported artifact and measure its own accuracy."""

        import numpy as np

        interpreter = tf.lite.Interpreter(model_content=model_bytes, num_threads=2)
        interpreter.allocate_tensors()
        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        correct = 0
        total = 0
        confusion = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
        for images, labels in validation:
            for image, expected in zip(images.numpy(), labels.numpy(), strict=True):
                sample = image[None, ...].astype(np.float32)
                if np.issubdtype(input_detail["dtype"], np.integer):
                    scale, zero = input_detail["quantization"]
                    if scale == 0:
                        raise RuntimeError("converted integer input has no quantization scale")
                    limits = np.iinfo(input_detail["dtype"])
                    sample = np.clip(np.rint(sample / scale + zero), limits.min, limits.max).astype(
                        input_detail["dtype"]
                    )
                else:
                    sample = sample.astype(input_detail["dtype"])
                interpreter.set_tensor(input_detail["index"], sample)
                interpreter.invoke()
                scores = interpreter.get_tensor(output_detail["index"])[0]
                predicted = int(np.argmax(scores))
                expected_index = int(expected)
                confusion[expected_index, predicted] += 1
                correct += int(predicted == expected_index)
                total += 1
        if total == 0:
            raise RuntimeError("validation dataset is empty")
        return correct / total, confusion.tolist()

    quantized_accuracy, confusion_matrix = evaluate_tflite(quantized)

    metadata = {
        "classes": class_names,
        "middle_class": next(iter(middle)),
        "validation_accuracy": float(accuracy),
        "validation_loss": float(loss),
        "quantized_validation_accuracy": float(quantized_accuracy),
        "quantized_confusion_matrix": {
            "row_and_column_labels": class_names,
            "rows_are_actual_columns_are_predicted": confusion_matrix,
        },
        "input_size": args.image_size,
        "input_range": "uint8 RGB 0..255",
        "validation_source": validation_source,
        "training_image_counts": class_counts,
        "class_weights": None if args.no_class_balance else {
            class_names[index]: float(weight) for index, weight in class_weights.items()
        },
        "has_other_class": has_other_class,
        "routing_note": (
            "other never opens a lid"
            if has_other_class
            else "classification/dry-run only; live actuation requires an other class"
        ),
    }
    (output / "training_summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Model: {output / 'waste_classifier_int8.tflite'}")


if __name__ == "__main__":
    main()
