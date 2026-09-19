# Waste-bin AI improvement report

## Outcome

The previous custom classifier was upgraded into a robustness-oriented MobileNetV3-Small system while keeping the three requested ordered classes:

- `empty`: exactly 0%
- `half-full`: 1–74%
- `full`: 75–100%

The new system adds conservative preprocessing, multi-view inference, an ordinal-aware loss, reproducible dataset generation, controlled robustness evaluation, and ONNX export for the NXP path. Its deployed output is restricted to exactly `empty`, `half-full`, or `full`.

## Dataset audit and cleaning

The original preparation stage received 254 files and removed 14 exact duplicate images, leaving 240 unique originals:

| Class | Unique originals |
|---|---:|
| Empty | 46 |
| Half-full | 96 |
| Full | 98 |

The audit found no corrupt files, flagged 9 blurry images and 9 ambiguous low-confidence model predictions, and found no cross-split near-duplicate pair at the configured perceptual-hash threshold. Manual visual review identified 12 related training images labeled `full` that did not safely demonstrate the new 75% threshold. They were excluded from training and placed in `manual-review`; they were not automatically relabeled.

The final v2 layout is:

| Split | Empty | Half-full | Full | Total | Synthetic allowed? |
|---|---:|---:|---:|---:|---|
| Train | 600 | 600 | 600 | 1,800 | Yes, training only |
| Validation | 8 | 16 | 16 | 40 | No |
| Test | 8 | 16 | 16 | 40 | No |
| Manual review | 0 | 0 | 12 | 12 | No |

The 1,800-image training split contains 148 originals and 1,652 synthetic variants. Synthetic samples inherit only from training originals and never enter validation or test. Nearby source photographs remain grouped by the earlier split assignment.

## Synthetic coverage

The generator simulates label-preserving capture variation:

- close, medium, and far apparent bin scale
- zoom, padding, translation, and off-center framing
- small rotation and perspective changes
- brightness, contrast, saturation, and color-temperature variation
- shadows and highlights/reflections
- blur, sensor noise, JPEG degradation, and reduced resolution
- minor edge obstruction
- randomized blurred/tinted surroundings

The ranges are configurable in `generate_robust_dataset.py` and recorded with the seed in `evaluation/robust_dataset_generation_report.json`. Severe transformations that could change the semantic fill level are intentionally avoided.

## Model and training

- Architecture: MobileNetV3-Small
- Parameters: 1,520,931
- Initialization: earlier public-data checkpoint
- Fine-tuning: final five feature blocks plus classifier
- Loss: cross-entropy plus expected ordinal class-distance penalty
- Regularization: dropout 0.35, AdamW weight decay, augmentation, gradient clipping, early stopping
- Calibration: validation-selected temperature = 0.5
- Reproducibility seed: 20260919
- Training completed: 9 epochs before early stopping

The ordinal term penalizes the severe `empty`↔`full` error more strongly than an adjacent-class mistake.

## Independent original-only test

| Metric | Previous model | Robust v2 |
|---|---:|---:|
| Accuracy | 97.5% | 100.0% |
| Macro-F1 | 0.968 | 1.000 |
| Weighted F1 | 0.975 | 1.000 |
| Empty precision / recall / F1 | 1.000 / 0.875 / 0.933 | 1.000 / 1.000 / 1.000 |
| Half-full precision / recall / F1 | 0.941 / 1.000 / 0.970 | 1.000 / 1.000 / 1.000 |
| Full precision / recall / F1 | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |

Previous confusion matrix (`empty`, `half-full`, `full`):

```text
[[7, 1, 0],
 [0,16, 0],
 [0, 0,16]]
```

Robust v2 confusion matrix:

```text
[[8, 0, 0],
 [0,16, 0],
 [0, 0,16]]
```

## Controlled robustness comparison

Each condition contains the same 40 held-out originals. The deployed version always returns its best-scoring one of the three classes.

| Condition | Previous model | Robust v2 |
|---|---:|---:|
| Medium/original | 97.5% | 100.0% |
| Close | 100.0% | 100.0% |
| Far | 67.5% | 100.0% |
| Angle | 100.0% | 100.0% |
| Lighting | 97.5% | 97.5% |
| Background/position | 77.5% | 100.0% |
| Low resolution | 82.5% | 100.0% |
| Blur/noise | 100.0% | 100.0% |

Across the 280 transformed (non-medium) views:

- Previous model: 89.3% accuracy and 88.2% consistency.
- Robust v2: 99.6% accuracy and 99.6% consistency.
- One altered-lighting view was the only changed prediction in the transformed evaluation.

The transformed tests are controlled simulations, not independent physical recordings.

## Inference behavior

Inference preserves the full image with letterboxing and evaluates three views: full frame, padded/far view, and a conservative ROI proposal. Probabilities are averaged and the highest-scoring class is returned.

The user-facing command and webcam return only `empty`, `half-full`, or `full`. They do not display an estimated fill percentage or create an additional uncertainty category.

## Size and measured desktop speed

| Artifact | Size |
|---|---:|
| Previous TorchScript | 6,505,980 bytes |
| Robust v2 TorchScript | 6,510,297 bytes |
| Robust v2 ONNX | 6,096,230 bytes |
| Dynamic-INT8 TorchScript CPU reference | 4,759,031 bytes |

Measured CPU single-view export benchmark was about 20.2 ms for float and 20.9 ms for dynamic-INT8. Robust multi-view evaluation averaged about 31.4 ms per image in the evaluation environment. These are desktop timings and are not i.MX93 measurements.

## Failure patterns and remaining risks

- Far-away bins and low-resolution frames remain the riskiest conditions and should be minimized in the camera setup.
- Extreme lighting caused the only raw changed prediction in the transformed evaluation.
- The dataset contains strongly related capture sessions and bin appearances, so shortcut learning is still possible.
- The 40-image test set is small and comes from the same broad collection process.
- The exact 0% and 75% boundaries cannot always be judged from one perspective; those samples need human review or a measured sensor/reference.
- A new real deployment capture session remains the most valuable next data collection step.

## NXP FRDM i.MX93 preparation

The exported ONNX model is the portable interchange artifact. The i.MX93 target uses an Arm Ethos-U65 NPU, so the deployment artifact must be a supported fully quantized TFLite model compiled through the Arm Vela/eIQ path for the installed NXP BSP.

Recommended sequence:

1. Capture representative calibration frames from the final camera and environment.
2. Reproduce the desktop letterbox, RGB ordering, normalization/quantization, and label order exactly.
3. Convert ONNX through NXP eIQ's conversion/quantization workflow to a fully integer-quantized TFLite model.
4. Check that all operators map to the supported Ethos-U set, then compile with the Vela flow supplied/recommended by the installed BSP.
5. Compare board outputs against the TorchScript/ONNX reference using the same images.
6. Validate the three-class outputs on real board-camera frames and measure latency, memory, and thermals on the board.

The included dynamically quantized TorchScript model quantizes only eligible CPU linear layers. It is useful as a size/runtime reference but is not an Ethos-U65 deployment binary.

Official references:

- NXP eIQ conversion and quantization: https://eiq.nxp.com/learning-hub/convQuant/index.html
- i.MX Machine Learning User's Guide: https://www.nxp.com/docs/en/user-guide/UG10166.pdf
- i.MX93 product page: https://www.nxp.com/products/processors-and-microcontrollers/arm-processors/i-mx-applications-processors/i-mx-9-processors/i-mx-93-applications-processor-family-arm-cortex-a55-ml-acceleration-power-efficient-mpu%3Ai.MX93
- NXP eIQ Toolkit release notes: https://www.nxp.com/docs/en/release-note/EIQTRN-v21.pdf

## Deliverables

- `robust_waste_bin_dataset_v2.zip`: clean split, synthetic training set, manifest, report, and manual-review images
- `model/robust_v2_training_report.json`: complete metrics and history
- `model/robust_v2_test_predictions.csv`: per-image independent-test predictions
- `evaluation/dataset_audit.json`: full audit records
- `evaluation/robustness_report.json`: summarized old/new evaluation
- `evaluation/robustness_predictions.csv`: every controlled-condition prediction
- `evaluation/confusion_matrix.png`
- `evaluation/training_history.png`
- `evaluation/robustness_comparison.png`
- `evaluation/prediction_examples.jpg`
- `model/edge_export_report.json`
