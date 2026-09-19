# Waste-bin fill-level classifier

This component classifies a complete bin view into exactly three categories:

- `empty`: exactly 0% occupied
- `half-full`: 1–74% occupied
- `full`: 75–100% occupied

The webcam and command-line interfaces display only one of these three categories. They do not display an estimated fill percentage or create an additional uncertainty class.

## Current model

- Architecture: MobileNetV3-Small
- Parameters: 1,520,931
- Training set: 1,800 balanced images, 600 per class
- Validation set: 40 original-only images
- Held-out test set: 40 original-only images
- Held-out accuracy: 100% (40/40)
- Controlled transformed-view accuracy: 99.6% across 280 views

The included test set is small and comes from the same broad collection process. A separate recording session with the final camera position remains necessary for a trustworthy deployment estimate.

## Windows setup

From the repository root:

```powershell
py -3.12 -m venv .venv-fill-level
.\.venv-fill-level\Scripts\Activate.ps1
python -m pip install -r .\fill_level_ai\requirements.txt
```

Python 3.12 is recommended for compatibility with the rest of this repository.

## Webcam test

```powershell
.\fill_level_ai\run_webcam.ps1 -Camera 0
```

If the external webcam has a different OpenCV index, try `-Camera 1` or `-Camera 2`. Keep the complete bin interior inside the yellow guide and press `Q` to quit.

The overlay displays only `EMPTY`, `HALF-FULL`, or `FULL`.

## Predict an image or folder

```powershell
.\fill_level_ai\run_predict.ps1 -ImagePath "C:\path\to\bin.jpg"
.\fill_level_ai\run_predict.ps1 -ImagePath "C:\path\to\photos" -OutputPath "C:\path\to\predictions.csv"
```

Single-image output contains only the predicted category. Folder output contains a path and category for every image.

## Collect more real images

```powershell
.\fill_level_ai\run_collect_training_data.ps1 -Camera 0
```

Use `E` for empty, `H` for half-full, `F` for full, and `Q` to quit. Images are saved under `fill_level_ai/data/collected/` and are intentionally ignored by Git.

## Dataset and training

The generated dataset archive is not committed because it is about 126 MB and the repository intentionally excludes collected/generated datasets. Put prepared originals in:

```text
fill_level_ai/data/prepared-custom-data/
```

Then run:

```powershell
.\fill_level_ai\run_audit.ps1
.\fill_level_ai\run_generate_robust_dataset.ps1 -TargetPerClass 600
.\fill_level_ai\run_train_robust.ps1
.\fill_level_ai\run_evaluate_robustness.ps1
```

Synthetic images are generated only from training originals. Validation and test remain original-only. Twelve questionable `full` images are listed in `manual_review_list.txt` and excluded from training rather than silently relabeled.

## Included model artifacts

- `model/waste_bin_robust_v2.torchscript.pt`: desktop and webcam inference
- `model/waste_bin_robust_v2.pth`: training checkpoint
- `model/waste_bin_robust_v2.onnx`: edge interchange model
- `model/waste_bin_fill_level_mobilenet_v3_small.pth`: reproducible base checkpoint
- `model/waste_bin_custom_v1_baseline.torchscript.pt`: previous model used for comparison

## NXP FRDM i.MX93

The ONNX model is an interchange artifact. For the i.MX93 Ethos-U65, perform full-integer post-training quantization with representative camera frames, produce a supported TFLite model using NXP eIQ, and compile it through the Arm Vela flow matching the installed BSP.

The complete audit, results, limitations, and edge notes are in [IMPROVEMENT_REPORT.md](IMPROVEMENT_REPORT.md).
