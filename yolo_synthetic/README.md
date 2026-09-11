# Synthetic trash dataset for YOLO

This is a starter object-detection dataset for the trash project: 1,200 generated 640 x 640 images, with bounding boxes for plastic bottles, metal cans, and wrappers. Use it to test image loading, YOLO labels, training, and prediction displays.

The images are procedurally rendered, stylized synthetic objects. They are not real camera photographs. A high score on this test set measures performance on this generator's images; it does not establish accuracy on actual trash. Train and evaluate with real, separately collected and box-labeled camera photos before relying on live recognition.

## Dataset location and classes

Open `smart_recycling_ai/datasets/trash_yolo_synthetic_v1/`. The YOLO configuration is `data.yaml`.

| Split | Images | Purpose |
| --- | ---: | --- |
| `images/train` | 960 | Learn from these images |
| `images/val` | 120 | Monitor training and choose settings |
| `images/test` | 120 | Evaluate after choosing the model/settings |

| YOLO class ID | Class name | Corresponding project category |
| ---: | --- | --- |
| 0 | `plastic_bottle` | Plastic |
| 1 | `metal_can` | Metal |
| 2 | `wrapper` | General |

Exactly 12.5% of the scenes are empty negatives with empty label files: 120 training images, 15 validation images, and 15 test images. There is no `other` detection class: background is represented by the absence of boxes. That alone does not teach rejection of every unsupported real object.

Images and labels have matching basenames, in parallel `images/{split}` and `labels/{split}` folders. Each `.txt` label has one row per object with its class and normalized box center/size. For example, `0 0.5 0.5 0.2 0.4` describes a plastic bottle centered in the image with a box 20% of the image width and 40% of its height. This follows the [Ultralytics detection dataset format](https://docs.ultralytics.com/datasets/detect/).

Every scene is generated independently. Boxes are derived from the visible object pixels in the instance mask, with shadows excluded. The dataset also includes:

- `previews/contact_sheet.jpg`: example images with labels for a quick visual check.
- `masks/{split}/*.png`: instance masks for auditing the boxes; the YOLO detector reads the `.txt` boxes.
- `manifest.jsonl`: per-image generation and annotation records.
- `dataset_summary.json`: generated dataset counts and settings.

The sibling archive `smart_recycling_ai/datasets/trash_yolo_synthetic_v1.zip` is available for copying the dataset to another computer. The YAML uses relative split paths and intentionally omits `path:`, so keep `data.yaml`, `images`, and `labels` together in their existing folder structure when moving or extracting the dataset. Point the training command's `--data` option to the moved `data.yaml`.

This is a separate **YOLO detection** experiment. The existing `RUN_TRAINING.cmd` trains a MobileNet image classifier from class folders and does not consume these bounding-box labels. The YOLO weights produced below do not replace the existing TFLite model or connect to lid control.

## Set up YOLO on Windows

Open PowerShell in the `smart_recycling_ai` folder. Use a separate environment so YOLO's PyTorch dependencies do not change the existing TensorFlow setup. On this computer, use the project's working Python 3.12 environment to create the YOLO environment. No environment activation is required:

```powershell
.\.venv\Scripts\python.exe -m venv .venv-yolo
.\.venv-yolo\Scripts\python.exe -m pip install --upgrade pip
.\.venv-yolo\Scripts\python.exe -m pip install -r .\yolo_synthetic\requirements-yolo.txt
```

When setting up on another computer without this project's `.venv`, replace `.\.venv\Scripts\python.exe` in the first command with the full path to a working Python 3.12 executable. Installing Ultralytics downloads its dependencies; the first training run also downloads `yolo11n.pt` if absent. The dataset itself is already local.

## Run a small training test

From `smart_recycling_ai`:

```powershell
.\.venv-yolo\Scripts\python.exe .\yolo_synthetic\train_yolo.py
```

The wrapper defaults to YOLO11 nano, 3 epochs, image size 640, batch size 4, CPU, and zero data-loader workers. It resolves the default dataset from its own location, so the dataset does not depend on your current terminal folder. Three epochs are for checking that the pipeline runs, not for a finished model. CPU training can take a while.

The wrapper uses the documented [YOLO11 model](https://docs.ultralytics.com/models/yolo11/) and [training API](https://docs.ultralytics.com/modes/train/). Its Windows main guard and `workers=0` avoid the usual multiprocessing setup issue.

For a longer experiment, change the epoch count:

```powershell
.\.venv-yolo\Scripts\python.exe .\yolo_synthetic\train_yolo.py --epochs 30 --name trash_30epochs
```

With a working CUDA-enabled PyTorch installation and a compatible NVIDIA GPU, add `--device 0`. Otherwise keep the CPU default.

The first default run saves weights to `yolo_synthetic/runs/trash_smoke/weights/best.pt`. Repeated run names receive numbered output folders; the script prints the actual path. Use that printed path in the following commands if it differs.

## Test the trained detector

Evaluate the held-out synthetic test split after training. Use `best.pt` from your custom training run, not the original pretrained model:

```powershell
.\.venv-yolo\Scripts\yolo.exe detect val model="yolo_synthetic/runs/trash_smoke/weights/best.pt" data="datasets/trash_yolo_synthetic_v1/data.yaml" split=test imgsz=640 batch=4 device=cpu workers=0 project="yolo_synthetic/runs" name="trash_test"
```

The [Ultralytics validation command](https://docs.ultralytics.com/modes/val/) supports `split=test` and reports detection metrics such as mAP. Keep tuning decisions on the validation split to preserve the test split as a final check. All three splits use the same synthetic generator, so this remains a synthetic-domain result.

Save predictions on the test images:

```powershell
.\.venv-yolo\Scripts\yolo.exe detect predict model="yolo_synthetic/runs/trash_smoke/weights/best.pt" source="datasets/trash_yolo_synthetic_v1/images/test" imgsz=640 device=cpu save=True project="yolo_synthetic/runs" name="trash_predictions"
```

To explore the trained detector on your laptop camera, close the project's other camera window first, then run:

```powershell
.\.venv-yolo\Scripts\yolo.exe detect predict model="yolo_synthetic/runs/trash_smoke/weights/best.pt" source=0 imgsz=640 device=cpu show=True save=False
```

The [Ultralytics prediction interface](https://docs.ultralytics.com/modes/predict/) accepts camera index `0` and `show=True`. If the wrong camera opens, try `source=1`; stop the command with Ctrl+C. This only displays detections. Expect a gap between stylized training images and real webcam items.

For real evaluation, take photos of different physical bottles, cans, and wrappers across backgrounds, lighting conditions, and collection sessions. Draw a box around every target object and retain the same class IDs. Keep physical objects/sessions used for final evaluation out of training; add real empty and unsupported-object scenes as negatives.

## Generate another set or check the labels

The provided dataset is ready to use. Generation and label checks only need NumPy and Pillow, which are available in this computer's existing `.venv`; they do not require installing YOLO. From `smart_recycling_ai`, use a new output directory and a different seed range to generate another set:

```powershell
.\.venv\Scripts\python.exe .\yolo_synthetic\generate_dataset.py --output .\datasets\trash_yolo_synthetic_v2 --train 960 --val 120 --test 120 --seed 10930909
```

See each script's `--help` output for its available options. To audit the provided dataset, use the validator:

```powershell
.\.venv\Scripts\python.exe .\yolo_synthetic\validate_dataset.py --dataset .\datasets\trash_yolo_synthetic_v1
```
