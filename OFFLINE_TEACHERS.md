# Offline segmentation and detection teachers

These tools run only on the laptop. The deployed NXP model remains the small, fully-INT8 MobileNet model. Use a separate Python environment because EfficientViT-SAM and YOLO-World use PyTorch while EcoSort training uses TensorFlow.

## 1. Install EfficientViT-SAM

Create the teacher environment with Python 3.10, clone the official repository, and install the CUDA-enabled PyTorch build recommended for your GPU by the [official PyTorch installer](https://pytorch.org/get-started/locally/).

```powershell
cd "D:\NXP hackathon\indobantaimeichu-main\indobantaimeichu"
py -3.10 -m venv .teacher-venv
git clone https://github.com/mit-han-lab/efficientvit external\efficientvit
.\.teacher-venv\Scripts\python.exe -m pip install --upgrade pip
.\.teacher-venv\Scripts\python.exe -m pip install -r external\efficientvit\requirements.txt
```

Download the official L0 checkpoint into the location expected by the model zoo:

```powershell
New-Item -ItemType Directory -Force external\efficientvit\assets\checkpoints\efficientvit_sam
Invoke-WebRequest `
  "https://huggingface.co/mit-han-lab/efficientvit-sam/resolve/main/efficientvit_sam_l0.pt" `
  -OutFile "external\efficientvit\assets\checkpoints\efficientvit_sam\efficientvit_sam_l0.pt"
```

Test twenty images before processing the full dataset:

```powershell
.\.teacher-venv\Scripts\python.exe scripts\prepare_segmented_dataset.py `
  --data dataset_enhanced `
  --output dataset_segmented `
  --model efficientvit-sam-l0 `
  --limit 20
```

Inspect `dataset_segmented/review`, the generated masks, and the manifest. Continue without destroying completed work:

```powershell
.\.teacher-venv\Scripts\python.exe scripts\prepare_segmented_dataset.py `
  --data dataset_enhanced `
  --output dataset_segmented `
  --model efficientvit-sam-l0 `
  --resume
```

The output layout is:

```text
dataset_segmented/
├── data/                       # pass this folder to train.py
│   ├── general/
│   ├── metal/
│   ├── other/
│   └── plastic/
├── masks/                      # full-size object masks for audits
├── review/
│   ├── multiple_objects/
│   └── no_object/
├── segmentation_manifest.jsonl
└── segmentation_report.json
```

Known material images are segmented, tightly cropped, and composited onto class-independent solid, gradient, and textured backgrounds. `other` examples pass through unchanged so the validity head still sees realistic hands, empty scenes, unsafe objects, and clutter. Images with no reliable central object are excluded and placed under `review`. Automatic SAM masks are alternative region proposals rather than detected instances, so reliable multiple-object rejection uses YOLO-World boxes; more than one accepted detector box is placed under `review/multiple_objects`.

## 2. Optional YOLO-World boxes

YOLO-World is optional and much heavier. It can propose open-vocabulary boxes and flag multiple objects before segmentation. Install the [official YOLO-World repository](https://github.com/AILab-CVC/YOLO-World) in the teacher environment and download one of its official configs/checkpoints.

```powershell
git clone --recursive https://github.com/AILab-CVC/YOLO-World external\YOLO-World
.\.teacher-venv\Scripts\python.exe -m pip install -e external\YOLO-World
```

Export detections by replacing the example config and checkpoint with the files you downloaded:

```powershell
.\.teacher-venv\Scripts\python.exe scripts\yolo_world_teacher.py `
  --data dataset_enhanced `
  --config external\YOLO-World\configs\YOUR_CONFIG.py `
  --checkpoint external\YOLO-World\weights\YOUR_CHECKPOINT.pth `
  --output teacher_outputs\yolo_world.jsonl
```

Then create a fresh segmented output using those boxes:

```powershell
.\.teacher-venv\Scripts\python.exe scripts\prepare_segmented_dataset.py `
  --data dataset_enhanced `
  --output dataset_segmented_yolo `
  --detections-jsonl teacher_outputs\yolo_world.jsonl
```

When one detector box is present it prompts EfficientViT-SAM. More than one accepted detector box marks the scene for review. Images without a detector box fall back to automatic EfficientViT-SAM masks. Human review remains necessary because open-vocabulary suggestions are not ground truth.

## 3. Retrain with EOS and synthetic rejects

Use the main TensorFlow environment again. The updated trainer adds blurred, dark, empty, and CutMix/multiple-object reject examples and applies an entropic open-set loss to `other` examples.

```powershell
.\.venv\Scripts\python.exe train.py `
  --data dataset_segmented\data `
  --output artifacts_v3 `
  --epochs 15 `
  --fine-tune-epochs 5 `
  --synthetic-reject-ratio 0.25 `
  --eos-weight 0.15 `
  --target-route-precision 0.90
```

The trainer calibrates the validity threshold, per-material confidence thresholds, and prototype-distance limits. It evaluates the final decision policy—not only the raw material argmax—on both float and quantized models.

## 4. Run the background-only bias audit

This replaces every segmented foreground object with the median background colour while leaving the surrounding scene intact:

```powershell
.\.venv\Scripts\python.exe scripts\background_bias_audit.py `
  --data dataset_enhanced `
  --segmented dataset_segmented `
  --model artifacts_v3\waste_classifier_int8.tflite `
  --labels artifacts_v3\labels.txt `
  --output teacher_outputs\background_bias.json `
  --examples-dir teacher_outputs\biased_examples
```

The most important result is `background_only_routed_rate`: it should be close to zero. Background-only material accuracy should be near chance (about 33% for three materials). A substantially higher value indicates class-correlated backgrounds or capture sessions.

## Important limitations

- Automatic masks and open-vocabulary boxes can be wrong. Review samples before training.
- Segmentation should also be applied to an independent validation session; otherwise validation and training distributions differ.
- Do not copy EfficientViT-SAM, YOLO-World, PyTorch, or their checkpoints to the NXP board.
- Keep the original dataset. Generated datasets and teacher checkpoints are reproducible local artifacts and remain excluded from Git.

## Backbone comparison gate

MobileNetV2 remains the deployment baseline. If you later produce a MobileNetV4-Conv-Small full-INT8 artifact, compare both folders on the same machine—or preferably after Vela compilation on the board:

```powershell
.\.venv\Scripts\python.exe scripts\compare_models.py artifacts_v3 artifacts_mobilenetv4
```

The report places model size and median/P95 inference latency beside macro F1, unknown false-acceptance rate, and wrong-lid activation rate from each training summary. Do not switch backbones until the candidate converts fully, Vela accepts its operators, and its safety/latency trade-off beats the MobileNetV2 baseline.
