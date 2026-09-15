# EcoSort full model experiment

This experiment is separate from train.py and the existing artifacts_v2 or
artifacts_v3 models. It produces one compact model with four outputs:

~~~text
224x224 RGB
    |
MobileNetV2 0.35 shared encoder
    |-- 56x56 object mask
    |-- general/paper, metal, plastic
    |-- supported/reject probability
    +-- 64-value embedding
~~~

EfficientViT-SAM is an offline mask teacher only. The exported model does not
contain SAM, PyTorch, CUDA, Depth Anything, or GazeSAM. The i.MX93 receives RGB
and runs only the INT8 MobileNet student.

## 1. Software check

This creates synthetic shapes, trains for one epoch, exports full INT8, loads
the TFLite file, and checks every output:

~~~powershell
.\.venv\Scripts\python.exe train_full_model.py --smoke-test --output artifacts_full_smoke
~~~

artifacts_full_smoke proves that the pipeline works. Its weights are synthetic
and must never control a lid.

## 2. Prepare the real mask dataset

Use a new output folder and segment every class, including other:

~~~powershell
.\.teacher-venv\Scripts\python.exe scripts\prepare_segmented_dataset.py --data dataset_enhanced --output dataset_segmented_full --model efficientvit-sam-l0 --device cuda --segment-other --prompt-mode center --mask-only --resume
~~~

This fast command assumes one presented object crosses the center of each
image. Use --prompt-mode auto instead when objects are not centered; automatic
mode is more thorough but much slower. This step runs on the laptop GPU. Stop
it safely with Ctrl+C and use the same command with --resume to continue. Inspect:

~~~text
dataset_segmented_full\review
dataset_segmented_full\segmentation_report.json
dataset_segmented_full\segmentation_manifest.jsonl
~~~

Do not train until useful accepted masks exist for general (or paper), metal,
plastic, and other. Add empty views, multiple objects, glass, batteries,
electronics, food, and badly blurred frames to other. For labelled held-object
photos, keep the real material label but set supported to false in the manifest;
the material head still learns the object while the safety head learns that a
visible hand must block actuation.

## 3. Train the real model

TensorFlow on this Windows environment currently trains on CPU. The SAM teacher
uses the NVIDIA GPU, but the compact TensorFlow student does not.

~~~powershell
.\.venv\Scripts\python.exe train_full_model.py --manifest dataset_segmented_full\segmentation_manifest.jsonl --output artifacts_full --epochs 15 --fine-tune-epochs 5 --batch-size 32
~~~

The deployable result is:

~~~text
artifacts_full\full_waste_model_int8.tflite
artifacts_full\labels.txt
artifacts_full\full_model.json
~~~

The trainer refuses an incomplete class set and refuses to overwrite a nonempty
output folder unless --overwrite is explicitly supplied.

## 4. Preview the real model

This viewer is visual dry-run only. It can show the isolated item against white
or as a green overlay:

~~~powershell
.\.venv\Scripts\python.exe -m ecosort_ai.full_live_demo --model artifacts_full\full_waste_model_int8.tflite --labels artifacts_full\labels.txt --camera 0 --view white
~~~

Press V to switch view and Q or Escape to quit.

## Safety and limitations

- A lid may open only when both the validity and material thresholds pass.
- The current validity output learns hand rejection only from examples labelled
  as reject; there is not yet a separately annotated hand head.
- The mask is a binary foreground mask, not a per-pixel material map.
- A mixed-material object still receives one dominant material label.
- Depth-free geometry is represented here by RGB-only student inference,
  teacher masks, edge-preserving skip features, and a boundary-aware mask loss.
  Depth Anything supervision is not included yet because no depth pseudo-labels
  have been generated for this dataset.
- Vela compilation and latency must be tested on the actual i.MX93 BSP before
  enabling an actuator.

## Train the three dataset_new disposal routes

The `--taxonomy disposal` option predicts exactly the three top-level folders:
`recyclable - others`, `recyclable - paper`, and `regular trash`. Nested folders
such as Glass, Metals, Paper, Tissue, and Plastic Bags contribute to their
parent route. It does not predict GENERAL, METAL, or PLASTIC. The EfficientViT-SAM-L0
mask pass for `dataset_new` has already completed in `dataset_segmented_new_raw`.
Do not use the earlier `segmentation_manifest_mapped.jsonl` for this taxonomy.

Run from the project root on your laptop:

~~~powershell
.\.venv\Scripts\python.exe train_full_model.py --manifest dataset_segmented_new_raw\segmentation_manifest.jsonl --taxonomy disposal --skip-unsupported-images --output artifacts_disposal_new --epochs 15 --fine-tune-epochs 5 --batch-size 32
~~~

The skip option excludes the 13 WEBP and one MPO image whose `.jpg` extension
misrepresents their actual format; it does not modify the source files. Preview
the resulting RGB + mask + disposal-category model with:

~~~powershell
.\.venv\Scripts\python.exe -m ecosort_ai.full_live_demo --model artifacts_disposal_new\full_waste_model_int8.tflite --labels artifacts_disposal_new\labels.txt --camera 0 --view overlay --delegate none
~~~

This taxonomy has no separate unknown/reject examples. Its validity head is
untrained and the preview must not command lids. Add a reject dataset and
retrain before live actuation. Older models and data remain untouched.

## Laptop-only live EfficientViT-SAM outline

The same three-route model can be previewed with a live full-frame SAM outline.
This mode starts the existing GPU held-object worker, which uses YOLO detection,
optional wrist selection, and a box prompt for EfficientViT-SAM-L0. The
detector's padded RGB item crop is classified by `artifacts_disposal_new`; the
SAM mask is drawn on the full camera frame. It does not retrain or alter the
INT8 model. The first launch may download two Ultralytics checkpoints into the
ignored `teacher_models` directory.

~~~powershell
.\.teacher-venv\Scripts\python.exe -m pip install -r requirements-held-object.txt
.\.venv\Scripts\python.exe -m ecosort_ai.full_live_demo --model artifacts_disposal_new\full_waste_model_int8.tflite --labels artifacts_disposal_new\labels.txt --camera 0 --camera-width 640 --camera-height 480 --view overlay --delegate none --held-object --held-device cuda
~~~

If the detector misses your item, it cannot provide a SAM box. A COCO-only
detector may miss trash-specific objects. Try showing one item clearly in the
center, or supply compatible custom `--object-model` and `--pose-model`
checkpoints. The displayed FPS may be lower than the INT8-only preview because
the laptop runs detector, pose, and SAM for every frame. This option is visual
dry-run only and does not actuate lids.

On the i.MX93, omit `--held-object`: the board runs only the compact INT8
MobileNetV2 heads, including its own lower-resolution object mask. There is no
CUDA, YOLO, or EfficientViT-SAM dependency in the board-style path. Actual
Ethos-U65 acceleration is not established until Vela compilation and on-board
delegate/latency tests pass; CPU TFLite fallback may be possible. The disposal
taxonomy still lacks a reject class, so do not connect it to lid control.

On a board image that provides `vela`, compile the new model separately:

~~~sh
sh scripts/compile_vela.sh artifacts_disposal_new/full_waste_model_int8.tflite artifacts_disposal_new/vela
~~~

Inspect the Vela report for CPU fallbacks and benchmark the compiled model on
the actual board before deciding whether the mask head can run at camera rate.
