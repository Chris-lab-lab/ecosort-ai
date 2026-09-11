# Start here: EcoSort on Windows

Camera preview works before training. Item recognition works only after you collect labeled photos and successfully train the model.

Your first demo uses **plastic bottles**, **metal cans**, and **wrappers**. Label bottles with **1**, wrappers with **2 (General)**, and cans with **3**. Use **0 (Other)** for an empty square, an empty hand, unsupported items, or several objects together.

1. Double-click **OPEN_CAMERA.cmd**. Click inside the camera window. Put one item inside the green square, then press **1 for plastic**, **2 for general**, **3 for metal**, or **0 for other**. Each key saves one labeled photo into `data`. Press **Q** to close the camera.
2. Collect all four classes. Use different objects, angles, backgrounds, and lighting. Include empty views and unsuitable items as `other`. The training launcher needs at least 5 photos per class just to start; aim for 100 or more varied photos per class for an initial experiment. Many nearly identical photos are less useful than varied examples.
3. Double-click **RUN_TRAINING.cmd**. Keep its window open while it runs. The first run may download pretrained MobileNetV2 weights. A successful run creates `artifacts/waste_classifier_int8.tflite`, `artifacts/labels.txt`, and measured results in `artifacts/training_summary.json`.
4. Double-click **RUN_DEMO.cmd**. Keep one item inside the green square, hold it steady, then press **SPACE** to analyze 7 frames. Press **Q** to close. This is a dry-run: it displays recognition and routing decisions without moving lids. Uncertain items can be rejected. Try new objects to check how well the trained model generalizes.

The collection keys tell the program the correct label; they do not ask it to recognize the object. Recognition scores and accuracy come from your trained model and its evaluation, and are not guaranteed by collecting a particular number of photos.

If the wrong camera opens, open a terminal in this folder and run `./OPEN_CAMERA.cmd 1` or `./RUN_DEMO.cmd 1`. Try another index if needed, and close other apps using the camera.

## If Python setup is missing

The launchers prefer this folder's `.venv`. Training and recognition require it. If setup is already complete, skip these commands. Otherwise install Python 3.12 with the Python launcher, open a terminal in this folder, and run:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-train.txt
```

For camera collection only, `OPEN_CAMERA.cmd` can also use an existing `python` command when that Python already has OpenCV and NumPy installed. If a launcher fails, its window stays open so you can read the error. More advanced training and project details are in [README.md](README.md).
