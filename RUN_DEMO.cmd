@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
if errorlevel 1 goto folder_error
if /i "%~1"=="--help" goto help
if /i "%~1"=="/?" goto help
set "CAMERA_INDEX=0"
if not "%~1"=="" set "CAMERA_INDEX=%~1"
if not exist "artifacts\waste_classifier_int8.tflite" goto model_error
if not exist "artifacts\labels.txt" goto model_error
for %%F in ("artifacts\waste_classifier_int8.tflite" "artifacts\labels.txt") do if %%~zF EQU 0 goto model_error
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "KERAS_HOME=%~dp0.cache\keras"
if not exist "%PYTHON_EXE%" goto environment_error
"%PYTHON_EXE%" -c "import cv2, numpy, tensorflow"
if errorlevel 1 goto environment_error
echo.
echo Opening the recognition demo on camera %CAMERA_INDEX%.
echo Hold one item inside the green square, then press SPACE to analyze.
echo The demo combines 7 frames. Press Q to quit.
echo Dry-run mode displays decisions without moving physical lids.
"%PYTHON_EXE%" -m ecosort_ai.live_demo --model "artifacts\waste_classifier_int8.tflite" --labels "artifacts\labels.txt" --camera "%CAMERA_INDEX%" --frames 7 --dry-run
set "RESULT=%ERRORLEVEL%"
if "%RESULT%"=="0" goto finish
echo.
echo Recognition stopped with an error. Read the message above.
echo For another camera, run: RUN_DEMO.cmd 1
goto finish

:model_error
echo Recognition is not ready: the trained model or labels are missing or empty.
echo Required files: artifacts\waste_classifier_int8.tflite and artifacts\labels.txt
echo First open OPEN_CAMERA.cmd to collect labeled photos for all four classes.
echo Then run RUN_TRAINING.cmd successfully before opening this demo.
echo Camera preview and collection can run before training.
set "RESULT=1"
goto finish

:environment_error
echo.
echo The project's .venv Python environment or demo dependencies are missing.
echo Follow the setup commands in START_HERE.md, then try again.
set "RESULT=1"
goto finish

:folder_error
echo Could not open the project folder containing this launcher.
set "RESULT=1"
goto finish

:help
echo Usage: RUN_DEMO.cmd [camera_index]
echo The default camera index is 0. Try 1 for another camera.
echo Requires successful training and the model plus labels in artifacts.
echo Runs the recognition demo with 7 frames in dry-run mode.
exit /b 0

:finish
echo.
pause
exit /b %RESULT%
