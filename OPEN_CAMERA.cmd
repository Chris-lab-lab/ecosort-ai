@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
if errorlevel 1 goto folder_error
if /i "%~1"=="--help" goto help
if /i "%~1"=="/?" goto help
set "CAMERA_INDEX=0"
if not "%~1"=="" set "CAMERA_INDEX=%~1"
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PYTHON_EXE%" goto fallback
"%PYTHON_EXE%" -c "import cv2, numpy" >nul 2>&1
if not errorlevel 1 goto run

:fallback
set "PYTHON_EXE=python"
"%PYTHON_EXE%" -c "import cv2, numpy" >nul 2>&1
if not errorlevel 1 goto run
echo Camera dependencies are missing.
echo See START_HERE.md for Python setup, then try again.
echo The camera needs OpenCV and NumPy; it does not need a trained model.
set "RESULT=1"
goto finish

:run
echo Opening camera %CAMERA_INDEX% for preview and labeled photo collection.
echo Click the camera window, then press 1=plastic, 2=general, 3=metal, 0=other.
echo Each key saves one photo. Keep one item inside the green square.
echo Press Q to close. This window does not recognize items yet.
"%PYTHON_EXE%" dataset_studio.py --camera "%CAMERA_INDEX%" --middle general
set "RESULT=%ERRORLEVEL%"
if "%RESULT%"=="0" goto finish
echo.
echo Camera collection stopped with an error. Read the message above.
echo Close other camera apps. For another camera, run: OPEN_CAMERA.cmd 1
goto finish

:folder_error
echo Could not open the project folder containing this launcher.
set "RESULT=1"
goto finish

:help
echo Usage: OPEN_CAMERA.cmd [camera_index]
echo The default camera index is 0. Try 1 for another camera.
echo Opens the photo collector without requiring a trained model.
exit /b 0

:finish
echo.
pause
exit /b %RESULT%
