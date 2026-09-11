@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
if errorlevel 1 goto folder_error
if /i "%~1"=="--help" goto help
if /i "%~1"=="/?" goto help
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
set "KERAS_HOME=%~dp0.cache\keras"
if not exist "%PYTHON_EXE%" goto environment_error
if not exist "data\" goto data_error
"%PYTHON_EXE%" -c "from pathlib import Path; import sys; root=Path('data'); folders={p.name.lower():p for p in root.iterdir() if p.is_dir()}; names=set(folders); middle=names.intersection({'general','paper'}); valid=len(middle)==1 and names=={'plastic','metal','other'}.union(middle); sys.exit('Expected data folders: plastic, metal, other, and exactly one of general or paper.') if not valid else None; extensions={'.jpg','.jpeg','.png','.bmp','.gif'}; counts={name:sum(p.is_file() and p.suffix.lower() in extensions for p in folder.rglob('*')) for name,folder in folders.items()}; print('Images per class: '+', '.join(name+'='+str(count) for name,count in sorted(counts.items()))); missing=[name for name,count in counts.items() if count<5]; sys.exit('Please collect at least 5 images per class before this launcher can train. More varied images are needed for useful recognition.') if missing else None"
if errorlevel 1 goto data_error
"%PYTHON_EXE%" -c "import tensorflow, cv2, numpy, PIL"
if errorlevel 1 goto environment_error
echo.
echo Training from the labeled photos in data. This may take a while.
echo The first run may download MobileNetV2 weights and needs internet access.
echo Training writes model files and measured results into artifacts.
"%PYTHON_EXE%" train.py --data data --output artifacts
set "RESULT=%ERRORLEVEL%"
if not "%RESULT%"=="0" goto training_error
echo.
echo Training completed. You can now open RUN_DEMO.cmd.
goto finish

:data_error
echo.
echo Training has not started. Add labeled photos using OPEN_CAMERA.cmd.
echo Collect plastic, general, metal, and other examples, then try again.
echo Read START_HERE.md and any data error above for details.
set "RESULT=1"
goto finish

:environment_error
echo.
echo The project's .venv Python environment or training dependencies are missing.
echo Follow the setup commands in START_HERE.md, then try again.
set "RESULT=1"
goto finish

:training_error
echo.
echo Training stopped with an error. Read the message above.
echo A successful training run is required before using RUN_DEMO.cmd.
goto finish

:folder_error
echo Could not open the project folder containing this launcher.
set "RESULT=1"
goto finish

:help
echo Usage: RUN_TRAINING.cmd
echo Checks labeled images in data, then trains into artifacts using .venv.
echo Requires plastic, metal, other, and exactly one of general or paper.
echo This launcher requires at least 5 image files per class.
echo For advanced options, run: .venv\Scripts\python.exe train.py --help
exit /b 0

:finish
echo.
pause
exit /b %RESULT%
