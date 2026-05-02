@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Poster Reader GUI setup + launch script for Windows.
REM Put this file in the same folder as GUI_Main.py and requirements.txt.
REM Optional recreate:
REM   set FORCE_RECREATE=1
REM   run_gui_conda.bat

set "ENV_NAME=poster-reader"
set "PY_VER=3.11"
set "SCRIPT_DIR=%~dp0"

cd /d "%SCRIPT_DIR%"

where conda >nul 2>&1
if errorlevel 1 (
    echo ERROR: conda was not found on PATH.
    echo Open Anaconda Prompt/Miniforge Prompt, or add conda to PATH, then run this again.
    pause
    exit /b 1
)

echo.
echo === Using project folder ===
echo %SCRIPT_DIR%

if /I "%FORCE_RECREATE%"=="1" (
    echo.
    echo === Removing old conda environment: %ENV_NAME% ===
    call conda env remove -y -n "%ENV_NAME%"
)

echo.
echo === Checking conda environment: %ENV_NAME% ===
call conda env list | findstr /R /C:"^%ENV_NAME%[ ]" >nul 2>&1
if errorlevel 1 (
    echo Creating conda environment "%ENV_NAME%" with Python %PY_VER%...
    call conda create -y -n "%ENV_NAME%" python=%PY_VER%
    if errorlevel 1 (
        echo ERROR: Failed to create conda environment.
        pause
        exit /b 1
    )
) else (
    echo Environment already exists.
)

echo.
echo === Installing/updating pip tools ===
call conda run -n "%ENV_NAME%" python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo ERROR: Failed to upgrade pip tools.
    pause
    exit /b 1
)

echo.
echo === Installing requirements ===
if not exist "%SCRIPT_DIR%requirements.txt" (
    echo ERROR: requirements.txt not found in:
    echo %SCRIPT_DIR%
    pause
    exit /b 1
)

call conda run -n "%ENV_NAME%" python -m pip install --no-cache-dir -r "%SCRIPT_DIR%requirements.txt"
if errorlevel 1 (
    echo ERROR: Failed to install requirements.
    pause
    exit /b 1
)

echo.
echo === Checking required project files ===
set "MISSING=0"
for %%F in (GUI_Main.py yolo_engine.py ocr_engine.py caption_engine.py audio_engine.py) do (
    if not exist "%SCRIPT_DIR%%%F" (
        echo MISSING: %%F
        set "MISSING=1"
    )
)

if "%MISSING%"=="1" (
    echo.
    echo ERROR: One or more required project files are missing.
    echo Rename your uploaded files to their normal names, for example:
    echo   GUI_Main^(54^).py       --^> GUI_Main.py
    echo   yolo_engine^(47^).py    --^> yolo_engine.py
    echo   ocr_engine^(23^).py     --^> ocr_engine.py
    echo   caption_engine^(14^).py --^> caption_engine.py
    echo   audio_engine^(6^).py    --^> audio_engine.py
    pause
    exit /b 1
)

echo.
echo === Starting Poster Reader GUI ===
call conda run -n "%ENV_NAME%" python "%SCRIPT_DIR%GUI_Main.py"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo GUI exited with code %EXIT_CODE%.
pause
exit /b %EXIT_CODE%
