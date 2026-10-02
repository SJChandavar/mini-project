@echo off
setlocal enabledelayedexpansion

:: Automatically determine project root directory
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

echo ========================================================
echo   Automated Tooth Segmentation - Application Launcher  
echo ========================================================
echo.

:: 1. Verify Virtual Environment
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment was not found!
    echo Expected: %PROJECT_DIR%.venv\Scripts\python.exe
    echo.
    echo Please ensure the .venv virtual environment is set up before running.
    echo.
    pause
    exit /b 1
)

:: 2. Verify Application Entry Point
if not exist "app.py" (
    echo [ERROR] Application script app.py not found in project directory!
    echo Target directory: %PROJECT_DIR%
    echo.
    pause
    exit /b 1
)

:: 3. Launch application via tools/start_app.py
.venv\Scripts\python.exe tools\start_app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application failed to start or exited with code %ERRORLEVEL%.
    echo.
    pause
)
