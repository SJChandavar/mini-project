@echo off
setlocal enabledelayedexpansion

:: Automatically determine project root directory
set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

echo ========================================================
echo   Automated Tooth Segmentation - Stop Application      
echo ========================================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment python.exe not found at .venv\Scripts\python.exe
    echo.
    pause
    exit /b 1
)

if not exist "tools\stop_app.py" (
    echo [ERROR] Script tools\stop_app.py not found in project directory!
    echo.
    pause
    exit /b 1
)

.venv\Scripts\python.exe tools\stop_app.py

echo.
pause
