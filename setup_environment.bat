@echo off
cls
title Thesis Pipeline - Environment Setup (Windows)
color 0A

echo ======================================================================
echo          THESIS DEEP LEARNING PIPELINE - REMOTE PC SETUP
echo ======================================================================
echo  This script will:
echo   1. Verify Python 3.10+ installation
echo   2. Create a clean virtual environment ('venv')
echo   3. Install PyTorch with NVIDIA CUDA acceleration
echo   4. Install all data science and vision dependencies
echo ======================================================================
echo.

:: Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not found in your PATH.
    echo Please install Python 3.10 or 3.11 from https://www.python.org/
    echo Make sure to check 'Add Python to PATH' during installation.
    pause
    exit /b 1
)

echo [Step 1/4] Found Python:
python --version
echo.

:: Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [Step 2/4] Creating virtual environment 'venv'...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo Virtual environment created successfully!
) else (
    echo [Step 2/4] Virtual environment 'venv' already exists.
)
echo.

:: Upgrade pip
echo [Step 3/4] Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip
echo.

:: Install PyTorch with CUDA 12.4
echo [Step 4/4] Installing PyTorch with CUDA 12.4 support and dependencies...
echo Installing PyTorch with CUDA 12.4...
venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

echo.
echo Installing remaining requirements from requirements.txt...
venv\Scripts\python.exe -m pip install -r requirements.txt

echo.
echo ======================================================================
echo                     VERIFYING GPU ACCELERATION
echo ======================================================================
venv\Scripts\python.exe -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None (CPU Mode)\"}')"
echo ======================================================================
echo.
echo [SETUP COMPLETE] You can now launch 'run_pipeline.bat' to start training!
pause
