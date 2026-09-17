@echo off
cls
set PYTHONUNBUFFERED=1
title DDI and EfficientNetV2 Thesis Pipeline
color 0B

:MENU
echo ======================================================================
echo             DDI AND EFFICIENTNETV2 THESIS PIPELINE CONTROL
echo ======================================================================
echo  This script runs on Python 3.11 with NVIDIA RTX 3050 GPU Acceleration.
echo ======================================================================
echo.
echo  [1] Run Pipeline with Mock/Dummy Data (Verify system instantly)
echo  [2] Bulk Download Stanford DDI Dataset from Redivis + Run Pipeline
echo  [3] Run Pipeline on Existing Stanford DDI Dataset (if already downloaded)
echo  [4] Download and Train on Google SCIN Dataset
echo  [5] Run Pipeline on Existing Google SCIN Dataset (if already downloaded)
echo  [6] Run Pipeline on Existing PAD-UFES-20 Dataset (if already downloaded)
echo  [7] Exit
echo.
echo ======================================================================
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto MOCK
if "%choice%"=="2" goto DOWNLOAD
if "%choice%"=="3" goto REAL_EXISTING
if "%choice%"=="4" goto SCIN_DOWNLOAD
if "%choice%"=="5" goto SCIN_EXISTING
if "%choice%"=="6" goto PADUFES
if "%choice%"=="7" goto EXIT
echo Invalid choice. Please try again.
pause
cls
goto MENU

:MOCK
cls
echo Starting Mock/Dummy Pipeline...
echo -------------------------------------------------------------
echo [Step 1] Activating Python Virtual Environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    goto MENU
)
echo.
echo [Step 2] Generating Mock Dataset (150 images + metadata)...
python generate_dummy_ddi.py
if %errorlevel% neq 0 (
    echo [ERROR] Mock dataset generation failed.
    pause
    goto MENU
)
echo.
echo [Step 3] Training and Evaluating EfficientNetV2 on Mock Data (3 epochs)...
python train_efficientnet.py --data_dir DDI --epochs 3
if %errorlevel% neq 0 (
    echo [ERROR] Training pipeline failed.
    pause
    goto MENU
)
echo.
echo [SUCCESS] Mock pipeline completed successfully!
echo Training curves and ROC subgroup plots are in the 'results' folder.
echo Trained model weights are in the 'models' folder.
pause
cls
goto MENU

:DOWNLOAD
cls
echo Starting Stanford AIMI Redivis Bulk Downloader...
echo -------------------------------------------------------------
echo [Step 1] Activating Python Virtual Environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    goto MENU
)
echo.
echo [Step 2] Starting program download_ddi.py...
python download_ddi.py
if %errorlevel% neq 0 (
    echo [ERROR] Download process failed.
    pause
    goto MENU
)
echo.
echo =============================================================
echo Choose Model Architecture:
echo   [1] EfficientNetV2-S (CNN)
echo   [2] Swin Transformer V2-S (ViT)
echo   [3] DINOv2-S (Foundation ViT)
echo =============================================================
set /p mchoice="Select option (1-3) [Default 1]: "
set MODEL_FLAG=--model efficientnet
if "%mchoice%"=="2" set MODEL_FLAG=--model swin
if "%mchoice%"=="3" set MODEL_FLAG=--model dinov2
echo.
echo [Step 3] Training and Evaluating on REAL Stanford DDI...
echo We will train for up to 50 epochs (at least 25 epochs) on your RTX 3050 GPU.
python train_efficientnet.py --data_dir DDI --epochs 50 --batch_size 32 --patience 5 --start_epoch 25 %MODEL_FLAG%
if %errorlevel% neq 0 (
    echo [ERROR] Training pipeline on real data failed.
    pause
    goto MENU
)
echo.
echo [SUCCESS] Real DDI pipeline completed successfully!
echo Training curves and ROC subgroup plots are in the 'results' folder.
echo Trained model weights are in the 'models' folder.
pause
cls
goto MENU

:REAL_EXISTING
cls
echo Starting Pipeline on Existing DDI Dataset...
echo -------------------------------------------------------------
echo [Step 1] Activating Python Virtual Environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    goto MENU
)
echo.
echo =============================================================
echo Choose Model Architecture:
echo   [1] EfficientNetV2-S (CNN)
echo   [2] Swin Transformer V2-S (ViT)
echo   [3] DINOv2-S (Foundation ViT)
echo =============================================================
set /p mchoice="Select option (1-3) [Default 1]: "
set MODEL_FLAG=--model efficientnet
if "%mchoice%"=="2" set MODEL_FLAG=--model swin
if "%mchoice%"=="3" set MODEL_FLAG=--model dinov2
echo.
echo [Step 2] Training and Evaluating on Existing DDI...
python train_efficientnet.py --data_dir DDI --epochs 50 --batch_size 32 --patience 5 --start_epoch 25 %MODEL_FLAG%
if %errorlevel% neq 0 (
    echo [ERROR] Training pipeline failed. Make sure the 'DDI' folder exists and has 'images' and 'ddi_metadata.csv'.
    pause
    goto MENU
)
echo.
echo [SUCCESS] Real DDI pipeline completed successfully!
echo Training curves and ROC subgroup plots are in the 'results' folder.
echo Trained model weights are in the 'models' folder.
pause
cls
goto MENU

:SCIN_DOWNLOAD
cls
echo Starting Google SCIN Downloader and Training Pipeline...
echo -------------------------------------------------------------
echo [Step 1] Activating Python Virtual Environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    goto MENU
)
echo.
echo [Step 2] Starting program download_scin.py...
python download_scin.py
if %errorlevel% neq 0 (
    echo [ERROR] Download process failed.
    pause
    goto MENU
)
echo.
echo =============================================================
echo Choose Model Architecture:
echo   [1] EfficientNetV2-S (CNN)
echo   [2] Swin Transformer V2-S (ViT)
echo   [3] DINOv2-S (Foundation ViT)
echo =============================================================
set /p mchoice="Select option (1-3) [Default 1]: "
set MODEL_FLAG=--model efficientnet
if "%mchoice%"=="2" set MODEL_FLAG=--model swin
if "%mchoice%"=="3" set MODEL_FLAG=--model dinov2
echo.
echo [Step 3] Training and Evaluating on REAL Google SCIN...
echo We will train for up to 50 epochs (at least 25 epochs) with Early Stopping.
python train_scin.py --scin_dir SCIN --epochs 50 --batch_size 32 --patience 5 --start_epoch 25 %MODEL_FLAG%
if %errorlevel% neq 0 (
    echo [ERROR] Training pipeline on SCIN failed.
    pause
    goto MENU
)
echo.
echo [SUCCESS] Google SCIN pipeline completed successfully!
echo Training curves and evaluation logs are in the 'results_scin' folder.
echo Trained model weights are in the 'models' folder.
pause
cls
goto MENU

:SCIN_EXISTING
cls
echo Starting SCIN Pipeline on Existing Dataset...
echo -------------------------------------------------------------
echo [Step 1] Activating Python Virtual Environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    goto MENU
)
echo.
echo =============================================================
echo Choose Model Architecture:
echo   [1] EfficientNetV2-S (CNN)
echo   [2] Swin Transformer V2-S (ViT)
echo   [3] DINOv2-S (Foundation ViT)
echo =============================================================
set /p mchoice="Select option (1-3) [Default 1]: "
set MODEL_FLAG=--model efficientnet
if "%mchoice%"=="2" set MODEL_FLAG=--model swin
if "%mchoice%"=="3" set MODEL_FLAG=--model dinov2
echo.
echo [Step 2] Training and Evaluating on Existing SCIN...
python train_scin.py --scin_dir SCIN --epochs 50 --batch_size 32 --patience 5 --start_epoch 25 %MODEL_FLAG%
if %errorlevel% neq 0 (
    echo [ERROR] Training pipeline failed. Make sure the 'SCIN' folder exists and contains 'images' and 'scin_metadata.csv'.
    pause
    goto MENU
)
echo.
echo [SUCCESS] SCIN pipeline completed successfully!
echo Training curves and evaluation logs are in the 'results_scin' folder.
echo Trained model weights are in the 'models' folder.
pause
cls
goto MENU

:PADUFES
cls
echo ======================================================================
echo   PAD-UFES-20: TRAINING AND EVALUATION PIPELINE
echo ======================================================================
echo.
echo [Step 1] Activating Python Virtual Environment...
call venv\Scripts\activate
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    goto MENU
)
echo.
echo =============================================================
echo Choose Model Architecture:
echo   [1] EfficientNetV2-S (CNN)
echo   [2] Swin Transformer V2-S (ViT)
echo   [3] DINOv2-S (Foundation ViT)
echo =============================================================
set /p mchoice="Select option (1-3) [Default 1]: "
set MODEL_FLAG=--model efficientnet
if "%mchoice%"=="2" set MODEL_FLAG=--model swin
if "%mchoice%"=="3" set MODEL_FLAG=--model dinov2
echo.
echo [Step 2] Training and Evaluating on PAD-UFES-20...
venv\Scripts\python.exe train_padufes.py --data_dir PAD-UFES-20 --epochs 50 --batch_size 32 --patience 5 --start_epoch 25 %MODEL_FLAG%
if %errorlevel% neq 0 (
    echo [ERROR] Training pipeline failed. Make sure 'PAD-UFES-20' folder exists and has 'metadata.csv'.
    pause
    goto MENU
)
echo.
echo [SUCCESS] PAD-UFES-20 pipeline completed successfully!
echo Training curves and evaluation logs are in the 'results_padufes' folder.
echo Trained model weights are in the 'models' folder.
pause
cls
goto MENU

:EXIT
echo Exiting. Good luck with your thesis!
exit
