#!/usr/bin/env bash
# ======================================================================
# Thesis Deep Learning Pipeline - Linux / Ubuntu Environment Setup
# ======================================================================

set -e

echo "======================================================================"
echo "         THESIS DEEP LEARNING PIPELINE - LINUX REMOTE PC SETUP"
echo "======================================================================"

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found. Please install Python 3.10+ (sudo apt update && sudo apt install python3 python3-venv python3-pip)"
    exit 1
fi

echo "[Step 1/4] Found Python: $(python3 --version)"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "[Step 2/4] Creating virtual environment 'venv'..."
    python3 -m venv venv
else
    echo "[Step 2/4] Virtual environment 'venv' already exists."
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
echo "[Step 3/4] Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA 12.4
echo "[Step 4/4] Installing PyTorch with CUDA 12.4 support and dependencies..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

echo ""
echo "======================================================================"
echo "                     VERIFYING GPU ACCELERATION"
echo "======================================================================"
python3 -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None (CPU Mode)\"}')"
echo "======================================================================"
echo ""
echo "[SETUP COMPLETE] You can now run './run_pipeline.sh' or execute python scripts directly!"
