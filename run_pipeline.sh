#!/usr/bin/env bash
# ======================================================================
# Thesis Deep Learning Pipeline Control Script (Linux / macOS)
# ======================================================================

set -e

# Activate venv
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "[WARNING] 'venv' not found. Running with global python3."
fi

show_menu() {
    clear
    echo "======================================================================"
    echo "            THESIS DEEP LEARNING PIPELINE CONTROL (LINUX)"
    echo "======================================================================"
    echo "  [1] Run Pipeline with Mock/Dummy Data (Verify system instantly)"
    echo "  [2] Bulk Download Stanford DDI Dataset from Redivis + Run Pipeline"
    echo "  [3] Run Pipeline on Existing Stanford DDI Dataset"
    echo "  [4] Download and Train on Google SCIN Dataset"
    echo "  [5] Run Pipeline on Existing Google SCIN Dataset"
    echo "  [6] Run Pipeline on Existing PAD-UFES-20 Dataset"
    echo "  [7] Exit"
    echo "======================================================================"
    read -p "Enter your choice (1-7): " choice
    
    case $choice in
        1)
            echo "Starting Mock/Dummy Pipeline..."
            python generate_dummy_ddi.py
            python train_efficientnet.py --data_dir DDI --epochs 3
            ;;
        2)
            echo "Starting Stanford DDI Downloader..."
            python download_ddi.py
            echo "Choose Model: [1] EfficientNetV2-S  [2] Swin Transformer V2-S  [3] DINOv2-S"
            read -p "Select (1-3) [Default 1]: " mc
            mflag="--model efficientnet"
            [ "$mc" == "2" ] && mflag="--model swin"
            [ "$mc" == "3" ] && mflag="--model dinov2"
            python train_efficientnet.py --data_dir DDI --epochs 50 --batch_size 32 --patience 5 --start_epoch 25 $mflag
            ;;
        3)
            echo "Choose Model: [1] EfficientNetV2-S  [2] Swin Transformer V2-S  [3] DINOv2-S"
            read -p "Select (1-3) [Default 1]: " mc
            mflag="--model efficientnet"
            [ "$mc" == "2" ] && mflag="--model swin"
            [ "$mc" == "3" ] && mflag="--model dinov2"
            python train_efficientnet.py --data_dir DDI --epochs 50 --batch_size 32 --patience 5 --start_epoch 25 $mflag
            ;;
        4)
            echo "Starting Google SCIN Downloader..."
            python download_scin.py
            echo "Choose Model: [1] EfficientNetV2-S  [2] Swin Transformer V2-S  [3] DINOv2-S"
            read -p "Select (1-3) [Default 1]: " mc
            mflag="--model efficientnet"
            [ "$mc" == "2" ] && mflag="--model swin"
            [ "$mc" == "3" ] && mflag="--model dinov2"
            python train_scin.py --scin_dir SCIN --epochs 50 --batch_size 32 --patience 5 --start_epoch 25 $mflag
            ;;
        5)
            echo "Choose Model: [1] EfficientNetV2-S  [2] Swin Transformer V2-S  [3] DINOv2-S"
            read -p "Select (1-3) [Default 1]: " mc
            mflag="--model efficientnet"
            [ "$mc" == "2" ] && mflag="--model swin"
            [ "$mc" == "3" ] && mflag="--model dinov2"
            python train_scin.py --scin_dir SCIN --epochs 50 --batch_size 32 --patience 5 --start_epoch 25 $mflag
            ;;
        6)
            echo "Choose Model: [1] EfficientNetV2-S  [2] Swin Transformer V2-S  [3] DINOv2-S"
            read -p "Select (1-3) [Default 1]: " mc
            mflag="--model efficientnet"
            [ "$mc" == "2" ] && mflag="--model swin"
            [ "$mc" == "3" ] && mflag="--model dinov2"
            python train_padufes.py --data_dir PAD-UFES-20 --epochs 50 --batch_size 32 --patience 5 --start_epoch 25 $mflag
            ;;
        7)
            echo "Exiting."
            exit 0
            ;;
        *)
            echo "Invalid option."
            ;;
    esac
    read -p "Press Enter to return to menu..."
    show_menu
}

show_menu
