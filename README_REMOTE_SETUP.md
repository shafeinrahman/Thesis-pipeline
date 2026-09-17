# Thesis Deep Learning Pipeline - Remote Research PC Setup & Export Guide

This repository contains the complete deep learning training and evaluation pipeline for skin lesion classification across Fitzpatrick skin tone subgroups, supporting **EfficientNetV2-S**, **Swin Transformer V2-S**, and **DINOv2-S** across the **Stanford DDI**, **Google SCIN**, **PAD-UFES-20**, and **SkinCON** datasets.

---

## 1. Quick Transfer to Remote Research PC

To copy this pipeline to your remote PC without datasets, virtual environments, or caches:

### Option A: Use the Pre-Packaged ZIP Archive
A lightweight zip file containing only the source code and configuration files is provided:
`thesis_pipeline_export.zip` (~100 KB)

### Option B: Direct Copy via SCP / SFTP (if using SSH)
```bash
# From your local machine to remote PC:
scp -r ./export_pipeline user@remote-pc-ip:/path/to/destination/
```

### Option C: Git / Cloud Drive
Compress the folder or push to a private Git repository, then clone or extract on the remote PC.

---

## 2. One-Click Environment Setup on Remote PC

### On Windows Remote PC:
1. Open the project folder in File Explorer or Command Prompt.
2. Double-click or run:
   ```cmd
   setup_environment.bat
   ```
   *This automatically creates `venv`, installs PyTorch with NVIDIA CUDA acceleration, and installs all requirements.*

### On Linux / Ubuntu Remote PC:
1. Open terminal in the project folder and make scripts executable:
   ```bash
   chmod +x setup_environment.sh run_pipeline.sh
   ./setup_environment.sh
   ```

---

## 3. Attaching Your Datasets

Once the code is on the remote PC, create or copy the dataset folders inside the project root matching the exact layout below:

### Dataset 1: Stanford DDI (`DDI/`)
```
DDI/
├── ddi_metadata.csv
└── images/
    ├── 000001.png
    ├── 000002.png
    └── ...
```
*(Alternatively, run Option `[2]` in `run_pipeline.bat` to automatically download it via Redivis).*

---

### Dataset 2: Google SCIN (`SCIN/`)
```
SCIN/
├── scin_metadata.csv
└── images/
    ├── scin_00000_1.png
    ├── scin_00000_2.png
    └── ...
```
*(Alternatively, run `python download_scin.py` to auto-download and format the dataset from Hugging Face).*

---

### Dataset 3: PAD-UFES-20 (`PAD-UFES-20/`)
```
PAD-UFES-20/
├── metadata.csv
└── images/
    ├── PAT_100_393_595.png
    ├── PAT_101_1041_651.png
    └── ... (or imgs_part_1.zip, imgs_part_2.zip, imgs_part_3.zip)
```
*(If the images are in zip files `imgs_part_*.zip`, the pipeline will automatically unpack them on first run).*

---

### Dataset 4: SkinCON (`SkinCon/`)
```
SkinCon/
├── annotations_ddi.csv
├── annotations_fitzpatrick17k.csv
├── fitzpatrick17k.csv
└── fitzpatrick17k_images/
    ├── 000001.jpg
    └── ...
```

---

## 4. Running the Training Pipeline

### Interactive Menu (Recommended):
- **Windows**: Double-click `run_pipeline.bat`
- **Linux**: Run `./run_pipeline.sh`

### Command-Line Execution (for Headless / SSH Sessions):

#### A. Stanford DDI Dataset:
```bash
# EfficientNetV2-S
python train_efficientnet.py --model efficientnet --epochs 50 --batch_size 32 --start_epoch 25

# Swin Transformer V2-S
python train_efficientnet.py --model swin --epochs 50 --batch_size 32 --start_epoch 25

# DINOv2-S
python train_efficientnet.py --model dinov2 --epochs 50 --batch_size 32 --start_epoch 25
```

#### B. Google SCIN Dataset:
```bash
python train_scin.py --model efficientnet --epochs 50 --batch_size 32 --start_epoch 25
python train_scin.py --model swin --epochs 50 --batch_size 32 --start_epoch 25
python train_scin.py --model dinov2 --epochs 50 --batch_size 32 --start_epoch 25
```

#### C. PAD-UFES-20 Dataset:
```bash
# EfficientNetV2-S
python train_padufes.py --model efficientnet --epochs 50 --batch_size 32 --start_epoch 25

# Swin Transformer V2-S
python train_padufes.py --model swin --epochs 50 --batch_size 32 --start_epoch 25

# DINOv2-S
python train_padufes.py --model dinov2 --epochs 50 --batch_size 32 --start_epoch 25
```

---

## 5. Output Folders & Generated Artifacts

When training runs, all outputs are automatically organized into:
* `models/`: Saved `.pth` weight checkpoints for each architecture and dataset.
* `results/`: DDI evaluation metrics, classification reports, and ROC curves.
* `results_scin/`: Google SCIN evaluation metrics and training curves.
* `results_padufes/`: PAD-UFES-20 evaluation metrics, ROC curves, and multi-class curves.
