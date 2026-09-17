import os
# Force PyTorch Hub to use local directory for cache
os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "torch")
import time
import zipfile
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

# =====================================================================
# 1. Early Stopping Helper Class
# =====================================================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, start_epoch=25):
        self.patience = patience
        self.min_delta = min_delta
        self.start_epoch = start_epoch
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_weights = None

    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            print(f"Validation loss initialized to {val_loss:.4f}. Saving weights.")
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            print(f"Validation loss decreased to {val_loss:.4f}. Saving best weights.")
        else:
            if epoch >= self.start_epoch:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter} out of {self.patience} (Best Val Loss: {self.best_loss:.4f})")
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                print(f"Validation loss did not improve (Best Val Loss: {self.best_loss:.4f}). Early stopping is inactive until epoch {self.start_epoch}.")

from concurrent.futures import ThreadPoolExecutor

def ensure_images_extracted(data_dir):
    """
    Checks if PAD-UFES-20 image archives are extracted.
    If not, automatically extracts imgs_part_1.zip, imgs_part_2.zip, imgs_part_3.zip.
    """
    images_dir = os.path.join(data_dir, "images")
    zip_files = glob.glob(os.path.join(images_dir, "*.zip"))
    if len(zip_files) > 0 and not os.path.exists(os.path.join(images_dir, "imgs_part_1")):
        print(f"[INFO] Found {len(zip_files)} zip archives in {images_dir}. Extracting...", flush=True)
        for zf in zip_files:
            print(f"  Extracting {os.path.basename(zf)}...", flush=True)
            with zipfile.ZipFile(zf, 'r') as zip_ref:
                zip_ref.extractall(images_dir)
        print("[INFO] Extraction complete!", flush=True)

# Global RAM cache to avoid re-reading images across multiple model runs in the same session
_GLOBAL_IMAGE_CACHE = {}

def preload_padufes_images(data_dir, img_ids):
    global _GLOBAL_IMAGE_CACHE
    missing_ids = [img_id for img_id in img_ids if img_id not in _GLOBAL_IMAGE_CACHE]
    if not missing_ids:
        return _GLOBAL_IMAGE_CACHE

    images_dir = os.path.join(data_dir, "images")
    print(f"[INFO] Indexing image paths in {images_dir}...", flush=True)
    file_index = {}
    for root, _, files in os.walk(images_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_index[f] = os.path.join(root, f)

    def _load_single_image(img_id):
        img_path = file_index.get(img_id, os.path.join(images_dir, img_id))
        try:
            # Load and pre-resize to 224x224 RGB in memory
            im = Image.open(img_path).convert('RGB').resize((224, 224), Image.BILINEAR)
            return img_id, im
        except Exception as e:
            return img_id, None

    print(f"[INFO] Preloading and caching {len(missing_ids)} images into RAM for high-speed GPU training...", flush=True)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(_load_single_image, missing_ids)
        for img_id, im in results:
            if im is not None:
                _GLOBAL_IMAGE_CACHE[img_id] = im
    print(f"[SUCCESS] {len(_GLOBAL_IMAGE_CACHE)} images cached in RAM in {time.time()-t0:.2f}s! Epochs will run at maximum GPU speed.", flush=True)
    return _GLOBAL_IMAGE_CACHE

class PAD_UFES_Dataset(Dataset):
    def __init__(self, df, image_cache, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_cache = image_cache
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img_id = str(row["img_id"])
        
        # O(1) in-memory lookup
        img = self.image_cache.get(img_id)
        if img is None:
            raise RuntimeError(f"Image {img_id} not found in preloaded cache.")
            
        if self.transform:
            img = self.transform(img)
            
        label = int(row["class_idx"])
        skin_tone = int(row["skin_tone_group"])
        
        return img_id, img, label, skin_tone

# =====================================================================
# 3. Main PAD-UFES-20 Pipeline
# =====================================================================
def run_padufes_pipeline(data_dir="PAD-UFES-20", epochs=50, batch_size=32, lr=1e-4, patience=5, start_epoch=25, model_type="efficientnet"):
    print("=" * 60)
    print(f"STARTING PAD-UFES-20 {model_type.upper()} PIPELINE")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    csv_path = os.path.join(data_dir, "metadata.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] Metadata file not found at: {csv_path}")
        return False
        
    ensure_images_extracted(data_dir)
    
    df = pd.read_csv(csv_path)
    print(f"Total rows in metadata: {len(df)}")
    
    # Clean diagnostic labels
    df = df.dropna(subset=['diagnostic', 'img_id']).copy()
    classes = sorted(df['diagnostic'].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for idx, cls in enumerate(classes)}
    num_classes = len(classes)
    df['class_idx'] = df['diagnostic'].map(class_to_idx)
    
    print(f"Number of classes ({num_classes}): {classes}")
    for cls in classes:
        print(f"  - {cls}: {sum(df['diagnostic'] == cls)} samples")
        
    # Map Fitzpatrick skin tone scale:
    # 1.0, 2.0 -> 12 (Light: Fitzpatrick I-II)
    # 3.0, 4.0 -> 34 (Medium: Fitzpatrick III-IV)
    # 5.0, 6.0 -> 56 (Dark: Fitzpatrick V-VI)
    # NaN/other -> -1 (Unknown)
    def map_fitzpatrick(val):
        try:
            val = float(val)
            if val in [1.0, 2.0]:
                return 12
            elif val in [3.0, 4.0]:
                return 34
            elif val in [5.0, 6.0]:
                return 56
            else:
                return -1
        except (ValueError, TypeError):
            return -1
            
    df['skin_tone_group'] = df['fitspatrick'].apply(map_fitzpatrick)
    
    # Preload and cache all 2,298 resized images into RAM once
    image_cache = preload_padufes_images(data_dir, df['img_id'].dropna().tolist())
    
    # Preprocessing Transforms (Images are already 224x224 in RAM cache)
    means = [0.485, 0.456, 0.406]
    stds  = [0.229, 0.224, 0.225]
    
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(15),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=means, std=stds)
    ])
    
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=means, std=stds)
    ])
    
    # Stratified Train/Test Split (80% Train, 20% Test)
    stratify_col = df['class_idx'].values
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=stratify_col
    )
    
    print(f"\nTrain split size: {len(train_df)} samples", flush=True)
    print(f"Test split size:  {len(test_df)} samples", flush=True)
    
    train_dataset = PAD_UFES_Dataset(train_df, image_cache, transform=train_transform)
    test_dataset = PAD_UFES_Dataset(test_df, image_cache, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 3.1 Load Pre-trained Model Architecture
    model_key = model_type.lower()
    if model_key == "swin":
        print("\n[INFO] Loading pre-trained Swin Transformer V2 (Tiny)...", flush=True)
        model_name = "Swin Transformer V2-T"
        weight_filename = "swintransformer_padufes.pth"
        try:
            weights = models.Swin_V2_T_Weights.DEFAULT
            model = models.swin_v2_t(weights=weights)
            print("[SUCCESS] Pre-trained Swin Transformer weights loaded successfully!", flush=True)
        except Exception as e:
            print(f"[WARNING] Falling back to uninitialized Swin: {e}", flush=True)
            try:
                model = models.swin_v2_t()
            except Exception:
                model = models.swin_v2_s()
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
        
    elif model_key == "dinov2":
        print("\nLoading pre-trained DINOv2-S foundation model...")
        model_name = "DINOv2-S (ViT-S/14)"
        weight_filename = "dinov2_padufes.pth"
        try:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            print("Pre-trained DINOv2 weights loaded successfully!")
        except Exception as e:
            print(f"Failed to fetch DINOv2 weights: {e}")
            raise e
        in_features = getattr(model, 'embed_dim', 384)
        model.head = nn.Linear(in_features, num_classes)
        
    else:
        print("\nLoading pre-trained EfficientNetV2-S...")
        model_name = "EfficientNetV2-S"
        weight_filename = "efficientnetv2_padufes.pth"
        try:
            weights = models.EfficientNet_V2_S_Weights.DEFAULT
            model = models.efficientnet_v2_s(weights=weights)
            print("Pre-trained EfficientNet weights downloaded successfully!")
        except Exception as e:
            print(f"Initializing without pre-trained weights: {e}")
            model = models.efficientnet_v2_s()
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    early_stopping = EarlyStopping(patience=patience, start_epoch=start_epoch)
    
    # 3.2 Training Loop
    print(f"\nTraining for up to {epochs} epochs (Early stopping active after epoch {start_epoch}, patience={patience})...")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for _, images, targets, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_train_loss = running_loss / total
        epoch_train_acc = (correct / total) * 100
        
        # Validation Loop
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for _, images, targets, _ in test_loader:
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                running_val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        epoch_val_loss = running_val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100
        
        train_losses.append(epoch_train_loss)
        val_losses.append(epoch_val_loss)
        train_accs.append(epoch_train_acc)
        val_accs.append(epoch_val_acc)
        
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}% | Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}% | Time: {epoch_time:.1f}s")
        
        early_stopping(epoch_val_loss, model, epoch)
        if early_stopping.early_stop:
            print(f"\n[INFO] Early stopping triggered at epoch {epoch}! Restoring best weights.")
            model.load_state_dict(early_stopping.best_weights)
            break
            
    if not early_stopping.early_stop and early_stopping.best_weights is not None:
        print("\nRestoring best weights from training run.")
        model.load_state_dict(early_stopping.best_weights)
        
    # Save the model checkpoint
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", weight_filename)
    torch.save(model.state_dict(), model_path)
    print(f"\nTrained model weights saved to: {model_path}")
    
    # 3.3 Final Evaluation on Test Set
    print("\nRunning final evaluation on test split...")
    model.eval()
    
    all_targets = []
    all_preds = []
    all_probs = []
    all_skin_tones = []
    
    with torch.no_grad():
        for _, images, targets, tones in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_targets.extend(targets.numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_skin_tones.extend(tones.numpy())
            
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_skin_tones = np.array(all_skin_tones)
    
    # Overall Multi-Class Metrics
    overall_acc = accuracy_score(all_targets, all_preds)
    overall_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    try:
        overall_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
    except Exception:
        overall_auc = float('nan')
        
    # Subgroup Analysis by Skin Tone (Fitzpatrick)
    unique_tones = [12, 34, 56, -1]
    tone_names = {
        12: "Light (Fitzpatrick I-II)",
        34: "Medium (Fitzpatrick III-IV)",
        56: "Dark (Fitzpatrick V-VI)",
        -1: "Unspecified / Unknown"
    }
    
    subgroup_results = {}
    for tone in unique_tones:
        idx = np.where(all_skin_tones == tone)[0]
        if len(idx) == 0:
            continue
            
        sub_targets = all_targets[idx]
        sub_preds = all_preds[idx]
        sub_probs = all_probs[idx]
        
        sub_acc = accuracy_score(sub_targets, sub_preds)
        sub_f1 = f1_score(sub_targets, sub_preds, average='macro', zero_division=0)
        
        if len(np.unique(sub_targets)) > 1:
            try:
                sub_auc = roc_auc_score(sub_targets, sub_probs, multi_class='ovr', average='macro')
            except Exception:
                sub_auc = float('nan')
        else:
            sub_auc = float('nan')
            
        subgroup_results[tone] = {
            'accuracy': sub_acc,
            'f1_score': sub_f1,
            'auc': sub_auc,
            'count': len(idx)
        }
        
    # Write Evaluation Report
    os.makedirs("results_padufes", exist_ok=True)
    report_path = os.path.join("results_padufes", f"padufes_evaluation_report_{model_key}.txt")
    with open(report_path, "w") as rf:
        def log_print(text):
            print(text)
            rf.write(str(text) + "\n")
            
        log_print("=" * 60)
        log_print(f"PAD-UFES-20 DATASET EVALUATION REPORT - {model_name.upper()}")
        log_print("=" * 60)
        log_print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log_print(f"Model Architecture: {model_name}")
        log_print(f"Classes ({num_classes}): {classes}")
        log_print(f"Total Epochs Trained: {len(train_losses)} (Maximum set: {epochs})")
        log_print(f"Batch Size: {batch_size} | Learning Rate: {lr}")
        log_print(f"Device Used: {device}")
        if device.type == 'cuda':
            log_print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        log_print("=" * 60 + "\n")
        
        log_print("=" * 45)
        log_print("OVERALL PERFORMANCE EVALUATION")
        log_print("=" * 45)
        log_print(f"Test Accuracy: {overall_acc*100:.2f}%")
        log_print(f"Test F1-Score (Macro): {overall_f1:.4f}")
        log_print(f"Test ROC-AUC (Macro OVR): {overall_auc:.4f}" if not np.isnan(overall_auc) else "Test ROC-AUC: N/A")
        log_print("\nClassification Report:")
        log_print(classification_report(all_targets, all_preds, target_names=classes, zero_division=0))
        log_print("=" * 45 + "\n")
        
        log_print("=" * 45)
        log_print("SUBGROUP ANALYSIS BY FITZPATRICK SKIN TONE")
        log_print("=" * 45)
        for tone in unique_tones:
            if tone not in subgroup_results:
                continue
            res = subgroup_results[tone]
            log_print(f"Subgroup: {tone_names[tone]} (N = {res['count']})")
            log_print(f"  Accuracy: {res['accuracy']*100:.2f}%")
            log_print(f"  F1-Score (Macro): {res['f1_score']:.4f}")
            log_print(f"  ROC-AUC (Macro OVR): {res['auc']:.4f}" if not np.isnan(res['auc']) else "  ROC-AUC : N/A")
            log_print("-" * 35)
        log_print("=" * 45)
        
    print(f"\n[SUCCESS] PAD-UFES-20 evaluation report saved to: {report_path}")
    
    # 3.4 Save Training Curve Plots
    plt.figure(figsize=(12, 5))
    epochs_range = range(1, len(train_losses) + 1)
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs_range, val_losses, 'r-o', label='Val Loss')
    plt.title(f'Loss - {model_name} (PAD-UFES-20)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, 'b-o', label='Train Acc')
    plt.plot(epochs_range, val_accs, 'r-o', label='Val Acc')
    plt.title(f'Accuracy - {model_name} (PAD-UFES-20)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    curves_plot_path = os.path.join("results_padufes", f"padufes_training_curves_{model_key}.png")
    plt.savefig(curves_plot_path, dpi=150)
    plt.close()
    print(f"PAD-UFES-20 training curves saved to: {curves_plot_path}")
    
    # 3.5 Save Multi-Class ROC Plot
    try:
        y_bin = label_binarize(all_targets, classes=list(range(num_classes)))
        plt.figure(figsize=(8, 6))
        for i, cls_name in enumerate(classes):
            if num_classes == 2:
                fpr, tpr, _ = roc_curve(all_targets, all_probs[:, 1])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{cls_name} (AUC = {roc_auc:.3f})')
                break
            else:
                fpr, tpr, _ = roc_curve(y_bin[:, i], all_probs[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{cls_name} (AUC = {roc_auc:.3f})')
                
        plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing (AUC = 0.50)")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves by Class - {model_name} (PAD-UFES-20)")
        plt.legend(loc="lower right")
        plt.grid(True)
        roc_plot_path = os.path.join("results_padufes", f"padufes_roc_curves_{model_key}.png")
        plt.savefig(roc_plot_path, dpi=150)
        plt.close()
        print(f"PAD-UFES-20 ROC curves saved to: {roc_plot_path}")
    except Exception as e:
        print(f"[NOTE] Multi-class ROC plot skipped: {e}")
        
    print(f"\n[SUCCESS] {model_name} training and evaluation on PAD-UFES-20 completed successfully!")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="PAD-UFES-20", help="PAD-UFES-20 dataset directory")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    parser.add_argument('--start_epoch', type=int, default=25, help="Epoch at which early stopping becomes active")
    parser.add_argument('--model', type=str, default="efficientnet", choices=["efficientnet", "swin", "dinov2"], help="Model architecture")
    args = parser.parse_args()
    
    run_padufes_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        start_epoch=args.start_epoch,
        model_type=args.model
    )
