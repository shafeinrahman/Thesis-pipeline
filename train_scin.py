import os
# Force PyTorch Hub and Hugging Face to use local directory for cache
os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "torch")
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "huggingface")
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T
import torchvision.models as models

from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

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

# Global RAM cache for SCIN images to eliminate disk I/O bottleneck
_SCIN_IMAGE_CACHE = {}

def preload_scin_images(images_dir, filenames):
    global _SCIN_IMAGE_CACHE
    missing_files = [fn for fn in filenames if fn not in _SCIN_IMAGE_CACHE]
    if not missing_files:
        return _SCIN_IMAGE_CACHE
        
    print(f"[INFO] Preloading and caching {len(missing_files)} SCIN images into RAM...", flush=True)
    
    def _load_single(fn):
        path = os.path.join(images_dir, fn)
        try:
            im = Image.open(path).convert('RGB').resize((224, 224), Image.BILINEAR)
            return fn, im
        except Exception:
            return fn, None
            
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = ex.map(_load_single, missing_files)
        for fn, im in results:
            if im is not None:
                _SCIN_IMAGE_CACHE[fn] = im
                
    print(f"[SUCCESS] {len(_SCIN_IMAGE_CACHE)} SCIN images cached in RAM in {time.time()-t0:.2f}s! Training will run at maximum GPU speed.", flush=True)
    return _SCIN_IMAGE_CACHE

class SCIN_Dataset(Dataset):
    def __init__(self, df, image_cache, transform=None):
        self.df = df.reset_index(drop=True)
        self.image_cache = image_cache
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        fn = row["filename"]
        img = self.image_cache.get(fn)
        if img is None:
            raise RuntimeError(f"Image {fn} missing from RAM cache.")
            
        if self.transform:
            img = self.transform(img)
            
        label = int(row["class_idx"])
        skin_tone = int(row["skin_tone"])
        
        return fn, img, label, skin_tone

# =====================================================================
# 3. Main Pipeline
# =====================================================================
def run_scin_pipeline(scin_dir="SCIN", epochs=50, batch_size=32, lr=1e-4, patience=5, start_epoch=25, model_type="efficientnet"):
    print("=" * 60)
    print(f"STARTING GOOGLE SCIN {model_type.upper()} PIPELINE")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    csv_path = os.path.join(scin_dir, "scin_metadata.csv")
    images_dir = os.path.join(scin_dir, "images")
    
    if not os.path.exists(csv_path):
        print(f"[ERROR] Metadata file not found at: {csv_path}")
        print("Please run 'download_scin.py' first to download the dataset.")
        return
        
    # 3.1 Load and Filter Metadata
    print("Loading SCIN metadata...")
    df = pd.read_csv(csv_path)
    
    # Filter rows with valid skin tone and valid disease label
    df = df[df["skin_tone"].notna()]
    df = df[df["disease"].notna() & (df["disease"] != "unknown")]
    
    # Select the Top 5 most frequent disease classes to keep training stable and balanced
    top_diseases = df["disease"].value_counts().head(5).index.tolist()
    print(f"Top 5 disease classes selected for training: {top_diseases}")
    df = df[df["disease"].isin(top_diseases)].copy()
    
    # Map disease names to integer class indices
    disease_to_idx = {name: idx for idx, name in enumerate(top_diseases)}
    idx_to_disease = {idx: name for name, idx in disease_to_idx.items()}
    df["class_idx"] = df["disease"].map(disease_to_idx)
    
    total_samples = len(df)
    print(f"Total samples for Top 5 classes: {total_samples}")
    if total_samples < 50:
        print("[ERROR] Too few samples to train. Make sure you downloaded the complete dataset.")
        return
        
    # Preload and cache all Top 5 SCIN images into RAM
    image_cache = preload_scin_images(images_dir, df["filename"].tolist())
    
    # 3.2 Preprocessing Transforms (Images are already 224x224 in RAM cache)
    means = [0.485, 0.456, 0.406]
    stds  = [0.229, 0.224, 0.225]
    
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(15),
        T.ToTensor(),
        T.Normalize(mean=means, std=stds)
    ])
    
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=means, std=stds)
    ])
    
    # Stratified Split
    stratify_keys = [f"L{l}_S{s}" for l, s in zip(df["class_idx"], df["skin_tone"])]
    
    train_df, test_df = train_test_split(
        df, 
        test_size=0.20, 
        random_state=42, 
        stratify=stratify_keys
    )
    
    print(f"Train split size: {len(train_df)} samples", flush=True)
    print(f"Test split size: {len(test_df)} samples", flush=True)
    
    train_dataset = SCIN_Dataset(train_df, image_cache, transform=train_transform)
    test_dataset = SCIN_Dataset(test_df, image_cache, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 3.3 Load pre-trained model architecture
    if model_type.lower() == "swin":
        print("\n[INFO] Loading pre-trained Swin Transformer V2 (Tiny)...", flush=True)
        model_name = "Swin Transformer V2-T"
        weight_filename = "swintransformer_scin.pth"
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
            
        # Replace classification head for 5 classes
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, 5)
    elif model_type.lower() == "dinov2":
        print("\nLoading pre-trained DINOv2-S foundation model...")
        model_name = "DINOv2-S (ViT-S/14)"
        weight_filename = "dinov2_scin.pth"
        try:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            print("Pre-trained DINOv2 weights loaded successfully!")
        except Exception as e:
            print(f"Failed to fetch DINOv2 weights: {e}")
            raise e
        
        # Replace classification head for 5 classes
        in_features = getattr(model, 'embed_dim', 384)
        model.head = nn.Linear(in_features, 5)
    else:
        print("\nLoading pre-trained EfficientNetV2-S...")
        model_name = "EfficientNetV2-S"
        weight_filename = "efficientnetv2_scin.pth"
        try:
            weights = models.EfficientNet_V2_S_Weights.DEFAULT
            model = models.efficientnet_v2_s(weights=weights)
            print("Pre-trained weights downloaded successfully!")
        except Exception as e:
            print(f"Initializing without pre-trained weights: {e}")
            model = models.efficientnet_v2_s()
            
        # Replace classification head for 5 classes
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 5)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    early_stopping = EarlyStopping(patience=patience, start_epoch=start_epoch)
    
    # 3.4 Training Loop
    print(f"\nTraining for up to {epochs} epochs (at least {start_epoch} epochs before early stopping)...")
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
        
    # Save the model
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", weight_filename)
    torch.save(model.state_dict(), model_path)
    print(f"\nTrained model weights saved to: {model_path}")
    
    # 3.5 Evaluation
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
    
    # Overall multi-class metrics
    overall_acc = accuracy_score(all_targets, all_preds)
    overall_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    
    try:
        overall_auc = roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
    except Exception:
        overall_auc = float('nan')
        
    unique_tones = [12, 34, 56]
    tone_names = {12: "Light (Fitzpatrick 1-2)", 34: "Medium (Fitzpatrick 3-4)", 56: "Dark (Fitzpatrick 5-6)"}
    
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
        
        # Calculate AUC (only if we have more than one class present in subgroup)
        if len(np.unique(sub_targets)) > 1:
            try:
                # Filter probs to columns that are present in sub_targets if needed, or ovr
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
        
    # Write report
    os.makedirs("results_scin", exist_ok=True)
    report_path = os.path.join("results_scin", "scin_evaluation_report.txt")
    with open(report_path, "w") as rf:
        def log_print(text):
            print(text)
            rf.write(str(text) + "\n")
            
        log_print("=" * 60)
        log_print(f"GOOGLE SCIN DATASET EVALUATION REPORT - {model_name.upper()}")
        log_print("=" * 60)
        log_print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log_print(f"Model Architecture: {model_name}")
        log_print(f"Classes Trained On: {top_diseases}")
        log_print(f"Total Epochs Trained: {len(train_losses)} (Maximum set: {epochs})")
        log_print(f"Batch Size: {batch_size} | Learning Rate: {lr}")
        log_print(f"Device Used: {device}")
        if device.type == 'cuda':
            log_print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        log_print("=" * 60 + "\n")
        
        log_print("="*45)
        log_print("OVERALL PERFORMANCE EVALUATION")
        log_print("="*45)
        log_print(f"Test Accuracy: {overall_acc*100:.2f}%")
        log_print(f"Test F1-Score (Macro): {overall_f1:.4f}")
        log_print(f"Test ROC-AUC (Macro OVR): {overall_auc:.4f}" if not np.isnan(overall_auc) else "Test ROC-AUC: N/A")
        log_print("\nClassification Report:")
        log_print(classification_report(all_targets, all_preds, target_names=[top_diseases[i] for i in range(len(top_diseases))], zero_division=0))
        log_print("="*45 + "\n")
        
        log_print("="*45)
        log_print("SUBGROUP ANALYSIS BY SKIN TONE")
        log_print("="*45)
        for tone in unique_tones:
            if tone not in subgroup_results:
                log_print(f"No samples found in test split for skin tone group: {tone_names[tone]}")
                continue
            res = subgroup_results[tone]
            log_print(f"Subgroup: {tone_names[tone]} (N = {res['count']})")
            log_print(f"  Accuracy: {res['accuracy']*100:.2f}%")
            log_print(f"  F1-Score (Macro): {res['f1_score']:.4f}")
            log_print(f"  ROC-AUC (Macro OVR): {res['auc']:.4f}" if not np.isnan(res['auc']) else "  ROC-AUC : N/A")
            log_print("-" * 35)
        log_print("="*45)
        
    print(f"\n[SUCCESS] SCIN evaluation report successfully saved to: {report_path}")
    
    # Plot Training Curves
    plt.figure(figsize=(12, 5))
    epochs_range = range(1, len(train_losses) + 1)
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs_range, val_losses, 'r-o', label='Val Loss')
    plt.title('Training & Validation Loss (SCIN)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, 'b-o', label='Train Acc')
    plt.plot(epochs_range, val_accs, 'r-o', label='Val Acc')
    plt.title('Training & Validation Accuracy (SCIN)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    curves_plot_path = os.path.join("results_scin", "scin_training_curves.png")
    plt.savefig(curves_plot_path, dpi=150)
    plt.close()
    print(f"SCIN training curves plot saved to: {curves_plot_path}")
    print("[SUCCESS] SCIN Pipeline execution complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--scin_dir', type=str, default="SCIN", help="SCIN dataset directory path")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    parser.add_argument('--start_epoch', type=int, default=25, help="Epoch at which early stopping becomes active")
    parser.add_argument('--model', type=str, default="efficientnet", choices=["efficientnet", "swin", "dinov2"], help="Model architecture")
    args = parser.parse_args()
    
    run_scin_pipeline(
        scin_dir=args.scin_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        start_epoch=args.start_epoch,
        model_type=args.model
    )
