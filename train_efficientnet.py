import os
# Force PyTorch Hub to use local directory for cache
os.environ["TORCH_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "torch")
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
from torchvision.datasets import ImageFolder

from sklearn.metrics import classification_report, roc_curve, auc, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# =====================================================================
# 1. Early Stopping Helper Class
# =====================================================================
class EarlyStopping:
    """
    Early stopping helper to monitor validation loss. 
    Stops training if validation loss does not improve for a certain number of epochs (patience).
    Saves and restores the best model weights.
    Only triggers early stopping after start_epoch.
    """
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

# =====================================================================
# 2. Dataset Class Definition (Fixes Windows file path separator bug)
# =====================================================================
class DDI_Dataset(ImageFolder):
    """
    Custom DDI Dataset class extending PyTorch's ImageFolder.
    It reads images from a directory and maps them to labels (malignant/benign)
    and skin tones (12, 34, 56) from the 'ddi_metadata.csv' file.
    """
    def __init__(self, root, csv_path=None, transform=None, *args, **kwargs):
        if csv_path is None:
            csv_path = os.path.join(root, "ddi_metadata.csv")
            
        assert os.path.exists(csv_path), f"Metadata CSV not found at: {csv_path}"
        super(DDI_Dataset, self).__init__(root, *args, transform=transform, **kwargs)
        
        # Load the metadata
        self.annotations = pd.read_csv(csv_path)
        
        # Ensure 'malignant' column exists.
        m_key = 'malignant'
        if m_key not in self.annotations.columns:
            if 'malignancy(malig=1)' in self.annotations.columns:
                self.annotations[m_key] = self.annotations['malignancy(malig=1)'].apply(lambda x: x == 1)
            else:
                raise KeyError("Could not find 'malignant' or 'malignancy(malig=1)' column in metadata CSV.")

    def __getitem__(self, index):
        img, _ = super(DDI_Dataset, self).__getitem__(index)
        path = self.imgs[index][0]
        
        # BUG FIX: extract filename robustly for both Windows and Linux paths
        filename = path.replace("\\", "/").split("/")[-1]
        
        row = self.annotations[self.annotations.DDI_file == filename]
        if row.empty:
            raise KeyError(f"Image filename '{filename}' not found in metadata CSV index.")
            
        is_malignant = int(row['malignant'].iloc[0]) # 1 if malignant, 0 if benign
        skin_tone = int(row['skin_tone'].iloc[0])     # 12, 34, or 56
        
        return path, img, is_malignant, skin_tone

# =====================================================================
# 3. Main Training & Evaluation Pipeline
# =====================================================================
def run_pipeline(data_dir="DDI", epochs=50, batch_size=32, lr=1e-4, patience=5, start_epoch=25, model_type="efficientnet"):
    print("=" * 60)
    print(f"STARTING {model_type.upper()} PIPELINE WITH EARLY STOPPING")
    print("=" * 60)
    
    # 3.1 Device Configuration (GPU vs CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        
    # 3.2 Transforms & Data Preprocessing
    means = [0.485, 0.456, 0.406]
    stds  = [0.229, 0.224, 0.225]
    
    train_transform = T.Compose([
        lambda x: x.convert('RGB'),
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(), # Data augmentation
        T.RandomRotation(15),     # Data augmentation
        T.ToTensor(),
        T.Normalize(mean=means, std=stds)
    ])
    
    val_transform = T.Compose([
        lambda x: x.convert('RGB'),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=means, std=stds)
    ])
    
    # 3.3 Load Dataset
    print(f"\nLoading dataset from: {data_dir}...")
    try:
        train_dataset = DDI_Dataset(data_dir, transform=train_transform)
        val_dataset = DDI_Dataset(data_dir, transform=val_transform)
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        return
        
    num_samples = len(train_dataset)
    print(f"Dataset successfully loaded. Total samples: {num_samples}")
    
    labels = []
    skin_tones = []
    for i in range(num_samples):
        _, _, label, skin_tone = train_dataset[i]
        labels.append(label)
        skin_tones.append(skin_tone)
        
    # Stratified split to ensure skin tone and label balance
    stratify_keys = [f"L{l}_S{s}" for l, s in zip(labels, skin_tones)]
    
    indices = np.arange(num_samples)
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=0.20, 
        random_state=42, 
        stratify=stratify_keys
    )
    
    print(f"Train split size: {len(train_idx)} samples")
    print(f"Test split size: {len(test_idx)} samples")
    
    train_subset = Subset(train_dataset, train_idx)
    test_subset = Subset(val_dataset, test_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # 3.4 Load pre-trained model architecture
    # 3.4 Load pre-trained model architecture
    if model_type.lower() == "swin":
        print("\n[INFO] Loading pre-trained Swin Transformer V2 (Tiny)...", flush=True)
        model_name = "Swin Transformer V2-T"
        weight_filename = "swintransformer_ddi.pth"
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
        
        # Modify the classification head for Swin
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, 2)
    elif model_type.lower() == "dinov2":
        print("\nLoading pre-trained DINOv2-S foundation model...")
        model_name = "DINOv2-S (ViT-S/14)"
        weight_filename = "dinov2_ddi.pth"
        try:
            model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            print("Pre-trained DINOv2 weights loaded successfully!")
        except Exception as e:
            print(f"Failed to fetch DINOv2 weights: {e}")
            raise e
        
        # Modify the classification head for DINOv2
        in_features = getattr(model, 'embed_dim', 384)
        model.head = nn.Linear(in_features, 2)
    else:
        print("\nLoading pre-trained EfficientNetV2-S model...")
        model_name = "EfficientNetV2-S"
        weight_filename = "efficientnetv2_ddi.pth"
        try:
            weights = models.EfficientNet_V2_S_Weights.DEFAULT
            model = models.efficientnet_v2_s(weights=weights)
            print("Pre-trained weights downloaded successfully!")
        except Exception as e:
            print(f"Failed to fetch weights automatically: {e}")
            print("Initializing model without pre-trained weights...")
            model = models.efficientnet_v2_s()
            
        # Modify the classification head for EfficientNet
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, 2)
    
    model = model.to(device)
    
    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    
    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=patience, start_epoch=start_epoch)
    
    # 3.5 Training & Validation Loop
    print(f"\nTraining for up to {epochs} epochs with Early Stopping (patience={patience})...")
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(1, epochs + 1):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        for batch_idx, (_, images, targets, _) in enumerate(train_loader):
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
        
        # --- Validation Phase ---
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
        
        # Check Early Stopping
        early_stopping(epoch_val_loss, model, epoch)
        if early_stopping.early_stop:
            print(f"\n[INFO] Early stopping triggered at epoch {epoch}! Restoring best weights.")
            model.load_state_dict(early_stopping.best_weights)
            break
            
    # If training completed without triggering early stopping, restore best weights anyway
    if not early_stopping.early_stop and early_stopping.best_weights is not None:
        print("\nRestoring best weights from training run.")
        model.load_state_dict(early_stopping.best_weights)
        
    # Save the best model parameters
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", weight_filename)
    torch.save(model.state_dict(), model_path)
    print(f"\nTrained model weights saved to: {model_path}")
    
    # 3.6 Overall & Subgroup Evaluation on Test Split
    print("\nRunning final evaluation on test split using best model...")
    model.eval()
    
    all_paths = []
    all_targets = []
    all_preds = []
    all_probs = []
    all_skin_tones = []
    
    with torch.no_grad():
        for paths, images, targets, tones in test_loader:
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of malignancy
            _, predicted = outputs.max(1)
            
            all_paths.extend(paths)
            all_targets.extend(targets.numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_skin_tones.extend(tones.numpy())
            
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    all_skin_tones = np.array(all_skin_tones)
    
    # Compute overall metrics
    overall_acc = accuracy_score(all_targets, all_preds)
    overall_f1 = f1_score(all_targets, all_preds, zero_division=0)
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    overall_auc = auc(fpr, tpr)
    
    # Compute subgroup metrics by skin tone
    unique_tones = [12, 34, 56]
    tone_names = {12: "Light (Fitzpatrick 1-2)", 34: "Medium (Fitzpatrick 3-4)", 56: "Dark (Fitzpatrick 5-6)"}
    
    subgroup_results = {}
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guessing (AUC = 0.50)")
    
    for tone in unique_tones:
        idx = np.where(all_skin_tones == tone)[0]
        if len(idx) == 0:
            continue
            
        sub_targets = all_targets[idx]
        sub_preds = all_preds[idx]
        sub_probs = all_probs[idx]
        
        sub_acc = accuracy_score(sub_targets, sub_preds)
        sub_f1 = f1_score(sub_targets, sub_preds, zero_division=0)
        
        if len(np.unique(sub_targets)) > 1:
            sub_fpr, sub_tpr, _ = roc_curve(sub_targets, sub_probs)
            sub_auc = auc(sub_fpr, sub_tpr)
            plt.plot(sub_fpr, sub_tpr, label=f"{tone_names[tone]} (AUC = {sub_auc:.3f}, N = {len(idx)})")
        else:
            sub_auc = float('nan')
            
        subgroup_results[tone] = {
            'accuracy': sub_acc,
            'f1_score': sub_f1,
            'auc': sub_auc,
            'count': len(idx)
        }

    # Open log file to save results
    os.makedirs("results", exist_ok=True)
    report_path = os.path.join("results", "evaluation_report.txt")
    
    with open(report_path, "w") as rf:
        def log_print(text):
            print(text)
            rf.write(str(text) + "\n")
            
        log_print("=" * 60)
        log_print(f"DDI DATASET EVALUATION REPORT - {model_name.upper()}")
        log_print("=" * 60)
        log_print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log_print(f"Model Architecture: {model_name}")
        log_print(f"Total Epochs Trained: {len(train_losses)} (Maximum set: {epochs})")
        log_print(f"Learning Rate: {lr}")
        log_print(f"Batch Size: {batch_size}")
        log_print(f"Device Used: {device}")
        if device.type == 'cuda':
            log_print(f"GPU Model: {torch.cuda.get_device_name(0)}")
        log_print("=" * 60 + "\n")
        
        log_print("="*45)
        log_print("OVERALL PERFORMANCE EVALUATION (BEST MODEL)")
        log_print("="*45)
        log_print(f"Test Accuracy: {overall_acc*100:.2f}%")
        log_print(f"Test F1-Score: {overall_f1:.4f}")
        log_print(f"Test Area Under ROC (AUC): {overall_auc:.4f}")
        log_print("\nClassification Report:")
        log_print(classification_report(all_targets, all_preds, target_names=["Benign", "Malignant"], zero_division=0))
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
            log_print(f"  F1-Score: {res['f1_score']:.4f}")
            log_print(f"  ROC-AUC : {res['auc']:.4f}" if not np.isnan(res['auc']) else "  ROC-AUC : N/A")
            
            if np.isnan(res['auc']):
                log_print(f"  [Warning] Subgroup has only one class present; AUC cannot be computed.")
            log_print("-" * 35)
            
        log_print("="*45)
        
    print(f"\n[SUCCESS] Console output report successfully saved to: {report_path}")
    
    # 3.7 Save Charts and Plots
    # Save the ROC curve
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Subgroup Analysis by Fitzpatrick Skin Tone")
    plt.legend(loc="lower right")
    roc_plot_path = os.path.join("results", "subgroup_roc_curve.png")
    plt.savefig(roc_plot_path, dpi=150)
    plt.close()
    print(f"\nSubgroup ROC analysis plot saved to: {roc_plot_path}")
    
    # Plot training curves (Loss and Accuracy)
    plt.figure(figsize=(12, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    epochs_range = range(1, len(train_losses) + 1)
    plt.plot(epochs_range, train_losses, 'b-o', label='Train Loss')
    plt.plot(epochs_range, val_losses, 'r-o', label='Val Loss')
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accs, 'b-o', label='Train Acc')
    plt.plot(epochs_range, val_accs, 'r-o', label='Val Acc')
    plt.title('Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    curves_plot_path = os.path.join("results", "training_curves.png")
    plt.savefig(curves_plot_path, dpi=150)
    plt.close()
    print(f"Training curves plot saved to: {curves_plot_path}")
    
    print("\n[SUCCESS] Pipeline execution complete!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="DDI", help="Dataset directory path")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--patience', type=int, default=5, help="Early stopping patience")
    parser.add_argument('--start_epoch', type=int, default=25, help="Epoch at which early stopping becomes active")
    parser.add_argument('--model', type=str, default="efficientnet", choices=["efficientnet", "swin", "dinov2"], help="Model architecture")
    args = parser.parse_args()
    
    run_pipeline(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        start_epoch=args.start_epoch,
        model_type=args.model
    )
