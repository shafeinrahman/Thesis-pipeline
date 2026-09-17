import os
# Force Hugging Face to use local directory for cache
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache", "huggingface")

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

def download_scin():
    print("=" * 60)
    print("Google SCIN Dataset Downloader via Hugging Face")
    print("=" * 60)
    print("This dataset is gated. To download it, you must:")
    print("1. Log in to https://huggingface.co/datasets/google/scin in your browser")
    print("2. Agree to the dataset terms and click 'Request Access'")
    print("3. Generate a 'Read' API token in your settings: https://huggingface.co/settings/tokens")
    print("=" * 60)
    
    dataset = None
    try:
        print("\nChecking for cached Hugging Face credentials...")
        dataset = load_dataset("google/scin", split="train")
        print("Successfully authenticated using cached credentials!")
    except Exception as credential_err:
        print(f"Cached credentials failed or not found: {credential_err}")
        print("Please authenticate manually:")
        token = input("Enter your Hugging Face Token: ").strip()
        if not token:
            print("[ERROR] Token is required to download gated datasets.")
            return
        try:
            from huggingface_hub import login
            login(token)
            print("\nLoading SCIN dataset from Hugging Face... (This will download metadata and cache files)")
            dataset = load_dataset("google/scin", split="train")
            print("Dataset loaded successfully!")
        except Exception as e:
            print(f"\n[ERROR] Failed to load dataset with provided token: {e}")
            print("\nPlease double check that:")
            print("1. Your HF Token is correct and active.")
            print("2. You have requested and been approved for access on: https://huggingface.co/datasets/google/scin")
            return

    if dataset is None:
        print("[ERROR] Dataset failed to load.")
        return

    scin_dir = "SCIN"
    images_dir = os.path.join(scin_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    metadata_records = []
    
    print("\nSaving images locally and extracting metadata...")
    # Iterating through dataset and saving images
    for idx, item in enumerate(tqdm(dataset)):
        case_id = item.get("case_id")
        
        # Get Fitzpatrick skin tone (check dermatologist estimates, then fallback to self-report)
        fst = None
        for fst_key in ["dermatologist_fitzpatrick_skin_type_label_1", "dermatologist_fitzpatrick_skin_type_label_2", "dermatologist_fitzpatrick_skin_type_label_3", "fitzpatrick_skin_type"]:
            val = item.get(fst_key)
            if isinstance(val, str) and val.startswith("FST"):
                fst = val
                break
                
        # Group Fitzpatrick scale string to numeric groupings:
        # 12: Light (FST1, FST2)
        # 34: Medium (FST3, FST4)
        # 56: Dark (FST5, FST6)
        fst_val = None
        if fst:
            try:
                num = int(fst[3:])
                if num in [1, 2]:
                    fst_val = 12
                elif num in [3, 4]:
                    fst_val = 34
                elif num in [5, 6]:
                    fst_val = 56
            except ValueError:
                pass

        # Extract primary condition label from weighted dermatologists consensus
        weighted_labels_str = item.get("weighted_skin_condition_label")
        primary_label = "unknown"
        if isinstance(weighted_labels_str, str) and weighted_labels_str.strip() != "":
            try:
                import ast
                weighted_labels = ast.literal_eval(weighted_labels_str)
                if isinstance(weighted_labels, dict) and len(weighted_labels) > 0:
                    primary_label = max(weighted_labels, key=weighted_labels.get)
            except Exception:
                pass
            
        # Iterate over possible images in the case
        for img_key in ["image_1_path", "image_2_path", "image_3_path"]:
            img = item.get(img_key)
            if img is None:
                continue
                
            img_num = img_key.split("_")[1] # e.g., "1", "2", or "3"
            filename = f"scin_{idx:05d}_{img_num}.png"
            dest_path = os.path.join(images_dir, filename)
            
            # Save PIL image as PNG if it doesn't already exist
            if not os.path.exists(dest_path):
                img.save(dest_path)
            
            metadata_records.append({
                "filename": filename,
                "case_id": case_id,
                "skin_tone": fst_val,
                "raw_fst": fst,
                "disease": primary_label
            })
            
    # Save metadata to CSV
    df = pd.DataFrame(metadata_records)
    csv_dest = os.path.join(scin_dir, "scin_metadata.csv")
    df.to_csv(csv_dest, index=False)
    
    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    print(f"Total Images Saved: {len(df)}")
    print(f"Metadata File: {csv_dest}")
    print("=" * 50)
    print("\nGoogle SCIN Dataset is fully set up in the 'SCIN' directory!")

if __name__ == "__main__":
    download_scin()
