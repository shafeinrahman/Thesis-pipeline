import os
import pandas as pd
import redivis
from tqdm import tqdm

class MockTable:
    pass

def download_dataset():
    print("=" * 60)
    print("Stanford DDI Dataset Direct Bulk Downloader (Arrow Bypass Mode)")
    print("=" * 60)
    print("To download the dataset, you need a Redivis API Token.")
    print("If you haven't generated one yet, please:")
    print("1. Log in to https://stanford.redivis.com")
    print("2. Click your profile picture in the top-right and select 'Settings'")
    print("3. Scroll to 'API tokens' and click 'Generate new token'")
    print("=" * 60)

    token = input("Enter your Redivis API Token (press Enter to skip if already authenticated): ").strip()
    if token:
        redivis.api_token = token
    
    try:
        # Reference the DDI dataset
        print("\nConnecting to Stanford DDI dataset on Redivis...")
        dataset = redivis.organization("aimi").dataset("ddi_diverse_dermatology_images:3r16")
        
        # 1. Download metadata CSV
        os.makedirs("DDI", exist_ok=True)
        csv_path = os.path.join("DDI", "ddi_metadata.csv")
        if not os.path.exists(csv_path):
            print("Locating metadata table...")
            metadata_table = dataset.table("ddi_metadata:kc7n")
            print(f"Downloading metadata CSV to: {csv_path}...")
            metadata_table.download(path=csv_path, format="csv", overwrite=True)
            print("Metadata downloaded successfully!")
        else:
            print("Metadata CSV already exists. Skipping download.")
            
        # 2. Download files list index as a CSV to get file_ids (avoids Arrow bug!)
        files_list_path = "files_list.csv"
        if not os.path.exists(files_list_path):
            print("Locating images index table...")
            files_table = dataset.table("files:srry")
            print("Downloading images list index as CSV (bypassing Arrow streams)...")
            files_table.download(path=files_list_path, format="csv", overwrite=True)
            print("Images index downloaded successfully!")
        else:
            print("Images index already exists. Skipping download.")
            
        # 3. Download each image file by constructing File objects manually
        print("\nLoading files list index...")
        df = pd.read_csv(files_list_path)
        print(f"Total files to download: {len(df)}")
        
        images_dir = os.path.join("DDI", "images")
        os.makedirs(images_dir, exist_ok=True)
        
        print("\nDownloading images individually... (using direct HTTP download engine)")
        success_count = 0
        skip_count = 0
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            filename = row["file_name"]
            file_id = row["file_id"]
            file_size = int(row["size"])
            file_hash = row["md5_hash"]
            added_at = row["added_at"]
            
            dest_path = os.path.join(images_dir, filename)
            
            # Skip if file already exists and has the correct size
            if os.path.exists(dest_path) and os.path.getsize(dest_path) == file_size:
                skip_count += 1
                continue
                
            try:
                properties = {
                    "size": file_size,
                    "md5_hash": file_hash,
                    "added_at": added_at
                }
                file_obj = redivis.classes.File.File(
                    id=file_id,
                    name=filename,
                    table=MockTable(),
                    directory=None,
                    properties=properties
                )
                
                # Download with progress bar hidden for individual files to keep output clean
                file_obj.download(images_dir, overwrite=True, progress=False)
                success_count += 1
            except Exception as fe:
                print(f"\n[Warning] Failed to download {filename} (ID: {file_id}): {fe}")
                
        print("\n" + "=" * 50)
        print("DOWNLOAD SUMMARY")
        print("=" * 50)
        print(f"Already Existed (Skipped): {skip_count}")
        print(f"Successfully Downloaded  : {success_count}")
        print(f"Total Images in Folder   : {len(os.listdir(images_dir))}")
        print("=" * 50)
        print("\nDDI Dataset is fully set up in the 'DDI' directory!")
        
    except Exception as e:
        print("\n[ERROR] Failed to download dataset.")
        print(f"Details: {e}")
        print("\nPlease double-check that:")
        print("1. Your API Token is correct and active.")
        print("2. You have been approved for access to the 'DDI - Diverse Dermatology Images' dataset on Redivis.")

if __name__ == "__main__":
    download_dataset()
