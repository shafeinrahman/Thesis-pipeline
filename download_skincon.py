import os
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def download_file(url, dest_path):
    """Downloads a single file from a URL to a destination path."""
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            f.write(response.content)
        return True
    except Exception as e:
        return f"Error: {e}"

def download_image_task(img_id, url, dest_dir):
    """Downloads a single image and handles potential failures."""
    dest_path = os.path.join(dest_dir, img_id)
    if os.path.exists(dest_path):
        return img_id, "Skipped (Exists)"
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, timeout=10, headers=headers)
        if response.status_code == 200:
            with open(dest_path, 'wb') as f:
                f.write(response.content)
            return img_id, "Success"
        else:
            return img_id, f"HTTP Status {response.status_code}"
    except Exception as e:
        return img_id, f"Exception: {e}"

def download_skincon():
    print("=" * 60)
    print("SkinCON Dataset Downloader")
    print("=" * 60)
    
    # Define directories
    skincon_dir = "SkinCon"
    images_dir = os.path.join(skincon_dir, "fitzpatrick17k_images")
    os.makedirs(images_dir, exist_ok=True)
    
    # 1. Download Metadata and Annotations
    urls = {
        "annotations_ddi.csv": "https://skincon-dataset.github.io/files/annotations_ddi.csv",
        "annotations_fitzpatrick17k.csv": "https://skincon-dataset.github.io/files/annotations_fitzpatrick17k.csv",
        "fitzpatrick17k.csv": "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"
    }
    
    print("\n[Step 1] Downloading annotation and metadata CSVs...")
    for filename, url in urls.items():
        dest = os.path.join(skincon_dir, filename)
        if os.path.exists(dest):
            print(f"  - {filename} already exists. Skipping download.")
            continue
        print(f"  - Downloading {filename}...")
        res = download_file(url, dest)
        if res is not True:
            print(f"  [ERROR] Failed to download {filename}: {res}")
            return
    print("[SUCCESS] All metadata files are ready.")
    
    # 2. Load and merge annotations to find the images to download
    print("\n[Step 2] Resolving image URLs for Fitzpatrick17k subset...")
    try:
        ann_fitz = pd.read_csv(os.path.join(skincon_dir, "annotations_fitzpatrick17k.csv"))
        meta_fitz = pd.read_csv(os.path.join(skincon_dir, "fitzpatrick17k.csv"))
    except Exception as e:
        print(f"[ERROR] Failed to read downloaded metadata files: {e}")
        return
    
    # Get the unique ImageIDs that SkinCON annotated
    annotated_ids = set(ann_fitz['ImageID'].dropna().unique())
    print(f"  - Total Fitzpatrick17k images annotated by SkinCON: {len(annotated_ids)}")
    
    # Map md5hash -> url from the original fitzpatrick17k dataset (adding '.jpg' to match SkinCON ImageID)
    url_map = dict(zip(meta_fitz['md5hash'].astype(str) + '.jpg', meta_fitz['url']))
    
    # Build list of download tasks
    download_tasks = []
    missing_urls = 0
    for img_id in annotated_ids:
        url = url_map.get(img_id)
        if isinstance(url, str) and url.strip() != "":
            # Clean url if needed (e.g. fix leading slashes)
            if url.startswith("//"):
                url = "https:" + url
            download_tasks.append((img_id, url))
        else:
            missing_urls += 1
            
    if missing_urls > 0:
        print(f"  [Warning] {missing_urls} annotated images do not have a URL in fitzpatrick17k.csv")
    
    print(f"  - Resolved {len(download_tasks)} image URLs for download.")
    
    # 3. Parallel Image Download
    print("\n[Step 3] Starting parallel download of Fitzpatrick17k images (Thread Pool size: 16)...")
    success_count = 0
    skipped_count = 0
    fail_count = 0
    failures = []
    
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_image_task, img_id, url, images_dir): img_id for img_id, url in download_tasks}
        
        # Use tqdm to track progress bar
        for future in tqdm(as_completed(futures), total=len(download_tasks), desc="Downloading Images"):
            img_id = futures[future]
            try:
                img_id, result = future.result()
                if result == "Success":
                    success_count += 1
                elif result == "Skipped (Exists)":
                    skipped_count += 1
                else:
                    fail_count += 1
                    failures.append((img_id, result))
            except Exception as e:
                fail_count += 1
                failures.append((img_id, f"Executor Error: {e}"))
                
    # 4. Final summary
    print("\n" + "=" * 50)
    print("DOWNLOAD SUMMARY")
    print("=" * 50)
    print(f"Successfully Downloaded : {success_count}")
    print(f"Already Existed (Skipped): {skipped_count}")
    print(f"Failed to Download      : {fail_count}")
    print("=" * 50)
    
    if fail_count > 0:
        print("\nNote: Some original image URLs might be deprecated or timed out.")
        print(f"The first 5 failures: {failures[:5]}")
        print("To download remaining failed images, you can rerun this downloader later.")
    
    print("\nSkinCON dataset files are successfully configured in 'SkinCon' folder!")

if __name__ == "__main__":
    download_skincon()
