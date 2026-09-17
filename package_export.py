import os
import shutil
import zipfile

def create_export_package():
    export_dir = "export_pipeline"
    zip_filename = "thesis_pipeline_export.zip"
    
    # Clean old export artifacts if present
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    if os.path.exists(zip_filename):
        os.remove(zip_filename)
        
    os.makedirs(export_dir, exist_ok=True)
    
    files_to_copy = [
        "requirements.txt",
        "setup_environment.bat",
        "setup_environment.sh",
        "run_pipeline.bat",
        "run_pipeline.sh",
        "README_REMOTE_SETUP.md",
        "train_efficientnet.py",
        "train_scin.py",
        "train_padufes.py",
        "download_ddi.py",
        "download_scin.py",
        "download_skincon.py",
        "generate_dummy_ddi.py",
        "test_download_via_id.py",
        "files_list.csv"
    ]
    
    print("=" * 60)
    print("PACKAGING THESIS DEEP LEARNING PIPELINE FOR REMOTE PC")
    print("=" * 60)
    
    copied_count = 0
    for filename in files_to_copy:
        if os.path.exists(filename):
            dest_path = os.path.join(export_dir, filename)
            shutil.copy2(filename, dest_path)
            size_kb = os.path.getsize(dest_path) / 1024.0
            print(f"  [+] Copied: {filename:<28} ({size_kb:.1f} KB)")
            copied_count += 1
        else:
            print(f"  [!] Warning: {filename} not found, skipping.")
            
    # Also create empty placeholder directories with .gitkeep so folder structure is clear
    placeholder_dirs = ["DDI", "SCIN", "PAD-UFES-20", "SkinCon", "models", "results", "results_scin", "results_padufes"]
    for pdir in placeholder_dirs:
        full_pdir = os.path.join(export_dir, pdir)
        os.makedirs(full_pdir, exist_ok=True)
        with open(os.path.join(full_pdir, ".gitkeep"), "w") as f:
            f.write(f"# Placeholder directory for {pdir}\n")
            
    # Also copy README_REMOTE_SETUP.md as README.md in export dir
    shutil.copy2("README_REMOTE_SETUP.md", os.path.join(export_dir, "README.md"))
    
    # Create ZIP archive
    print(f"\nCompressing into '{zip_filename}'...")
    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zip_out:
        for root, dirs, files in os.walk(export_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, export_dir)
                zip_out.write(abs_path, rel_path)
                
    zip_size_kb = os.path.getsize(zip_filename) / 1024.0
    print("=" * 60)
    print(f"[SUCCESS] Export Package Ready!")
    print(f"  * Directory: '{os.path.abspath(export_dir)}'")
    print(f"  * ZIP File:  '{os.path.abspath(zip_filename)}' ({zip_size_kb:.1f} KB)")
    print("=" * 60)

if __name__ == "__main__":
    create_export_package()
