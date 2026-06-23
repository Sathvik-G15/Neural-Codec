import os
import json
import subprocess
import argparse
from pathlib import Path

def setup_kaggle_dataset(target_dir, username):
    """
    Creates the metadata and uploads the extracted dataset to Kaggle.
    """
    # 1. Verify the directory exists
    if not os.path.exists(target_dir):
        print(f"Error: Could not find dataset directory at {target_dir}")
        print("Did you finish running download_vimeo90k.py yet?")
        return

    print(f"Preparing to upload dataset from: {target_dir}")
    
    # 2. Check if Kaggle CLI is installed
    try:
        subprocess.run(["kaggle", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\n[!] Kaggle CLI is not installed or not in your PATH.")
        print("    Please run 'pip install kaggle' in your terminal before running this script.")
        print("    Also, make sure you've placed your kaggle.json in C:\\Users\\sathv\\.kaggle\\kaggle.json")
        return

    # 3. Create the dataset metadata
    metadata_path = os.path.join(target_dir, 'dataset-metadata.json')
    dataset_name = "vimeo90k-septuplet-custom-v3"
    
    metadata = {
      "title": f"Vimeo90K Septuplet Extracted {username} v3",
      "id": f"{username}/{dataset_name}",
      "licenses": [{"name": "CC0-1.0"}]
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Created metadata: {metadata_path}")
    print(f"Dataset will be named: {username}/{dataset_name}")

    # 4. Upload!
    print("\nStarting upload to Kaggle... This will take a long time (up to 18 hours depending on internet speed).")
    print("Please do not close this window.")
    
    try:
        # We run the command using subprocess so it streams output properly to the terminal
        subprocess.run(["kaggle", "datasets", "create", "-p", target_dir, "-r", "tar"], check=True)
        print("\n[SUCCESS] Upload finished successfully!")
        print("You can now find your dataset in Kaggle under your 'Datasets' tab.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Kaggle upload failed. {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload Vimeo90k to Kaggle.")
    parser.add_argument("--username", type=str, required=True, 
                        help="Your Kaggle username (required to generate the dataset ID)")
    args = parser.parse_args()

    # I found this from your scripts/download_vimeo90k.py!
    # IMPORTANT: We upload ONLY the extracted folder, NOT the parent folder.
    # If we uploaded F:\Vimeo90k_Dataset, it would upload BOTH the 82GB zip AND the 82GB extracted files!
    dataset_dir = r"F:\Vimeo90k_Dataset\vimeo_septuplet"
    
    setup_kaggle_dataset(dataset_dir, args.username)
