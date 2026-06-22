import os
import requests
import zipfile
from pathlib import Path

def download_and_extract(url, target_dir):
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    zip_path = os.path.join(target_dir, "vimeo_septuplet.zip")
    
    if not os.path.exists(zip_path):
        print(f"Downloading Vimeo-90k septuplets (this is ~82GB, it will take a while)...")
        # Use requests with stream=True to write chunks DIRECTLY to the F: drive.
        # This prevents Python from caching the 82GB file in the C: drive's temp folder.
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Download complete!")
    else:
        print(f"Found existing zip at {zip_path}")

    extract_path = os.path.join(target_dir, "vimeo_septuplet")
    if not os.path.exists(extract_path):
        print("Extracting zip file...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        print("Extraction complete!")
    else:
        print(f"Dataset already extracted at {extract_path}")

if __name__ == "__main__":
    vimeo_url = "http://data.csail.mit.edu/tofu/dataset/vimeo_septuplet.zip"
    # Download directly into the new external hard drive to save space
    target_dir = r"F:\Vimeo90k_Dataset"
    download_and_extract(vimeo_url, target_dir)

