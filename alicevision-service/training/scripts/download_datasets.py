#!/usr/bin/env python3
"""
=============================================================================
AIWardrobe - Dataset Downloader
=============================================================================

Downloads FREE clothing datasets for training AI:

1. iMaterialist Fashion 2019: 1M+ images with fine-grained labels
2. DeepFashion2: 800K+ images, 13 categories  
3. ModaNet: 55K street fashion images
4. Fashion-MNIST: 70K grayscale images (for testing)

Total: 2M+ images for FREE

Usage:
    python download_datasets.py --all
    python download_datasets.py --dataset imaterialist
"""

import os
import subprocess
import zipfile
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm
import json

# Dataset output directory
OUTPUT_DIR = Path(__file__).parent.parent / "datasets"


def download_file(url: str, output_path: Path, desc: str = None):
    """Download file with progress bar."""
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f, tqdm(
        desc=desc or output_path.name,
        total=total,
        unit='B',
        unit_scale=True
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_imaterialist():
    """
    Download iMaterialist Fashion 2019 (1M+ images)
    
    Requires Kaggle API credentials.
    Get them from: https://www.kaggle.com/settings -> Create New Token
    Place kaggle.json in ~/.kaggle/
    """
    print("\n" + "="*60)
    print("📦 iMaterialist Fashion 2019")
    print("   1M+ images, 46 fine-grained categories")
    print("="*60 + "\n")
    
    output_dir = OUTPUT_DIR / "imaterialist"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for Kaggle credentials
    kaggle_config = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_config.exists():
        print("❌ Kaggle API key not found!")
        print("\n   To download iMaterialist:")
        print("   1. Go to: https://www.kaggle.com/settings")
        print("   2. Click 'Create New Token' under API section")
        print("   3. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("   4. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("   5. Re-run this script")
        return False
    
    try:
        import kaggle
        
        print("📥 Downloading from Kaggle...")
        kaggle.api.competition_download_files(
            'imaterialist-fashion-2019-FGVC6',
            path=str(output_dir)
        )
        
        # Extract
        for zip_file in output_dir.glob("*.zip"):
            print(f"📂 Extracting {zip_file.name}...")
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(output_dir)
        
        print(f"✅ iMaterialist downloaded to {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n   Manual download:")
        print("   https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/data")
        return False


def download_modanet():
    """
    Download ModaNet (55K images)
    """
    print("\n" + "="*60)
    print("📦 ModaNet")
    print("   55K street fashion images, 13 categories")
    print("="*60 + "\n")
    
    output_dir = OUTPUT_DIR / "modanet"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download annotations from the repo
    print("📥 Downloading annotations...")
    anno_url = "https://raw.githubusercontent.com/eBay/modanet/master/annotations/modanet2018_instances_train.json"
    download_file(anno_url, output_dir / "instances_train.json", "Train annotations")
    
    anno_url_val = "https://raw.githubusercontent.com/eBay/modanet/master/annotations/modanet2018_instances_val.json"
    download_file(anno_url_val, output_dir / "instances_val.json", "Val annotations")
    
    print("\n⚠️  ModaNet images need to be downloaded separately:")
    print("   Images are from Paperdoll dataset")
    print("   Download from: http://vision.is.tohoku.ac.jp/~kyamagu/research/paperdoll/")
    print(f"\n   Place in: {output_dir / 'images'}/")
    
    return True


def download_deepfashion2():
    """
    Download DeepFashion2 (800K images)
    
    Requires Google Drive access.
    """
    print("\n" + "="*60)
    print("📦 DeepFashion2")
    print("   800K+ images, 13 clothing categories")
    print("="*60 + "\n")
    
    output_dir = OUTPUT_DIR / "deepfashion2"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("⚠️  DeepFashion2 requires manual download from Google Drive:")
    print("\n   1. Go to: https://github.com/switchablenorms/DeepFashion2")
    print("   2. Fill out the Google Form to get access")
    print("   3. Download the dataset (~25GB)")
    print(f"   4. Extract to: {output_dir}/")
    
    print("\n   Dataset structure:")
    print("   deepfashion2/")
    print("   ├── train/")
    print("   │   ├── image/ (191,961 images)")
    print("   │   └── annos/")
    print("   └── validation/")
    print("       ├── image/")
    print("       └── annos/")
    
    return False


def download_fashion_mnist():
    """
    Download Fashion-MNIST (70K images)
    Quick test dataset - grayscale 28x28 images.
    """
    print("\n" + "="*60)
    print("📦 Fashion-MNIST")
    print("   70K grayscale images, 10 categories (testing only)")
    print("="*60 + "\n")
    
    output_dir = OUTPUT_DIR / "fashion_mnist"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz"
    ]
    
    for f in files:
        print(f"📥 Downloading {f}...")
        download_file(base_url + f, output_dir / f, f)
    
    print(f"✅ Fashion-MNIST downloaded to {output_dir}")
    return True


def download_lookbook():
    """
    Download LookBook dataset (9K images)
    Fashion influencer photos.
    """
    print("\n" + "="*60)
    print("📦 LookBook Dataset")
    print("   9K fashion influencer photos")
    print("="*60 + "\n")
    
    output_dir = OUTPUT_DIR / "lookbook"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("⚠️  LookBook dataset:")
    print("   Download from: https://dgyoo.github.io/")
    print(f"   Extract to: {output_dir}/")
    
    return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download clothing datasets")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--dataset", choices=[
        "imaterialist", "modanet", "deepfashion2", "fashion_mnist", "lookbook"
    ], help="Specific dataset to download")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎨 AIWardrobe Dataset Downloader")
    print("   Total Available: 2M+ free clothing images")
    print("="*60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    datasets = {
        "imaterialist": download_imaterialist,
        "modanet": download_modanet,
        "deepfashion2": download_deepfashion2,
        "fashion_mnist": download_fashion_mnist,
        "lookbook": download_lookbook,
    }
    
    if args.all:
        for name, func in datasets.items():
            func()
    elif args.dataset:
        datasets[args.dataset]()
    else:
        # Interactive
        print("\n📊 Available Datasets:\n")
        print("  1. iMaterialist   - 1M+ images (Kaggle API required)")
        print("  2. ModaNet        - 55K images")
        print("  3. DeepFashion2   - 800K images (Google Drive)")
        print("  4. Fashion-MNIST  - 70K images (small, for testing)")
        print("  5. LookBook       - 9K images")
        print("\n  A. Download ALL\n")
        
        choice = input("Select (1-5 or A): ").strip().lower()
        
        if choice == 'a':
            for func in datasets.values():
                func()
        elif choice == '1':
            download_imaterialist()
        elif choice == '2':
            download_modanet()
        elif choice == '3':
            download_deepfashion2()
        elif choice == '4':
            download_fashion_mnist()
        elif choice == '5':
            download_lookbook()
    
    print("\n" + "="*60)
    print("📂 Datasets saved to:", OUTPUT_DIR)
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
