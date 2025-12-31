#!/usr/bin/env python3
"""
=============================================================================
AIWardrobe - YOLOv8 Clothing Detection Training Script
=============================================================================

This script fine-tunes YOLOv8 on clothing datasets for improved detection.

DATASETS USED (Free):
1. DeepFashion2 - 800K+ images, 13 clothing categories
2. Fashion-MNIST - 70K grayscale images
3. iMaterialist - 1M+ images with fine-grained labels

TRAINING PLATFORM: Google Colab (free GPU)

Usage:
    python train_yolo_clothing.py --dataset deepfashion2 --epochs 100
"""

import os
import argparse
import yaml
import requests
import zipfile
from pathlib import Path
import shutil

# ============================================
# DATASET CONFIGURATIONS
# ============================================

DATASETS = {
    "deepfashion2": {
        "name": "DeepFashion2",
        "url": "https://drive.google.com/file/d/1mYbWDBDGXF3YdW_-QXz1xJjj-YRdQCKd/view",
        "size": "~25GB",
        "categories": 13,
        "images": 800000,
        "description": "Largest free clothing detection dataset",
        "classes": [
            "short_sleeve_top", "long_sleeve_top", "short_sleeve_outwear", 
            "long_sleeve_outwear", "vest", "sling", "shorts", "trousers",
            "skirt", "short_sleeve_dress", "long_sleeve_dress", 
            "vest_dress", "sling_dress"
        ]
    },
    "fashionpedia": {
        "name": "Fashionpedia",
        "url": "https://s3.amazonaws.com/ifashionist-dataset/images/val2020.zip",
        "size": "~5GB",
        "categories": 46,
        "images": 50000,
        "description": "Fine-grained fashion attributes",
        "classes": [
            "shirt", "top", "sweater", "cardigan", "jacket", "vest",
            "pants", "shorts", "skirt", "coat", "dress", "jumpsuit",
            "cape", "glasses", "hat", "headband", "tie", "glove",
            "watch", "belt", "sock", "shoe", "bag", "scarf"
        ]
    },
    "modanet": {
        "name": "ModaNet",
        "url": "https://github.com/eBay/modanet",
        "size": "~8GB",
        "categories": 13,
        "images": 55000,
        "description": "Street fashion dataset",
        "classes": [
            "bag", "belt", "boots", "footwear", "outer", "dress",
            "sunglasses", "pants", "top", "shorts", "skirt", 
            "headwear", "scarf"
        ]
    }
}

# AIWardrobe custom clothing classes (unified from all datasets)
AIWARDROBE_CLASSES = [
    # Tops (10)
    "t-shirt", "shirt", "polo", "sweater", "hoodie",
    "cardigan", "tank_top", "blouse", "crop_top", "turtleneck",
    
    # Outerwear (8)
    "jacket", "blazer", "coat", "parka", "bomber",
    "leather_jacket", "denim_jacket", "windbreaker",
    
    # Bottoms (8)
    "jeans", "pants", "trousers", "chinos", "joggers",
    "shorts", "skirt", "leggings",
    
    # Dresses (4)
    "dress", "maxi_dress", "midi_dress", "mini_dress",
    
    # Footwear (8)
    "sneakers", "boots", "loafers", "heels", "sandals",
    "flats", "oxford_shoes", "combat_boots",
    
    # Accessories (8)
    "bag", "backpack", "handbag", "hat", "cap",
    "belt", "scarf", "sunglasses",
    
    # Total: 46 classes
]


def download_dataset(dataset_name: str, output_dir: str):
    """Download a clothing dataset."""
    if dataset_name not in DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available: {list(DATASETS.keys())}")
        return False
    
    dataset = DATASETS[dataset_name]
    print(f"\n{'='*60}")
    print(f"📦 Downloading {dataset['name']}")
    print(f"   Size: {dataset['size']}")
    print(f"   Images: {dataset['images']:,}")
    print(f"   Categories: {dataset['categories']}")
    print(f"{'='*60}\n")
    
    # For Google Drive links, user needs to download manually
    if "drive.google.com" in dataset["url"]:
        print(f"⚠️  This dataset requires manual download from Google Drive:")
        print(f"   {dataset['url']}")
        print(f"\n   After downloading, extract to: {output_dir}")
        return False
    
    # Direct download
    output_path = Path(output_dir) / f"{dataset_name}.zip"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading to {output_path}...")
    response = requests.get(dataset["url"], stream=True)
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size:
                pct = downloaded / total_size * 100
                print(f"\r   Progress: {pct:.1f}%", end="")
    
    print(f"\n✅ Downloaded {output_path}")
    
    # Extract
    print(f"📂 Extracting...")
    with zipfile.ZipFile(output_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    
    print(f"✅ Extracted to {output_dir}")
    return True


def create_yolo_config(classes: list, output_path: str):
    """Create YOLO dataset configuration file."""
    config = {
        "path": "./datasets/clothing",  # Relative path
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)}
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"✅ Created YOLO config: {output_path}")
    return output_path


def train_yolov8(
    dataset_yaml: str,
    model_size: str = "m",  # n, s, m, l, x
    epochs: int = 100,
    batch_size: int = 16,
    img_size: int = 640,
    device: str = "0"  # GPU ID or "cpu"
):
    """Train YOLOv8 on clothing dataset."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        return None
    
    print(f"\n{'='*60}")
    print(f"🚀 Training YOLOv8{model_size} for Clothing Detection")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Image size: {img_size}")
    print(f"   Device: {device}")
    print(f"{'='*60}\n")
    
    # Load pretrained model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project="runs/clothing",
        name=f"yolov8{model_size}_clothing",
        exist_ok=True,
        verbose=True,
        
        # Data augmentation
        hsv_h=0.015,  # Color hue augmentation
        hsv_s=0.7,    # Saturation augmentation
        hsv_v=0.4,    # Value/brightness augmentation
        degrees=10,   # Rotation
        translate=0.1,
        scale=0.5,
        fliplr=0.5,   # Horizontal flip
        mosaic=1.0,   # Mosaic augmentation
        
        # Training settings
        patience=50,   # Early stopping patience
        save_period=10,  # Save every N epochs
        workers=8,
        amp=True,  # Mixed precision training
    )
    
    print(f"\n✅ Training complete!")
    print(f"   Best model: runs/clothing/yolov8{model_size}_clothing/weights/best.pt")
    
    return results


def export_model(model_path: str, format: str = "onnx"):
    """Export trained model to different formats."""
    from ultralytics import YOLO
    
    model = YOLO(model_path)
    
    # Export
    exported = model.export(format=format)
    print(f"✅ Exported to {format}: {exported}")
    
    return exported


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 for clothing detection")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="modanet",
                       help="Dataset to use for training")
    parser.add_argument("--download", action="store_true",
                       help="Download dataset before training")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16,
                       help="Batch size")
    parser.add_argument("--model", choices=["n", "s", "m", "l", "x"], default="m",
                       help="YOLOv8 model size")
    parser.add_argument("--device", default="0",
                       help="GPU device (0, 1, etc.) or 'cpu'")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎨 AIWardrobe - Clothing Detection AI Training")
    print("="*60 + "\n")
    
    # Setup paths
    base_dir = Path(__file__).parent
    datasets_dir = base_dir / "datasets"
    config_path = base_dir / "clothing_dataset.yaml"
    
    # Download dataset if requested
    if args.download:
        download_dataset(args.dataset, str(datasets_dir / args.dataset))
    
    # Create YOLO config
    dataset = DATASETS[args.dataset]
    create_yolo_config(dataset["classes"], str(config_path))
    
    # Train
    train_yolov8(
        dataset_yaml=str(config_path),
        model_size=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device
    )


if __name__ == "__main__":
    main()
