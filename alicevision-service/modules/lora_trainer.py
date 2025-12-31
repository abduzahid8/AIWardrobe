"""
🎓 MASSIMO DUTTI LORA TRAINER

Scripts for:
1. Preparing training data (resize, caption)
2. Training LoRA on Replicate
3. Using the trained model
"""

import os
import json
import base64
import requests
from PIL import Image
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Configuration
TRAINING_DATA_DIR = Path(__file__).parent.parent / "assets" / "training_data"
TRIGGER_WORD = "MASSIMODUTTI"
TARGET_SIZE = (1024, 1024)


def prepare_training_data():
    """
    Prepare images for LoRA training:
    1. Resize all images to 1024x1024
    2. Generate captions
    3. Create metadata.jsonl
    """
    all_images = []
    
    # Find all images
    for category in ["tops", "pants", "shoes"]:
        category_dir = TRAINING_DATA_DIR / category
        if category_dir.exists():
            for img_file in category_dir.glob("*.png"):
                all_images.append((img_file, category))
            for img_file in category_dir.glob("*.jpg"):
                all_images.append((img_file, category))
            for img_file in category_dir.glob("*.jpeg"):
                all_images.append((img_file, category))
    
    if not all_images:
        logger.warning("No training images found in assets/training_data/")
        logger.info("Please add Massimo Dutti product images to:")
        logger.info(f"  - {TRAINING_DATA_DIR}/tops/")
        logger.info(f"  - {TRAINING_DATA_DIR}/pants/")
        logger.info(f"  - {TRAINING_DATA_DIR}/shoes/")
        return None
    
    # Prepare output directory
    output_dir = TRAINING_DATA_DIR / "prepared"
    output_dir.mkdir(exist_ok=True)
    
    metadata = []
    
    for idx, (img_path, category) in enumerate(all_images):
        try:
            # Load and resize image
            img = Image.open(img_path)
            img = img.convert("RGB")
            
            # Resize to target size (center crop + resize)
            img = resize_and_crop(img, TARGET_SIZE)
            
            # Save resized image
            output_path = output_dir / f"{idx:04d}.jpg"
            img.save(output_path, "JPEG", quality=95)
            
            # Generate caption
            caption = generate_caption(img_path.stem, category)
            
            metadata.append({
                "file_name": output_path.name,
                "text": caption
            })
            
            logger.info(f"Prepared: {img_path.name} -> {output_path.name}")
            
        except Exception as e:
            logger.error(f"Failed to process {img_path}: {e}")
    
    # Save metadata
    metadata_path = output_dir / "metadata.jsonl"
    with open(metadata_path, "w") as f:
        for item in metadata:
            f.write(json.dumps(item) + "\n")
    
    logger.info(f"\n✅ Prepared {len(metadata)} images for training")
    logger.info(f"   Output directory: {output_dir}")
    logger.info(f"   Metadata file: {metadata_path}")
    
    return output_dir


def resize_and_crop(img: Image.Image, target_size: tuple) -> Image.Image:
    """Resize image to target size with center crop."""
    # Calculate aspect ratios
    target_ratio = target_size[0] / target_size[1]
    img_ratio = img.width / img.height
    
    if img_ratio > target_ratio:
        # Image is wider - crop width
        new_width = int(img.height * target_ratio)
        left = (img.width - new_width) // 2
        img = img.crop((left, 0, left + new_width, img.height))
    else:
        # Image is taller - crop height
        new_height = int(img.width / target_ratio)
        top = (img.height - new_height) // 2
        img = img.crop((0, top, img.width, top + new_height))
    
    # Resize to target
    return img.resize(target_size, Image.LANCZOS)


def generate_caption(filename: str, category: str) -> str:
    """Generate training caption for an image."""
    # Base caption with trigger word
    if category == "tops":
        return f"{TRIGGER_WORD} style, professional e-commerce product photo, clothing item, ghost mannequin photography, clean off-white gradient background, studio lighting, soft shadows, high-end fashion catalog, 8k quality"
    elif category == "pants":
        return f"{TRIGGER_WORD} style, professional e-commerce product photo, pants, elegant flat lay presentation, clean off-white gradient background, studio lighting, soft shadows, high-end fashion catalog, 8k quality"
    elif category == "shoes":
        return f"{TRIGGER_WORD} style, professional e-commerce product photo, shoes, angled pair display, clean off-white gradient background, studio lighting, soft ground shadows, high-end fashion catalog, 8k quality"
    else:
        return f"{TRIGGER_WORD} style, professional e-commerce product photo, clean off-white gradient background, studio lighting, high-end fashion catalog, 8k quality"


def train_lora_on_replicate(prepared_dir: Path):
    """
    Train LoRA on Replicate using their FLUX trainer.
    
    Prerequisites:
    - REPLICATE_API_TOKEN environment variable set
    - Training images prepared in prepared_dir
    """
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        logger.error("REPLICATE_API_TOKEN not set")
        return None
    
    try:
        import replicate
        
        # Create ZIP file of training data
        import shutil
        zip_path = TRAINING_DATA_DIR / "training_data.zip"
        shutil.make_archive(
            str(zip_path.with_suffix("")),
            "zip",
            prepared_dir
        )
        
        logger.info(f"Created training ZIP: {zip_path}")
        
        # Upload and train
        logger.info("Starting LoRA training on Replicate...")
        logger.info("This may take 1-2 hours...")
        
        # Use Replicate's training API
        # Note: This uses Replicate's training feature
        training = replicate.trainings.create(
            version="ostris/flux-dev-lora-trainer:d995297071a44dcb72244e6c19462111649ec86a9f1e0a0f4c4cd8cdee61dfe3",
            input={
                "input_images": open(zip_path, "rb"),
                "trigger_word": TRIGGER_WORD,
                "steps": 1500,
                "lora_rank": 32,
                "learning_rate": 1e-4,
                "batch_size": 1,
                "resolution": "1024",
                "autocaption": False,  # We provide our own captions
            },
            destination=f"zohidvohidjonov/massimo-dutti-style"  # Your model destination
        )
        
        logger.info(f"Training started: {training.id}")
        logger.info(f"Monitor at: https://replicate.com/trainings/{training.id}")
        
        return training.id
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def use_trained_lora(item_description: str, lora_url: str = None) -> str:
    """
    Generate product image using our trained LoRA.
    
    Args:
        item_description: Description of the item (e.g., "navy blue wool cardigan")
        lora_url: URL to trained LoRA weights (or use default)
    
    Returns:
        Base64 encoded generated image
    """
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    if not api_token:
        logger.error("REPLICATE_API_TOKEN not set")
        return None
    
    try:
        import replicate
        
        # Construct prompt with trigger word
        prompt = f"{TRIGGER_WORD} style, professional e-commerce product photo of {item_description}, ghost mannequin photography, clean off-white gradient background, studio lighting, soft shadows, high-end fashion catalog, 8k quality"
        
        # If we have a custom LoRA URL, use it
        if lora_url:
            output = replicate.run(
                "lucataco/flux-dev-lora:a22c463f11808638ad5e2ebd582e07a469031f48dd567366fb4c6fdab91d614d",
                input={
                    "prompt": prompt,
                    "hf_lora": lora_url,
                    "num_outputs": 1,
                    "aspect_ratio": "4:5",
                    "output_format": "jpg",
                    "output_quality": 95
                }
            )
        else:
            # Fallback to base FLUX
            output = replicate.run(
                "black-forest-labs/flux-1.1-pro",
                input={
                    "prompt": prompt,
                    "num_outputs": 1,
                    "aspect_ratio": "4:5",
                    "output_format": "jpg",
                    "output_quality": 95
                }
            )
        
        if output:
            result_url = output[0] if isinstance(output, list) else output
            response = requests.get(result_url, timeout=60)
            if response.status_code == 200:
                return base64.b64encode(response.content).decode()
        
        return None
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return None


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🎓 MASSIMO DUTTI LORA TRAINER")
    print("=" * 60)
    
    # Step 1: Prepare training data
    print("\n📁 Step 1: Checking training data...")
    prepared_dir = prepare_training_data()
    
    if prepared_dir:
        print("\n🚀 Step 2: Ready to train!")
        print("   Run: python lora_trainer.py --train")
    else:
        print("\n⚠️ Please add training images first:")
        print(f"   1. Download 20-30 images from massimodutti.com")
        print(f"   2. Save to: {TRAINING_DATA_DIR}/tops/")
        print(f"   3. Save to: {TRAINING_DATA_DIR}/pants/")
        print(f"   4. Save to: {TRAINING_DATA_DIR}/shoes/")
        print(f"   5. Run this script again")
