#!/usr/bin/env python3
"""
Massimo Dutti LoRA Training Script
Creates training data with consistent captions for professional product photography style
"""

import os
import base64
import json
import zipfile
from pathlib import Path

# Configuration
TRAINING_DATA_DIR = "/Users/zohidvohidjonov/Desktop/AIWardrobe/alicevision-service/training_data/massimo_dutti"
OUTPUT_DIR = "/Users/zohidvohidjonov/Desktop/AIWardrobe/alicevision-service/training_data"

# LoRA training trigger word - this is what you'll use to activate the style
TRIGGER_WORD = "MDSTYLE"

# Base caption that describes the consistent style across all training images
BASE_CAPTION = f"{TRIGGER_WORD} professional product photograph, ghost mannequin presentation, clean white background, soft studio lighting, floating garment display, high-end fashion catalog photography, Massimo Dutti aesthetic, sharp fabric texture details, natural drape and shadows, premium e-commerce product shot"

def create_training_zip() -> str:
    """Create a ZIP file with images and captions in Replicate format"""
    print("� Creating training ZIP file...")
    
    training_dir = Path(TRAINING_DATA_DIR)
    zip_path = Path(OUTPUT_DIR) / "massimo_dutti_training.zip"
    
    # Get all images
    images = list(training_dir.glob("*.png")) + list(training_dir.glob("*.jpg")) + list(training_dir.glob("*.jpeg"))
    
    print(f"   Found {len(images)} images")
    
    captions = {}
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, img_path in enumerate(images, 1):
            img_name = img_path.name
            
            # Clean filename for caption file (no spaces)
            clean_name = img_path.stem.replace(" ", "_")
            new_img_name = f"{clean_name}{img_path.suffix}"
            
            # Add image with cleaned name
            zf.write(img_path, new_img_name)
            
            # Create caption file (same name but .txt extension)
            caption_name = f"{clean_name}.txt"
            zf.writestr(caption_name, BASE_CAPTION)
            
            captions[img_name] = BASE_CAPTION
            print(f"   [{i}/{len(images)}] Added: {new_img_name}")
    
    # Save captions to JSON for reference
    captions_file = Path(OUTPUT_DIR) / "captions.json"
    with open(captions_file, "w") as f:
        json.dump(captions, f, indent=2)
    
    print(f"\n   ✅ Created ZIP: {zip_path}")
    print(f"   📊 Size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return str(zip_path)

def print_training_instructions(zip_path: str):
    """Print instructions for training on Replicate"""
    print(f"""
{'='*60}
🚀 TRAINING INSTRUCTIONS
{'='*60}

📁 Your training ZIP is ready: {zip_path}

🏷️ Trigger word: {TRIGGER_WORD}

{'='*60}
OPTION 1: Train via Replicate Web UI (Recommended)
{'='*60}

1. Go to: https://replicate.com/ostris/flux-dev-lora-trainer/train

2. Click "Create a new model" or select existing destination

3. Upload the ZIP file:
   {zip_path}

4. Set these parameters:
   - trigger_word: {TRIGGER_WORD}
   - steps: 1500 (good balance for 31 images)
   - lora_rank: 16 (balanced quality/size)
   - learning_rate: 0.0004
   - batch_size: 1
   - resolution: 1024

5. Click "Create training" and wait (~20-40 min)

6. Once trained, you'll get model URL like:
   https://replicate.com/YOUR-USERNAME/massimo-dutti-style

{'='*60}
OPTION 2: Use Trained LoRA in your code
{'='*60}

After training, update product_card_generator.py:

```python
output = replicate.run(
    "YOUR-USERNAME/massimo-dutti-style:latest",
    input={{
        "prompt": "{TRIGGER_WORD} professional product photograph of your clothing item",
        "num_outputs": 1
    }}
)
```

The trigger word "{TRIGGER_WORD}" activates the Massimo Dutti style!

{'='*60}
""")

def main():
    print("=" * 60)
    print("🎨 Massimo Dutti LoRA Training Preparation")
    print("=" * 60)
    print(f"\n📝 Using trigger word: {TRIGGER_WORD}")
    print(f"📝 Base caption: {BASE_CAPTION[:60]}...\n")
    
    # Create training ZIP
    zip_path = create_training_zip()
    
    # Print instructions
    print_training_instructions(zip_path)
    
    print("✅ Training data preparation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
