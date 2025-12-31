# Grounded SAM2 + FashionCLIP - Quick Start Guide

## 🚀 The Most Powerful AI Vision for Clothing Analysis

This implementation combines:
- **Grounded SAM2**: Text-prompted object detection + segmentation
- **FashionCLIP**: Specialized fashion attribute extraction
- **Card Generator**: AI-to-AI prompt generation for product photos

## Installation

### 1. Install Python Dependencies

```bash
cd alicevision-service
pip install -r requirements.txt
```

### 2. Download Model Weights

```bash
chmod +x setup_models.sh
./setup_models.sh
```

This downloads (~3GB total):
- Grounding DINO weights (~700MB)
- SAM2 base model (~350MB)  
- SAM2 large model (~2GB, optional)

### 3. Start the Service

```bash
python main.py
```

Service runs on `http://localhost:5050`

## 🎯 API Endpoints

### V2 Endpoints (Grounded SAM2 + FashionCLIP)

#### 1. Text-Prompted Clothing Detection
```bash
POST /api/v2/detect-clothing
```

**Request:**
```json
{
  "image": "base64_image_string",
  "prompts": ["shirt", "pants", "jacket"],
  "return_masks": true
}
```

**Response:**
```json
{
  "success": true,
  "detections": [
    {
      "category": "shirt",
      "confidence": 0.95,
      "bbox": [100, 50, 300, 400]
    }
  ],
  "mask_base64": "base64_mask_string",
  "processing_time": 2.3,
  "model": "grounded_sam2"
}
```

#### 2. Fashion Attribute Extraction
```bash
POST /api/v2/extract-fashion-attributes
```

**Request:**
```json
{
  "image": "base64_image_string",
  "roi": [100, 50, 300, 400]
}
```

**Response:**
```json
{
  "success": true,
  "category": "shirt",
  "subcategory": "t-shirt",
  "colors": [
    {"name": "navy blue", "confidence": 0.92},
    {"name": "white", "confidence": 0.15}
  ],
  "patterns": [
    {"name": "solid color", "confidence": 0.89}
  ],
  "styles": [
    {"name": "casual", "confidence": 0.87},
    {"name": "sporty", "confidence": 0.45}
  ],
  "fabric": "cotton",
  "details": {
    "neckline": "round neck",
    "sleeve_length": "short sleeve",
    "fit": "regular fit"
  },
  "description": "navy blue solid color cotton casual t-shirt"
}
```

#### 3. Generate Card Prompt
```bash
POST /api/v2/generate-card-prompt
```

**Request:**
```json
{
  "attributes": {
    "category": "shirt",
    "description": "navy blue cotton casual t-shirt",
    "colors": [{"name": "navy blue", "confidence": 0.92}],
    "details": {"neckline": "round neck"}
  },
  "style": "ecommerce",
  "include_model": false
}
```

**Response:**
```json
{
  "success": true,
  "prompt": "navy blue cotton casual t-shirt, front view, flat lay perspective, round neck, even studio lighting, no harsh shadows, pure white background (#FFFFFF), product photography, maximum detail, color accurate, straight-on angle, centered composition",
  "negative_prompt": "blurry, out of focus, low quality, pixelated, watermark, text, logo, person, human, model",
  "tags": ["t-shirt", "casual", "navy blue"],
  "metadata": {
    "category": "shirt",
    "colors": ["navy blue"],
    "style": "ecommerce",
    "include_model": false
  }
}
```

#### 4. Full V2 Pipeline
```bash
POST /api/v2/process-full-pipeline
```

**Request:**
```json
{
  "image": "base64_image_string",
  "prompts": ["shirt"],
  "style": "massimo_dutti"
}
```

**Response:**
```json
{
  "success": true,
  "segmentation": {
    "detections": [...],
    "mask_available": true,
    "processing_time": 2.1
  },
  "attributes": {
    "category": "shirt",
    "description": "...",
    ...
  },
  "card_prompt": {
    "prompt": "...",
    "negative_prompt": "...",
    ...
  },
  "total_processing_time": 3.5
}
```

## 📊 Style Presets

Available in `/api/v2/generate-card-prompt`:

- `ecommerce` - Clean product photography (#FFFFFF background)
- `massimo_dutti` - Premium fashion aesthetic
- `zara` - Natural window light, minimalist
- `hm` - Bright catalog style

## 🔧 Python Usage

```python
from modules import (
    get_grounded_sam,
    get_fashion_clip,
    generate_card_prompt
)

# Initialize models
grounded_sam = get_grounded_sam()
fashion_clip = get_fashion_clip()

# Detect clothing
result = grounded_sam.segment_clothing(
    image,
    prompts=["shirt", "pants"]
)

# Extract attributes
attrs = fashion_clip.extract_attributes(
    image,
    roi=result.detections[0].bbox
)

# Generate card prompt
prompt = generate_card_prompt(
    attrs,
    style="ecommerce"
)
```

## 💾 Model Files Structure

```
alicevision-service/
├── weights/
│   ├── groundingdino_swint_ogc.pth
│   ├── sam2_hiera_base.pt
│   └── sam2_hiera_large.pt
├── groundingdino/
│   └── config/
├── sam2/
│   └── configs/
└── modules/
    ├── grounded_sam.py
    ├── fashion_clip.py
    └── card_generator.py
```

## ⚡ Performance

- **Grounded SAM2**: 2-5 seconds per image (GPU)
- **FashionCLIP**: 0.5-1 second per image
- **Full Pipeline**: 3-6 seconds per image

## 🎨 Features

### Grounded SAM2
✅ Text-prompted detection ("find the shirt")
✅ 18+ clothing categories
✅ High-quality segmentation masks
✅ Multi-item detection

### FashionCLIP
✅ 30+ clothing categories
✅ 20+ color detection
✅ 10+ pattern types
✅ Style classification (casual, formal, sporty, etc.)
✅ Fabric detection
✅ Detailed features (neckline, sleeves, fit)

### Card Generator
✅ 4 style presets
✅ Optimized for Stable Diffusion/DALL-E
✅ Detailed negative prompts
✅ Automatic tag extraction

## 🌐 API Documentation

Interactive docs: `http://localhost:5050/docs`

## 🐛 Troubleshooting

### Models not loading
```bash
# Re-run setup
./setup_models.sh

# Check weights exist
ls -lh weights/
```

### GPU not detected
```python
# Check in Python
import torch
print(torch.cuda.is_available())
```

### Out of memory
- Use SAM2 base instead of large
- Process images one at a time
- Reduce image resolution before processing

## 📖 Integration with Main API

```javascript
// In api/routes/ai.js
const analyzeClothing = async (imageBase64) => {
  const response = await fetch('http://localhost:5050/api/v2/process-full-pipeline', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image: imageBase64,
      prompts: ['shirt', 'pants', 'dress', 'jacket'],
      style: 'ecommerce'
    })
  });
  
  const result = await response.json();
  
  // Use attributes for card generation
  const cardPrompt = result.card_prompt.prompt;
  const attributes = result.attributes;
  
  return { cardPrompt, attributes };
};
```

## 🚀 Next Steps

1. ✅ Models installed
2. ✅ Service running
3. ✅ Test endpoints with Postman/curl
4. 🔄 Integrate with main AIWardrobe API
5. 🔄 Test with real clothing images
6. 🔄 Deploy to production server with GPU
