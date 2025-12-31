"""
🎯 ULTIMATE DETECTOR - Simplified, Accurate Clothing Detection

This module provides a clean, reliable detection pipeline:
1. SegFormer for segmentation (what areas are clothing)
2. Simple CLIP for classification (what type of clothing)
3. rembg for clean cutouts (professional product images)

NO complex ensembles, NO texture analysis, NO YOLO validation.
Just simple, accurate detection.
"""

import cv2
import numpy as np
import base64
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ============================================
# 📦 DATA STRUCTURES
# ============================================

@dataclass
class DetectedClothingItem:
    """A single detected clothing item"""
    category: str  # SegFormer category (upper_clothes, pants, etc.)
    label: str  # User-friendly label (Jacket, Pants, Shoes)
    specific_type: str  # Specific type from CLIP (denim jacket, sneakers)
    color: str  # Primary color name
    color_hex: str  # Hex color code
    confidence: float
    bbox: List[int]  # [x, y, width, height]
    mask: Optional[np.ndarray] = None
    cutout_image: Optional[str] = None  # Base64 cutout


# ============================================
# 🏷️ SIMPLE LABEL MAPPING
# ============================================

# Map SegFormer categories to simple user-friendly labels
CATEGORY_LABELS = {
    "upper_clothes": "Top",
    "pants": "Pants",
    "dress": "Dress",
    "skirt": "Skirt",
    "hat": "Hat",
    "left_shoe": "Shoes",
    "right_shoe": "Shoes",
    "shoes": "Shoes",
    "bag": "Bag",
    "scarf": "Scarf",
    "sunglasses": "Sunglasses",
    "belt": "Belt",
    "coat": "Coat",
    "jacket": "Jacket",
    "shorts": "Shorts",
    "gloves": "Gloves",
}

# CLIP categories for classification - SIMPLE and ACCURATE
CLIP_CATEGORIES = {
    "upper": [
        "t-shirt", "shirt", "blouse", "sweater", "hoodie", 
        "jacket", "blazer", "cardigan", "polo shirt", "tank top",
        "crop top", "turtleneck", "vest"
    ],
    "outerwear": [
        "jacket", "coat", "blazer", "bomber jacket", "leather jacket",
        "denim jacket", "puffer jacket", "trench coat", "parka",
        "windbreaker", "fleece jacket"
    ],
    "bottoms": [
        "pants", "jeans", "trousers", "shorts", "skirt",
        "joggers", "leggings", "chinos", "cargo pants", "sweatpants"
    ],
    "footwear": [
        "sneakers", "shoes", "boots", "sandals", "loafers",
        "heels", "flats", "slides", "running shoes", "dress shoes"
    ],
    "accessories": [
        "hat", "cap", "beanie", "scarf", "bag", "backpack",
        "sunglasses", "belt", "watch", "jewelry"
    ]
}


def get_simple_label(category: str) -> str:
    """Get a simple user-friendly label for a category."""
    return CATEGORY_LABELS.get(category.lower(), "Clothing")


def get_clip_candidates(category: str) -> List[str]:
    """Get CLIP classification candidates for a category."""
    cat_lower = category.lower()
    
    if cat_lower in ["upper_clothes", "top"]:
        return CLIP_CATEGORIES["upper"] + CLIP_CATEGORIES["outerwear"]
    elif cat_lower in ["pants", "shorts", "skirt"]:
        return CLIP_CATEGORIES["bottoms"]
    elif "shoe" in cat_lower:
        return CLIP_CATEGORIES["footwear"]
    elif cat_lower in ["hat", "bag", "scarf", "belt", "sunglasses"]:
        return CLIP_CATEGORIES["accessories"]
    else:
        # Return all categories
        all_cats = []
        for cats in CLIP_CATEGORIES.values():
            all_cats.extend(cats)
        return all_cats


# ============================================
# 🎨 COLOR EXTRACTION
# ============================================

def extract_color(image: np.ndarray, mask: np.ndarray = None) -> Tuple[str, str]:
    """Extract the dominant color from an image region.
    
    Returns:
        (color_name, hex_code)
    """
    try:
        if mask is not None:
            # Get pixels from masked region
            pixels = image[mask > 127]
        else:
            pixels = image.reshape(-1, 3)
        
        if len(pixels) < 10:
            return "unknown", "#808080"
        
        # Sample for speed
        if len(pixels) > 1000:
            indices = np.random.choice(len(pixels), 1000, replace=False)
            pixels = pixels[indices]
        
        # Calculate average color
        avg_bgr = np.mean(pixels, axis=0).astype(int)
        avg_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))
        
        # Get color name
        color_name = _rgb_to_name(avg_rgb)
        hex_code = "#{:02x}{:02x}{:02x}".format(*avg_rgb)
        
        return color_name, hex_code
        
    except Exception as e:
        logger.warning(f"Color extraction failed: {e}")
        return "unknown", "#808080"


def _rgb_to_name(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB to a simple color name."""
    r, g, b = rgb
    
    # Calculate basic properties
    brightness = (r + g + b) / 3
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    saturation = (max_val - min_val) / max(max_val, 1)
    
    # Very dark = Black
    if brightness < 30:
        return "Black"
    
    # Very bright = White
    if brightness > 230 and saturation < 0.1:
        return "White"
    
    # Low saturation = Gray
    if saturation < 0.15:
        if brightness < 100:
            return "Dark Gray"
        elif brightness > 180:
            return "Light Gray"
        else:
            return "Gray"
    
    # Determine hue-based color
    if r > g and r > b:
        if g > b * 1.5:
            return "Orange" if brightness > 150 else "Brown"
        else:
            return "Red" if saturation > 0.5 else "Pink"
    elif g > r and g > b:
        return "Green"
    elif b > r and b > g:
        if r > g:
            return "Purple"
        else:
            return "Blue"
    elif r > 180 and g > 180 and b < 100:
        return "Yellow"
    elif r > 150 and g < 100 and b > 150:
        return "Purple"
    elif r > 100 and g > 80 and b < 80:
        return "Brown"
    else:
        return "Gray"


# ============================================
# 🔍 SIMPLE CLIP CLASSIFICATION
# ============================================

_clip_model = None
_clip_preprocess = None


def get_clip_model():
    """Load CLIP model lazily."""
    global _clip_model, _clip_preprocess
    
    if _clip_model is None:
        try:
            import open_clip
            
            # Use fashion-clip if available, otherwise ViT-B-32
            try:
                _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32',
                    pretrained='laion2b_s34b_b79k'
                )
                logger.info("Loaded CLIP ViT-B-32")
            except:
                _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
                    'ViT-B-32',
                    pretrained='openai'
                )
                logger.info("Loaded CLIP ViT-B-32 (OpenAI)")
                
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            return None, None
    
    return _clip_model, _clip_preprocess


def classify_with_clip(image: np.ndarray, candidates: List[str]) -> Tuple[str, float]:
    """Classify an image using CLIP.
    
    Args:
        image: BGR image of the clothing item
        candidates: List of possible clothing types
        
    Returns:
        (best_match, confidence)
    """
    try:
        import torch
        import open_clip
        from PIL import Image
        
        model, preprocess = get_clip_model()
        if model is None:
            return candidates[0] if candidates else "clothing", 0.5
        
        # Convert BGR to RGB and preprocess
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        
        # Prepare image
        image_tensor = preprocess(pil_image).unsqueeze(0)
        
        # Prepare text prompts with clear format
        prompts = [f"a photo of {item}" for item in candidates]
        
        # Tokenize
        tokenizer = open_clip.get_tokenizer('ViT-B-32')
        text_tokens = tokenizer(prompts)
        
        # Get embeddings
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tokens)
            
            # Normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            
            # Get best match
            values, indices = similarity[0].topk(1)
            best_idx = indices[0].item()
            confidence = values[0].item()
            
        return candidates[best_idx], confidence
        
    except Exception as e:
        logger.warning(f"CLIP classification failed: {e}")
        return candidates[0] if candidates else "clothing", 0.3


# ============================================
# ✂️ CUTOUT GENERATION
# ============================================

def create_cutout(image: np.ndarray, mask: np.ndarray, bbox: List[int] = None) -> str:
    """Create a clean cutout image with white background.
    
    Args:
        image: Full BGR image
        mask: Binary mask for the item
        bbox: Optional bounding box [x, y, w, h]
        
    Returns:
        Base64 encoded cutout image
    """
    try:
        h, w = image.shape[:2]
        
        # Create white background
        white_bg = np.ones_like(image) * 255
        
        # Apply mask
        mask_3ch = cv2.merge([mask, mask, mask])
        cutout = np.where(mask_3ch > 127, image, white_bg).astype(np.uint8)
        
        # Crop to bounding box if provided
        if bbox:
            x, y, bw, bh = bbox
            padding = 15
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + bw + padding)
            y2 = min(h, y + bh + padding)
            cutout = cutout[y1:y2, x1:x2]
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', cutout, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"
        
    except Exception as e:
        logger.error(f"Cutout creation failed: {e}")
        return None


# ============================================
# 🚀 MAIN DETECTION FUNCTION
# ============================================

def detect_clothing_ultimate(
    image: np.ndarray,
    create_cutouts: bool = True,
    min_area_percent: float = 0.5
) -> List[DetectedClothingItem]:
    """
    🎯 ULTIMATE CLOTHING DETECTION
    
    Simple, reliable, accurate.
    
    Pipeline:
    1. SegFormer detects clothing regions
    2. CLIP classifies each region
    3. Color extraction for each item
    4. Clean cutouts with white background
    
    Args:
        image: BGR input image
        create_cutouts: Whether to generate cutout images
        min_area_percent: Minimum area (% of image) for item detection
        
    Returns:
        List of detected items with labels, colors, and cutouts
    """
    from modules.segmentation import get_advanced_segmentor
    
    logger.info("=" * 60)
    logger.info("🎯 ULTIMATE DETECTION: Starting simple, accurate pipeline...")
    
    h, w = image.shape[:2]
    total_pixels = h * w
    min_area = total_pixels * (min_area_percent / 100)
    
    # Step 1: SegFormer segmentation
    logger.info("📊 Step 1: SegFormer segmentation...")
    segmentor = get_advanced_segmentor()
    seg_result = segmentor.segment(image, add_white_bg=False, return_items=True)
    
    logger.info(f"   Found {len(seg_result.items)} raw segments")
    
    # Step 2: Process each segment
    detected_items = []
    shoes_found = []
    
    for seg_item in seg_result.items:
        category = seg_item.category
        mask = seg_item.mask
        bbox = list(seg_item.bbox) if seg_item.bbox else None
        
        # Skip small items
        area = np.sum(mask > 127)
        if area < min_area:
            logger.info(f"   Skipping {category} (too small: {area/total_pixels*100:.1f}%)")
            continue
        
        # Skip if dress exists with separate top/bottom
        if category == "dress" and any(i.category in ["upper_clothes", "pants"] for i in detected_items):
            logger.info(f"   Skipping dress (have separate top/bottom)")
            continue
        
        # Group shoes
        if "shoe" in category.lower():
            shoes_found.append((seg_item, mask, bbox))
            continue
        
        # Get simple label
        simple_label = get_simple_label(category)
        
        # Crop region for CLIP
        if bbox:
            x, y, bw, bh = bbox
            x1, y1 = max(0, x - 10), max(0, y - 10)
            x2, y2 = min(w, x + bw + 10), min(h, y + bh + 10)
            cropped = image[y1:y2, x1:x2]
        else:
            cropped = image
        
        # Step 2a: CLIP classification
        candidates = get_clip_candidates(category)
        specific_type, clip_conf = classify_with_clip(cropped, candidates)
        
        logger.info(f"   {category} → CLIP: {specific_type} ({clip_conf:.2f})")
        
        # Step 2b: Color extraction
        color_name, color_hex = extract_color(image, mask)
        
        # Step 2c: Create cutout
        cutout_b64 = None
        if create_cutouts and mask is not None:
            cutout_b64 = create_cutout(image, mask, bbox)
        
        # Create item
        detected_items.append(DetectedClothingItem(
            category=category,
            label=simple_label,
            specific_type=specific_type,
            color=color_name,
            color_hex=color_hex,
            confidence=clip_conf,
            bbox=bbox,
            mask=mask,
            cutout_image=cutout_b64
        ))
    
    # Merge shoes
    if shoes_found:
        first_shoe = shoes_found[0]
        seg_item, mask, bbox = first_shoe
        
        # Merge all shoe masks
        combined_mask = mask.copy()
        for _, other_mask, _ in shoes_found[1:]:
            combined_mask = np.maximum(combined_mask, other_mask)
        
        # Classify shoes
        if bbox:
            x, y, bw, bh = bbox
            x1, y1 = max(0, x - 10), max(0, y - 10)
            x2, y2 = min(w, x + bw + 10), min(h, y + bh + 10)
            cropped = image[y1:y2, x1:x2]
        else:
            cropped = image
        
        candidates = get_clip_candidates("shoes")
        specific_type, clip_conf = classify_with_clip(cropped, candidates)
        color_name, color_hex = extract_color(image, combined_mask)
        
        cutout_b64 = None
        if create_cutouts:
            cutout_b64 = create_cutout(image, combined_mask, bbox)
        
        detected_items.append(DetectedClothingItem(
            category="shoes",
            label="Shoes",
            specific_type=specific_type,
            color=color_name,
            color_hex=color_hex,
            confidence=clip_conf,
            bbox=bbox,
            mask=combined_mask,
            cutout_image=cutout_b64
        ))
        
        logger.info(f"   shoes → CLIP: {specific_type} ({clip_conf:.2f})")
    
    logger.info(f"🎯 ULTIMATE DETECTION: {len(detected_items)} items found")
    for item in detected_items:
        logger.info(f"   ✅ {item.label}: {item.specific_type} ({item.color})")
    logger.info("=" * 60)
    
    return detected_items
