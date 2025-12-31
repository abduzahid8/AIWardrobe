"""
AIWardrobe Custom Clothing Classifier

Uses our trained YOLOv8 classifier for highly accurate clothing type detection.
Trained on 10K images with 91.5% accuracy.

Classes: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
from PIL import Image
import io
import base64

logger = logging.getLogger(__name__)

# Model path
MODEL_PATH = Path(__file__).parent.parent / "weights" / "clothing_classifier.pt"

# Class mapping from our trained model
CLOTHING_CLASSES = [
    "Ankle boot",
    "Bag", 
    "Coat",
    "Dress",
    "Pullover",
    "Sandal",
    "Shirt",
    "Sneaker",
    "T-shirt_top",
    "Trouser"
]

# Map to unified names used in AIWardrobe
CLASS_MAPPING = {
    "Ankle boot": "boots",
    "Bag": "bag",
    "Coat": "coat",
    "Dress": "dress",
    "Pullover": "sweater",
    "Sandal": "sandals",
    "Shirt": "shirt",
    "Sneaker": "sneakers",
    "T-shirt_top": "t-shirt",
    "Trouser": "pants"
}

# Singleton model instance
_model = None


def get_classifier():
    """Load the trained clothing classifier (lazy loading)."""
    global _model
    
    if _model is None:
        if not MODEL_PATH.exists():
            logger.warning(f"Clothing classifier not found at {MODEL_PATH}")
            return None
        
        try:
            from ultralytics import YOLO
            logger.info(f"Loading custom clothing classifier from {MODEL_PATH}")
            _model = YOLO(str(MODEL_PATH))
            logger.info("✅ Custom clothing classifier loaded!")
        except Exception as e:
            logger.error(f"Failed to load clothing classifier: {e}")
            return None
    
    return _model


def classify_clothing(
    image: Image.Image,
    top_k: int = 3
) -> List[Dict]:
    """
    Classify clothing in an image using our trained model.
    
    Args:
        image: PIL Image of the clothing item
        top_k: Number of top predictions to return
        
    Returns:
        List of {class_name, unified_name, confidence} dicts
    """
    model = get_classifier()
    if model is None:
        return []
    
    try:
        # Run classification
        results = model.predict(image, verbose=False)
        
        if not results or len(results) == 0:
            return []
        
        # Get probabilities
        probs = results[0].probs
        
        # Get top k predictions
        top_indices = probs.top5[:top_k]
        top_confs = probs.top5conf[:top_k]
        
        predictions = []
        for idx, conf in zip(top_indices, top_confs):
            idx = int(idx)
            conf = float(conf)
            
            class_name = CLOTHING_CLASSES[idx]
            unified_name = CLASS_MAPPING.get(class_name, class_name.lower())
            
            predictions.append({
                "class_name": class_name,
                "unified_name": unified_name,
                "confidence": conf
            })
        
        return predictions
        
    except Exception as e:
        logger.error(f"Classification error: {e}")
        return []


def classify_from_base64(
    b64_image: str,
    top_k: int = 3
) -> List[Dict]:
    """
    Classify clothing from base64 encoded image.
    
    Args:
        b64_image: Base64 encoded image string
        top_k: Number of top predictions
        
    Returns:
        List of predictions
    """
    try:
        # Decode base64
        if "," in b64_image:
            b64_image = b64_image.split(",")[1]
        
        img_data = base64.b64decode(b64_image)
        image = Image.open(io.BytesIO(img_data)).convert("RGB")
        
        return classify_clothing(image, top_k)
        
    except Exception as e:
        logger.error(f"Base64 classification error: {e}")
        return []


def get_specific_type(
    image: Image.Image,
    category: str
) -> Tuple[str, float]:
    """
    Get specific clothing type using our trained classifier.
    
    This is designed to be called from segmentation.py to enhance
    the specific_type detection.
    
    Args:
        image: PIL Image of the clothing item
        category: Base category from SegFormer (e.g., "upper_clothes")
        
    Returns:
        (specific_type, confidence) tuple
    """
    predictions = classify_clothing(image, top_k=1)
    
    if not predictions:
        return (category, 0.5)
    
    top_pred = predictions[0]
    return (top_pred["unified_name"], top_pred["confidence"])


# Test function
if __name__ == "__main__":
    import sys
    
    print("Testing AIWardrobe Clothing Classifier")
    print("="*50)
    
    model = get_classifier()
    if model is None:
        print("❌ Model not loaded")
        sys.exit(1)
    
    print(f"✅ Model loaded: {MODEL_PATH}")
    print(f"   Classes: {len(CLOTHING_CLASSES)}")
    print(f"   {CLOTHING_CLASSES}")
    
    # Test with a sample image if provided
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        image = Image.open(img_path)
        
        results = classify_clothing(image)
        print(f"\nClassification results for {img_path}:")
        for pred in results:
            print(f"   {pred['unified_name']}: {pred['confidence']:.2%}")
