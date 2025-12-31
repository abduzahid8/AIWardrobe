"""
Advanced Clothing Segmentation with SegFormer
AI-powered 18-category clothing detection for AIWardrobe
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List, Any
from PIL import Image
from dataclasses import dataclass
import logging
import io
import base64

logger = logging.getLogger(__name__)

# Clothing categories from SegFormer model (raw IDs)
CLOTHING_CATEGORIES = {
    0: "background",
    1: "hat",
    2: "hair",
    3: "sunglasses",
    4: "upper_clothes",
    5: "skirt",
    6: "pants",
    7: "dress",
    8: "belt",
    9: "left_shoe",
    10: "right_shoe",
    11: "face",
    12: "left_leg",
    13: "right_leg",
    14: "left_arm",
    15: "right_arm",
    16: "bag",
    17: "scarf",
}

# Human-readable category names (improved naming)
CATEGORY_DISPLAY_NAMES = {
    "hat": "Hat",
    "sunglasses": "Sunglasses",
    "upper_clothes": "Top",
    "skirt": "Skirt",
    "pants": "Pants",
    "dress": "Dress",
    "belt": "Belt",
    "left_shoe": "Shoes",
    "right_shoe": "Shoes",
    "bag": "Bag",
    "scarf": "Scarf",
}

# Expanded clothing type aliases for better detection understanding
CLOTHING_TYPE_EXPANSIONS = {
    # Upper body categories
    "upper_clothes": [
        "shirt", "blouse", "top", "sweater", "hoodie", "cardigan",
        "t-shirt", "polo", "tank top", "crop top", "vest", "turtleneck",
        "sweatshirt", "henley", "button-down", "flannel", "jersey",
        "pullover", "tunic", "camisole", "bodysuit"
    ],
    "jacket": [
        "jacket", "blazer", "coat", "parka", "bomber", "denim jacket",
        "leather jacket", "windbreaker", "puffer", "peacoat", "trench coat",
        "overcoat", "raincoat", "anorak", "varsity jacket", "fleece",
        "quilted jacket", "safari jacket", "biker jacket"
    ],
    
    # Lower body categories
    "pants": [
        "pants", "jeans", "trousers", "chinos", "joggers", "leggings",
        "cargo pants", "dress pants", "sweatpants", "khakis", "corduroys",
        "wide-leg pants", "slim fit pants", "straight pants", "cropped pants",
        "palazzo pants", "culottes", "track pants"
    ],
    "skirt": [
        "skirt", "mini skirt", "maxi skirt", "midi skirt", "pleated skirt",
        "pencil skirt", "a-line skirt", "wrap skirt", "denim skirt",
        "leather skirt", "tulle skirt", "flared skirt"
    ],
    "shorts": [
        "shorts", "bermudas", "athletic shorts", "denim shorts", "cargo shorts",
        "chino shorts", "swim shorts", "running shorts", "boxer shorts"
    ],
    
    # Full body categories
    "dress": [
        "dress", "maxi dress", "midi dress", "mini dress", "gown",
        "sundress", "cocktail dress", "evening dress", "wrap dress",
        "shift dress", "sheath dress", "a-line dress", "bodycon dress",
        "shirt dress", "slip dress", "maxi dress", "kaftan"
    ],
    "jumpsuit": [
        "jumpsuit", "romper", "overalls", "dungarees", "playsuit",
        "coveralls", "one-piece"
    ],
    
    # Footwear categories
    "left_shoe": [
        "shoes", "sneakers", "loafers", "oxfords", "boots", "heels",
        "sandals", "flats", "mules", "espadrilles", "trainers", "brogues",
        "derby shoes", "monk straps", "chelsea boots", "ankle boots",
        "knee boots", "combat boots", "hiking boots", "slip-ons",
        "high tops", "low tops", "running shoes", "basketball shoes",
        "slides", "flip flops", "wedges", "platforms", "stilettos"
    ],
    "right_shoe": [
        "shoes", "sneakers", "loafers", "oxfords", "boots", "heels",
        "sandals", "flats", "mules", "espadrilles", "trainers", "brogues"
    ],
    
    # Accessories categories
    "bag": [
        "bag", "handbag", "purse", "backpack", "tote", "clutch",
        "crossbody", "shoulder bag", "messenger bag", "satchel",
        "bucket bag", "hobo bag", "weekender", "duffel", "fanny pack",
        "belt bag", "wristlet", "briefcase", "laptop bag"
    ],
    "sunglasses": [
        "sunglasses", "glasses", "eyewear", "aviators", "wayfarers",
        "round glasses", "cat-eye glasses", "sport glasses"
    ],
    "hat": [
        "hat", "cap", "beanie", "beret", "fedora", "bucket hat",
        "baseball cap", "snapback", "trucker hat", "sun hat", "visor",
        "panama hat", "newsboy cap", "flat cap", "cowboy hat"
    ],
    "scarf": [
        "scarf", "wrap", "shawl", "bandana", "neckerchief", "infinity scarf",
        "pashmina", "stole", "blanket scarf"
    ],
    "belt": [
        "belt", "waist belt", "leather belt", "chain belt", "fabric belt",
        "braided belt", "dress belt", "casual belt"
    ],
    
    # Jewelry and accessories (detected via other means)
    "jewelry": [
        "watch", "necklace", "bracelet", "earrings", "ring",
        "pendant", "chain", "bangle", "cuff", "anklet", "brooch"
    ],
}

# Pattern detection keywords
PATTERN_TYPES = [
    "solid", "striped", "plaid", "checkered", "floral", "paisley",
    "polka dot", "animal print", "leopard", "zebra", "camouflage",
    "geometric", "abstract", "tropical", "tie-dye", "ombre",
    "herringbone", "houndstooth", "argyle", "gingham", "tartan"
]

# Material detection keywords
MATERIAL_TYPES = [
    "cotton", "denim", "leather", "suede", "wool", "cashmere",
    "silk", "satin", "velvet", "linen", "polyester", "nylon",
    "fleece", "corduroy", "tweed", "jersey", "chiffon", "lace",
    "mesh", "canvas", "rubber", "faux leather", "synthetic"
]

# ============================================
# 🎯 4-LAYER ASSURANCE STACK - PHASE 1
# Confidence Thresholds for High Reliability
# ============================================

# Confidence thresholds (Layer 3: Conformal Prediction)
# NOTE: Lowered from original values - they were too strict and rejecting good detections
CONFIDENCE_THRESHOLDS = {
    "auto_accept": 0.70,    # 70%+ → process automatically, high reliability
    "flag_review": 0.40,    # 40-70% → accept but mark as needs_review
    "reject": 0.25,         # <25% → reject, obvious noise only
}

# Minimum confidence by category (LOWERED - some items have naturally low confidence)
CATEGORY_MIN_CONFIDENCE = {
    "upper_clothes": 0.35,  # Tops - lowered from 0.55
    "pants": 0.35,          # Lowered from 0.55
    "shoes": 0.20,          # Often small in frame, low confidence - lowered from 0.50
    "dress": 0.40,
    "skirt": 0.35,          # Often confused with pants - lowered from 0.60
    "bag": 0.30,
    "hat": 0.25,            # Often small - lowered from 0.45
    "belt": 0.20,           # Very small
    "scarf": 0.20,
    "sunglasses": 0.20,
}

# Multi-frame consensus thresholds (Layer 2)
MULTI_FRAME_CONSENSUS = {
    "min_frames_required": 3,       # Need at least 3 frames
    "agreement_threshold": 0.60,    # Item must appear in 60%+ of frames
    "best_frame_strategy": "highest_confidence",  # Use frame with best detection
}


def apply_confidence_filter(detections: list, min_threshold: float = None) -> list:
    """
    Layer 3: Apply conformal prediction confidence filtering.
    
    Removes low-confidence detections that would cause errors.
    Returns filtered list with confidence status flags.
    """
    filtered = []
    
    for item in detections:
        conf = getattr(item, 'confidence', 0.5)
        category = getattr(item, 'category', 'unknown')
        
        # Get category-specific minimum confidence
        if min_threshold is None:
            min_threshold = CATEGORY_MIN_CONFIDENCE.get(category, CONFIDENCE_THRESHOLDS["reject"])
        
        # Filter by confidence
        if conf < min_threshold:
            logger.info(f"  ❌ REJECTED {category} (conf={conf:.2f} < {min_threshold:.2f})")
            continue
        
        # Determine status
        if conf >= CONFIDENCE_THRESHOLDS["auto_accept"]:
            status = "verified"
        elif conf >= CONFIDENCE_THRESHOLDS["flag_review"]:
            status = "needs_review"
        else:
            status = "low_confidence"
        
        # Add status to item
        item.confidence_status = status
        filtered.append(item)
        logger.info(f"  ✅ ACCEPTED {category} (conf={conf:.2f}, status={status})")
    
    return filtered


def apply_multi_frame_consensus(frame_detections: List[List], min_agreement: float = 0.60) -> List[dict]:
    """
    Layer 2: Multi-frame consensus for video analysis.
    
    Items must appear consistently across frames to be accepted.
    This filters out random noise and single-frame misdetections.
    
    Args:
        frame_detections: List of detection lists, one per frame
        min_agreement: Minimum fraction of frames item must appear in (default 60%)
        
    Returns:
        List of consensus items with best frame info
    """
    if len(frame_detections) < MULTI_FRAME_CONSENSUS["min_frames_required"]:
        # Not enough frames for consensus, return all from last frame
        logger.warning(f"Not enough frames for consensus ({len(frame_detections)} < {MULTI_FRAME_CONSENSUS['min_frames_required']})")
        return frame_detections[-1] if frame_detections else []
    
    # Track item appearances across frames
    item_appearances = {}  # category -> list of (frame_idx, item, confidence)
    
    for frame_idx, frame_items in enumerate(frame_detections):
        for item in frame_items:
            category = getattr(item, 'category', str(item))
            specific_type = getattr(item, 'specific_type', category)
            conf = getattr(item, 'confidence', 0.5)
            
            key = category  # Group by base category
            if key not in item_appearances:
                item_appearances[key] = []
            item_appearances[key].append({
                'frame_idx': frame_idx,
                'item': item,
                'confidence': conf,
                'specific_type': specific_type
            })
    
    # Determine consensus items
    total_frames = len(frame_detections)
    consensus_items = []
    
    for category, appearances in item_appearances.items():
        agreement = len(appearances) / total_frames
        
        if agreement >= min_agreement:
            # Item appears in enough frames - ACCEPT
            # Find best frame (highest confidence)
            best = max(appearances, key=lambda x: x['confidence'])
            best_item = best['item']
            
            # Add consensus metadata
            best_item.consensus_agreement = agreement
            best_item.consensus_frame_count = len(appearances)
            best_item.consensus_total_frames = total_frames
            
            consensus_items.append(best_item)
            logger.info(f"  ✅ CONSENSUS: {category} appeared in {len(appearances)}/{total_frames} frames ({agreement:.0%}) - ACCEPTED")
        else:
            # Item doesn't appear consistently - REJECT
            logger.info(f"  ❌ NO CONSENSUS: {category} appeared in {len(appearances)}/{total_frames} frames ({agreement:.0%}) - REJECTED")
    
    logger.info(f"Multi-frame consensus: {len(consensus_items)}/{len(item_appearances)} items passed")
    return consensus_items


# Comprehensive color name mapping (RGB ranges to color names)
COLOR_NAMES = {
    # Neutrals
    "Black": (0, 0, 0),
    "Charcoal": (54, 69, 79),
    "Dark Gray": (64, 64, 64),
    "Gray": (128, 128, 128),
    "Silver": (192, 192, 192),
    "Light Gray": (211, 211, 211),
    "White": (255, 255, 255),
    "Off-White": (250, 249, 246),
    "Ivory": (255, 255, 240),
    "Cream": (255, 253, 208),
    "Beige": (245, 245, 220),
    "Tan": (210, 180, 140),
    "Taupe": (72, 60, 50),
    
    # Browns
    "Chocolate": (123, 63, 0),
    "Brown": (139, 69, 19),
    "Saddle Brown": (139, 90, 43),
    "Cognac": (154, 70, 20),
    "Caramel": (255, 213, 154),
    "Camel": (193, 154, 107),
    "Khaki": (195, 176, 145),
    
    # Reds
    "Burgundy": (128, 0, 32),
    "Maroon": (128, 0, 0),
    "Wine": (114, 47, 55),
    "Crimson": (220, 20, 60),
    "Red": (255, 0, 0),
    "Cherry": (222, 49, 99),
    "Coral": (255, 127, 80),
    "Salmon": (250, 128, 114),
    "Rose": (255, 0, 127),
    "Blush": (255, 111, 255),
    
    # Pinks
    "Hot Pink": (255, 105, 180),
    "Pink": (255, 192, 203),
    "Light Pink": (255, 182, 193),
    "Dusty Rose": (194, 129, 140),
    "Mauve": (224, 176, 255),
    "Fuchsia": (255, 0, 255),
    
    # Oranges
    "Burnt Orange": (204, 85, 0),
    "Orange": (255, 165, 0),
    "Tangerine": (255, 204, 0),
    "Peach": (255, 218, 185),
    "Apricot": (251, 206, 177),
    "Rust": (183, 65, 14),
    "Terracotta": (226, 114, 91),
    
    # Yellows
    "Mustard": (255, 219, 88),
    "Gold": (255, 215, 0),
    "Yellow": (255, 255, 0),
    "Lemon": (255, 247, 0),
    "Butter": (255, 255, 191),
    "Pale Yellow": (255, 255, 224),
    
    # Greens
    "Forest Green": (34, 139, 34),
    "Hunter Green": (53, 94, 59),
    "Dark Green": (0, 100, 0),
    "Green": (0, 128, 0),
    "Olive": (128, 128, 0),
    "Army Green": (75, 83, 32),
    "Sage": (188, 184, 138),
    "Mint": (189, 252, 201),
    "Seafoam": (159, 226, 191),
    "Lime": (50, 205, 50),
    "Emerald": (80, 200, 120),
    "Teal": (0, 128, 128),
    
    # Blues
    "Navy Blue": (0, 0, 128),
    "Dark Blue": (0, 0, 139),
    "Cobalt": (0, 71, 171),
    "Royal Blue": (65, 105, 225),
    "Denim Blue": (21, 96, 189),
    "Blue": (0, 0, 255),
    "Cornflower": (100, 149, 237),
    "Steel Blue": (70, 130, 180),
    "Sky Blue": (135, 206, 235),
    "Light Blue": (173, 216, 230),
    "Baby Blue": (137, 207, 240),
    "Powder Blue": (176, 224, 230),
    "Ice Blue": (153, 255, 255),
    "Turquoise": (64, 224, 208),
    "Aqua": (0, 255, 255),
    "Cyan": (0, 255, 255),
    
    # Purples
    "Eggplant": (97, 64, 81),
    "Plum": (142, 69, 133),
    "Purple": (128, 0, 128),
    "Violet": (138, 43, 226),
    "Lavender": (230, 230, 250),
    "Lilac": (200, 162, 200),
    "Orchid": (218, 112, 214),
    "Magenta": (255, 0, 255),
    "Indigo": (75, 0, 130),
}


def get_color_name(rgb: Tuple[int, int, int]) -> str:
    """Convert RGB to nearest color name"""
    min_dist = float('inf')
    closest_color = "Unknown"
    
    for name, color_rgb in COLOR_NAMES.items():
        dist = sum((a - b) ** 2 for a, b in zip(rgb, color_rgb))
        if dist < min_dist:
            min_dist = dist
            closest_color = name
    
    return closest_color


def extract_dominant_colors(image: np.ndarray, mask: np.ndarray = None, n_colors: int = 3) -> List[Dict]:
    """
    Extract dominant colors from an image region using K-means clustering.
    
    Args:
        image: BGR image
        mask: Optional mask to limit color extraction region
        n_colors: Number of dominant colors to extract
    
    Returns:
        List of dicts with color info: name, hex, rgb, percentage
    """
    from sklearn.cluster import KMeans
    
    # Convert to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply mask if provided
    if mask is not None:
        # Ensure mask is 2D
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # Get pixels where mask is non-zero
        pixels = rgb_image[mask > 127].reshape(-1, 3)
    else:
        pixels = rgb_image.reshape(-1, 3)
    
    if len(pixels) < n_colors:
        return [{"name": "Unknown", "hex": "#000000", "rgb": (0, 0, 0), "percentage": 100}]
    
    # Sample pixels for speed (max 10000)
    if len(pixels) > 10000:
        indices = np.random.choice(len(pixels), 10000, replace=False)
        pixels = pixels[indices]
    
    # K-means clustering
    kmeans = KMeans(n_clusters=min(n_colors, len(pixels)), random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Get cluster centers and counts
    colors = kmeans.cluster_centers_.astype(int)
    labels = kmeans.labels_
    counts = np.bincount(labels)
    
    # Sort by frequency
    sorted_indices = np.argsort(counts)[::-1]
    
    result = []
    total = len(labels)
    
    for idx in sorted_indices:
        # Convert numpy.int64 to Python int for JSON serialization
        rgb = tuple(int(x) for x in colors[idx])
        percentage = float(counts[idx]) / total * 100
        hex_color = '#{:02x}{:02x}{:02x}'.format(*rgb)
        name = get_color_name(rgb)
        
        result.append({
            "name": name,
            "hex": hex_color,
            "rgb": rgb,
            "percentage": round(percentage, 1)
        })
    
    return result


def infer_specific_type(category: str, primary_color: str, image: np.ndarray = None, mask: np.ndarray = None) -> str:
    """
    Infer specific clothing type based on category and visual features.
    Returns specific type like 'denim jacket', 'skinny jeans', 'high-top sneakers'.
    
    Args:
        category: Base category (e.g., 'upper_clothes', 'pants', 'left_shoe')
        primary_color: Detected primary color name
        image: Optional BGR image for texture analysis
        mask: Optional mask for item region
        
    Returns:
        Specific type string
    """
    color_lower = primary_color.lower() if primary_color else ""
    
    # === UPPER BODY - ADVANCED DETECTION ===
    if category in ["upper_clothes", "top"]:
        
        # Analyze visual features if image is available
        is_thick_fabric = False
        is_structured = False
        has_collar = False
        is_heavy = False
        aspect_ratio = 1.0
        edge_density = 0.0
        
        if image is not None and mask is not None:
            try:
                # Get masked region for analysis
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Calculate aspect ratio of the garment
                contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cnt = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect_ratio = w / h if h > 0 else 1.0
                
                # Edge density analysis (jackets/coats have more structure = more edges)
                edges = cv2.Canny(gray, 50, 150)
                masked_edges = edges & mask.astype(np.uint8)
                edge_density = np.sum(masked_edges > 0) / max(np.sum(mask > 0), 1) * 100
                
                # Texture variance (thick fabrics like wool have more texture)
                masked_pixels = gray[mask > 127]
                if len(masked_pixels) > 0:
                    texture_variance = np.var(masked_pixels)
                    is_thick_fabric = texture_variance > 400  # Wool, fleece, heavy cotton
                    is_structured = edge_density > 15  # Jackets, blazers have structure
                
                # Check for collar region (top 15% of garment)
                if h > 0:
                    collar_region = mask[y:y+int(h*0.15), :]
                    collar_edges = edges[y:y+int(h*0.15), :]
                    collar_edge_density = np.sum(collar_edges > 0) / max(np.sum(collar_region > 0), 1) * 100
                    has_collar = collar_edge_density > 20
                
            except Exception as e:
                pass  # Fall back to color-only analysis
        
        # === JACKET/COAT DETECTION ===
        # Jackets: wider aspect ratio, more structure, heavier colors
        if is_structured or edge_density > 12 or aspect_ratio > 0.9:
            # Likely a jacket or structured garment
            if "denim" in color_lower or _is_denim_texture(image, mask) if image is not None else False:
                return "denim jacket"
            elif color_lower in ["black", "brown", "cognac", "saddle brown", "chocolate"]:
                if is_thick_fabric or edge_density > 18:
                    return "leather jacket"
                else:
                    return "blazer"
            elif color_lower in ["olive", "army green", "khaki", "forest green", "hunter green"]:
                return "bomber jacket"
            elif color_lower in ["navy blue", "charcoal", "dark gray"]:
                return "blazer"
            elif color_lower in ["tan", "camel", "beige"]:
                return "trench coat"
            elif is_thick_fabric:
                if color_lower in ["gray", "heather", "light gray"]:
                    return "fleece jacket"
                else:
                    return "puffer jacket"
            else:
                return "jacket"
        
        # === HOODIE DETECTION ===
        # Hoodies: thick fabric, typically gray/black/navy, casual colors
        if is_thick_fabric or color_lower in ["heather", "charcoal"]:
            if color_lower in ["gray", "heather", "dark gray", "light gray", "charcoal"]:
                return "hoodie"
            elif color_lower in ["black", "navy blue", "midnight blue"]:
                return "zip-up hoodie"
            elif color_lower in ["burgundy", "forest green", "olive"]:
                return "pullover hoodie"
        
        # === SWEATER DETECTION ===
        if is_thick_fabric:
            if color_lower in ["cream", "ivory", "off-white", "beige"]:
                return "cable knit sweater"
            elif color_lower in ["burgundy", "wine", "maroon"]:
                return "crewneck sweater"
            elif color_lower in ["navy blue", "forest green"]:
                return "v-neck sweater"
            elif color_lower in ["gray", "charcoal"]:
                return "cashmere sweater"
            else:
                return "sweater"
        
        # === CARDIGAN DETECTION ===
        if has_collar and not is_structured:
            if color_lower in ["cream", "beige", "tan", "gray"]:
                return "cardigan"
        
        # === SHIRT DETECTION ===
        if has_collar:
            if "plaid" in color_lower or "checkered" in color_lower or color_lower in ["red", "green"]:
                return "flannel shirt"
            elif color_lower in ["white", "off-white", "light blue", "sky blue"]:
                return "button-down shirt"
            elif color_lower in ["navy blue", "midnight blue", "dark blue"]:
                return "polo shirt"
            elif color_lower in ["pink", "light pink"]:
                return "dress shirt"
            else:
                return "button-down shirt"
        
        # === SWEATSHIRT DETECTION ===
        if color_lower in ["gray", "heather", "dark gray", "light gray"]:
            return "crewneck sweatshirt"
        
        # === T-SHIRT (Default for upper body) ===
        if color_lower in ["white", "off-white"]:
            return "white t-shirt"
        elif color_lower in ["black"]:
            return "black t-shirt"
        elif color_lower in ["navy blue"]:
            return "navy t-shirt"
        elif color_lower in ["red", "crimson"]:
            return "graphic tee"
        else:
            return "t-shirt"
    
    # === JACKETS/OUTERWEAR (separate category) ===
    if category in ["jacket", "coat", "outerwear"]:
        if "denim" in color_lower or (image is not None and _is_denim_texture(image, mask)):
            return "denim jacket"
        elif color_lower in ["black", "brown", "cognac", "saddle brown"]:
            return "leather jacket"
        elif color_lower in ["olive", "army green", "khaki", "tan"]:
            return "bomber jacket"
        elif color_lower in ["navy blue", "dark blue", "charcoal"]:
            return "blazer"
        elif color_lower in ["gray", "heather"]:
            return "fleece jacket"
        elif color_lower in ["tan", "camel", "beige"]:
            return "trench coat"
        else:
            return "jacket"

    
    # === PANTS ===
    if category == "pants":
        if "denim" in color_lower or (image is not None and _is_denim_texture(image, mask)):
            # Check darkness for wash type
            if color_lower in ["black", "charcoal", "dark gray"]:
                return "black jeans"
            elif color_lower in ["navy blue", "denim blue", "dark blue", "raw denim"]:
                return "dark wash jeans"
            elif color_lower in ["light blue", "sky blue", "light wash"]:
                return "light wash jeans"
            else:
                return "jeans"
        elif color_lower in ["black", "charcoal"]:
            return "dress pants"
        elif color_lower in ["khaki", "tan", "beige", "sand"]:
            return "chinos"
        elif color_lower in ["gray", "light gray", "heather"]:
            return "joggers"
        elif color_lower in ["olive", "army green"]:
            return "cargo pants"
        else:
            return "pants"
    
    # === SHORTS ===
    if category == "shorts":
        if "denim" in color_lower or (image is not None and _is_denim_texture(image, mask)):
            return "denim shorts"
        elif color_lower in ["khaki", "tan", "beige"]:
            return "chino shorts"
        else:
            return "shorts"
    
    # === SKIRTS ===
    if category == "skirt":
        if "denim" in color_lower or (image is not None and _is_denim_texture(image, mask)):
            return "denim skirt"
        elif color_lower in ["black", "navy blue"]:
            return "pencil skirt"
        else:
            return "skirt"
    
    # === DRESSES ===
    if category == "dress":
        if color_lower in ["black", "charcoal"]:
            return "cocktail dress"
        elif color_lower in ["white", "off-white", "cream", "ivory"]:
            return "sundress"
        elif "floral" in color_lower or "print" in color_lower:
            return "sundress"
        else:
            return "midi dress"
    
    # === FOOTWEAR ===
    if category in ["left_shoe", "right_shoe", "shoes"]:
        if color_lower in ["white", "off-white"]:
            return "sneakers"
        elif color_lower in ["black", "brown", "cognac", "saddle brown"]:
            # Dark leather-like colors
            if image is not None and _is_smooth_texture(image, mask):
                return "dress shoes"
            else:
                return "boots"
        elif color_lower in ["tan", "beige", "sand"]:
            return "loafers"
        elif "canvas" in color_lower:
            return "canvas sneakers"
        else:
            return "sneakers"
    
    # === ACCESSORIES ===
    if category == "hat":
        if color_lower in ["black", "gray", "charcoal"]:
            return "beanie"
        elif color_lower in ["khaki", "tan", "beige"]:
            return "baseball cap"
        else:
            return "cap"
    
    if category == "bag":
        if color_lower in ["brown", "tan", "cognac", "saddle brown"]:
            return "leather bag"
        elif color_lower in ["black"]:
            return "backpack"
        else:
            return "tote bag"
    
    if category == "belt":
        return "leather belt"
    
    if category == "scarf":
        if color_lower in ["gray", "charcoal", "heather"]:
            return "wool scarf"
        else:
            return "scarf"
    
    if category == "sunglasses":
        return "sunglasses"
    
    # Default: return category display name
    return CATEGORY_DISPLAY_NAMES.get(category, category)


# ============================================
# 🎯 HELPER FUNCTIONS FOR SIMPLER LABELS
# ============================================

def simplify_clothing_type(clip_label: str) -> str:
    """
    Simplify overly formal/specific CLIP labels to more casual/general ones.
    Users expect 'jacket' not 'sport coat', 'pants' not 'dress pants'.
    """
    simplifications = {
        # Jackets - unify to common names
        "sport coat": "jacket",
        "blazer": "blazer",  # Keep this one
        "trucker jacket": "denim jacket",
        "moto jacket": "leather jacket",
        "biker jacket": "leather jacket",
        "varsity jacket": "jacket",
        "field jacket": "jacket",
        "safari jacket": "jacket",
        "quilted jacket": "jacket",
        "anorak": "jacket",
        "windbreaker": "jacket",
        "shacket": "jacket",
        "overshirt": "shirt",
        
        # Pants - simplify formal
        "dress pants": "pants",
        "trousers": "pants",
        "slacks": "pants",
        "corduroys": "pants",
        "palazzo pants": "pants",
        "culottes": "pants",
        "cropped pants": "pants",
        
        # Shoes - simplify
        "running shoes": "sneakers",
        "high-top sneakers": "sneakers",
        "low-top sneakers": "sneakers",
        "basketball shoes": "sneakers",
        "tennis shoes": "sneakers",
        "skate shoes": "sneakers",
        "dress shoes": "shoes",
        "derby shoes": "shoes",
        "brogues": "shoes",
        "oxfords": "oxford shoes",
        
        # Tops - simplify
        "crewneck sweater": "sweater",
        "v-neck sweater": "sweater",
        "cable knit sweater": "sweater",
        "pullover hoodie": "hoodie",
        "zip-up hoodie": "hoodie",
        "crewneck sweatshirt": "sweatshirt",
        "graphic tee": "t-shirt",
        "henley shirt": "shirt",
        "thermal top": "long sleeve shirt",
        "flannel shirt": "flannel",
        "oxford shirt": "shirt",
        "dress shirt": "shirt",
    }
    
    return simplifications.get(clip_label, clip_label)


def get_simple_category_label(category: str) -> str:
    """
    Return a simple, user-friendly label when classification fails.
    These are generic but accurate categories.
    """
    simple_labels = {
        "upper_clothes": "top",
        "top": "top",
        "pants": "pants",
        "left_shoe": "shoes",
        "right_shoe": "shoes",
        "shoes": "shoes",
        "dress": "dress",
        "skirt": "skirt",
        "hat": "hat",
        "bag": "bag",
        "belt": "belt",
        "scarf": "scarf",
        "sunglasses": "sunglasses",
    }
    return simple_labels.get(category, category)


# ============================================
# 🚀 V2 TYPE INFERENCE - ULTIMATE ACCURACY
# ============================================

def infer_specific_type_v2(category: str, image: np.ndarray, mask: np.ndarray = None) -> Tuple[str, float]:
    """
    🎯 V2 ULTIMATE TYPE CLASSIFICATION
    
    CLIP-first approach with visual texture validation for maximum accuracy.
    This is the most powerful clothing type detection available.
    
    Strategy:
    1. Use Fashion-CLIP for direct classification (primary)
    2. Validate with visual texture analysis (secondary)
    3. Combine results for high-confidence output
    
    Args:
        category: Base category from SegFormer (e.g., 'upper_clothes', 'pants')
        image: BGR image of the clothing item cutout
        mask: Optional mask for item region
        
    Returns:
        Tuple of (specific_type, confidence)
    """
    from typing import Tuple
    
    try:
        # STEP 0: Our TRAINED CLASSIFIER (highest accuracy - 91.5%)
        custom_type = None
        custom_conf = 0.0
        
        try:
            from modules.clothing_classifier import classify_clothing
            from PIL import Image
            
            # Convert numpy array to PIL Image
            if image is not None:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)
                
                predictions = classify_clothing(pil_image, top_k=1)
                if predictions:
                    custom_type = predictions[0]["unified_name"]
                    custom_conf = predictions[0]["confidence"]
                    logger.info(f"  🎯 CUSTOM MODEL says: {custom_type} (conf={custom_conf:.2f})")
        except Exception as custom_err:
            logger.debug(f"  Custom classifier not available: {custom_err}")
        
        # If custom classifier is highly confident, use its result directly
        if custom_type and custom_conf > 0.75:
            logger.info(f"  ✅ Final type: {custom_type} (from trained model, conf={custom_conf:.2f})")
            return custom_type, custom_conf
        
        # STEP 1: CLIP Classification (fallback for uncertain cases)
        clip_type = None
        clip_conf = 0.0
        
        try:
            from modules.fashion_clip import get_fashion_clip
            clip = get_fashion_clip()
            clip_type, clip_conf = clip.classify_specific_type(image, category_hint=category)
            logger.info(f"  🏷️ CLIP says: {clip_type} (conf={clip_conf:.2f})")
        except Exception as clip_err:
            logger.warning(f"  ⚠️ CLIP classification failed: {clip_err}")
        
        # STEP 2: Visual Texture Analysis (Secondary validation)
        texture_type, texture_conf = analyze_texture_v2(image, mask, category)
        logger.info(f"  🔬 Texture says: {texture_type} (conf={texture_conf:.2f})")
        
        # STEP 3: Combine Results (Ensemble)
        # 🚀 PRIORITIZE CLIP over texture for better accuracy
        if clip_type and clip_conf > 0.25:  # CLIP threshold
            # Check if texture is reliable (only material like "denim" or "leather", not type)
            texture_is_material = texture_type in ["denim", "leather", "fleece", "knit", "wool"]
            
            if texture_type and texture_conf > 0.6 and texture_is_material:
                # Only use texture if it's a MATERIAL (not a garment type)
                # Merge material into CLIP type: "denim" + "jacket" = "denim jacket"
                simplified_type = simplify_clothing_type(clip_type)
                if "jacket" in simplified_type and texture_type == "denim":
                    final_type = "denim jacket"
                elif "jacket" in simplified_type and texture_type == "leather":
                    final_type = "leather jacket"
                elif "pants" in clip_type.lower() and texture_type == "denim":
                    final_type = "jeans"
                else:
                    final_type = simplified_type
                final_conf = min(0.98, (clip_conf * 0.7 + texture_conf * 0.3))
            else:
                # Trust CLIP result - simplify overly formal labels
                final_type = simplify_clothing_type(clip_type)
                final_conf = clip_conf + 0.1
        elif texture_type and texture_conf > 0.5:
            # CLIP failed, use texture
            final_type = texture_type
            final_conf = texture_conf
        else:
            # Both failed - use SIMPLE fallback labels
            final_type = get_simple_category_label(category)
            final_conf = 0.5
        
        logger.info(f"  ✅ Final type: {final_type} (conf={final_conf:.2f})")
        return final_type, final_conf
        
    except Exception as e:
        logger.error(f"V2 type inference failed: {e}")
        # Ultimate fallback
        return CATEGORY_DISPLAY_NAMES.get(category, category), 0.3


def analyze_texture_v2(image: np.ndarray, mask: np.ndarray = None, category: str = None) -> Tuple[str, float]:
    """
    Advanced texture analysis to determine material/fabric type.
    Uses multiple visual cues for high accuracy.
    
    Returns:
        Tuple of (material_type, confidence)
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Apply mask if provided
        if mask is not None and mask.size > 0:
            mask_2d = mask if len(mask.shape) == 2 else cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            valid_mask = mask_2d > 127
        else:
            valid_mask = np.ones(gray.shape, dtype=bool)
        
        valid_pixels = np.sum(valid_mask)
        if valid_pixels < 100:
            return None, 0.0
        
        # Feature extraction
        h, s, v = cv2.split(hsv)
        
        # 1. Blue ratio (for denim)
        blue_mask = (h >= 95) & (h <= 135) & (s > 25)
        blue_ratio = np.sum(blue_mask & valid_mask) / valid_pixels
        
        # 2. Edge density (structured fabrics have more edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum((edges > 0) & valid_mask) / valid_pixels
        
        # 3. Texture variance (thick fabrics have more variance)
        gray_pixels = gray[valid_mask]
        texture_var = np.var(gray_pixels) if len(gray_pixels) > 0 else 0
        
        # 4. Smoothness (leather/satin are smooth)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        smoothness = 1.0 / (1.0 + np.var(laplacian[valid_mask]))
        
        # 5. Color saturation
        sat_pixels = s[valid_mask]
        avg_sat = np.mean(sat_pixels) if len(sat_pixels) > 0 else 0
        
        # 6. Brightness uniformity
        val_pixels = v[valid_mask]
        brightness_std = np.std(val_pixels) if len(val_pixels) > 0 else 0
        
        # 7. Reflectivity (shiny materials like leather)
        high_v = (v > 230) & valid_mask
        reflectivity = np.sum(high_v) / valid_pixels
        
        # CLASSIFICATION RULES
        
        # DENIM: STRICT blue hue + HIGH saturation (must look blue, not gray)
        # Hue 100-125 is true blue, 95-135 was too wide
        if blue_ratio > 0.4 and avg_sat > 50:  # STRICTER thresholds
            if category in ["upper_clothes", "top"]:
                return "denim", 0.85  # Return MATERIAL, not garment type

            elif category == "pants":
                return "jeans", 0.88
            elif category == "skirt":
                return "denim skirt", 0.80
            else:
                return "denim", 0.75
        
        # LEATHER: Smooth + dark + reflective
        if smoothness > 0.15 and reflectivity > 0.02 and texture_var < 2000:
            avg_brightness = np.mean(val_pixels) if len(val_pixels) > 0 else 0
            if avg_brightness < 100:  # Dark
                if category in ["upper_clothes", "top"]:
                    return "leather jacket", 0.78
                elif category in ["left_shoe", "right_shoe", "shoes"]:
                    return "leather shoes", 0.75
                else:
                    return "leather", 0.70
        
        # KNIT/WOOL: High texture variance
        if texture_var > 1500 and edge_density > 0.08:
            if category in ["upper_clothes", "top"]:
                if texture_var > 2500:
                    return "cable knit sweater", 0.72
                else:
                    return "sweater", 0.68
        
        # FLEECE: Fluffy texture, low edges
        if texture_var > 800 and edge_density < 0.05:
            if category in ["upper_clothes", "top"]:
                return "fleece jacket", 0.65
        
        # 🚀 IMPROVED PANTS DETECTION (fixed mean_v bug)
        mean_v = np.mean(val_pixels) if len(val_pixels) > 0 else 128
        if category == "pants":
            # Very dark = black pants (could be jeans or dress pants)
            if mean_v < 60:
                return "black pants", 0.60
            # Dark with texture = jeans
            elif mean_v < 100 and texture_var > 300:
                return "jeans", 0.55
            # Medium texture, casual = pants (simple label)
            elif texture_var < 800:
                return "pants", 0.50
            # High texture casual = joggers
            elif texture_var > 1000 and edge_density < 0.06:
                return "joggers", 0.55
            else:
                return "pants", 0.45
        
        # COTTON: Default for tops
        if edge_density < 0.12 and texture_var < 800:
            if category in ["upper_clothes", "top"]:
                return "t-shirt", 0.55
        
        return None, 0.0
        
    except Exception as e:
        logger.debug(f"Texture analysis failed: {e}")
        return None, 0.0


def merge_type_with_texture(clip_type: str, texture_type: str) -> str:
    """
    Merge CLIP classification with texture analysis.
    E.g., CLIP says "jacket" + texture says "denim" = "denim jacket"
    """
    clip_lower = clip_type.lower()
    texture_lower = texture_type.lower()
    
    # If texture already in clip type, return clip type
    if texture_lower in clip_lower:
        return clip_type
    
    # Specific merge rules
    texture_material = None
    if "denim" in texture_lower:
        texture_material = "denim"
    elif "leather" in texture_lower:
        texture_material = "leather"
    elif "fleece" in texture_lower:
        texture_material = "fleece"
    elif "knit" in texture_lower or "sweater" in texture_lower:
        texture_material = "knit"
    
    if texture_material:
        # Add material prefix if it makes sense
        if "jacket" in clip_lower and texture_material not in clip_lower:
            return f"{texture_material} jacket"
        elif "pants" in clip_lower and texture_material == "denim":
            return "jeans"
    
    return clip_type


def extract_primary_color_quick(image: np.ndarray, mask: np.ndarray = None) -> str:
    """Quick color extraction for fallback."""
    try:
        if mask is not None:
            pixels = image[mask > 127]
        else:
            pixels = image.reshape(-1, 3)
        
        if len(pixels) < 10:
            return "unknown"
        
        # Sample for speed
        if len(pixels) > 1000:
            indices = np.random.choice(len(pixels), 1000, replace=False)
            pixels = pixels[indices]
        
        avg_bgr = np.mean(pixels, axis=0).astype(int)
        avg_rgb = (int(avg_bgr[2]), int(avg_bgr[1]), int(avg_bgr[0]))
        
        return get_color_name(avg_rgb)
    except:
        return "unknown"


def _is_denim_texture(image: np.ndarray, mask: np.ndarray = None) -> bool:
    """Check if texture appears to be denim (blue tones + textured)."""
    try:
        # Convert to HSV for color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        if mask is not None:
            # Apply mask
            h = hsv[:, :, 0][mask > 127]
            s = hsv[:, :, 1][mask > 127]
        else:
            h = hsv[:, :, 0].flatten()
            s = hsv[:, :, 1].flatten()
        
        if len(h) == 0:
            return False
        
        # Denim is typically blue (hue 100-130) with moderate saturation
        avg_hue = np.mean(h)
        avg_sat = np.mean(s)
        
        return 90 <= avg_hue <= 140 and avg_sat > 30
    except:
        return False


def _is_smooth_texture(image: np.ndarray, mask: np.ndarray = None) -> bool:
    """Check if texture appears smooth (like leather or satin)."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if mask is not None:
            # Analyze only masked region
            pixels = gray[mask > 127]
        else:
            pixels = gray.flatten()
        
        if len(pixels) == 0:
            return False
        
        # Low variance = smooth texture
        variance = np.var(pixels)
        return variance < 500  # Arbitrary threshold
    except:
        return False


@dataclass
class ClothingItem:
    """Detected clothing item with mask and metadata"""
    category: str
    category_id: int
    mask: np.ndarray
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    area_percentage: float
    primary_color: str = "Unknown"  # Color name
    color_hex: str = "#000000"  # Hex code
    colors: List[Dict] = None  # All detected colors
    specific_type: str = None  # Specific type (e.g., "denim jacket")


@dataclass
class AdvancedSegmentationResult:
    """Full segmentation result with per-item masks"""
    original_image: np.ndarray
    full_mask: np.ndarray
    segmented_image: np.ndarray
    items: List[ClothingItem]
    primary_confidence: float
    processing_time_ms: float


# Global singleton cache for models (persists between requests)
_SEGFORMER_CACHE = {
    "model": None,
    "processor": None,
    "loaded": False
}


def get_cached_segformer():
    """Get or create cached SegFormer model instance."""
    global _SEGFORMER_CACHE
    
    if _SEGFORMER_CACHE["loaded"]:
        return _SEGFORMER_CACHE["model"], _SEGFORMER_CACHE["processor"]
    
    try:
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        import torch
        
        logger.info("🚀 Loading SegFormer model into global cache...")
        
        model_name = "mattmdjaga/segformer_b2_clothes"
        
        processor = SegformerImageProcessor.from_pretrained(model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(model_name)
        
        if torch.cuda.is_available():
            model = model.to("cuda")
            logger.info("✅ Using CUDA for SegFormer")
        
        model.eval()
        
        _SEGFORMER_CACHE["model"] = model
        _SEGFORMER_CACHE["processor"] = processor
        _SEGFORMER_CACHE["loaded"] = True
        
        logger.info("✅ SegFormer model cached globally (will be reused)")
        return model, processor
        
    except Exception as e:
        logger.error(f"Failed to load SegFormer: {e}")
        return None, None

class AdvancedClothingSegmentor:
    """
    State-of-the-art clothing segmentation using SegFormer-B2-Clothes.
    
    Features:
    - 18-category clothing detection
    - Per-item segmentation masks
    - Confidence scoring per category
    - Automatic background removal
    - Edge refinement for clean cutouts
    """
    
    def __init__(self, use_segformer: bool = True, device: str = "cpu"):
        self.use_segformer = use_segformer
        self.device = device
        self._segformer_model = None
        self._segformer_processor = None
        self._rembg_session = None
        self._model_loaded = False
    
    def _load_segformer(self):
        """Load SegFormer model from global cache"""
        if self._model_loaded:
            return True
        
        # Use global cache for faster loading
        model, processor = get_cached_segformer()
        
        if model is not None and processor is not None:
            self._segformer_model = model
            self._segformer_processor = processor
            self._model_loaded = True
            logger.info("✅ Using cached SegFormer model")
            return True
        
        logger.warning("SegFormer cache unavailable, using fallback")
        self.use_segformer = False
        return False
    
    def _get_rembg_session(self):
        """Lazy load rembg as fallback"""
        if self._rembg_session is None:
            try:
                from rembg import new_session
                self._rembg_session = new_session("u2net_cloth_seg")
                logger.info("Loaded rembg fallback model")
            except ImportError:
                logger.warning("rembg not available")
        return self._rembg_session
    
    def segment_with_segformer(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Segment clothing using SegFormer transformer model.
        Uses optimal sizing for better detection of all items.
        
        Returns:
            Dictionary with per-category masks
        """
        import torch
        import torch.nn.functional as F
        
        if not self._load_segformer():
            return None
        
        original_h, original_w = image.shape[:2]
        
        # CRITICAL: Resize to optimal size for SegFormer
        # SMALLER is BETTER for detecting all items (caps, shoes, etc.)
        # Large images cause SegFormer to focus only on dominant items
        MAX_SIZE = 512   # REDUCED from 768 - more aggressive resize for better detection
        MIN_SIZE = 384
        
        # Calculate resize ratio
        max_dim = max(original_h, original_w)
        min_dim = min(original_h, original_w)
        
        # ALWAYS resize large images down
        if max_dim > MAX_SIZE:
            scale = MAX_SIZE / max_dim
            new_h = int(original_h * scale)
            new_w = int(original_w * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"⬇️ Resized {original_w}x{original_h} → {new_w}x{new_h} for BETTER detection")
        elif min_dim < MIN_SIZE:
            scale = MIN_SIZE / min_dim
            new_h = int(original_h * scale)
            new_w = int(original_w * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            logger.info(f"⬆️ Upscaled {original_w}x{original_h} → {new_w}x{new_h}")
        else:
            resized = image
            new_h, new_w = original_h, original_w
            logger.info(f"📐 Using original size {original_w}x{original_h}")
        
        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        
        # Process image
        inputs = self._segformer_processor(images=pil_image, return_tensors="pt")
        
        if self.device == "cuda":
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = self._segformer_model(**inputs)
        
        # Get segmentation logits
        logits = outputs.logits
        
        # Upsample to ORIGINAL size (not resized size)
        upsampled = F.interpolate(
            logits,
            size=(original_h, original_w),  # Back to original dimensions
            mode="bilinear",
            align_corners=False
        )
        
        # Get predictions
        predictions = upsampled.argmax(dim=1).squeeze().cpu().numpy()
        probabilities = torch.softmax(upsampled, dim=1).squeeze().cpu().numpy()
        
        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "categories": CLOTHING_CATEGORIES
        }
    
    def segment_with_rembg(self, image: np.ndarray) -> np.ndarray:
        """Fallback segmentation with rembg"""
        try:
            from rembg import remove
            
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            session = self._get_rembg_session()
            
            if session:
                result = remove(pil_image, session=session)
            else:
                result = remove(pil_image)
            
            return np.array(result)
            
        except Exception as e:
            logger.error(f"rembg segmentation failed: {e}")
            return None
    
    def extract_clothing_items(
        self, 
        image: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray
    ) -> List[ClothingItem]:
        """Extract individual clothing items from segmentation with improved detection"""
        
        items = []
        
        # Safety check for empty predictions
        if predictions is None or predictions.size == 0 or len(predictions.shape) < 2:
            logger.warning("Empty or invalid predictions array, returning empty items")
            return items
            
        h, w = predictions.shape
        total_pixels = h * w
        
        # All clothing categories we want to detect
        # IDs: 1=hat, 3=sunglasses, 4=upper_clothes, 5=skirt, 6=pants, 7=dress, 8=belt, 9=left_shoe, 10=right_shoe, 16=bag, 17=scarf
        clothing_ids = [1, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17]
        
        logger.info(f"Scanning for items in image {w}x{h} ({total_pixels} pixels)")
        
        # First pass: check what categories exist in the predictions
        detected_categories = []
        for cat_id in clothing_ids:
            mask_pixels = np.sum(predictions == cat_id)
            if mask_pixels > 0:
                pct = (mask_pixels / total_pixels) * 100
                detected_categories.append((cat_id, CLOTHING_CATEGORIES[cat_id], pct))
        
        logger.info(f"Initial scan found {len(detected_categories)} potential categories: {[(c[1], f'{c[2]:.1f}%') for c in detected_categories]}")
        
        for cat_id in clothing_ids:
            mask = (predictions == cat_id).astype(np.uint8) * 255
            
            # Check if this category exists (VERY permissive: 0.05% minimum)
            mask_area = np.sum(mask > 0)
            area_pct = (mask_area / total_pixels) * 100
            
            # ULTRA-LOW THRESHOLD: 0.05% catches small accessories like watches, belts, jewelry
            if mask_area == 0 or area_pct < 0.05:  # LOWERED from 0.1% to 0.05% for max recall
                continue
            
            category_name = CLOTHING_CATEGORIES[cat_id]
            logger.info(f"Processing {category_name} (cat_id={cat_id}, area={area_pct:.2f}%)")
            
            # SPECIAL HANDLING: Check if "dress" (7) might actually be separate top + pants
            if cat_id == 7:  # dress
                upper_mask = (predictions == 4).astype(np.uint8) * 255  # upper_clothes
                pants_mask = (predictions == 6).astype(np.uint8) * 255  # pants
                
                # If we already detected separate upper and pants, skip the dress
                if np.sum(upper_mask) > total_pixels * 0.005 and np.sum(pants_mask) > total_pixels * 0.005:
                    logger.info("Skipping dress - already have separate upper clothes and pants")
                    continue
                
                # Check if the "dress" should be split
                dress_rows = np.where(np.any(mask > 0, axis=1))[0]
                if len(dress_rows) > 0:
                    dress_top = dress_rows[0]
                    dress_bottom = dress_rows[-1]
                    dress_height = dress_bottom - dress_top
                    
                    # If dress is very tall (covers most of body), try to split it
                    if dress_height > h * 0.45:  # More than 45% of image height
                        mid_point = dress_top + dress_height // 2
                        
                        # Check for a "waist" area with fewer pixels
                        waist_start = max(0, int(mid_point - dress_height * 0.12))
                        waist_end = min(h, int(mid_point + dress_height * 0.12))
                        waist_pixels = [np.sum(mask[row] > 0) for row in range(waist_start, waist_end)]
                        
                        if len(waist_pixels) > 0:
                            avg_waist = np.mean(waist_pixels)
                            min_waist = np.min(waist_pixels)
                            
                            # If waist area has significantly fewer pixels, split
                            if min_waist < avg_waist * 0.65 or min_waist < w * 0.15:
                                logger.info(f"Splitting 'dress' into upper + lower (waist: min={min_waist:.0f}, avg={avg_waist:.0f})")
                                
                                # Create upper portion
                                upper_portion = mask.copy()
                                upper_portion[mid_point:, :] = 0
                                
                                # Create lower portion
                                lower_portion = mask.copy()
                                lower_portion[:mid_point, :] = 0
                                
                                # Add upper as "Top"
                                if np.sum(upper_portion) > total_pixels * 0.005:
                                    item = self._create_clothing_item(
                                        image, upper_portion, probabilities, 4, h, w, total_pixels,
                                        override_category="upper_clothes"
                                    )
                                    if item:
                                        items.append(item)
                                        logger.info(f"  ✅ Added upper_clothes from split")
                                
                                # Add lower as "Pants"
                                if np.sum(lower_portion) > total_pixels * 0.005:
                                    item = self._create_clothing_item(
                                        image, lower_portion, probabilities, 6, h, w, total_pixels,
                                        override_category="pants"
                                    )
                                    if item:
                                        items.append(item)
                                        logger.info(f"  ✅ Added pants from split")
                                
                                continue  # Skip adding as dress
            
            # Standard item extraction
            item = self._create_clothing_item(image, mask, probabilities, cat_id, h, w, total_pixels)
            if item:
                items.append(item)
                logger.info(f"  ✅ Added {category_name} ({item.primary_color}, conf={item.confidence:.2f})")
            else:
                logger.info(f"  ❌ Skipped {category_name} (too small or invalid)")
        
        # Sort by area (largest first)
        items.sort(key=lambda x: x.area_percentage, reverse=True)
        
        # Merge left/right shoes into single "Shoes" item
        items = self._merge_shoe_pairs(items)
        
        logger.info(f"Before filtering: {len(items)} clothing items detected")
        for item in items:
            logger.info(f"  - {item.category}: {item.primary_color} ({item.area_percentage:.1f}%, conf={item.confidence:.2f})")
        
        # 🎯 LAYER 3: Apply confidence filter (4-Layer Assurance Stack)
        logger.info("🎯 Applying 4-Layer confidence filter...")
        items = apply_confidence_filter(items)
        
        logger.info(f"After filtering: {len(items)} items passed confidence threshold")
        for item in items:
            status = getattr(item, 'confidence_status', 'unknown')
            logger.info(f"  ✅ {item.category}: {item.primary_color} (conf={item.confidence:.2f}, status={status})")
        
        return items
    
    def _create_clothing_item(
        self, 
        image: np.ndarray, 
        mask: np.ndarray, 
        probabilities: np.ndarray,
        cat_id: int,
        h: int,
        w: int, 
        total_pixels: int,
        override_category: str = None
    ) -> ClothingItem:
        """Create a ClothingItem from mask"""
        
        # Find bounding box
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(largest)
        
        # Skip only VERY tiny items (noise)
        if bw < 5 or bh < 5:
            return None  # LOWERED from 10x10 to 5x5
        
        # Calculate confidence
        if cat_id < probabilities.shape[0]:
            category_probs = probabilities[cat_id]
            confidence = float(np.mean(category_probs[mask > 0])) if np.sum(mask > 0) > 0 else 0.5
        else:
            confidence = 0.7
        
        # Calculate area percentage
        area_pct = float(np.sum(mask > 0)) / total_pixels * 100
        
        # Extract dominant colors for this item
        primary_color = "Unknown"
        color_hex = "#000000"
        colors = []
        
        try:
            colors = extract_dominant_colors(image, mask, n_colors=3)
            if colors:
                primary_color = colors[0]["name"]
                color_hex = colors[0]["hex"]
        except Exception as e:
            category_name = override_category or CLOTHING_CATEGORIES.get(cat_id, "unknown")
            logger.warning(f"Color extraction failed for {category_name}: {e}")
        
        category = override_category or CLOTHING_CATEGORIES.get(cat_id, "unknown")
        
        # 🚀 V2 ULTIMATE TYPE INFERENCE - CLIP + Texture ensemble
        # This makes ALL endpoints return specific types like "denim jacket", "skinny jeans"
        specific_type = None
        
        try:
            # Crop the item region for better CLIP classification
            padding = 5
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + bw + padding)
            y2 = min(h, y + bh + padding)
            
            item_crop = image[y1:y2, x1:x2]
            item_mask_crop = mask[y1:y2, x1:x2] if mask is not None else None
            
            if item_crop.size > 0:
                # Use V2 CLIP-first inference
                specific_type, type_conf = infer_specific_type_v2(category, item_crop, item_mask_crop)
                logger.info(f"  🚀 CLIP Type: {specific_type} (conf={type_conf:.2f})")
            else:
                # Fallback to heuristics
                specific_type = infer_specific_type(category, primary_color, image, mask)
                
        except Exception as v2_err:
            logger.debug(f"  V2 type inference failed: {v2_err}, using fallback")
            # Fallback to heuristic method
            specific_type = infer_specific_type(category, primary_color, image, mask)
        
        return ClothingItem(
            category=category,
            category_id=cat_id,
            mask=mask,
            confidence=confidence,
            bbox=(x, y, bw, bh),
            area_percentage=area_pct,
            primary_color=primary_color,
            color_hex=color_hex,
            colors=colors,
            specific_type=specific_type
        )
    
    def _merge_shoe_pairs(self, items: List[ClothingItem]) -> List[ClothingItem]:
        """Merge left_shoe and right_shoe into a single 'Shoes' item"""
        
        left_shoe = None
        right_shoe = None
        other_items = []
        
        for item in items:
            if item.category == "left_shoe":
                left_shoe = item
            elif item.category == "right_shoe":
                right_shoe = item
            else:
                other_items.append(item)
        
        # If we have both shoes, merge them
        if left_shoe and right_shoe:
            # Combine bboxes
            x1 = min(left_shoe.bbox[0], right_shoe.bbox[0])
            y1 = min(left_shoe.bbox[1], right_shoe.bbox[1])
            x2 = max(left_shoe.bbox[0] + left_shoe.bbox[2], right_shoe.bbox[0] + right_shoe.bbox[2])
            y2 = max(left_shoe.bbox[1] + left_shoe.bbox[3], right_shoe.bbox[1] + right_shoe.bbox[3])
            
            # Combine masks
            combined_mask = cv2.bitwise_or(left_shoe.mask, right_shoe.mask)
            
            # 🚀 Preserve the best specific_type from either shoe
            shoe_type = left_shoe.specific_type or right_shoe.specific_type or "shoes"
            
            merged = ClothingItem(
                category="shoes",
                category_id=9,  # Use left_shoe id
                mask=combined_mask,
                confidence=(left_shoe.confidence + right_shoe.confidence) / 2,
                bbox=(x1, y1, x2 - x1, y2 - y1),
                area_percentage=left_shoe.area_percentage + right_shoe.area_percentage,
                primary_color=left_shoe.primary_color,
                color_hex=left_shoe.color_hex,
                colors=left_shoe.colors,
                specific_type=shoe_type  # 🚀 Keep "sneakers", "dress shoes" etc.
            )
            other_items.append(merged)
            logger.info(f"Merged left/right shoes into single 'Shoes' item (type: {shoe_type})")
        elif left_shoe:
            left_shoe.category = "shoes"
            other_items.append(left_shoe)
        elif right_shoe:
            right_shoe.category = "shoes"
            other_items.append(right_shoe)
        
        return other_items
    
    def create_combined_mask(
        self, 
        predictions: np.ndarray,
        include_categories: Optional[List[int]] = None
    ) -> np.ndarray:
        """Create combined mask for selected categories"""
        
        if include_categories is None:
            # Default: all clothing items
            include_categories = [1, 3, 4, 5, 6, 7, 8, 9, 10, 16, 17]
        
        mask = np.zeros(predictions.shape, dtype=np.uint8)
        
        for cat_id in include_categories:
            mask = np.logical_or(mask, predictions == cat_id)
        
        return (mask * 255).astype(np.uint8)
    
    def refine_edges(self, mask: np.ndarray, feather_radius: int = 5, quality: str = "high") -> np.ndarray:
        """
        Refine mask edges for professional-quality cutouts.
        
        Args:
            mask: Binary mask (0-255)
            feather_radius: Edge feathering amount
            quality: 'low', 'medium', 'high' - processing quality level
        """
        
        # Step 1: Clean up noise with morphological operations
        kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        
        # Remove small holes inside the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
        
        # Remove small noise outside
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        
        if quality == "low":
            return mask
        
        # Step 2: Contour smoothing for cleaner edges
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        smooth_mask = np.zeros_like(mask)
        
        for contour in contours:
            if cv2.contourArea(contour) < 100:  # Skip tiny contours
                continue
            # Approximate contour with fewer points (smoother)
            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            cv2.fillPoly(smooth_mask, [approx], 255)
        
        mask = smooth_mask if np.sum(smooth_mask) > 0 else mask
        
        if quality == "medium":
            # Simple Gaussian blur for feathering
            mask_float = mask.astype(np.float32) / 255.0
            blurred = cv2.GaussianBlur(mask_float, (feather_radius*2+1, feather_radius*2+1), 0)
            return (blurred * 255).astype(np.uint8)
        
        # Step 3: High quality - Alpha matting with edge refinement
        mask_float = mask.astype(np.float32) / 255.0
        
        # Create trimap: definite foreground, definite background, unknown
        dilated = cv2.dilate(mask, kernel_medium, iterations=2)
        eroded = cv2.erode(mask, kernel_medium, iterations=2)
        
        # Unknown region (edges)
        unknown = dilated - eroded
        
        # Gaussian blur for soft edges in unknown region
        edge_blur = cv2.GaussianBlur(mask_float, (9, 9), 0)
        
        # Combine: use sharp inside, blurred edges outside
        refined = mask_float.copy()
        unknown_mask = unknown > 0
        refined[unknown_mask] = edge_blur[unknown_mask]
        
        # Bilateral filter for edge-aware smoothing (preserves sharp edges)
        refined_uint8 = (refined * 255).astype(np.uint8)
        refined_uint8 = cv2.bilateralFilter(refined_uint8, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Final anti-aliasing pass
        final = cv2.GaussianBlur(refined_uint8, (3, 3), 0.5)
        
        return final
    
    def apply_mask_to_image(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        add_white_bg: bool = True
    ) -> np.ndarray:
        """Apply segmentation mask to create cutout"""
        
        # Ensure mask is correct size
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Convert to float
        alpha = mask.astype(np.float32) / 255.0
        
        if len(alpha.shape) == 2:
            alpha = alpha[:, :, np.newaxis]
        
        # Convert image to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        if add_white_bg:
            # Create white background
            white_bg = np.ones_like(image, dtype=np.uint8) * 255
            result = (image * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        else:
            # Create RGBA with transparency
            result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
            result[:, :, :3] = image
            result[:, :, 3] = mask
        
        return result
    
    def segment(
        self, 
        image: np.ndarray,
        add_white_bg: bool = True,
        refine: bool = True,
        return_items: bool = True
    ) -> AdvancedSegmentationResult:
        """
        Full advanced segmentation pipeline.
        
        Args:
            image: Input BGR image
            add_white_bg: Add white background to result
            refine: Apply edge refinement
            return_items: Extract individual clothing items
        
        Returns:
            AdvancedSegmentationResult with full analysis
        """
        import time
        start_time = time.time()
        
        items = []
        
        # Try SegFormer first
        if self.use_segformer:
            result = self.segment_with_segformer(image)
            
            if result is not None:
                predictions = result["predictions"]
                probabilities = result["probabilities"]
                
                # Extract items
                if return_items:
                    items = self.extract_clothing_items(image, predictions, probabilities)
                
                # Create combined mask
                full_mask = self.create_combined_mask(predictions)
                
                # Calculate confidence
                clothing_mask = full_mask > 0
                if np.any(clothing_mask):
                    primary_confidence = float(np.max(probabilities[:, clothing_mask].mean(axis=1)))
                else:
                    primary_confidence = 0.5
                
                logger.info(f"SegFormer: Detected {len(items)} clothing items")
            else:
                # Fallback
                full_mask = None
                primary_confidence = 0.0
        else:
            full_mask = None
            primary_confidence = 0.0
        
        # Fallback to rembg
        if full_mask is None or np.sum(full_mask) == 0:
            logger.info("Using rembg fallback")
            rembg_result = self.segment_with_rembg(image)
            
            if rembg_result is not None and rembg_result.shape[2] == 4:
                full_mask = rembg_result[:, :, 3]
                primary_confidence = 0.85
            else:
                # Last resort: simple background detection
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, full_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
                primary_confidence = 0.5
        
        # Refine edges
        if refine:
            full_mask = self.refine_edges(full_mask)
        
        # Apply mask
        segmented = self.apply_mask_to_image(image, full_mask, add_white_bg)
        
        processing_time = (time.time() - start_time) * 1000
        
        return AdvancedSegmentationResult(
            original_image=image,
            full_mask=full_mask,
            segmented_image=segmented,
            items=items,
            primary_confidence=primary_confidence,
            processing_time_ms=processing_time
        )


def segment_clothing_from_base64(
    image_base64: str,
    add_white_bg: bool = True,
    use_advanced: bool = True
) -> Dict:
    """
    Utility function for base64 image segmentation.
    
    Args:
        image_base64: Base64-encoded image
        add_white_bg: Add white background
        use_advanced: Use SegFormer (vs basic rembg)
    
    Returns:
        Dictionary with segmented image and metadata
    """
    # Remove data URL prefix if present
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]
    
    # Decode base64 to image
    img_bytes = base64.b64decode(image_base64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "Could not decode image"}
    
    # Segment
    segmentor = AdvancedClothingSegmentor(use_segformer=use_advanced)
    result = segmentor.segment(image, add_white_bg=add_white_bg)
    
    # Encode result to base64
    if add_white_bg:
        _, buffer = cv2.imencode('.png', cv2.cvtColor(
            result.segmented_image, cv2.COLOR_RGB2BGR
        ))
    else:
        _, buffer = cv2.imencode('.png', result.segmented_image)
    
    result_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Build items list with color info and specificType
    items_data = []
    for item in result.items:
        items_data.append({
            "category": item.category,
            "specificType": item.specific_type,  # 🚀 V2 type like "t-shirt", "denim jacket"
            "confidence": round(item.confidence, 3),
            "bbox": item.bbox,
            "areaPercentage": round(item.area_percentage, 2),
            "primaryColor": item.primary_color,
            "colorHex": item.color_hex,
            "colors": item.colors or []
        })
    
    return {
        "segmentedImage": f"data:image/png;base64,{result_base64}",
        "confidence": round(result.primary_confidence, 4),
        "items": items_data,
        "itemCount": len(items_data),
        "processingTimeMs": round(result.processing_time_ms, 1),
        "hasTransparency": not add_white_bg
    }
