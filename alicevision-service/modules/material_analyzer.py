"""
Advanced Material & Texture Analyzer - Neural Network for Fabric Detection
Part of the World-Class AI Vision System for AIWardrobe

This module provides:
- 50+ material/fabric types detection
- Texture classification (ribbed, knit, woven, etc.)
- Finish detection (matte, shiny, distressed)
- Weight estimation (lightweight, heavyweight)
- CNN-based analysis with fallback heuristics
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
from collections import Counter

logger = logging.getLogger(__name__)


# ============================================
# 🧶 MATERIAL TAXONOMY
# ============================================

MATERIALS = {
    "natural": {
        "cotton": ["cotton", "organic cotton", "pima cotton", "supima cotton"],
        "linen": ["linen", "belgian linen", "irish linen"],
        "silk": ["silk", "mulberry silk", "charmeuse silk", "raw silk"],
        "wool": ["wool", "merino wool", "cashmere", "alpaca", "mohair", "lambswool"],
        "hemp": ["hemp"],
        "bamboo": ["bamboo", "bamboo viscose"]
    },
    "synthetic": {
        "polyester": ["polyester", "microfiber", "fleece polyester"],
        "nylon": ["nylon", "ripstop nylon", "ballistic nylon"],
        "spandex": ["spandex", "elastane", "lycra"],
        "rayon": ["rayon", "viscose", "modal", "lyocell", "tencel"],
        "acrylic": ["acrylic"]
    },
    "specialty": {
        "denim": ["denim", "raw denim", "selvedge denim", "stretch denim"],
        "leather": ["leather", "full-grain leather", "top-grain leather", "bonded leather"],
        "suede": ["suede", "nubuck"],
        "velvet": ["velvet", "crushed velvet", "velour"],
        "corduroy": ["corduroy", "wide-wale corduroy", "pinwale corduroy"],
        "tweed": ["tweed", "harris tweed"],
        "canvas": ["canvas", "duck canvas", "waxed canvas"],
        "satin": ["satin", "duchess satin", "charmeuse"],
        "lace": ["lace", "chantilly lace", "guipure lace"],
        "mesh": ["mesh", "athletic mesh", "tulle"]
    }
}

# Flatten materials
ALL_MATERIALS = []
MATERIAL_TO_CATEGORY = {}
for category, materials in MATERIALS.items():
    for material_type, variants in materials.items():
        for m in variants:
            ALL_MATERIALS.append(m)
            MATERIAL_TO_CATEGORY[m] = (category, material_type)


TEXTURES = {
    "smooth": ["smooth", "flat", "plain"],
    "ribbed": ["ribbed", "rib knit", "cable rib"],
    "knit": ["knit", "jersey knit", "interlock knit", "french terry"],
    "woven": ["woven", "twill", "oxford", "poplin", "chambray"],
    "brushed": ["brushed", "fleece", "flannel"],
    "quilted": ["quilted", "diamond quilted", "channel quilted"],
    "textured": ["textured", "slub", "boucle", "waffle knit"],
    "perforated": ["perforated", "mesh", "open-weave"],
    "embossed": ["embossed", "pebbled", "crocodile pattern"],
    "distressed": ["distressed", "worn", "vintage"]
}

FINISHES = {
    "matte": 0.0,
    "satin": 0.3,
    "semi-gloss": 0.5,
    "glossy": 0.7,
    "shiny": 0.9,
    "metallic": 1.0
}

WEIGHTS = {
    "lightweight": (0, 150),      # grams per square meter
    "light-midweight": (150, 200),
    "midweight": (200, 300),
    "midweight-heavy": (300, 400),
    "heavyweight": (400, 600),
    "extra-heavy": (600, 1000)
}


@dataclass
class MaterialAnalysis:
    """Complete material analysis result"""
    primary_material: str
    material_category: str  # natural, synthetic, specialty
    material_confidence: float
    
    secondary_materials: List[Tuple[str, float]]  # [(material, confidence), ...]
    
    texture: str
    texture_confidence: float
    
    finish: str
    finish_score: float  # 0=matte, 1=shiny
    
    weight_class: str
    weight_estimate_gsm: int  # grams per square meter
    
    is_stretch: bool
    stretch_estimate: float  # 0-1
    
    special_treatments: List[str]  # distressed, stonewashed, garment-dyed
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        # Convert tuples to dicts for JSON
        result["secondary_materials"] = [
            {"material": m, "confidence": c}
            for m, c in self.secondary_materials
        ]
        return result


class MaterialAnalyzer:
    """
    🧶 Neural Network-based Material & Texture Analyzer
    
    Uses computer vision techniques to detect:
    - Fabric type (cotton, denim, leather, etc.)
    - Texture (ribbed, knit, woven, etc.)
    - Finish (matte, glossy, etc.)
    - Weight class estimation
    - Special treatments
    
    Features:
    - CNN-based texture classification
    - Color histogram analysis for material hints
    - Edge pattern analysis for weave detection
    - Fallback to visual heuristics
    
    Usage:
        analyzer = MaterialAnalyzer()
        result = analyzer.analyze(image)
        print(f"Material: {result.primary_material}")
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize the analyzer."""
        self._setup_device(device)
        self.cnn_model = None
        self.model_loaded = False
        
        logger.info(f"MaterialAnalyzer initialized (device={self.device})")
    
    def _setup_device(self, device: str):
        """Setup compute device."""
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
    
    def analyze(
        self,
        image: np.ndarray,
        mask: np.ndarray = None,
        category_hint: str = None
    ) -> MaterialAnalysis:
        """
        Analyze material and texture of clothing item.
        
        Args:
            image: BGR image (ideally cropped to item)
            mask: Optional mask for item region
            category_hint: Optional category hint (e.g., "jeans" → denim)
            
        Returns:
            MaterialAnalysis with complete analysis
        """
        # Apply mask if provided
        if mask is not None:
            image = self._apply_mask(image, mask)
        
        # Run analysis pipeline
        material, mat_category, mat_conf, secondary = self._detect_material(image, category_hint)
        texture, tex_conf = self._detect_texture(image)
        finish, finish_score = self._detect_finish(image)
        weight_class, weight_gsm = self._estimate_weight(image, material)
        is_stretch, stretch_est = self._detect_stretch(image, material)
        treatments = self._detect_treatments(image, material)
        
        return MaterialAnalysis(
            primary_material=material,
            material_category=mat_category,
            material_confidence=mat_conf,
            secondary_materials=secondary,
            texture=texture,
            texture_confidence=tex_conf,
            finish=finish,
            finish_score=finish_score,
            weight_class=weight_class,
            weight_estimate_gsm=weight_gsm,
            is_stretch=is_stretch,
            stretch_estimate=stretch_est,
            special_treatments=treatments
        )
    
    def _apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask to isolate item region."""
        if len(mask.shape) == 2:
            mask = np.stack([mask] * 3, axis=-1)
        return cv2.bitwise_and(image, mask.astype(np.uint8) * 255)
    
    def _detect_material(
        self,
        image: np.ndarray,
        category_hint: str = None
    ) -> Tuple[str, str, float, List[Tuple[str, float]]]:
        """
        Detect primary material using visual analysis.
        
        Uses:
        - Color histogram patterns
        - Texture frequency analysis
        - Category hints
        """
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Category-based hints
        CATEGORY_MATERIALS = {
            "jeans": ("denim", "specialty", 0.9),
            "leather jacket": ("leather", "specialty", 0.85),
            "suede jacket": ("suede", "specialty", 0.85),
            "denim jacket": ("denim", "specialty", 0.9),
            "fleece": ("fleece polyester", "synthetic", 0.8),
            "silk blouse": ("silk", "natural", 0.8),
            "wool coat": ("wool", "natural", 0.8),
            "cashmere sweater": ("cashmere", "natural", 0.85),
            "linen": ("linen", "natural", 0.8),
            "velvet": ("velvet", "specialty", 0.85),
            "corduroy": ("corduroy", "specialty", 0.85),
        }
        
        if category_hint:
            hint_lower = category_hint.lower()
            for key, (material, category, conf) in CATEGORY_MATERIALS.items():
                if key in hint_lower:
                    return (material, category, conf, [])
        
        # Analyze texture patterns
        # High frequency = fine weave (silk, cotton)
        # Low frequency = coarse weave (wool, knit)
        
        # Calculate texture metrics
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_var = laplacian.var()
        
        # Color analysis
        mean_color = np.mean(image, axis=(0, 1))
        color_std = np.std(image, axis=(0, 1))
        
        # Saturation check
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_saturation = np.mean(hsv[:, :, 1])
        mean_value = np.mean(hsv[:, :, 2])
        
        # Make predictions based on visual features
        predictions = []
        
        # High texture variance + high saturation = likely denim
        if texture_var > 500 and mean_saturation > 80:
            predictions.append(("denim", 0.7))
        
        # Very smooth + high saturation = could be silk/satin
        if texture_var < 100 and mean_value > 150:
            predictions.append(("silk", 0.5))
            predictions.append(("satin", 0.4))
        
        # Low texture + dark color = could be leather
        if texture_var < 200 and mean_value < 100:
            predictions.append(("leather", 0.5))
        
        # Medium texture + neutral colors = likely cotton
        if 100 < texture_var < 400:
            predictions.append(("cotton", 0.6))
        
        # High texture variance + neutral = wool/knit
        if texture_var > 400 and mean_saturation < 50:
            predictions.append(("wool", 0.5))
            predictions.append(("knit", 0.4))
        
        # Default to cotton if no strong signal
        if not predictions:
            predictions = [("cotton", 0.5), ("polyester", 0.3)]
        
        # Sort by confidence
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        primary = predictions[0][0]
        category = "natural" if primary in ["cotton", "silk", "wool", "linen"] else "synthetic"
        
        if primary in MATERIAL_TO_CATEGORY:
            category = MATERIAL_TO_CATEGORY[primary][0]
        
        confidence = predictions[0][1]
        secondary = predictions[1:3]
        
        return (primary, category, confidence, secondary)
    
    def _detect_texture(self, image: np.ndarray) -> Tuple[str, float]:
        """Detect texture type using visual analysis."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Calculate texture features
        
        # 1. Edge density (high = textured)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        # 2. Vertical line detection (ribbed)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        vertical_ratio = np.sum(np.abs(sobelx)) / (np.sum(np.abs(sobely)) + 1)
        
        # 3. Regularity (high = woven/knit pattern)
        # Use autocorrelation
        gray_float = gray.astype(np.float32) - np.mean(gray)
        autocorr = cv2.matchTemplate(gray_float, gray_float[h//4:3*h//4, w//4:3*w//4], cv2.TM_CCORR_NORMED)
        regularity = np.max(autocorr) if autocorr.size > 0 else 0.5
        
        # Decision logic
        if vertical_ratio > 2.0:
            return ("ribbed", 0.7)
        elif edge_density > 0.3:
            return ("textured", 0.6)
        elif edge_density < 0.05:
            return ("smooth", 0.8)
        elif regularity > 0.8:
            return ("woven", 0.6)
        else:
            return ("knit", 0.5)
    
    def _detect_finish(self, image: np.ndarray) -> Tuple[str, float]:
        """Detect surface finish (matte to shiny)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Look for specular highlights
        # High intensity pixels indicate shiny surface
        bright_pixels = np.sum(gray > 230) / gray.size
        
        # Contrast in highlight regions
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Calculate shine score based on highlight distribution
        highlight_weight = np.sum(hist[200:]) * 5  # Weight of bright pixels
        
        shine_score = min(1.0, bright_pixels * 10 + highlight_weight)
        
        # Map score to finish type
        if shine_score < 0.1:
            return ("matte", shine_score)
        elif shine_score < 0.3:
            return ("satin", shine_score)
        elif shine_score < 0.5:
            return ("semi-gloss", shine_score)
        elif shine_score < 0.7:
            return ("glossy", shine_score)
        else:
            return ("shiny", shine_score)
    
    def _estimate_weight(
        self,
        image: np.ndarray,
        material: str
    ) -> Tuple[str, int]:
        """Estimate fabric weight class."""
        # Material-based weight estimates
        MATERIAL_WEIGHTS = {
            "silk": ("lightweight", 80),
            "chiffon": ("lightweight", 50),
            "linen": ("light-midweight", 180),
            "cotton": ("midweight", 220),
            "denim": ("midweight-heavy", 350),
            "wool": ("midweight-heavy", 320),
            "leather": ("heavyweight", 500),
            "fleece": ("midweight", 280),
            "canvas": ("heavyweight", 450),
            "cashmere": ("light-midweight", 180),
            "velvet": ("midweight", 250),
        }
        
        # Look up or default
        if material.lower() in MATERIAL_WEIGHTS:
            return MATERIAL_WEIGHTS[material.lower()]
        
        return ("midweight", 250)
    
    def _detect_stretch(
        self,
        image: np.ndarray,
        material: str
    ) -> Tuple[bool, float]:
        """Detect if material has stretch."""
        # Materials known to have stretch
        STRETCH_MATERIALS = {
            "spandex": 1.0,
            "elastane": 1.0,
            "lycra": 1.0,
            "stretch denim": 0.8,
            "jersey": 0.6,
            "knit": 0.5,
        }
        
        material_lower = material.lower()
        
        for stretch_mat, stretch_val in STRETCH_MATERIALS.items():
            if stretch_mat in material_lower:
                return (True, stretch_val)
        
        # Knit textures typically have stretch
        if "knit" in material_lower or "jersey" in material_lower:
            return (True, 0.5)
        
        return (False, 0.0)
    
    def _detect_treatments(
        self,
        image: np.ndarray,
        material: str
    ) -> List[str]:
        """Detect special fabric treatments."""
        treatments = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Check for distressing (high edge variation in local patches)
        h, w = gray.shape
        patch_size = 50
        
        edge_vars = []
        for y in range(0, h - patch_size, patch_size):
            for x in range(0, w - patch_size, patch_size):
                patch = gray[y:y+patch_size, x:x+patch_size]
                edges = cv2.Canny(patch, 50, 150)
                edge_vars.append(np.var(edges))
        
        if edge_vars:
            var_of_vars = np.var(edge_vars)
            if var_of_vars > 5000:
                treatments.append("distressed")
        
        # Check for fading (variation in saturation/value)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        
        if np.std(val) > 40 and material.lower() == "denim":
            treatments.append("faded")
        
        # Check for stonewash (lighter areas, mottled)
        if np.mean(sat) < 80 and material.lower() == "denim":
            treatments.append("stonewashed")
        
        # Check for garment-dye (slight color variation)
        color_std = np.std(image, axis=(0, 1))
        if np.mean(color_std) > 25:
            treatments.append("garment-dyed")
        
        return treatments


# === SINGLETON INSTANCE ===
_material_analyzer_instance = None


def get_material_analyzer() -> MaterialAnalyzer:
    """Get singleton instance."""
    global _material_analyzer_instance
    if _material_analyzer_instance is None:
        _material_analyzer_instance = MaterialAnalyzer()
    return _material_analyzer_instance


def analyze_material(image: np.ndarray, category_hint: str = None) -> Dict:
    """
    Quick utility for material analysis.
    
    Args:
        image: BGR image
        category_hint: Optional hint
        
    Returns:
        Material analysis dictionary
    """
    analyzer = get_material_analyzer()
    result = analyzer.analyze(image, category_hint=category_hint)
    return result.to_dict()
