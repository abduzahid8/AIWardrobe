"""
3D Garment Understanding Module
Analyzes garment shape, drape, and structural properties from 2D images

This module provides:
- Garment shape analysis (silhouette, structure)
- Drape and fit estimation
- Depth/layering understanding
- Body-garment relationship analysis
- Pose-aware garment fitting
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


# ============================================
# 🎯 GARMENT SHAPE TAXONOMY
# ============================================

SILHOUETTES = {
    "fitted": {
        "description": "Close to the body, tailored",
        "volume": 0.2,
        "applies_to": ["blazer", "dress shirt", "bodycon dress"]
    },
    "slim": {
        "description": "Slightly fitted, modern cut",
        "volume": 0.3,
        "applies_to": ["slim jeans", "polo", "fitted t-shirt"]
    },
    "regular": {
        "description": "Standard fit, comfortable",
        "volume": 0.5,
        "applies_to": ["classic t-shirt", "straight jeans", "regular shirt"]
    },
    "relaxed": {
        "description": "Loose but not oversized",
        "volume": 0.6,
        "applies_to": ["casual shirt", "relaxed jeans", "a-line dress"]
    },
    "oversized": {
        "description": "Intentionally large, boxy",
        "volume": 0.8,
        "applies_to": ["oversized hoodie", "boyfriend jeans", "oversized coat"]
    },
    "voluminous": {
        "description": "Very full, structured volume",
        "volume": 0.9,
        "applies_to": ["puffer jacket", "ball gown", "wide palazzo pants"]
    }
}

STRUCTURES = {
    "structured": {
        "description": "Maintains shape, often with padding/lining",
        "stiffness": 0.9,
        "examples": ["blazer", "structured bag", "tailored coat"]
    },
    "semi-structured": {
        "description": "Some shape retention, flexible",
        "stiffness": 0.6,
        "examples": ["denim jacket", "chinos", "leather jacket"]
    },
    "unstructured": {
        "description": "Soft, drapes with body",
        "stiffness": 0.3,
        "examples": ["cardigan", "jersey dress", "soft sweater"]
    },
    "fluid": {
        "description": "Flows freely, liquid drape",
        "stiffness": 0.1,
        "examples": ["silk blouse", "maxi skirt", "chiffon dress"]
    }
}

DRAPE_TYPES = {
    "stiff": "Minimal drape, holds shape",
    "soft": "Gentle drape, follows curves",
    "flowing": "Continuous fluid movement",
    "crisp": "Clean lines, sharp edges",
    "relaxed": "Natural settling, casual"
}

NECKLINE_DEPTH = {
    "high": (0, 0.15),      # Crew, turtle, mock
    "standard": (0.15, 0.25),  # Regular crew, boat
    "medium": (0.25, 0.35),    # V-neck, scoop
    "low": (0.35, 0.50),       # Deep V, plunging
    "very_low": (0.50, 1.0)    # Extreme plunge
}

HEMLINE_TYPES = {
    "straight": "Horizontal even hem",
    "curved": "Rounded hem edge",
    "asymmetric": "Uneven, artistic hem",
    "high-low": "Short front, long back",
    "slit": "Opening for movement",
    "raw": "Unfinished edge",
    "cuffed": "Turned-up hem"
}


@dataclass
class GarmentDimensions:
    """Estimated garment dimensions"""
    shoulder_width: float  # Relative to image width
    chest_width: float
    waist_width: float
    hip_width: float
    length: float  # Top to bottom
    sleeve_length: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DrapeAnalysis:
    """Garment drape characteristics"""
    drape_type: str  # "stiff", "soft", "flowing", etc.
    drape_quality: float  # 0-1, how pronounced
    fold_count: int  # Number of visible folds
    fold_depth: str  # "shallow", "medium", "deep"
    movement_potential: float  # 0-1, how much it would move
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class LayeringInfo:
    """Information about garment layering"""
    layer_position: str  # "base", "mid", "outer"
    layer_weight: str  # "sheer", "light", "medium", "heavy"
    stackable: bool  # Can other layers go over
    requires_base: bool  # Needs something under
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass 
class GarmentAnalysis3D:
    """Complete 3D garment analysis"""
    # Silhouette
    silhouette: str
    silhouette_confidence: float
    volume_score: float  # 0-1
    
    # Structure
    structure: str
    stiffness_score: float  # 0-1
    
    # Dimensions
    dimensions: GarmentDimensions
    aspect_ratio: float  # width/height
    
    # Drape
    drape: DrapeAnalysis
    
    # Neckline/hemline (for tops/dresses)
    neckline_depth: Optional[str] = None
    hemline_type: Optional[str] = None
    
    # Layering
    layering: Optional[LayeringInfo] = None
    
    # Fit estimation
    estimated_fit: str = "regular"  # "tight", "fitted", "regular", "loose", "oversized"
    body_conformity: float = 0.5  # 0=boxy, 1=body-hugging
    
    # Construction details
    seam_visibility: str = "standard"  # "hidden", "standard", "exposed"
    construction_quality: str = "standard"  # "budget", "standard", "high", "luxury"
    
    def to_dict(self) -> Dict:
        result = {
            "silhouette": self.silhouette,
            "silhouetteConfidence": self.silhouette_confidence,
            "volumeScore": self.volume_score,
            "structure": self.structure,
            "stiffnessScore": self.stiffness_score,
            "dimensions": self.dimensions.to_dict(),
            "aspectRatio": self.aspect_ratio,
            "drape": self.drape.to_dict(),
            "necklineDepth": self.neckline_depth,
            "hemlineType": self.hemline_type,
            "layering": self.layering.to_dict() if self.layering else None,
            "estimatedFit": self.estimated_fit,
            "bodyConformity": self.body_conformity,
            "seamVisibility": self.seam_visibility,
            "constructionQuality": self.construction_quality
        }
        return result


class GarmentAnalyzer3D:
    """
    🎯 3D Garment Understanding from 2D Images
    
    Infers 3D properties of garments using:
    - Contour analysis for silhouette
    - Edge detection for structure
    - Texture analysis for drape
    - Proportion estimation for dimensions
    
    Note: This is pseudo-3D analysis from 2D images.
    For true 3D, would need depth sensors or multiple views.
    
    Usage:
        analyzer = GarmentAnalyzer3D()
        result = analyzer.analyze(image, category="Top")
        print(f"Silhouette: {result.silhouette}")
        print(f"Fit: {result.estimated_fit}")
    """
    
    def __init__(self):
        """Initialize 3D garment analyzer."""
        logger.info("GarmentAnalyzer3D initialized")
    
    def analyze(
        self,
        image: np.ndarray,
        category: str = None,
        mask: np.ndarray = None
    ) -> GarmentAnalysis3D:
        """
        Analyze 3D properties of garment.
        
        Args:
            image: BGR image (ideally cropped to garment)
            category: Optional category hint
            mask: Optional mask for garment region
            
        Returns:
            GarmentAnalysis3D with complete analysis
        """
        # Apply mask if provided
        if mask is not None:
            image = self._apply_mask(image, mask)
        
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Analyze components
        silhouette, sil_conf, volume = self._analyze_silhouette(gray, category)
        structure, stiffness = self._analyze_structure(gray, category)
        dimensions = self._estimate_dimensions(gray, category)
        drape = self._analyze_drape(gray, image)
        neckline = self._detect_neckline(gray) if category in ["Top", "Dress"] else None
        hemline = self._detect_hemline(gray)
        layering = self._infer_layering(category, structure)
        fit, conformity = self._estimate_fit(gray, silhouette, structure)
        seam_vis, quality = self._analyze_construction(gray, image)
        
        return GarmentAnalysis3D(
            silhouette=silhouette,
            silhouette_confidence=sil_conf,
            volume_score=volume,
            structure=structure,
            stiffness_score=stiffness,
            dimensions=dimensions,
            aspect_ratio=w / h,
            drape=drape,
            neckline_depth=neckline,
            hemline_type=hemline,
            layering=layering,
            estimated_fit=fit,
            body_conformity=conformity,
            seam_visibility=seam_vis,
            construction_quality=quality
        )
    
    def _apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask to isolate garment."""
        if len(mask.shape) == 2:
            mask = np.stack([mask] * 3, axis=-1)
        return cv2.bitwise_and(image, mask.astype(np.uint8) * 255)
    
    def _analyze_silhouette(
        self,
        gray: np.ndarray,
        category: str = None
    ) -> Tuple[str, float, float]:
        """Analyze garment silhouette."""
        h, w = gray.shape
        
        # Find contours
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return ("regular", 0.5, 0.5)
        
        # Get largest contour
        main_contour = max(contours, key=cv2.contourArea)
        
        # Calculate bounding rect
        x, y, bw, bh = cv2.boundingRect(main_contour)
        
        # Calculate fill ratio (how much of bounding box is filled)
        contour_area = cv2.contourArea(main_contour)
        bbox_area = bw * bh
        fill_ratio = contour_area / bbox_area if bbox_area > 0 else 0.5
        
        # Analyze width variation along height
        widths = []
        for row in range(y, y + bh, max(1, bh // 20)):
            row_pixels = np.where(binary[row, x:x+bw] > 0)[0]
            if len(row_pixels) > 0:
                widths.append(len(row_pixels) / bw)
        
        if widths:
            width_variance = np.var(widths)
            avg_width = np.mean(widths)
        else:
            width_variance = 0.1
            avg_width = 0.5
        
        # Determine silhouette based on fill and variance
        if fill_ratio > 0.85:
            silhouette = "fitted"
            volume = 0.25
        elif fill_ratio > 0.75:
            silhouette = "slim"
            volume = 0.35
        elif fill_ratio > 0.60:
            silhouette = "regular"
            volume = 0.5
        elif fill_ratio > 0.45:
            silhouette = "relaxed"
            volume = 0.65
        else:
            silhouette = "oversized"
            volume = 0.8
        
        # Category-based adjustments
        if category:
            cat_lower = category.lower()
            if "puffer" in cat_lower or "puffy" in cat_lower:
                silhouette = "voluminous"
                volume = 0.9
            elif "skinny" in cat_lower or "slim" in cat_lower:
                silhouette = "slim"
                volume = 0.3
        
        confidence = 0.7 + (0.3 * (1 - width_variance * 2))
        
        return (silhouette, min(1.0, confidence), volume)
    
    def _analyze_structure(
        self,
        gray: np.ndarray,
        category: str = None
    ) -> Tuple[str, float]:
        """Analyze garment structure."""
        # Calculate edge density and straightness
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Detect lines for structure
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        line_count = len(lines) if lines is not None else 0
        
        # More lines + edges = more structured
        if line_count > 20 and edge_density > 0.1:
            structure = "structured"
            stiffness = 0.85
        elif line_count > 10 or edge_density > 0.08:
            structure = "semi-structured"
            stiffness = 0.6
        elif edge_density > 0.04:
            structure = "unstructured"
            stiffness = 0.35
        else:
            structure = "fluid"
            stiffness = 0.15
        
        # Category adjustments
        if category:
            cat_lower = category.lower()
            STRUCTURED_ITEMS = ["blazer", "suit", "coat", "leather jacket"]
            FLUID_ITEMS = ["silk", "chiffon", "jersey", "knit"]
            
            if any(item in cat_lower for item in STRUCTURED_ITEMS):
                structure = "structured"
                stiffness = max(stiffness, 0.8)
            elif any(item in cat_lower for item in FLUID_ITEMS):
                structure = "fluid"
                stiffness = min(stiffness, 0.2)
        
        return (structure, stiffness)
    
    def _estimate_dimensions(
        self,
        gray: np.ndarray,
        category: str = None
    ) -> GarmentDimensions:
        """Estimate garment dimensions."""
        h, w = gray.shape
        
        # Find contour
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return GarmentDimensions(0.8, 0.8, 0.7, 0.7, 0.9)
        
        main_contour = max(contours, key=cv2.contourArea)
        x, y, bw, bh = cv2.boundingRect(main_contour)
        
        # Measure widths at different heights
        def width_at_pct(pct):
            row = int(y + bh * pct)
            if 0 <= row < h:
                row_pixels = np.where(binary[row, :] > 0)[0]
                if len(row_pixels) > 0:
                    return (max(row_pixels) - min(row_pixels)) / w
            return 0
        
        shoulder_width = width_at_pct(0.1)
        chest_width = width_at_pct(0.3)
        waist_width = width_at_pct(0.5)
        hip_width = width_at_pct(0.7)
        
        # Estimate sleeve length for tops
        sleeve_length = None
        if category and "top" in category.lower():
            # Look for horizontal extension
            left_extent = np.mean([np.min(np.where(row > 0)[0]) if np.any(row > 0) else w//2 for row in binary[int(h*0.2):int(h*0.5)]])
            right_extent = np.mean([np.max(np.where(row > 0)[0]) if np.any(row > 0) else w//2 for row in binary[int(h*0.2):int(h*0.5)]])
            sleeve_length = (right_extent - left_extent) / w
        
        return GarmentDimensions(
            shoulder_width=round(shoulder_width, 2),
            chest_width=round(chest_width, 2),
            waist_width=round(waist_width, 2),
            hip_width=round(hip_width, 2),
            length=round(bh / h, 2),
            sleeve_length=round(sleeve_length, 2) if sleeve_length else None
        )
    
    def _analyze_drape(
        self,
        gray: np.ndarray,
        color_image: np.ndarray
    ) -> DrapeAnalysis:
        """Analyze fabric drape characteristics."""
        h, w = gray.shape
        
        # Detect folds through gradient analysis
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Vertical lines indicate folds in hanging garments
        vertical_energy = np.sum(np.abs(sobelx))
        horizontal_energy = np.sum(np.abs(sobely))
        
        fold_ratio = vertical_energy / (horizontal_energy + 1)
        
        # Count peaks in vertical gradient (fold lines)
        avg_vertical = np.mean(np.abs(sobelx), axis=0)
        peaks = self._count_peaks(avg_vertical)
        
        # Determine drape type
        if fold_ratio < 0.5:
            drape_type = "stiff"
            quality = 0.2
        elif fold_ratio < 0.8:
            drape_type = "crisp"
            quality = 0.4
        elif fold_ratio < 1.2:
            drape_type = "soft"
            quality = 0.6
        else:
            drape_type = "flowing"
            quality = 0.85
        
        # Fold depth based on gradient magnitude
        max_gradient = np.max(np.abs(sobelx))
        if max_gradient > 100:
            fold_depth = "deep"
        elif max_gradient > 50:
            fold_depth = "medium"
        else:
            fold_depth = "shallow"
        
        movement_potential = min(1.0, quality + 0.1)
        
        return DrapeAnalysis(
            drape_type=drape_type,
            drape_quality=round(quality, 2),
            fold_count=peaks,
            fold_depth=fold_depth,
            movement_potential=round(movement_potential, 2)
        )
    
    def _count_peaks(self, signal: np.ndarray, threshold: float = 0.3) -> int:
        """Count peaks in signal."""
        if len(signal) < 3:
            return 0
        
        signal = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-6)
        peaks = 0
        
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1] and signal[i] > threshold:
                peaks += 1
        
        return peaks
    
    def _detect_neckline(self, gray: np.ndarray) -> Optional[str]:
        """Detect neckline depth."""
        h, w = gray.shape
        
        # Look at top portion of garment
        top_region = gray[:int(h * 0.3), :]
        
        # Find the lowest point of the neckline (darkest region at top center)
        center_third = top_region[:, w//3:2*w//3]
        
        # Find where garment starts
        col_means = np.mean(center_third, axis=1)
        threshold = np.mean(col_means)
        
        neckline_row = 0
        for i, val in enumerate(col_means):
            if val > threshold:
                neckline_row = i
                break
        
        depth_ratio = neckline_row / (h * 0.3)
        
        for depth_name, (low, high) in NECKLINE_DEPTH.items():
            if low <= depth_ratio < high:
                return depth_name
        
        return "standard"
    
    def _detect_hemline(self, gray: np.ndarray) -> str:
        """Detect hemline type."""
        h, w = gray.shape
        
        # Look at bottom portion
        bottom = gray[int(h * 0.85):, :]
        
        # Find edge
        edges = cv2.Canny(bottom, 50, 150)
        
        # Get bottom edge profile
        edge_rows = np.argmax(edges[::-1, :] > 0, axis=0)
        
        if len(edge_rows) == 0:
            return "straight"
        
        # Analyze variation
        variation = np.std(edge_rows)
        
        if variation < 3:
            return "straight"
        elif variation < 8:
            return "curved"
        else:
            # Check for high-low pattern
            left_avg = np.mean(edge_rows[:w//3])
            right_avg = np.mean(edge_rows[2*w//3:])
            center_avg = np.mean(edge_rows[w//3:2*w//3])
            
            if abs(left_avg - right_avg) > 10:
                return "asymmetric"
            elif abs(center_avg - (left_avg + right_avg) / 2) > 10:
                return "high-low"
        
        return "curved"
    
    def _infer_layering(
        self,
        category: str,
        structure: str
    ) -> LayeringInfo:
        """Infer layering properties."""
        if not category:
            return LayeringInfo("mid", "medium", True, False)
        
        cat_lower = category.lower()
        
        # Base layers
        if any(x in cat_lower for x in ["t-shirt", "tank", "camisole", "undershirt"]):
            return LayeringInfo("base", "light", True, False)
        
        # Mid layers
        if any(x in cat_lower for x in ["sweater", "cardigan", "hoodie", "shirt"]):
            return LayeringInfo("mid", "medium", True, True)
        
        # Outer layers
        if any(x in cat_lower for x in ["jacket", "coat", "blazer", "parka"]):
            weight = "heavy" if "coat" in cat_lower or "parka" in cat_lower else "medium"
            return LayeringInfo("outer", weight, False, True)
        
        return LayeringInfo("mid", "medium", True, False)
    
    def _estimate_fit(
        self,
        gray: np.ndarray,
        silhouette: str,
        structure: str
    ) -> Tuple[str, float]:
        """Estimate garment fit."""
        # Map silhouette to fit
        FIT_MAP = {
            "fitted": ("fitted", 0.85),
            "slim": ("fitted", 0.7),
            "regular": ("regular", 0.5),
            "relaxed": ("loose", 0.35),
            "oversized": ("oversized", 0.2),
            "voluminous": ("oversized", 0.15)
        }
        
        fit, conformity = FIT_MAP.get(silhouette, ("regular", 0.5))
        
        # Adjust by structure
        if structure == "structured":
            conformity = min(conformity + 0.1, 1.0)
        elif structure == "fluid":
            conformity = max(conformity - 0.1, 0.0)
        
        return fit, round(conformity, 2)
    
    def _analyze_construction(
        self,
        gray: np.ndarray,
        color_image: np.ndarray
    ) -> Tuple[str, str]:
        """Analyze construction quality indicators."""
        # Look for visible seams
        edges = cv2.Canny(gray, 30, 100)
        edge_density = np.sum(edges > 0) / edges.size
        
        if edge_density < 0.02:
            seam_vis = "hidden"
        elif edge_density < 0.06:
            seam_vis = "standard"
        else:
            seam_vis = "exposed"
        
        # Quality heuristic based on color consistency
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        color_var = np.std(hsv[:, :, 1]) + np.std(hsv[:, :, 2])
        
        if color_var < 30:
            quality = "luxury"
        elif color_var < 50:
            quality = "high"
        elif color_var < 80:
            quality = "standard"
        else:
            quality = "budget"
        
        return seam_vis, quality


# === SINGLETON INSTANCE ===
_garment_3d_instance = None


def get_garment_analyzer_3d() -> GarmentAnalyzer3D:
    """Get singleton instance."""
    global _garment_3d_instance
    if _garment_3d_instance is None:
        _garment_3d_instance = GarmentAnalyzer3D()
    return _garment_3d_instance


def analyze_garment_3d(image: np.ndarray, category: str = None) -> Dict:
    """Quick utility for 3D garment analysis."""
    analyzer = get_garment_analyzer_3d()
    result = analyzer.analyze(image, category)
    return result.to_dict()
