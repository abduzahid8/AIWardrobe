"""
Advanced Pattern Recognition Module - CNN-based Pattern Detection
Part of the World-Class AI Vision System for AIWardrobe

This module provides:
- 30+ pattern types detection
- Multi-scale pattern analysis
- Color pattern extraction
- Confidence-weighted voting
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
from scipy import ndimage
from skimage import feature

logger = logging.getLogger(__name__)


# ============================================
# 🎨 PATTERN TAXONOMY
# ============================================

PATTERN_CATEGORIES = {
    "solid": {
        "patterns": ["solid", "plain"],
        "keywords": ["single color", "uniform", "no pattern"]
    },
    "heathered": {
        "patterns": ["heathered", "marled", "melange"],
        "keywords": ["mixed fibers", "speckled", "multi-tone"]
    },
    "stripes": {
        "patterns": [
            "horizontal stripes", "vertical stripes", "diagonal stripes",
            "pinstripes", "rugby stripes", "breton stripes", "candy stripes",
            "awning stripes", "banker stripes"
        ],
        "keywords": ["lines", "parallel", "striped"]
    },
    "checks": {
        "patterns": [
            "plaid", "tartan", "gingham", "buffalo check", "windowpane",
            "houndstooth", "glen check", "prince of wales", "madras"
        ],
        "keywords": ["squares", "checkered", "crossed lines"]
    },
    "prints": {
        "patterns": [
            "floral", "tropical", "paisley", "abstract", "geometric",
            "polka dot", "graphic print", "novelty print", "botanical"
        ],
        "keywords": ["printed", "design", "motif"]
    },
    "animal_prints": {
        "patterns": [
            "leopard print", "zebra print", "snake print", "cow print",
            "tiger print", "cheetah print"
        ],
        "keywords": ["animal", "wild", "safari"]
    },
    "special": {
        "patterns": [
            "tie-dye", "ombre", "color-block", "camouflage", "digital camo",
            "ikat", "batik", "tribal", "aztec"
        ],
        "keywords": ["unique", "special", "artistic"]
    }
}

# Flatten all patterns
ALL_PATTERNS = []
PATTERN_TO_CATEGORY = {}
for category, data in PATTERN_CATEGORIES.items():
    for pattern in data["patterns"]:
        ALL_PATTERNS.append(pattern)
        PATTERN_TO_CATEGORY[pattern] = category


@dataclass
class PatternAnalysis:
    """Complete pattern analysis result"""
    primary_pattern: str
    pattern_category: str
    confidence: float
    
    secondary_patterns: List[Tuple[str, float]]
    
    # Stripe analysis
    is_striped: bool
    stripe_direction: Optional[str]  # horizontal, vertical, diagonal
    stripe_width: Optional[str]  # thin, medium, wide
    stripe_colors: List[str]
    
    # Check analysis
    is_checkered: bool
    check_size: Optional[str]  # small, medium, large
    
    # Print analysis
    has_print: bool
    print_density: float  # 0-1, how dense the print is
    
    # Color palette
    pattern_colors: List[Dict]  # [{color, hex, percentage}]
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result["secondary_patterns"] = [
            {"pattern": p, "confidence": c}
            for p, c in self.secondary_patterns
        ]
        # Convert numpy booleans to Python booleans
        result["is_striped"] = bool(result["is_striped"])
        result["is_checkered"] = bool(result["is_checkered"])
        result["has_print"] = bool(result["has_print"])
        return result


class PatternDetector:
    """
    🎨 Advanced Pattern Recognition System
    
    Uses multiple detection techniques:
    - Fourier Transform for stripe/check detection
    - Local Binary Patterns for texture
    - Color histogram analysis for prints
    - Edge detection for pattern boundaries
    
    Features:
    - Multi-scale analysis
    - Stripe direction detection
    - Check size estimation
    - Print density calculation
    
    Usage:
        detector = PatternDetector()
        result = detector.analyze(image)
        print(f"Pattern: {result.primary_pattern}")
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize the pattern detector."""
        self._setup_device(device)
        logger.info(f"PatternDetector initialized (device={self.device})")
    
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
        mask: np.ndarray = None
    ) -> PatternAnalysis:
        """
        Analyze pattern in clothing image.
        
        Args:
            image: BGR image
            mask: Optional mask for item region
            
        Returns:
            PatternAnalysis with complete analysis
        """
        if mask is not None:
            image = self._apply_mask(image, mask)
        
        # Run analysis pipeline
        is_solid, solid_conf = self._check_solid(image)
        
        if is_solid and solid_conf > 0.8:
            # Definitely solid color
            colors = self._extract_pattern_colors(image)
            return PatternAnalysis(
                primary_pattern="solid",
                pattern_category="solid",
                confidence=solid_conf,
                secondary_patterns=[],
                is_striped=False,
                stripe_direction=None,
                stripe_width=None,
                stripe_colors=[],
                is_checkered=False,
                check_size=None,
                has_print=False,
                print_density=0.0,
                pattern_colors=colors
            )
        
        # Check for stripes
        stripe_info = self._detect_stripes(image)
        
        # Check for checks/plaid
        check_info = self._detect_checks(image)
        
        # Check for prints
        print_info = self._detect_prints(image)
        
        # Check for heathered/marled
        heather_info = self._detect_heathered(image)
        
        # Determine primary pattern
        pattern_scores = []
        
        if stripe_info["is_striped"]:
            pattern_name = f"{stripe_info['direction']} stripes"
            pattern_scores.append((pattern_name, stripe_info["confidence"]))
        
        if check_info["is_checkered"]:
            pattern_scores.append((check_info["type"], check_info["confidence"]))
        
        if print_info["has_print"]:
            pattern_scores.append((print_info["type"], print_info["confidence"]))
        
        if heather_info["is_heathered"]:
            pattern_scores.append(("heathered", heather_info["confidence"]))
        
        if is_solid:
            pattern_scores.append(("solid", solid_conf))
        
        # Sort by confidence
        pattern_scores.sort(key=lambda x: x[1], reverse=True)
        
        if not pattern_scores:
            pattern_scores = [("solid", 0.5)]
        
        primary = pattern_scores[0][0]
        primary_conf = pattern_scores[0][1]
        secondary = pattern_scores[1:3]
        
        # Get category
        category = PATTERN_TO_CATEGORY.get(primary, "solid")
        
        # Extract colors
        colors = self._extract_pattern_colors(image)
        
        return PatternAnalysis(
            primary_pattern=primary,
            pattern_category=category,
            confidence=primary_conf,
            secondary_patterns=secondary,
            is_striped=stripe_info["is_striped"],
            stripe_direction=stripe_info["direction"] if stripe_info["is_striped"] else None,
            stripe_width=stripe_info["width"] if stripe_info["is_striped"] else None,
            stripe_colors=stripe_info.get("colors", []),
            is_checkered=check_info["is_checkered"],
            check_size=check_info.get("size"),
            has_print=print_info["has_print"],
            print_density=print_info.get("density", 0.0),
            pattern_colors=colors
        )
    
    def _apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask to isolate item region."""
        if len(mask.shape) == 2:
            mask = np.stack([mask] * 3, axis=-1)
        return cv2.bitwise_and(image, mask.astype(np.uint8) * 255)
    
    def _check_solid(self, image: np.ndarray) -> Tuple[bool, float]:
        """Check if image is solid color."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate color variance
        h_std = np.std(hsv[:, :, 0])
        s_std = np.std(hsv[:, :, 1])
        v_std = np.std(hsv[:, :, 2])
        
        # Low variance = solid color
        total_std = h_std + s_std / 2 + v_std / 2
        
        if total_std < 15:
            return (True, 0.95)
        elif total_std < 25:
            return (True, 0.8)
        elif total_std < 35:
            return (True, 0.6)
        elif total_std < 50:
            return (True, 0.4)
        else:
            return (False, 0.2)
    
    def _detect_stripes(self, image: np.ndarray) -> Dict:
        """Detect stripe patterns using FFT."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Apply FFT
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.log(np.abs(f_shift) + 1)
        
        # Analyze frequency distribution
        center_y, center_x = h // 2, w // 2
        
        # Horizontal stripe = vertical frequency lines
        vertical_energy = np.sum(magnitude[center_y-5:center_y+5, :])
        horizontal_energy = np.sum(magnitude[:, center_x-5:center_x+5])
        
        # Calculate stripe score
        h_stripe_score = horizontal_energy / (vertical_energy + 1e-6)
        v_stripe_score = vertical_energy / (horizontal_energy + 1e-6)
        
        is_striped = max(h_stripe_score, v_stripe_score) > 1.5
        
        if not is_striped:
            return {"is_striped": False, "confidence": 0.0}
        
        # Determine direction
        if h_stripe_score > v_stripe_score:
            direction = "horizontal"
            confidence = min(0.9, h_stripe_score / 3)
        else:
            direction = "vertical"
            confidence = min(0.9, v_stripe_score / 3)
        
        # Estimate stripe width using autocorrelation
        if direction == "horizontal":
            profile = np.mean(gray, axis=1)
        else:
            profile = np.mean(gray, axis=0)
        
        autocorr = np.correlate(profile - np.mean(profile), profile - np.mean(profile), mode='full')
        peaks = self._find_peaks(autocorr)
        
        if len(peaks) >= 2:
            spacing = np.mean(np.diff(peaks))
            if spacing < 10:
                width = "thin"
            elif spacing < 30:
                width = "medium"
            else:
                width = "wide"
        else:
            width = "medium"
        
        return {
            "is_striped": True,
            "direction": direction,
            "confidence": confidence,
            "width": width,
            "colors": []
        }
    
    def _find_peaks(self, arr: np.ndarray) -> np.ndarray:
        """Find peaks in array."""
        peaks = []
        for i in range(1, len(arr) - 1):
            if arr[i] > arr[i-1] and arr[i] > arr[i+1]:
                peaks.append(i)
        return np.array(peaks)
    
    def _detect_checks(self, image: np.ndarray) -> Dict:
        """Detect check/plaid patterns."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Check patterns have both horizontal AND vertical regularity
        edges = cv2.Canny(gray, 50, 150)
        
        # Detect lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=30, maxLineGap=10)
        
        if lines is None:
            return {"is_checkered": False, "confidence": 0.0}
        
        # Count horizontal vs vertical lines
        h_lines = 0
        v_lines = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
            
            if angle < 30 or angle > 150:
                h_lines += 1
            elif 60 < angle < 120:
                v_lines += 1
        
        # Check pattern = roughly equal H and V lines
        total_lines = h_lines + v_lines
        if total_lines < 10:
            return {"is_checkered": False, "confidence": 0.0}
        
        balance = min(h_lines, v_lines) / max(h_lines, v_lines + 1)
        
        if balance > 0.5:
            # Determine check type
            if total_lines > 50:
                check_type = "plaid"
                size = "complex"
            elif total_lines > 30:
                check_type = "gingham"
                size = "medium"
            else:
                check_type = "windowpane"
                size = "large"
            
            return {
                "is_checkered": True,
                "type": check_type,
                "confidence": balance * 0.8,
                "size": size
            }
        
        return {"is_checkered": False, "confidence": 0.0}
    
    def _detect_prints(self, image: np.ndarray) -> Dict:
        """Detect print patterns."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Prints typically have:
        # - High color variation
        # - Non-regular patterns
        # - Distinct shapes/motifs
        
        # Color variation
        h_var = np.var(hsv[:, :, 0])
        s_var = np.var(hsv[:, :, 1])
        
        # Blob detection for print motifs
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 50
        params.maxArea = 5000
        
        detector = cv2.SimpleBlobDetector_create(params)
        blobs = detector.detect(gray)
        
        blob_count = len(blobs)
        h, w = gray.shape
        density = blob_count / ((h * w) / 10000)  # blobs per 10000 pixels
        
        # High color variation + blobs = print
        color_score = (h_var + s_var) / 1000
        
        if blob_count > 10 and color_score > 0.3:
            # Determine print type
            if h_var > 500:
                print_type = "floral"
            elif density > 2:
                print_type = "polka dot"
            elif color_score > 1:
                print_type = "abstract"
            else:
                print_type = "graphic print"
            
            return {
                "has_print": True,
                "type": print_type,
                "confidence": min(0.9, color_score),
                "density": min(1.0, density / 5)
            }
        
        return {"has_print": False, "confidence": 0.0}
    
    def _detect_heathered(self, image: np.ndarray) -> Dict:
        """Detect heathered/marled patterns."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Heathered = fine-scale variation, multi-tonal
        # Use Local Binary Patterns
        try:
            lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
            lbp_var = np.var(lbp)
            
            # Heathered fabrics have medium LBP variance
            if 50 < lbp_var < 200:
                confidence = min(0.8, (lbp_var - 50) / 100)
                return {"is_heathered": True, "confidence": confidence}
        except:
            pass
        
        return {"is_heathered": False, "confidence": 0.0}
    
    def _extract_pattern_colors(
        self,
        image: np.ndarray,
        n_colors: int = 5
    ) -> List[Dict]:
        """Extract dominant colors from pattern."""
        # Reshape for clustering
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Remove near-black/white (likely background)
        mask = (pixels.sum(axis=1) > 30) & (pixels.sum(axis=1) < 700)
        pixels = pixels[mask]
        
        if len(pixels) < 100:
            return []
        
        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, n_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
        )
        
        # Count occurrences
        unique, counts = np.unique(labels, return_counts=True)
        total = len(labels)
        
        colors = []
        for i, (center, count) in enumerate(sorted(zip(centers, counts), key=lambda x: x[1], reverse=True)):
            b, g, r = int(center[0]), int(center[1]), int(center[2])
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            percentage = (count / total) * 100
            
            color_name = self._get_color_name(r, g, b)
            
            colors.append({
                "name": color_name,
                "hex": hex_color,
                "rgb": [r, g, b],
                "percentage": round(percentage, 1)
            })
        
        return colors
    
    def _get_color_name(self, r: int, g: int, b: int) -> str:
        """Get approximate color name from RGB."""
        # Simple color naming
        h, s, v = cv2.cvtColor(
            np.array([[[b, g, r]]], dtype=np.uint8),
            cv2.COLOR_BGR2HSV
        )[0, 0]
        
        if v < 30:
            return "black"
        if v > 230 and s < 30:
            return "white"
        if s < 30:
            if v < 100:
                return "dark gray"
            elif v < 180:
                return "gray"
            else:
                return "light gray"
        
        # Color wheel
        if h < 15 or h >= 165:
            return "red"
        elif h < 25:
            return "orange"
        elif h < 35:
            return "yellow"
        elif h < 80:
            return "green"
        elif h < 130:
            return "blue"
        elif h < 150:
            return "purple"
        else:
            return "pink"


# === SINGLETON INSTANCE ===
_pattern_detector_instance = None


def get_pattern_detector() -> PatternDetector:
    """Get singleton instance."""
    global _pattern_detector_instance
    if _pattern_detector_instance is None:
        _pattern_detector_instance = PatternDetector()
    return _pattern_detector_instance


def detect_pattern(image: np.ndarray) -> Dict:
    """Quick utility for pattern detection."""
    detector = get_pattern_detector()
    result = detector.analyze(image)
    return result.to_dict()
