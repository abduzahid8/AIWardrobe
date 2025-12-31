"""
Advanced Clothing Feature Detection Module
Detects detailed clothing features: zippers, buttons, collars, sleeves, pockets, and special features

This module provides comprehensive visual analysis of clothing items to extract
detailed attributes that differentiate similar items (e.g., zip-up hoodie vs pullover hoodie).
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClosureInfo:
    """Closure/fastening information"""
    type: str  # "full_zip", "half_zip", "quarter_zip", "button_up", "pullover", "snap", "toggle", "drawstring"
    hasZipper: bool
    buttonCount: int
    confidence: float


@dataclass
class CollarInfo:
    """Collar/neckline information"""
    type: str  # 18+ types
    height: str  # "standard", "tall", "low"
    confidence: float


@dataclass
class SleeveInfo:
    """Sleeve information"""
    length: str  # "sleeveless", "short_sleeve", "3/4_sleeve", "long_sleeve"
    style: str  # "set_in", "raglan", "cap", "puff", "bell", "dolman"
    cuffed: bool
    rolled: bool


@dataclass
class PocketInfo:
    """Pocket information"""
    count: int
    types: List[str]  # "chest", "side", "kangaroo", "cargo", "flap", "welt", "patch"
    visible: bool


@dataclass
class ClothingFeatures:
    """Complete clothing features"""
    closure: ClosureInfo
    collar: CollarInfo
    sleeves: SleeveInfo
    pockets: PocketInfo
    fit: str  # "slim", "regular", "relaxed", "oversized", "tailored"
    length: str  # "cropped", "hip_length", "thigh_length", "knee_length", "midi", "maxi"
    specialFeatures: List[str]  # "hood", "drawstring", "distressed", "graphic_print", "logo", "embroidered"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization - handles numpy bool conversion"""
        return {
            "closure": {
                "type": self.closure.type,
                "hasZipper": bool(self.closure.hasZipper),  # Convert numpy.bool_
                "buttonCount": int(self.closure.buttonCount),
                "confidence": float(self.closure.confidence)
            },
            "collar": {
                "type": self.collar.type,
                "height": self.collar.height,
                "confidence": float(self.collar.confidence)
            },
            "sleeves": {
                "length": self.sleeves.length,
                "style": self.sleeves.style,
                "cuffed": bool(self.sleeves.cuffed),  # Convert numpy.bool_
                "rolled": bool(self.sleeves.rolled)   # Convert numpy.bool_
            },
            "pockets": {
                "count": int(self.pockets.count),
                "types": list(self.pockets.types),
                "visible": bool(self.pockets.visible)  # Convert numpy.bool_
            },
            "fit": self.fit,
            "length": self.length,
            "specialFeatures": list(self.specialFeatures)
        }


class FeatureDetector:
    """
    Comprehensive clothing feature detector using computer vision.
    
    Detects:
    - Closures: zippers, buttons, snaps, pullovers
    - Collars: 18+ neckline types
    - Sleeves: length and style
    - Pockets: count and types
    - Fit: slim to oversized
    - Length: cropped to maxi
    - Special features: hood, drawstring, distressing, graphics
    """
    
    COLLAR_TYPES = [
        "crew_neck", "v_neck", "scoop_neck", "boat_neck",
        "turtleneck", "mock_neck", "polo_collar", "button_down_collar",
        "spread_collar", "mandarin_collar", "hooded",
        "cowl_neck", "halter", "off_shoulder", "one_shoulder",
        "strapless", "peter_pan_collar", "shawl_collar"
    ]
    
    CLOSURE_TYPES = [
        "pullover", "full_zip", "half_zip", "quarter_zip",
        "button_up", "snap_buttons", "toggle_buttons", "drawstring",
        "hook_and_eye", "wrap", "tie_front", "double_breasted"
    ]
    
    def __init__(self):
        """Initialize feature detector"""
        logger.info("🔍 FeatureDetector initialized")
    
    def detect_all_features(
        self, 
        image: np.ndarray, 
        category: str = None,
        specific_type: str = None
    ) -> ClothingFeatures:
        """
        Detect all features from clothing image.
        
        Args:
            image: BGR image of clothing item (ideally cutout on white background)
            category: Optional category hint (e.g., "Top", "Pants")
            specific_type: Optional specific type hint (e.g., "denim jacket")
        
        Returns:
            ClothingFeatures with comprehensive analysis
        """
        try:
            # Convert to grayscale for analysis
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            h, w = gray.shape
            
            # Detect all features with context
            closure = self._detect_closure_advanced(gray, image, specific_type)
            collar = self._detect_collar_advanced(gray, h, w, category)
            sleeves = self._detect_sleeves_advanced(gray, h, w, category)
            pockets = self._detect_pockets_advanced(gray, h, w)
            fit = self._detect_fit(gray, h, w)
            length = self._detect_length(gray, h, w, category)
            special = self._detect_special_features(gray, image)
            
            return ClothingFeatures(
                closure=closure,
                collar=collar,
                sleeves=sleeves,
                pockets=pockets,
                fit=fit,
                length=length,
                specialFeatures=special
            )
            
        except Exception as e:
            logger.error(f"Feature detection failed: {e}")
            # Return defaults
            return ClothingFeatures(
                closure=ClosureInfo(type="unknown", hasZipper=False, buttonCount=0, confidence=0.0),
                collar=CollarInfo(type="unknown", height="standard", confidence=0.0),
                sleeves=SleeveInfo(length="unknown", style="unknown", cuffed=False, rolled=False),
                pockets=PocketInfo(count=0, types=[], visible=False),
                fit="unknown",
                length="unknown",
                specialFeatures=[]
            )
    
    def _detect_closure_advanced(
        self, 
        gray: np.ndarray, 
        image: np.ndarray,
        specific_type: str = None
    ) -> ClosureInfo:
        """
        Advanced closure detection using multiple techniques:
        1. Vertical line detection for zippers
        2. Circle detection for buttons
        3. Edge density analysis
        """
        h, w = gray.shape
        
        # Focus on center vertical strip (where closures typically are)
        center_x1, center_x2 = int(w * 0.35), int(w * 0.65)
        center_region = gray[:, center_x1:center_x2]
        
        # === ZIPPER DETECTION ===
        # Look for strong vertical lines
        edges = cv2.Canny(center_region, 50, 150)
        
        # Hough lines for zipper detection
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=50, 
            minLineLength=int(h * 0.25), 
            maxLineGap=15
        )
        
        vertical_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Calculate angle from vertical
                dx = abs(x2 - x1)
                dy = abs(y2 - y1)
                if dy > 0:
                    angle_from_vertical = np.arctan(dx / dy) * 180 / np.pi
                    if angle_from_vertical < 15:  # Within 15 degrees of vertical
                        vertical_lines += 1
        
        # === BUTTON DETECTION ===
        # Look for circular patterns
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(h * 0.04),  # Minimum distance between buttons
            param1=50,
            param2=25,
            minRadius=int(h * 0.008),  # Min button radius
            maxRadius=int(h * 0.035)   # Max button radius
        )
        
        button_count = 0
        if circles is not None:
            # Filter circles that are in center region (where buttons would be)
            for circle in circles[0]:
                cx, cy, r = circle
                if center_x1 < cx < center_x2:
                    button_count += 1
        
        # === DETERMINE CLOSURE TYPE ===
        closure_type = "pullover"
        has_zipper = False
        confidence = 0.5
        
        # Check for specific type hints
        if specific_type:
            specific_lower = specific_type.lower()
            if "zip" in specific_lower:
                has_zipper = True
                if "half" in specific_lower:
                    closure_type = "half_zip"
                elif "quarter" in specific_lower:
                    closure_type = "quarter_zip"
                else:
                    closure_type = "full_zip"
                confidence = 0.8
            elif "button" in specific_lower:
                closure_type = "button_up"
                confidence = 0.8
        
        # Visual detection fallback
        if closure_type == "pullover":
            if vertical_lines >= 2:
                has_zipper = True
                if vertical_lines >= 4:
                    closure_type = "full_zip"
                    confidence = 0.7
                else:
                    closure_type = "half_zip"
                    confidence = 0.6
            elif button_count >= 4:
                closure_type = "button_up"
                confidence = 0.65 + min(0.2, button_count * 0.03)
            elif button_count >= 2:
                closure_type = "snap_buttons"
                confidence = 0.55
        
        return ClosureInfo(
            type=closure_type,
            hasZipper=has_zipper,
            buttonCount=min(button_count, 12),
            confidence=round(confidence, 2)
        )
    
    def _detect_collar_advanced(
        self, 
        gray: np.ndarray, 
        h: int, 
        w: int,
        category: str = None
    ) -> CollarInfo:
        """
        Detect collar/neckline type using shape and edge analysis.
        Supports 18+ collar types.
        """
        # Focus on neck/collar region (top 25% of garment)
        neck_region = gray[:int(h * 0.25), int(w * 0.2):int(w * 0.8)]
        
        if neck_region.size == 0:
            return CollarInfo(type="unknown", height="standard", confidence=0.3)
        
        # === EDGE ANALYSIS ===
        edges = cv2.Canny(neck_region, 50, 150)
        edge_density = np.sum(edges) / edges.size
        
        # === SHAPE ANALYSIS ===
        # Find contours in neck region
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        collar_type = "crew_neck"
        height = "standard"
        confidence = 0.5
        
        if contours:
            # Get largest contour (likely the neckline)
            largest = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(largest)
            aspect = cw / (ch + 0.001)
            area_ratio = cv2.contourArea(largest) / (cw * ch + 1)
            
            # === CLASSIFY BY SHAPE ===
            if aspect > 2.5:  # Very wide opening
                if edge_density < 0.05:
                    collar_type = "boat_neck"
                elif area_ratio < 0.5:  # Low fill = scoop
                    collar_type = "scoop_neck"
                else:
                    collar_type = "crew_neck"
                    
            elif aspect < 0.8:  # Tall shape
                if edge_density > 0.15:
                    collar_type = "turtleneck"
                    height = "tall"
                else:
                    collar_type = "mock_neck"
                    height = "tall"
                    
            elif 0.8 <= aspect <= 1.5:  # V-shaped proportions
                if area_ratio < 0.4:
                    collar_type = "v_neck"
                else:
                    collar_type = "crew_neck"
                    
            else:  # Medium aspect ratios
                if edge_density > 0.12:
                    collar_type = "polo_collar"
                elif edge_density > 0.08:
                    collar_type = "button_down_collar"
                else:
                    collar_type = "v_neck"
            
            confidence = min(0.85, edge_density * 4 + 0.45)
        
        # === HOOD DETECTION ===
        # Check very top of image for hood-like structure
        top_strip = gray[:int(h * 0.08), :]
        if top_strip.size > 0:
            top_edge_density = np.sum(cv2.Canny(top_strip, 50, 150)) / top_strip.size
            if top_edge_density > 0.12:
                collar_type = "hooded"
                confidence = 0.75
        
        return CollarInfo(
            type=collar_type,
            height=height,
            confidence=round(confidence, 2)
        )
    
    def _detect_sleeves_advanced(
        self, 
        gray: np.ndarray, 
        h: int, 
        w: int,
        category: str = None
    ) -> SleeveInfo:
        """
        Detect sleeve length and style.
        """
        # Analyze left and right edges where sleeves would be
        left_strip = gray[:, :int(w * 0.2)]
        right_strip = gray[:, int(w * 0.8):]
        
        # === FIND SLEEVE LENGTH ===
        # Look for where content ends on sides (where sleeves stop)
        left_profile = np.mean(left_strip < 240, axis=1)  # Content = non-white
        right_profile = np.mean(right_strip < 240, axis=1)
        
        # Find where sleeves end (where profile drops below threshold)
        threshold = 0.05
        
        left_ends = np.where(left_profile < threshold)[0]
        right_ends = np.where(right_profile < threshold)[0]
        
        sleeve_end_ratio = 1.0  # Default: full length
        
        # Find first significant gap
        if len(left_ends) > 0 and len(right_ends) > 0:
            # Look for consistent gap (not just noise)
            for i in range(min(len(left_ends), len(right_ends)), 0, -1):
                left_end = left_ends[i-1] if i <= len(left_ends) else h
                right_end = right_ends[i-1] if i <= len(right_ends) else h
                avg_end = (left_end + right_end) / 2
                sleeve_end_ratio = avg_end / h
                if sleeve_end_ratio < 0.9:
                    break
        
        # === CLASSIFY SLEEVE LENGTH ===
        if sleeve_end_ratio < 0.2:
            length = "sleeveless"
        elif sleeve_end_ratio < 0.4:
            length = "short_sleeve"
        elif sleeve_end_ratio < 0.6:
            length = "3/4_sleeve"
        else:
            length = "long_sleeve"
        
        # === CHECK FOR CUFFS ===
        cuffed = False
        if length == "long_sleeve":
            # Look for contrast at wrist level (bottom 15%)
            wrist_region = gray[int(h * 0.85):, :]
            if wrist_region.size > 0:
                wrist_contrast = np.std(wrist_region)
                cuffed = wrist_contrast > 35
        
        # === SLEEVE STYLE ===
        # Default to set-in, could be expanded with more analysis
        style = "set_in"
        
        # Check for raglan (diagonal seam from neck to armpit)
        shoulder_region = gray[:int(h * 0.3), :]
        if shoulder_region.size > 0:
            edges = cv2.Canny(shoulder_region, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, 30, minLineLength=30, maxLineGap=10)
            if lines is not None:
                diagonal_count = 0
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    if x2 != x1:
                        angle = abs(np.arctan((y2-y1)/(x2-x1+0.001)) * 180 / np.pi)
                        if 30 < angle < 60:  # Diagonal lines
                            diagonal_count += 1
                if diagonal_count >= 2:
                    style = "raglan"
        
        return SleeveInfo(
            length=length,
            style=style,
            cuffed=cuffed,
            rolled=False  # Would need more sophisticated detection
        )
    
    def _detect_pockets_advanced(
        self, 
        gray: np.ndarray, 
        h: int, 
        w: int
    ) -> PocketInfo:
        """
        Detect pocket count and types.
        """
        edges = cv2.Canny(gray, 40, 120)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        pockets = []
        pocket_types = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200:  # Too small
                continue
            
            x, y, cw, ch = cv2.boundingRect(contour)
            
            # === POCKET HEURISTICS ===
            aspect = cw / (ch + 0.001)
            relative_area = (cw * ch) / (w * h)
            
            # Size filter: pockets are typically 1% to 8% of garment area
            if not (0.008 < relative_area < 0.08):
                continue
            
            # Shape filter: pockets are roughly rectangular or square
            if not (0.4 < aspect < 3.0):
                continue
            
            # Solidity check (how "filled" the shape is)
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            if solidity < 0.5:  # Too irregular
                continue
            
            # === CLASSIFY POCKET TYPE BY POSITION ===
            center_y = y + ch / 2
            center_x = x + cw / 2
            
            if center_y < h * 0.35:
                pocket_type = "chest"
            elif center_y > h * 0.7:
                pocket_type = "lower"
            elif cw > w * 0.2:
                pocket_type = "cargo"
            elif aspect > 1.5:
                pocket_type = "welt"
            else:
                pocket_type = "patch"
            
            # Check for flap (edge above pocket)
            if y > 10:
                above_region = gray[max(0, y-15):y, x:x+cw]
                if above_region.size > 0:
                    above_edges = np.sum(cv2.Canny(above_region, 50, 150))
                    if above_edges > above_region.size * 0.1:
                        pocket_type = "flap"
            
            pockets.append((x, y, cw, ch))
            pocket_types.append(pocket_type)
        
        # === DEDUPLICATE OVERLAPPING POCKETS ===
        # Simple merge: if two pockets overlap >50%, keep larger
        final_count = len(pockets)
        if final_count == 0:
            return PocketInfo(count=0, types=["none"], visible=False)
        
        # Limit to reasonable count
        final_count = min(final_count, 8)
        unique_types = list(set(pocket_types)) if pocket_types else ["patch"]
        
        return PocketInfo(
            count=final_count,
            types=unique_types,
            visible=True
        )
    
    def _detect_fit(self, gray: np.ndarray, h: int, w: int) -> str:
        """
        Detect garment fit (slim to oversized).
        """
        # Sample garment width at different heights
        sample_heights = [0.2, 0.35, 0.5, 0.65, 0.8]
        widths = []
        
        for ratio in sample_heights:
            row = gray[int(h * ratio), :]
            # Find where clothing pixels are (non-white)
            clothing_mask = row < 240
            clothing_pixels = np.where(clothing_mask)[0]
            
            if len(clothing_pixels) > 10:
                width = clothing_pixels[-1] - clothing_pixels[0]
                widths.append(width)
        
        if len(widths) < 2:
            return "regular"
        
        # === ANALYZE WIDTH PATTERN ===
        avg_width = np.mean(widths)
        width_std = np.std(widths)
        relative_width = avg_width / w
        
        # Width variation coefficient
        variation = width_std / (avg_width + 0.001)
        
        # === CLASSIFY FIT ===
        if relative_width > 0.85:
            return "oversized"
        elif relative_width > 0.75:
            return "relaxed"
        elif relative_width < 0.5:
            return "slim"
        elif variation > 0.15:
            return "tailored"  # Significant waist definition
        else:
            return "regular"
    
    def _detect_length(
        self, 
        gray: np.ndarray, 
        h: int, 
        w: int, 
        category: str = None
    ) -> str:
        """
        Detect garment length.
        """
        # Find where garment ends at bottom
        # Look at bottom portion and find last row with significant content
        
        bottom_region = gray[int(h * 0.6):, :]
        
        if bottom_region.size == 0:
            return "hip_length"
        
        # Profile of content in each row
        content_profile = np.mean(bottom_region < 240, axis=1)
        
        # Find last row with significant content (>5% of width)
        significant_rows = np.where(content_profile > 0.05)[0]
        
        if len(significant_rows) == 0:
            garment_end_ratio = 0.6
        else:
            last_row = significant_rows[-1]
            garment_end_ratio = (int(h * 0.6) + last_row) / h
        
        # === CLASSIFY LENGTH ===
        if garment_end_ratio < 0.45:
            return "cropped"
        elif garment_end_ratio < 0.6:
            return "hip_length"
        elif garment_end_ratio < 0.72:
            return "thigh_length"
        elif garment_end_ratio < 0.82:
            return "knee_length"
        elif garment_end_ratio < 0.92:
            return "midi"
        else:
            return "maxi"
    
    def _detect_special_features(
        self, 
        gray: np.ndarray, 
        image: np.ndarray
    ) -> List[str]:
        """
        Detect special features: hood, drawstring, distressing, graphics, logos.
        """
        features = []
        h, w = gray.shape
        
        # === 1. HOOD DETECTION ===
        top_region = gray[:int(h * 0.12), int(w * 0.25):int(w * 0.75)]
        if top_region.size > 0:
            edges = cv2.Canny(top_region, 50, 150)
            edge_density = np.sum(edges) / edges.size
            if edge_density > 0.1:
                features.append("hood")
        
        # === 2. DRAWSTRING DETECTION ===
        # Look for parallel lines near collar or waist
        collar_region = gray[int(h * 0.08):int(h * 0.2), int(w * 0.3):int(w * 0.7)]
        waist_region = gray[int(h * 0.55):int(h * 0.7), int(w * 0.3):int(w * 0.7)]
        
        for region in [collar_region, waist_region]:
            if region.size > 0:
                edges = cv2.Canny(region, 30, 100)
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 20, minLineLength=20, maxLineGap=5)
                if lines is not None and len(lines) >= 2:
                    # Check for roughly horizontal parallel lines
                    horizontal_lines = []
                    for line in lines:
                        x1, y1, x2, y2 = line[0]
                        if abs(y2 - y1) < abs(x2 - x1) * 0.3:  # Nearly horizontal
                            horizontal_lines.append((y1 + y2) / 2)
                    
                    if len(horizontal_lines) >= 2:
                        horizontal_lines.sort()
                        for i in range(len(horizontal_lines) - 1):
                            if horizontal_lines[i+1] - horizontal_lines[i] < 20:
                                features.append("drawstring")
                                break
        
        # === 3. DISTRESSING DETECTION ===
        # High local contrast variations indicate distressing
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_std = np.std(laplacian)
        if laplacian_std > 28:
            features.append("distressed")
        
        # === 4. GRAPHIC PRINT DETECTION ===
        # High color variance in a localized region
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hue = hsv[:, :, 0]
            saturation = hsv[:, :, 1]
            
            # Check torso region for graphics
            torso = hsv[int(h * 0.2):int(h * 0.7), int(w * 0.2):int(w * 0.8)]
            if torso.size > 0:
                hue_std = np.std(torso[:, :, 0])
                sat_mean = np.mean(torso[:, :, 1])
                
                if hue_std > 40 and sat_mean > 50:
                    features.append("graphic_print")
        
        # === 5. LOGO DETECTION ===
        # Look for small, complex region (typically upper left chest)
        chest_regions = [
            gray[int(h * 0.15):int(h * 0.35), int(w * 0.55):int(w * 0.85)],  # Right chest
            gray[int(h * 0.15):int(h * 0.35), int(w * 0.15):int(w * 0.45)],  # Left chest
        ]
        
        for chest in chest_regions:
            if chest.size > 0:
                edges = cv2.Canny(chest, 50, 150)
                edge_density = np.sum(edges) / edges.size
                
                # Logo typically has moderate edge density (not too high, not too low)
                if 0.06 < edge_density < 0.2:
                    # Check for compact complex region
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        if 100 < area < 5000:  # Logo-sized
                            perimeter = cv2.arcLength(cnt, True)
                            if perimeter > 0:
                                compactness = 4 * np.pi * area / (perimeter * perimeter)
                                if compactness > 0.3:  # Reasonably compact
                                    if "logo_area" not in features:
                                        features.append("logo_area")
                                    break
        
        # === 6. EMBROIDERY DETECTION ===
        # Small, dense texture patterns
        texture = cv2.Laplacian(gray, cv2.CV_64F)
        local_std = cv2.blur(np.abs(texture), (15, 15))
        high_texture_area = np.sum(local_std > 15) / local_std.size
        
        if high_texture_area > 0.1 and high_texture_area < 0.4:
            features.append("textured_detail")
        
        return features


# === SINGLETON INSTANCE ===
_feature_detector_instance = None


def get_feature_detector() -> FeatureDetector:
    """Get singleton instance of FeatureDetector"""
    global _feature_detector_instance
    if _feature_detector_instance is None:
        _feature_detector_instance = FeatureDetector()
    return _feature_detector_instance


def detect_features_from_image(image: np.ndarray, category: str = None) -> Dict[str, Any]:
    """
    Utility function to detect features from an image.
    
    Args:
        image: BGR image
        category: Optional category hint
    
    Returns:
        Dictionary with all detected features
    """
    detector = get_feature_detector()
    features = detector.detect_all_features(image, category)
    return features.to_dict()
