"""
Color Harmony Analyzer - Advanced Color Theory for Fashion
Analyzes color relationships and provides color-based recommendations

This module provides:
- Color harmony analysis (complementary, analogous, triadic, etc.)
- Color temperature detection
- Skin tone matching
- Color palette extraction
- Color combination recommendations
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
import colorsys

logger = logging.getLogger(__name__)


# ============================================
# 🎨 COLOR THEORY DEFINITIONS
# ============================================

# Named colors with HSV ranges
NAMED_COLORS = {
    # Neutrals
    "black": {"h": (0, 180), "s": (0, 30), "v": (0, 30)},
    "white": {"h": (0, 180), "s": (0, 30), "v": (220, 255)},
    "gray": {"h": (0, 180), "s": (0, 30), "v": (30, 220)},
    
    # Warm colors
    "red": {"h": (0, 10), "s": (100, 255), "v": (50, 255)},
    "orange": {"h": (10, 25), "s": (100, 255), "v": (50, 255)},
    "yellow": {"h": (25, 35), "s": (100, 255), "v": (50, 255)},
    "gold": {"h": (20, 30), "s": (100, 200), "v": (150, 255)},
    
    # Cool colors
    "green": {"h": (35, 80), "s": (50, 255), "v": (30, 255)},
    "teal": {"h": (80, 100), "s": (50, 255), "v": (30, 255)},
    "blue": {"h": (100, 130), "s": (50, 255), "v": (30, 255)},
    "navy": {"h": (100, 130), "s": (100, 255), "v": (20, 100)},
    "purple": {"h": (130, 150), "s": (50, 255), "v": (30, 255)},
    
    # Other
    "pink": {"h": (150, 170), "s": (30, 150), "v": (150, 255)},
    "magenta": {"h": (150, 170), "s": (150, 255), "v": (50, 255)},
    "brown": {"h": (10, 25), "s": (50, 200), "v": (30, 150)},
    "beige": {"h": (20, 35), "s": (20, 80), "v": (180, 255)},
    "cream": {"h": (30, 45), "s": (10, 50), "v": (220, 255)},
}

# Color harmony rules
HARMONY_TYPES = {
    "complementary": {
        "description": "Colors opposite on the color wheel",
        "hue_offset": 180,
        "energy": "high contrast, bold"
    },
    "analogous": {
        "description": "Colors adjacent on the color wheel",
        "hue_offset": 30,
        "energy": "harmonious, cohesive"
    },
    "triadic": {
        "description": "Three colors equally spaced",
        "hue_offset": 120,
        "energy": "vibrant, balanced"
    },
    "split-complementary": {
        "description": "Base color with two adjacent to complement",
        "hue_offset": [150, 210],
        "energy": "balanced contrast"
    },
    "monochromatic": {
        "description": "Variations of a single hue",
        "hue_offset": 0,
        "energy": "elegant, unified"
    },
    "neutral": {
        "description": "Black, white, gray, beige",
        "hue_offset": None,
        "energy": "timeless, versatile"
    }
}

# Season-based color palettes
SEASONAL_PALETTES = {
    "spring": {
        "colors": ["coral", "peach", "mint", "sky blue", "lavender", "warm yellow"],
        "characteristics": "warm, bright, clear"
    },
    "summer": {
        "colors": ["dusty rose", "powder blue", "soft lavender", "sage", "mauve", "cool gray"],
        "characteristics": "cool, soft, muted"
    },
    "autumn": {
        "colors": ["rust", "olive", "burnt orange", "mustard", "burgundy", "camel"],
        "characteristics": "warm, muted, earthy"
    },
    "winter": {
        "colors": ["black", "white", "true red", "royal blue", "emerald", "hot pink"],
        "characteristics": "cool, clear, bold"
    }
}


@dataclass
class ColorInfo:
    """Information about a single color"""
    name: str
    hex: str
    rgb: Tuple[int, int, int]
    hsv: Tuple[int, int, int]
    percentage: float
    temperature: str  # "warm", "cool", "neutral"
    lightness: str  # "light", "medium", "dark"


@dataclass
class ColorHarmonyAnalysis:
    """Complete color harmony analysis"""
    # Dominant colors
    dominant_colors: List[ColorInfo]
    primary_color: ColorInfo
    
    # Harmony type
    harmony_type: str
    harmony_description: str
    harmony_score: float  # 0-1, how well colors harmonize
    
    # Temperature
    overall_temperature: str  # "warm", "cool", "neutral"
    temperature_score: float  # -1 (cool) to 1 (warm)
    
    # Brightness/contrast
    brightness_level: str  # "light", "medium", "dark"
    contrast_level: str  # "low", "medium", "high"
    
    # Seasonal color analysis
    best_season: str
    season_scores: Dict[str, float]
    
    # Recommendations
    complementary_colors: List[str]
    avoid_colors: List[str]
    styling_suggestions: List[str]
    
    def to_dict(self) -> Dict:
        result = {
            "dominantColors": [asdict(c) for c in self.dominant_colors],
            "primaryColor": asdict(self.primary_color),
            "harmonyType": self.harmony_type,
            "harmonyDescription": self.harmony_description,
            "harmonyScore": self.harmony_score,
            "overallTemperature": self.overall_temperature,
            "temperatureScore": self.temperature_score,
            "brightnessLevel": self.brightness_level,
            "contrastLevel": self.contrast_level,
            "bestSeason": self.best_season,
            "seasonScores": self.season_scores,
            "complementaryColors": self.complementary_colors,
            "avoidColors": self.avoid_colors,
            "stylingSuggestions": self.styling_suggestions
        }
        return result


class ColorHarmonyAnalyzer:
    """
    🎨 Advanced Color Harmony Analysis for Fashion
    
    Analyzes clothing colors using color theory principles:
    - Extracts dominant colors with clustering
    - Identifies color harmony type
    - Determines warm/cool temperature
    - Matches to seasonal color palettes
    - Provides color combination recommendations
    
    Usage:
        analyzer = ColorHarmonyAnalyzer()
        result = analyzer.analyze(image)
        print(f"Harmony: {result.harmony_type}")
        print(f"Temperature: {result.overall_temperature}")
    """
    
    def __init__(self):
        """Initialize color harmony analyzer."""
        logger.info("ColorHarmonyAnalyzer initialized")
    
    def analyze(
        self,
        image: np.ndarray,
        mask: np.ndarray = None,
        n_colors: int = 5
    ) -> ColorHarmonyAnalysis:
        """
        Analyze color harmony in image.
        
        Args:
            image: BGR image
            mask: Optional mask to limit analysis region
            n_colors: Number of dominant colors to extract
            
        Returns:
            ColorHarmonyAnalysis with complete analysis
        """
        # Apply mask if provided
        if mask is not None:
            image = self._apply_mask(image, mask)
        
        # Extract dominant colors
        dominant_colors = self._extract_dominant_colors(image, n_colors)
        
        if not dominant_colors:
            # Return default analysis if no colors found
            return self._default_analysis()
        
        primary_color = dominant_colors[0]
        
        # Determine harmony type
        harmony_type, harmony_desc, harmony_score = self._determine_harmony(dominant_colors)
        
        # Calculate temperature
        temp, temp_score = self._calculate_temperature(dominant_colors)
        
        # Brightness and contrast
        brightness = self._determine_brightness(dominant_colors)
        contrast = self._determine_contrast(dominant_colors)
        
        # Season matching
        best_season, season_scores = self._match_seasons(dominant_colors)
        
        # Generate recommendations
        complementary = self._get_complementary_colors(primary_color)
        avoid = self._get_colors_to_avoid(primary_color)
        suggestions = self._generate_suggestions(
            harmony_type, temp, primary_color
        )
        
        return ColorHarmonyAnalysis(
            dominant_colors=dominant_colors,
            primary_color=primary_color,
            harmony_type=harmony_type,
            harmony_description=harmony_desc,
            harmony_score=harmony_score,
            overall_temperature=temp,
            temperature_score=temp_score,
            brightness_level=brightness,
            contrast_level=contrast,
            best_season=best_season,
            season_scores=season_scores,
            complementary_colors=complementary,
            avoid_colors=avoid,
            styling_suggestions=suggestions
        )
    
    def _apply_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask to isolate region."""
        if len(mask.shape) == 2:
            mask = np.stack([mask] * 3, axis=-1)
        return cv2.bitwise_and(image, mask.astype(np.uint8) * 255)
    
    def _extract_dominant_colors(
        self,
        image: np.ndarray,
        n_colors: int
    ) -> List[ColorInfo]:
        """Extract dominant colors using k-means clustering."""
        # Reshape for clustering
        pixels = image.reshape(-1, 3).astype(np.float32)
        
        # Remove near-black/white
        mask = (pixels.sum(axis=1) > 30) & (pixels.sum(axis=1) < 700)
        pixels = pixels[mask]
        
        if len(pixels) < n_colors * 10:
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
        for i in sorted(range(len(centers)), key=lambda x: counts[x], reverse=True):
            center = centers[i]
            b, g, r = int(center[0]), int(center[1]), int(center[2])
            
            # Convert to HSV
            hsv = cv2.cvtColor(
                np.array([[[b, g, r]]], dtype=np.uint8),
                cv2.COLOR_BGR2HSV
            )[0, 0]
            h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
            
            # Get color name
            name = self._get_color_name(h, s, v)
            
            # Hex
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            
            # Temperature
            temp = self._get_color_temperature(h, s)
            
            # Lightness
            lightness = "dark" if v < 80 else "light" if v > 180 else "medium"
            
            # Percentage
            pct = (counts[i] / total) * 100
            
            colors.append(ColorInfo(
                name=name,
                hex=hex_color,
                rgb=(r, g, b),
                hsv=(h, s, v),
                percentage=round(pct, 1),
                temperature=temp,
                lightness=lightness
            ))
        
        return colors
    
    def _get_color_name(self, h: int, s: int, v: int) -> str:
        """Get color name from HSV values."""
        # Check neutrals first
        if s < 30:
            if v < 30:
                return "black"
            elif v > 220:
                return "white"
            else:
                return "gray"
        
        # Check chromatic colors
        if h < 10 or h >= 170:
            return "red"
        elif h < 20:
            return "orange"
        elif h < 35:
            return "yellow"
        elif h < 80:
            return "green"
        elif h < 100:
            return "teal"
        elif h < 130:
            return "blue"
        elif h < 150:
            return "purple"
        else:
            return "pink"
    
    def _get_color_temperature(self, h: int, s: int) -> str:
        """Determine if color is warm, cool, or neutral."""
        if s < 30:
            return "neutral"
        
        # Warm: red, orange, yellow (0-35, 170+)
        # Cool: green, blue, purple (35-170)
        if h < 35 or h >= 170:
            return "warm"
        else:
            return "cool"
    
    def _determine_harmony(
        self,
        colors: List[ColorInfo]
    ) -> Tuple[str, str, float]:
        """Determine color harmony type."""
        if len(colors) < 2:
            return ("monochromatic", HARMONY_TYPES["monochromatic"]["description"], 1.0)
        
        # Check if all neutral
        if all(c.temperature == "neutral" for c in colors):
            return ("neutral", HARMONY_TYPES["neutral"]["description"], 0.95)
        
        # Get hue values (excluding neutrals)
        hues = [c.hsv[0] for c in colors if c.temperature != "neutral"]
        
        if not hues:
            return ("neutral", HARMONY_TYPES["neutral"]["description"], 0.9)
        
        if len(hues) == 1:
            return ("monochromatic", HARMONY_TYPES["monochromatic"]["description"], 0.95)
        
        # Calculate hue differences
        hue_diffs = []
        for i in range(len(hues)):
            for j in range(i + 1, len(hues)):
                diff = abs(hues[i] - hues[j])
                if diff > 90:
                    diff = 180 - diff
                hue_diffs.append(diff)
        
        avg_diff = np.mean(hue_diffs)
        
        # Classify harmony
        if avg_diff < 15:
            harmony = "monochromatic"
            score = 0.95 - avg_diff / 30
        elif avg_diff < 45:
            harmony = "analogous"
            score = 0.9 - abs(avg_diff - 30) / 60
        elif 75 < avg_diff < 105:
            harmony = "complementary"
            score = 0.85 - abs(avg_diff - 90) / 60
        elif 50 < avg_diff < 70:
            harmony = "triadic"
            score = 0.8 - abs(avg_diff - 60) / 40
        else:
            harmony = "analogous"
            score = 0.7
        
        return (
            harmony,
            HARMONY_TYPES[harmony]["description"],
            max(0.5, min(1.0, score))
        )
    
    def _calculate_temperature(
        self,
        colors: List[ColorInfo]
    ) -> Tuple[str, float]:
        """Calculate overall color temperature."""
        warm_weight = 0
        cool_weight = 0
        
        for color in colors:
            weight = color.percentage / 100
            if color.temperature == "warm":
                warm_weight += weight
            elif color.temperature == "cool":
                cool_weight += weight
        
        score = warm_weight - cool_weight
        
        if abs(score) < 0.2:
            temp = "neutral"
        elif score > 0:
            temp = "warm"
        else:
            temp = "cool"
        
        return temp, score
    
    def _determine_brightness(self, colors: List[ColorInfo]) -> str:
        """Determine overall brightness."""
        avg_v = np.mean([c.hsv[2] for c in colors])
        
        if avg_v < 80:
            return "dark"
        elif avg_v > 180:
            return "light"
        else:
            return "medium"
    
    def _determine_contrast(self, colors: List[ColorInfo]) -> str:
        """Determine contrast level."""
        if len(colors) < 2:
            return "low"
        
        v_values = [c.hsv[2] for c in colors]
        v_range = max(v_values) - min(v_values)
        
        if v_range < 50:
            return "low"
        elif v_range < 120:
            return "medium"
        else:
            return "high"
    
    def _match_seasons(
        self,
        colors: List[ColorInfo]
    ) -> Tuple[str, Dict[str, float]]:
        """Match colors to seasonal palettes."""
        scores = {}
        
        for season, data in SEASONAL_PALETTES.items():
            score = 0
            chars = data["characteristics"]
            
            # Temperature match
            if "warm" in chars:
                score += sum(
                    0.2 for c in colors if c.temperature == "warm"
                )
            elif "cool" in chars:
                score += sum(
                    0.2 for c in colors if c.temperature == "cool"
                )
            
            # Brightness match
            if "bright" in chars or "clear" in chars:
                score += sum(
                    0.15 for c in colors if c.lightness == "light"
                )
            elif "muted" in chars:
                score += sum(
                    0.15 for c in colors if c.lightness == "medium"
                )
            
            scores[season] = min(1.0, score)
        
        best = max(scores.items(), key=lambda x: x[1])
        return best[0], scores
    
    def _get_complementary_colors(self, color: ColorInfo) -> List[str]:
        """Get colors that complement the primary color."""
        if color.temperature == "neutral":
            return ["any color works well with neutrals"]
        
        # Complementary hue
        comp_hue = (color.hsv[0] + 90) % 180
        comp_name = self._get_color_name(comp_hue, 150, 150)
        
        # Analogous hue
        analog_hue = (color.hsv[0] + 15) % 180
        analog_name = self._get_color_name(analog_hue, 150, 150)
        
        return [
            comp_name,
            analog_name,
            "white",
            "black" if color.lightness == "light" else "cream"
        ]
    
    def _get_colors_to_avoid(self, color: ColorInfo) -> List[str]:
        """Get colors that clash with the primary color."""
        if color.temperature == "neutral":
            return []
        
        # Slightly off-complement often clashes
        clash_hue = (color.hsv[0] + 70) % 180
        clash_name = self._get_color_name(clash_hue, 150, 150)
        
        return [clash_name]
    
    def _generate_suggestions(
        self,
        harmony: str,
        temperature: str,
        primary: ColorInfo
    ) -> List[str]:
        """Generate styling suggestions based on color analysis."""
        suggestions = []
        
        if harmony == "monochromatic":
            suggestions.append("Add texture variation to create visual interest")
            suggestions.append("Incorporate metallic accents for dimension")
        
        if harmony == "complementary":
            suggestions.append("Use the bolder color as an accent")
            suggestions.append("Balance with neutral accessories")
        
        if temperature == "warm":
            suggestions.append("Pair with gold or bronze jewelry")
            suggestions.append("Earth-toned accessories work well")
        elif temperature == "cool":
            suggestions.append("Silver or platinum jewelry complements")
            suggestions.append("Blue-based or gray accessories work well")
        
        if primary.lightness == "dark":
            suggestions.append("Add lighter pieces for contrast")
        
        return suggestions[:3]
    
    def _default_analysis(self) -> ColorHarmonyAnalysis:
        """Return default analysis when no colors detected."""
        default_color = ColorInfo(
            name="neutral",
            hex="#808080",
            rgb=(128, 128, 128),
            hsv=(0, 0, 128),
            percentage=100.0,
            temperature="neutral",
            lightness="medium"
        )
        
        return ColorHarmonyAnalysis(
            dominant_colors=[default_color],
            primary_color=default_color,
            harmony_type="neutral",
            harmony_description="Neutral color palette",
            harmony_score=0.5,
            overall_temperature="neutral",
            temperature_score=0.0,
            brightness_level="medium",
            contrast_level="low",
            best_season="all-season",
            season_scores={s: 0.5 for s in SEASONAL_PALETTES.keys()},
            complementary_colors=["any"],
            avoid_colors=[],
            styling_suggestions=["Versatile neutral colors work with everything"]
        )


# === SINGLETON INSTANCE ===
_color_analyzer_instance = None


def get_color_harmony_analyzer() -> ColorHarmonyAnalyzer:
    """Get singleton instance."""
    global _color_analyzer_instance
    if _color_analyzer_instance is None:
        _color_analyzer_instance = ColorHarmonyAnalyzer()
    return _color_analyzer_instance


def analyze_color_harmony(image: np.ndarray) -> Dict:
    """Quick utility for color harmony analysis."""
    analyzer = get_color_harmony_analyzer()
    result = analyzer.analyze(image)
    return result.to_dict()
