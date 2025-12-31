"""
Outfit Recommender Module
Fashion knowledge + rule-based outfit composition system

Provides fashion theory intelligence:
- Color harmony analysis
- Style compatibility rules
- Occasion-based recommendations
- Body type considerations
"""

import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import colorsys
import math

logger = logging.getLogger(__name__)


class ColorFamily(Enum):
    """Color families for harmony matching."""
    RED = "red"
    ORANGE = "orange"
    YELLOW = "yellow"
    GREEN = "green"
    BLUE = "blue"
    PURPLE = "purple"
    PINK = "pink"
    BROWN = "brown"
    NEUTRAL = "neutral"  # black, white, gray, beige


class StyleCategory(Enum):
    """Style categories for matching."""
    CASUAL = "casual"
    SMART_CASUAL = "smart_casual"
    BUSINESS = "business"
    FORMAL = "formal"
    SPORTY = "sporty"
    BOHEMIAN = "bohemian"
    STREETWEAR = "streetwear"
    PREPPY = "preppy"
    MINIMALIST = "minimalist"
    VINTAGE = "vintage"


class Occasion(Enum):
    """Occasion types."""
    WORK = "work"
    CASUAL = "casual"
    DATE = "date"
    INTERVIEW = "interview"
    WEDDING = "wedding"
    PARTY = "party"
    WORKOUT = "workout"
    BEACH = "beach"
    FORMAL_EVENT = "formal_event"
    TRAVEL = "travel"


@dataclass
class ColorInfo:
    """Detailed color information."""
    name: str
    hex_code: str
    family: ColorFamily
    hue: float  # 0-360
    saturation: float  # 0-1
    lightness: float  # 0-1
    is_neutral: bool = False


@dataclass
class OutfitScore:
    """Detailed outfit scoring."""
    overall: float
    color_harmony: float
    style_consistency: float
    occasion_fit: float
    weather_suitability: float
    reasoning: str


class OutfitRecommender:
    """
    🎨 Smart Outfit Composition Engine
    
    Uses fashion theory and rules to compose perfect outfits:
    - Color harmony (complementary, analogous, triadic)
    - Style compatibility
    - Occasion requirements
    - Weather considerations
    - Personal preferences
    """
    
    # Color name to hex mapping (comprehensive)
    COLOR_DATABASE = {
        # Neutrals
        "black": {"hex": "#000000", "family": ColorFamily.NEUTRAL},
        "white": {"hex": "#FFFFFF", "family": ColorFamily.NEUTRAL},
        "gray": {"hex": "#808080", "family": ColorFamily.NEUTRAL},
        "grey": {"hex": "#808080", "family": ColorFamily.NEUTRAL},
        "charcoal": {"hex": "#36454F", "family": ColorFamily.NEUTRAL},
        "cream": {"hex": "#FFFDD0", "family": ColorFamily.NEUTRAL},
        "beige": {"hex": "#F5F5DC", "family": ColorFamily.NEUTRAL},
        "ivory": {"hex": "#FFFFF0", "family": ColorFamily.NEUTRAL},
        "taupe": {"hex": "#483C32", "family": ColorFamily.NEUTRAL},
        "khaki": {"hex": "#C3B091", "family": ColorFamily.NEUTRAL},
        "tan": {"hex": "#D2B48C", "family": ColorFamily.NEUTRAL},
        
        # Blues
        "navy": {"hex": "#1B3A57", "family": ColorFamily.BLUE},
        "navy blue": {"hex": "#1B3A57", "family": ColorFamily.BLUE},
        "blue": {"hex": "#2196F3", "family": ColorFamily.BLUE},
        "royal blue": {"hex": "#4169E1", "family": ColorFamily.BLUE},
        "sky blue": {"hex": "#87CEEB", "family": ColorFamily.BLUE},
        "baby blue": {"hex": "#89CFF0", "family": ColorFamily.BLUE},
        "teal": {"hex": "#008080", "family": ColorFamily.BLUE},
        "turquoise": {"hex": "#40E0D0", "family": ColorFamily.BLUE},
        "cobalt": {"hex": "#0047AB", "family": ColorFamily.BLUE},
        "denim": {"hex": "#1560BD", "family": ColorFamily.BLUE},
        
        # Greens
        "green": {"hex": "#4CAF50", "family": ColorFamily.GREEN},
        "olive": {"hex": "#808000", "family": ColorFamily.GREEN},
        "forest green": {"hex": "#228B22", "family": ColorFamily.GREEN},
        "emerald": {"hex": "#50C878", "family": ColorFamily.GREEN},
        "mint": {"hex": "#98FF98", "family": ColorFamily.GREEN},
        "sage": {"hex": "#B2AC88", "family": ColorFamily.GREEN},
        "hunter green": {"hex": "#355E3B", "family": ColorFamily.GREEN},
        
        # Reds
        "red": {"hex": "#E53935", "family": ColorFamily.RED},
        "burgundy": {"hex": "#800020", "family": ColorFamily.RED},
        "maroon": {"hex": "#800000", "family": ColorFamily.RED},
        "wine": {"hex": "#722F37", "family": ColorFamily.RED},
        "crimson": {"hex": "#DC143C", "family": ColorFamily.RED},
        "rust": {"hex": "#B7410E", "family": ColorFamily.RED},
        "coral": {"hex": "#FF7F50", "family": ColorFamily.RED},
        
        # Yellows
        "yellow": {"hex": "#FFC107", "family": ColorFamily.YELLOW},
        "gold": {"hex": "#FFD700", "family": ColorFamily.YELLOW},
        "mustard": {"hex": "#FFDB58", "family": ColorFamily.YELLOW},
        "lemon": {"hex": "#FFF44F", "family": ColorFamily.YELLOW},
        
        # Oranges
        "orange": {"hex": "#FF9800", "family": ColorFamily.ORANGE},
        "burnt orange": {"hex": "#CC5500", "family": ColorFamily.ORANGE},
        "peach": {"hex": "#FFDAB9", "family": ColorFamily.ORANGE},
        "terracotta": {"hex": "#E2725B", "family": ColorFamily.ORANGE},
        
        # Purples
        "purple": {"hex": "#9C27B0", "family": ColorFamily.PURPLE},
        "lavender": {"hex": "#E6E6FA", "family": ColorFamily.PURPLE},
        "violet": {"hex": "#EE82EE", "family": ColorFamily.PURPLE},
        "plum": {"hex": "#8B4789", "family": ColorFamily.PURPLE},
        "mauve": {"hex": "#E0B0FF", "family": ColorFamily.PURPLE},
        
        # Pinks
        "pink": {"hex": "#FFC0CB", "family": ColorFamily.PINK},
        "hot pink": {"hex": "#FF69B4", "family": ColorFamily.PINK},
        "blush": {"hex": "#DE5D83", "family": ColorFamily.PINK},
        "rose": {"hex": "#FF007F", "family": ColorFamily.PINK},
        "dusty pink": {"hex": "#D4A5A5", "family": ColorFamily.PINK},
        "salmon": {"hex": "#FA8072", "family": ColorFamily.PINK},
        
        # Browns
        "brown": {"hex": "#795548", "family": ColorFamily.BROWN},
        "chocolate": {"hex": "#7B3F00", "family": ColorFamily.BROWN},
        "camel": {"hex": "#C19A6B", "family": ColorFamily.BROWN},
        "coffee": {"hex": "#6F4E37", "family": ColorFamily.BROWN},
        "cognac": {"hex": "#9A463D", "family": ColorFamily.BROWN},
        "chestnut": {"hex": "#954535", "family": ColorFamily.BROWN},
    }
    
    # Style compatibility matrix
    # Higher score = more compatible
    STYLE_COMPATIBILITY = {
        StyleCategory.CASUAL: {
            StyleCategory.CASUAL: 1.0,
            StyleCategory.SMART_CASUAL: 0.7,
            StyleCategory.SPORTY: 0.6,
            StyleCategory.STREETWEAR: 0.8,
            StyleCategory.BOHEMIAN: 0.5,
        },
        StyleCategory.SMART_CASUAL: {
            StyleCategory.CASUAL: 0.7,
            StyleCategory.SMART_CASUAL: 1.0,
            StyleCategory.BUSINESS: 0.6,
            StyleCategory.PREPPY: 0.8,
            StyleCategory.MINIMALIST: 0.7,
        },
        StyleCategory.BUSINESS: {
            StyleCategory.SMART_CASUAL: 0.6,
            StyleCategory.BUSINESS: 1.0,
            StyleCategory.FORMAL: 0.7,
            StyleCategory.MINIMALIST: 0.6,
        },
        StyleCategory.FORMAL: {
            StyleCategory.BUSINESS: 0.7,
            StyleCategory.FORMAL: 1.0,
        },
        StyleCategory.SPORTY: {
            StyleCategory.CASUAL: 0.6,
            StyleCategory.SPORTY: 1.0,
            StyleCategory.STREETWEAR: 0.5,
        },
        StyleCategory.STREETWEAR: {
            StyleCategory.CASUAL: 0.8,
            StyleCategory.SPORTY: 0.5,
            StyleCategory.STREETWEAR: 1.0,
            StyleCategory.BOHEMIAN: 0.4,
        },
    }
    
    # Occasion requirements
    OCCASION_REQUIREMENTS = {
        Occasion.WORK: {
            "required_categories": [["upper_clothes", "pants"], ["dress"]],
            "preferred_styles": [StyleCategory.BUSINESS, StyleCategory.SMART_CASUAL],
            "avoid": ["ripped", "loud patterns", "casual shorts"],
            "color_guidance": "subdued, professional colors preferred"
        },
        Occasion.INTERVIEW: {
            "required_categories": [["upper_clothes", "pants", "jacket"], ["dress", "jacket"]],
            "preferred_styles": [StyleCategory.BUSINESS, StyleCategory.FORMAL],
            "avoid": ["casual", "flashy", "distracting patterns"],
            "color_guidance": "navy, black, gray, white, muted tones"
        },
        Occasion.CASUAL: {
            "required_categories": [["upper_clothes", "pants"], ["dress"]],
            "preferred_styles": [StyleCategory.CASUAL, StyleCategory.SMART_CASUAL],
            "avoid": [],
            "color_guidance": "any colors work"
        },
        Occasion.DATE: {
            "required_categories": [["upper_clothes", "pants"], ["dress"]],
            "preferred_styles": [StyleCategory.SMART_CASUAL, StyleCategory.MINIMALIST],
            "avoid": ["too casual", "workout clothes"],
            "color_guidance": "flattering colors, consider red accents"
        },
        Occasion.WEDDING: {
            "required_categories": [["dress"], ["upper_clothes", "pants", "jacket"]],
            "preferred_styles": [StyleCategory.FORMAL],
            "avoid": ["white", "black (sometimes)", "casual"],
            "color_guidance": "elegant colors, avoid white (bride's color)"
        },
        Occasion.PARTY: {
            "required_categories": [["upper_clothes", "pants"], ["dress", "skirt"]],
            "preferred_styles": [StyleCategory.SMART_CASUAL, StyleCategory.STREETWEAR],
            "avoid": [],
            "color_guidance": "have fun with colors and patterns"
        },
        Occasion.WORKOUT: {
            "required_categories": [["upper_clothes", "pants"]],
            "preferred_styles": [StyleCategory.SPORTY],
            "avoid": ["formal", "restrictive"],
            "color_guidance": "comfort first, any colors"
        },
    }
    
    def __init__(self):
        """Initialize the outfit recommender."""
        logger.info("OutfitRecommender initialized")
    
    def get_color_info(self, color_name: str) -> ColorInfo:
        """
        Get detailed color information from color name.
        
        Args:
            color_name: Color name (e.g., "navy blue")
            
        Returns:
            ColorInfo with all color details
        """
        color_lower = color_name.lower().strip()
        
        if color_lower in self.COLOR_DATABASE:
            data = self.COLOR_DATABASE[color_lower]
            hex_code = data["hex"]
            family = data["family"]
        else:
            # Default to neutral gray if unknown
            hex_code = "#808080"
            family = ColorFamily.NEUTRAL
        
        # Calculate HSL from hex
        r, g, b = self._hex_to_rgb(hex_code)
        h, l, s = colorsys.rgb_to_hls(r/255, g/255, b/255)
        
        return ColorInfo(
            name=color_name,
            hex_code=hex_code,
            family=family,
            hue=h * 360,
            saturation=s,
            lightness=l,
            is_neutral=family == ColorFamily.NEUTRAL
        )
    
    def calculate_color_harmony(self, colors: List[str]) -> Tuple[float, str]:
        """
        Calculate color harmony score for a set of colors.
        
        Args:
            colors: List of color names
            
        Returns:
            Tuple of (harmony_score 0-1, harmony_type description)
        """
        if len(colors) < 2:
            return 1.0, "single_color"
        
        color_infos = [self.get_color_info(c) for c in colors]
        
        # Check neutral + accent pattern (always works)
        neutrals = [c for c in color_infos if c.is_neutral]
        accents = [c for c in color_infos if not c.is_neutral]
        
        if len(neutrals) >= len(colors) - 1:
            return 0.95, "neutral_palette"
        
        if len(neutrals) >= len(colors) // 2:
            # Mostly neutrals with accents = generally safe
            return 0.85, "neutral_with_accents"
        
        # Check for monochromatic (same color family)
        families = set(c.family for c in color_infos if not c.is_neutral)
        if len(families) == 1:
            return 0.90, f"monochromatic_{list(families)[0].value}"
        
        # Check complementary (opposite on color wheel)
        hues = [c.hue for c in color_infos if not c.is_neutral]
        if len(hues) >= 2:
            hue_diff = abs(hues[0] - hues[1])
            if 150 < hue_diff < 210:  # Roughly opposite
                return 0.85, "complementary"
        
        # Check analogous (adjacent on color wheel)
        if len(hues) >= 2:
            max_diff = max(hues) - min(hues)
            if max_diff < 60:  # Adjacent colors
                return 0.80, "analogous"
        
        # Check triadic
        if len(hues) == 3:
            diffs = sorted([abs(hues[0] - hues[1]), abs(hues[1] - hues[2]), abs(hues[0] - hues[2])])
            if all(100 < d < 140 for d in diffs):
                return 0.75, "triadic"
        
        # Default - some harmony
        return 0.65, "varied"
    
    def calculate_style_compatibility(self, styles: List[str]) -> Tuple[float, str]:
        """
        Calculate style compatibility score.
        
        Args:
            styles: List of style tags
            
        Returns:
            Tuple of (compatibility_score 0-1, description)
        """
        if len(styles) < 2:
            return 1.0, "consistent_style"
        
        # Try to parse style tags into categories
        style_categories = []
        for style in styles:
            style_lower = style.lower()
            for cat in StyleCategory:
                if cat.value in style_lower:
                    style_categories.append(cat)
                    break
        
        if len(style_categories) < 2:
            return 0.80, "compatible_styles"
        
        # Check pairwise compatibility
        total_score = 0
        pairs = 0
        for i, s1 in enumerate(style_categories):
            for s2 in style_categories[i+1:]:
                compat = self.STYLE_COMPATIBILITY.get(s1, {}).get(s2, 0.5)
                total_score += compat
                pairs += 1
        
        if pairs > 0:
            avg_score = total_score / pairs
            return avg_score, "mixed_styles" if avg_score < 0.7 else "compatible_styles"
        
        return 0.75, "unknown_compatibility"
    
    def score_outfit_for_occasion(
        self,
        outfit_items: List[Dict],
        occasion: str
    ) -> Tuple[float, str]:
        """
        Score how well an outfit fits an occasion.
        
        Args:
            outfit_items: List of clothing item dicts
            occasion: Occasion name
            
        Returns:
            Tuple of (fit_score 0-1, explanation)
        """
        try:
            occ = Occasion(occasion.lower())
        except ValueError:
            return 0.70, f"Unknown occasion: {occasion}"
        
        requirements = self.OCCASION_REQUIREMENTS.get(occ, {})
        
        score = 1.0
        explanations = []
        
        # Check required categories
        categories = [item.get("category", "").lower() for item in outfit_items]
        required_combos = requirements.get("required_categories", [])
        
        if required_combos:
            combo_found = False
            for combo in required_combos:
                if all(any(req in cat for cat in categories) for req in combo):
                    combo_found = True
                    break
            
            if not combo_found:
                score -= 0.2
                explanations.append("Missing required clothing categories")
        
        # Check style compatibility
        preferred_styles = requirements.get("preferred_styles", [])
        item_styles = []
        for item in outfit_items:
            item_styles.extend(item.get("styleTags", []))
        
        if preferred_styles and item_styles:
            style_match = any(
                ps.value in s.lower() 
                for ps in preferred_styles 
                for s in item_styles
            )
            if not style_match:
                score -= 0.15
                explanations.append("Style doesn't match occasion")
        
        explanation = "; ".join(explanations) if explanations else "Good fit for occasion"
        return max(0, score), explanation
    
    def score_for_weather(
        self,
        outfit_items: List[Dict],
        weather: Dict
    ) -> Tuple[float, str]:
        """
        Score outfit for weather appropriateness.
        
        Args:
            outfit_items: List of clothing item dicts
            weather: Weather dict with temp, condition
            
        Returns:
            Tuple of (suitability_score 0-1, explanation)
        """
        temp = weather.get("temp", 20)  # Celsius
        condition = weather.get("condition", "clear").lower()
        
        score = 1.0
        explanations = []
        
        # Check materials for weather
        materials = []
        for item in outfit_items:
            if item.get("material"):
                materials.append(item["material"].lower())
        
        # Cold weather checks
        if temp < 10:
            warm_materials = ["wool", "fleece", "down", "cashmere"]
            if not any(m in mat for mat in materials for m in warm_materials):
                if not any("jacket" in item.get("category", "") or "coat" in item.get("specificType", "") 
                          for item in outfit_items):
                    score -= 0.3
                    explanations.append("May need warmer layers for cold weather")
        
        # Hot weather checks
        if temp > 28:
            heavy_materials = ["wool", "leather", "fleece"]
            if any(m in mat for mat in materials for m in heavy_materials):
                score -= 0.2
                explanations.append("Heavy materials may be uncomfortable in heat")
        
        # Rain checks
        if "rain" in condition:
            if not any("water" in mat or "rain" in mat for mat in materials):
                score -= 0.15
                explanations.append("Consider waterproof options for rain")
        
        explanation = "; ".join(explanations) if explanations else "Weather appropriate"
        return max(0, score), explanation
    
    def score_outfit(
        self,
        outfit_items: List[Dict],
        occasion: str = None,
        weather: Dict = None
    ) -> OutfitScore:
        """
        Calculate comprehensive outfit score.
        
        Args:
            outfit_items: List of clothing item dicts
            occasion: Optional occasion
            weather: Optional weather info
            
        Returns:
            OutfitScore with all scores and reasoning
        """
        # Color harmony
        colors = [item.get("primaryColor", "") for item in outfit_items if item.get("primaryColor")]
        color_score, color_type = self.calculate_color_harmony(colors)
        
        # Style consistency
        styles = []
        for item in outfit_items:
            styles.extend(item.get("styleTags", []))
        style_score, style_desc = self.calculate_style_compatibility(styles)
        
        # Occasion fit
        if occasion:
            occasion_score, occasion_desc = self.score_outfit_for_occasion(outfit_items, occasion)
        else:
            occasion_score, occasion_desc = 1.0, "No occasion specified"
        
        # Weather suitability
        if weather:
            weather_score, weather_desc = self.score_for_weather(outfit_items, weather)
        else:
            weather_score, weather_desc = 1.0, "No weather specified"
        
        # Calculate overall score
        overall = (color_score * 0.3 + style_score * 0.3 + 
                   occasion_score * 0.25 + weather_score * 0.15)
        
        reasoning = f"Color: {color_type} ({color_score:.0%}), Style: {style_desc} ({style_score:.0%})"
        if occasion:
            reasoning += f", Occasion: {occasion_desc}"
        if weather:
            reasoning += f", Weather: {weather_desc}"
        
        return OutfitScore(
            overall=overall,
            color_harmony=color_score,
            style_consistency=style_score,
            occasion_fit=occasion_score,
            weather_suitability=weather_score,
            reasoning=reasoning
        )
    
    def suggest_additions(
        self,
        current_items: List[Dict],
        occasion: str = None
    ) -> List[str]:
        """
        Suggest items to complete an outfit.
        
        Args:
            current_items: Current outfit items
            occasion: Target occasion
            
        Returns:
            List of suggested additions
        """
        suggestions = []
        categories = [item.get("category", "") for item in current_items]
        
        # Basic outfit completeness
        has_top = any(c in ["upper_clothes", "dress"] for c in categories)
        has_bottom = any(c in ["pants", "skirt", "dress", "shorts"] for c in categories)
        has_shoes = any(c in ["shoes", "footwear"] for c in categories)
        
        if not has_top:
            suggestions.append("Add a top (shirt, blouse, or sweater)")
        if not has_bottom:
            suggestions.append("Add bottoms (pants, skirt, or shorts)")
        if not has_shoes:
            suggestions.append("Don't forget shoes!")
        
        # Occasion-specific suggestions
        if occasion:
            try:
                occ = Occasion(occasion.lower())
                reqs = self.OCCASION_REQUIREMENTS.get(occ, {})
                
                if occ == Occasion.INTERVIEW:
                    if "jacket" not in categories:
                        suggestions.append("Consider adding a blazer for a professional look")
                
                if occ == Occasion.WEDDING:
                    suggestions.append("Consider accessories like a clutch or elegant jewelry")
                
            except ValueError:
                pass
        
        return suggestions
    
    def _hex_to_rgb(self, hex_code: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_code = hex_code.lstrip('#')
        return tuple(int(hex_code[i:i+2], 16) for i in (0, 2, 4))


# Singleton instance
_outfit_recommender_instance = None


def get_outfit_recommender() -> OutfitRecommender:
    """Get singleton instance of OutfitRecommender."""
    global _outfit_recommender_instance
    if _outfit_recommender_instance is None:
        _outfit_recommender_instance = OutfitRecommender()
    return _outfit_recommender_instance
