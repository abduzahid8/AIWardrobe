"""
Outfit Coherence Analyzer - AI for Complete Outfit Rating
Analyzes how well clothing items work together as an outfit

This module provides:
- Overall outfit coherence scoring
- Style consistency analysis  
- Color harmony evaluation
- Occasion appropriateness
- Improvement suggestions
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


@dataclass
class OutfitItem:
    """Single item in an outfit"""
    category: str  # "Top", "Bottom", "Shoes", etc.
    specific_type: str  # "denim jacket", "skinny jeans", etc.
    primary_color: str
    color_hex: str
    style: str  # "casual", "formal", etc.
    formality: float  # 0-1
    material: Optional[str] = None
    pattern: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OutfitAnalysis:
    """Complete outfit coherence analysis"""
    # Overall scores
    coherence_score: float  # 0-1, overall outfit coherence
    style_consistency: float  # 0-1, how consistent the style is
    color_harmony: float  # 0-1, how well colors work together
    formality_consistency: float  # 0-1, formality level consistency
    
    # Style analysis
    dominant_style: str
    style_breakdown: Dict[str, float]
    
    # Color analysis
    color_palette: List[str]
    color_harmony_type: str  # "complementary", "analogous", etc.
    color_temperature: str  # "warm", "cool", "neutral"
    
    # Occasion matching
    best_occasions: List[str]
    formality_level: str  # "casual", "smart casual", "business", "formal"
    
    # Issues and suggestions
    issues: List[str]
    suggestions: List[str]
    
    # Rating
    rating: str  # "excellent", "good", "fair", "poor"
    rating_emoji: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class OutfitCoherenceAnalyzer:
    """
    👗 AI-Powered Outfit Coherence Analysis
    
    Evaluates how well clothing items work together:
    - Style consistency across items
    - Color harmony and palette
    - Formality level matching
    - Occasion appropriateness
    
    Provides actionable suggestions for improvement.
    
    Usage:
        analyzer = OutfitCoherenceAnalyzer()
        items = [
            OutfitItem(category="Top", specific_type="denim jacket", ...),
            OutfitItem(category="Bottom", specific_type="chinos", ...),
        ]
        result = analyzer.analyze(items)
        print(f"Coherence: {result.coherence_score:.1%}")
    """
    
    # Style compatibility matrix (higher = more compatible)
    STYLE_COMPATIBILITY = {
        ("casual", "casual"): 1.0,
        ("casual", "streetwear"): 0.8,
        ("casual", "athletic"): 0.7,
        ("casual", "preppy"): 0.6,
        ("formal", "formal"): 1.0,
        ("formal", "business"): 0.9,
        ("formal", "luxury"): 0.8,
        ("business", "business"): 1.0,
        ("business", "preppy"): 0.7,
        ("business", "minimalist"): 0.8,
        ("streetwear", "streetwear"): 1.0,
        ("streetwear", "edgy"): 0.8,
        ("streetwear", "athletic"): 0.6,
        ("minimalist", "minimalist"): 1.0,
        ("minimalist", "casual"): 0.7,
        ("minimalist", "business"): 0.8,
        ("bohemian", "bohemian"): 1.0,
        ("bohemian", "vintage"): 0.8,
        ("romantic", "romantic"): 1.0,
        ("romantic", "bohemian"): 0.6,
    }
    
    # Formality levels
    FORMALITY_LEVELS = {
        (0.0, 0.25): "very casual",
        (0.25, 0.45): "casual",
        (0.45, 0.60): "smart casual",
        (0.60, 0.75): "business casual",
        (0.75, 0.90): "business",
        (0.90, 1.0): "formal"
    }
    
    def __init__(self):
        """Initialize outfit coherence analyzer."""
        logger.info("OutfitCoherenceAnalyzer initialized")
    
    def analyze(self, items: List[OutfitItem]) -> OutfitAnalysis:
        """
        Analyze outfit coherence.
        
        Args:
            items: List of OutfitItem objects
            
        Returns:
            OutfitAnalysis with complete evaluation
        """
        if not items:
            return self._empty_analysis()
        
        # Calculate individual scores
        style_score, style_breakdown, dominant_style = self._analyze_style_consistency(items)
        color_score, palette, harmony_type, temp = self._analyze_color_harmony(items)
        formality_score, formality_level = self._analyze_formality(items)
        
        # Calculate overall coherence
        coherence = (
            style_score * 0.35 +
            color_score * 0.35 +
            formality_score * 0.30
        )
        
        # Determine occasions
        occasions = self._determine_occasions(dominant_style, formality_level)
        
        # Generate issues and suggestions
        issues, suggestions = self._generate_feedback(
            items, style_score, color_score, formality_score
        )
        
        # Get rating
        rating, emoji = self._get_rating(coherence)
        
        return OutfitAnalysis(
            coherence_score=round(coherence, 2),
            style_consistency=round(style_score, 2),
            color_harmony=round(color_score, 2),
            formality_consistency=round(formality_score, 2),
            dominant_style=dominant_style,
            style_breakdown=style_breakdown,
            color_palette=palette,
            color_harmony_type=harmony_type,
            color_temperature=temp,
            best_occasions=occasions,
            formality_level=formality_level,
            issues=issues,
            suggestions=suggestions,
            rating=rating,
            rating_emoji=emoji
        )
    
    def analyze_from_detections(
        self,
        detections: List[Dict]
    ) -> OutfitAnalysis:
        """
        Analyze outfit from detection results.
        
        Args:
            detections: List of detection dicts from API
            
        Returns:
            OutfitAnalysis
        """
        items = []
        
        for det in detections:
            # Infer style from item type
            specific_type = det.get("specificType", det.get("category", "unknown"))
            style = self._infer_style(specific_type, det.get("material"))
            formality = self._infer_formality(specific_type, style)
            
            items.append(OutfitItem(
                category=det.get("category", "unknown"),
                specific_type=specific_type,
                primary_color=det.get("primaryColor", "unknown"),
                color_hex=det.get("colorHex", "#808080"),
                style=style,
                formality=formality,
                material=det.get("material"),
                pattern=det.get("pattern")
            ))
        
        return self.analyze(items)
    
    def _analyze_style_consistency(
        self,
        items: List[OutfitItem]
    ) -> Tuple[float, Dict[str, float], str]:
        """Analyze style consistency across items."""
        if len(items) < 2:
            return (1.0, {items[0].style: 1.0} if items else {}, items[0].style if items else "casual")
        
        # Count styles
        style_counts = {}
        for item in items:
            style_counts[item.style] = style_counts.get(item.style, 0) + 1
        
        total = len(items)
        style_breakdown = {s: c / total for s, c in style_counts.items()}
        
        # Dominant style
        dominant = max(style_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate compatibility score
        compatibility_scores = []
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                style_a = items[i].style
                style_b = items[j].style
                
                # Look up compatibility
                key = (style_a, style_b)
                if key not in self.STYLE_COMPATIBILITY:
                    key = (style_b, style_a)
                
                score = self.STYLE_COMPATIBILITY.get(key, 0.5)
                compatibility_scores.append(score)
        
        avg_compatibility = np.mean(compatibility_scores) if compatibility_scores else 1.0
        
        return avg_compatibility, style_breakdown, dominant
    
    def _analyze_color_harmony(
        self,
        items: List[OutfitItem]
    ) -> Tuple[float, List[str], str, str]:
        """Analyze color harmony."""
        # Collect colors
        colors = [(item.primary_color, item.color_hex) for item in items]
        palette = list(set(c[0] for c in colors))
        
        # Parse hex colors to get hue values
        hues = []
        temperatures = []
        
        for _, hex_color in colors:
            if hex_color and len(hex_color) >= 7:
                try:
                    r = int(hex_color[1:3], 16)
                    g = int(hex_color[3:5], 16)
                    b = int(hex_color[5:7], 16)
                    
                    # Convert to HSV
                    hsv = cv2.cvtColor(
                        np.array([[[b, g, r]]], dtype=np.uint8),
                        cv2.COLOR_BGR2HSV
                    )[0, 0]
                    
                    hues.append(hsv[0])
                    
                    # Temperature
                    if hsv[1] < 30:
                        temperatures.append("neutral")
                    elif hsv[0] < 35 or hsv[0] >= 170:
                        temperatures.append("warm")
                    else:
                        temperatures.append("cool")
                except:
                    pass
        
        # Determine harmony type
        if len(hues) < 2:
            harmony_type = "monochromatic"
            harmony_score = 1.0
        else:
            # Calculate hue differences
            hue_diffs = []
            for i in range(len(hues)):
                for j in range(i + 1, len(hues)):
                    diff = abs(hues[i] - hues[j])
                    if diff > 90:
                        diff = 180 - diff
                    hue_diffs.append(diff)
            
            avg_diff = np.mean(hue_diffs)
            
            if avg_diff < 20:
                harmony_type = "monochromatic"
                harmony_score = 0.95
            elif avg_diff < 45:
                harmony_type = "analogous"
                harmony_score = 0.9
            elif 75 < avg_diff < 105:
                harmony_type = "complementary"
                harmony_score = 0.85
            else:
                harmony_type = "mixed"
                harmony_score = 0.7
        
        # Determine temperature
        if not temperatures:
            temp = "neutral"
        else:
            temp_counts = {}
            for t in temperatures:
                temp_counts[t] = temp_counts.get(t, 0) + 1
            temp = max(temp_counts.items(), key=lambda x: x[1])[0]
        
        # Bonus for neutral colors (they go with everything)
        neutral_count = sum(1 for n, _ in colors if n.lower() in ["black", "white", "gray", "grey", "navy", "beige"])
        if neutral_count > 0:
            harmony_score = min(1.0, harmony_score + 0.05 * neutral_count)
        
        return harmony_score, palette, harmony_type, temp
    
    def _analyze_formality(
        self,
        items: List[OutfitItem]
    ) -> Tuple[float, str]:
        """Analyze formality consistency."""
        formalities = [item.formality for item in items]
        
        if not formalities:
            return (0.5, "casual")
        
        # Average formality
        avg_formality = np.mean(formalities)
        
        # Consistency (lower variance = more consistent)
        variance = np.var(formalities)
        consistency = 1 - min(1, variance * 4)  # Scale: 0.25 variance = 0 consistency
        
        # Determine level
        level = "casual"
        for (low, high), name in self.FORMALITY_LEVELS.items():
            if low <= avg_formality < high:
                level = name
                break
        
        return consistency, level
    
    def _determine_occasions(self, style: str, formality: str) -> List[str]:
        """Determine suitable occasions."""
        STYLE_OCCASIONS = {
            "casual": ["weekend", "brunch", "shopping", "casual lunch"],
            "streetwear": ["city outing", "concerts", "hanging out"],
            "formal": ["gala", "wedding", "formal dinner"],
            "business": ["office", "meeting", "interview"],
            "athletic": ["gym", "sports", "active outdoor"],
            "bohemian": ["festival", "beach", "casual events"],
            "romantic": ["date night", "romantic dinner"],
            "minimalist": ["any occasion"],
        }
        
        occasions = STYLE_OCCASIONS.get(style, ["general occasions"])
        
        # Adjust by formality
        if formality in ["formal", "business"]:
            occasions = [o for o in occasions if o not in ["gym", "beach"]]
            occasions.extend(["upscale venue", "professional event"])
        
        return occasions[:5]
    
    def _generate_feedback(
        self,
        items: List[OutfitItem],
        style_score: float,
        color_score: float,
        formality_score: float
    ) -> Tuple[List[str], List[str]]:
        """Generate issues and suggestions."""
        issues = []
        suggestions = []
        
        # Style issues
        if style_score < 0.6:
            styles = list(set(item.style for item in items))
            issues.append(f"Mixed styles: {', '.join(styles)} don't blend naturally")
            suggestions.append("Try items from the same style family")
        
        # Color issues
        if color_score < 0.7:
            issues.append("Color combination could be improved")
            suggestions.append("Add a neutral piece to balance the colors")
        
        # Formality issues
        if formality_score < 0.6:
            formal_items = [i.specific_type for i in items if i.formality > 0.6]
            casual_items = [i.specific_type for i in items if i.formality < 0.4]
            if formal_items and casual_items:
                issues.append(f"Formality mismatch: {formal_items[0]} with {casual_items[0]}")
                suggestions.append("Match formality levels across all pieces")
        
        # Check for missing essentials
        categories = [item.category.lower() for item in items]
        if "shoes" not in categories and "footwear" not in categories:
            suggestions.append("Complete the look with appropriate footwear")
        
        # Pattern suggestions
        patterns = [item.pattern for item in items if item.pattern and item.pattern != "solid"]
        if len(patterns) > 2:
            issues.append("Too many patterns can look busy")
            suggestions.append("Limit to 1-2 patterns and balance with solids")
        
        return issues, suggestions
    
    def _get_rating(self, coherence: float) -> Tuple[str, str]:
        """Get rating and emoji for coherence score."""
        if coherence >= 0.85:
            return ("excellent", "🌟")
        elif coherence >= 0.70:
            return ("good", "👍")
        elif coherence >= 0.55:
            return ("fair", "👌")
        else:
            return ("needs work", "🔧")
    
    def _infer_style(self, specific_type: str, material: str = None) -> str:
        """Infer style from item type."""
        type_lower = specific_type.lower() if specific_type else ""
        mat_lower = material.lower() if material else ""
        
        # Casual indicators
        if any(x in type_lower for x in ["jeans", "t-shirt", "hoodie", "sneakers", "sweatshirt"]):
            return "casual"
        
        # Formal indicators
        if any(x in type_lower for x in ["suit", "tuxedo", "gown", "dress pants", "oxford"]):
            return "formal"
        
        # Business indicators
        if any(x in type_lower for x in ["blazer", "chinos", "loafers", "button-down"]):
            return "business"
        
        # Athletic indicators
        if any(x in type_lower for x in ["shorts", "tank", "running", "yoga", "athletic"]):
            return "athletic"
        
        # Streetwear indicators
        if any(x in type_lower for x in ["bomber", "cargo", "high-top", "oversized"]):
            return "streetwear"
        
        # Material-based
        if "leather" in mat_lower:
            return "edgy"
        if "silk" in mat_lower or "satin" in mat_lower:
            return "romantic"
        
        return "casual"
    
    def _infer_formality(self, specific_type: str, style: str) -> float:
        """Infer formality from item type and style."""
        type_lower = specific_type.lower() if specific_type else ""
        
        FORMALITY_MAP = {
            "suit": 0.9, "tuxedo": 0.95, "gown": 0.9, "dress pants": 0.75,
            "blazer": 0.7, "oxford": 0.7, "loafers": 0.65,
            "chinos": 0.5, "dress": 0.6, "polo": 0.5,
            "jeans": 0.35, "sneakers": 0.3, "t-shirt": 0.25,
            "hoodie": 0.2, "shorts": 0.2, "tank": 0.15,
        }
        
        for item, formality in FORMALITY_MAP.items():
            if item in type_lower:
                return formality
        
        # Style-based fallback
        STYLE_FORMALITY = {
            "formal": 0.85, "business": 0.7, "preppy": 0.55,
            "minimalist": 0.5, "casual": 0.35, "streetwear": 0.3,
            "athletic": 0.2, "bohemian": 0.3
        }
        
        return STYLE_FORMALITY.get(style, 0.4)
    
    def _empty_analysis(self) -> OutfitAnalysis:
        """Return empty analysis."""
        return OutfitAnalysis(
            coherence_score=0.0,
            style_consistency=0.0,
            color_harmony=0.0,
            formality_consistency=0.0,
            dominant_style="unknown",
            style_breakdown={},
            color_palette=[],
            color_harmony_type="none",
            color_temperature="neutral",
            best_occasions=[],
            formality_level="unknown",
            issues=["No items to analyze"],
            suggestions=["Add clothing items to analyze"],
            rating="incomplete",
            rating_emoji="❓"
        )


# === SINGLETON INSTANCE ===
_outfit_analyzer_instance = None


def get_outfit_analyzer() -> OutfitCoherenceAnalyzer:
    """Get singleton instance."""
    global _outfit_analyzer_instance
    if _outfit_analyzer_instance is None:
        _outfit_analyzer_instance = OutfitCoherenceAnalyzer()
    return _outfit_analyzer_instance


def analyze_outfit(items: List[Dict]) -> Dict:
    """Quick utility for outfit analysis from detection dicts."""
    analyzer = get_outfit_analyzer()
    result = analyzer.analyze_from_detections(items)
    return result.to_dict()
