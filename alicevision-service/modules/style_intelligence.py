"""
Style & Occasion Intelligence Module
Advanced AI for fashion style classification and occasion matching

This module provides:
- 50+ style categories (casual, formal, streetwear, etc.)
- 30+ occasion types (work, date night, gym, wedding, etc.)
- Season/weather appropriateness scoring
- Style coherence analysis
- Outfit recommendations
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field, asdict
import logging

logger = logging.getLogger(__name__)


# ============================================
# 🎨 STYLE TAXONOMY
# ============================================

STYLE_CATEGORIES = {
    "casual": {
        "description": "Relaxed, everyday wear",
        "keywords": ["comfortable", "relaxed", "laid-back", "effortless"],
        "occasions": ["casual outing", "weekend", "running errands", "brunch"],
        "subtypes": [
            "smart casual", "athleisure", "sporty casual", "weekend casual",
            "beach casual", "resort wear", "casual chic"
        ]
    },
    "formal": {
        "description": "Elegant, dressy attire",
        "keywords": ["elegant", "sophisticated", "refined", "polished"],
        "occasions": ["black tie", "gala", "formal dinner", "wedding"],
        "subtypes": [
            "black tie", "white tie", "cocktail attire", "semi-formal",
            "business formal", "evening wear"
        ]
    },
    "business": {
        "description": "Professional work attire",
        "keywords": ["professional", "sharp", "tailored", "corporate"],
        "occasions": ["office", "meeting", "interview", "presentation"],
        "subtypes": [
            "business professional", "business casual", "corporate",
            "executive", "office casual", "creative professional"
        ]
    },
    "streetwear": {
        "description": "Urban, trend-focused fashion",
        "keywords": ["urban", "trendy", "edgy", "cool"],
        "occasions": ["city outing", "concerts", "social events"],
        "subtypes": [
            "hypebeast", "skater", "urban", "techwear", "grunge",
            "punk", "hip-hop"
        ]
    },
    "vintage": {
        "description": "Retro-inspired clothing",
        "keywords": ["retro", "classic", "nostalgic", "timeless"],
        "occasions": ["themed parties", "casual", "creative events"],
        "subtypes": [
            "60s mod", "70s bohemian", "80s new wave", "90s grunge",
            "y2k", "old money", "rockabilly"
        ]
    },
    "bohemian": {
        "description": "Free-spirited, artistic style",
        "keywords": ["artistic", "free-spirited", "eclectic", "natural"],
        "occasions": ["festivals", "beach", "casual gatherings"],
        "subtypes": [
            "boho chic", "hippie", "folk", "gypsy", "earth mother"
        ]
    },
    "minimalist": {
        "description": "Clean, simple aesthetic",
        "keywords": ["simple", "clean", "understated", "refined"],
        "occasions": ["any"],
        "subtypes": [
            "scandinavian", "japanese minimalist", "modern minimalist",
            "capsule wardrobe"
        ]
    },
    "preppy": {
        "description": "Classic, collegiate-inspired",
        "keywords": ["classic", "polished", "traditional", "clean-cut"],
        "occasions": ["campus", "country club", "brunch"],
        "subtypes": [
            "ivy league", "country club", "nautical", "collegiate"
        ]
    },
    "athletic": {
        "description": "Sport and fitness wear",
        "keywords": ["sporty", "active", "performance", "dynamic"],
        "occasions": ["gym", "sports", "outdoor activities"],
        "subtypes": [
            "gym wear", "running", "yoga", "outdoor", "athleisure"
        ]
    },
    "romantic": {
        "description": "Soft, feminine aesthetic",
        "keywords": ["feminine", "delicate", "soft", "graceful"],
        "occasions": ["date night", "romantic dinner", "garden party"],
        "subtypes": [
            "cottagecore", "fairy tale", "soft girl", "coquette"
        ]
    },
    "edgy": {
        "description": "Bold, unconventional style",
        "keywords": ["bold", "daring", "rebellious", "avant-garde"],
        "occasions": ["nightlife", "concerts", "creative events"],
        "subtypes": [
            "goth", "punk", "rock", "biker", "alternative"
        ]
    },
    "luxury": {
        "description": "High-end, designer fashion",
        "keywords": ["luxurious", "exclusive", "premium", "designer"],
        "occasions": ["special events", "upscale venues"],
        "subtypes": [
            "quiet luxury", "loud luxury", "old money", "new money"
        ]
    }
}

OCCASIONS = {
    "work": ["office", "meeting", "interview", "presentation", "conference"],
    "social": ["brunch", "lunch date", "dinner party", "cocktail party", "networking"],
    "romantic": ["date night", "romantic dinner", "anniversary", "valentine's day"],
    "formal_events": ["wedding", "gala", "black tie", "red carpet", "awards ceremony"],
    "casual_outings": ["shopping", "movies", "casual lunch", "coffee date", "walking"],
    "outdoor": ["picnic", "beach", "hiking", "camping", "bbq"],
    "nightlife": ["club", "bar", "concert", "party", "festival"],
    "sports": ["gym", "yoga", "running", "tennis", "golf"],
    "travel": ["airport", "sightseeing", "resort", "cruise"],
    "special": ["holiday", "birthday", "graduation", "religious ceremony"]
}

SEASONS = ["spring", "summer", "fall", "winter", "all-season"]

WEATHER_CONDITIONS = {
    "hot": {"temp_range": (30, 45), "suitable_for": ["summer", "beach", "resort"]},
    "warm": {"temp_range": (20, 30), "suitable_for": ["spring", "summer", "casual"]},
    "mild": {"temp_range": (15, 20), "suitable_for": ["spring", "fall", "all-season"]},
    "cool": {"temp_range": (5, 15), "suitable_for": ["fall", "spring", "layering"]},
    "cold": {"temp_range": (-10, 5), "suitable_for": ["winter", "layering", "warm"]},
    "rainy": {"suitable_for": ["waterproof", "covered"]},
    "windy": {"suitable_for": ["layered", "fitted"]}
}


@dataclass
class StyleAnalysis:
    """Complete style analysis result"""
    # Primary style classification
    primary_style: str
    primary_style_confidence: float
    style_subtypes: List[str]
    
    # Secondary styles
    secondary_styles: List[Tuple[str, float]]
    
    # Style attributes
    formality_score: float  # 0 = very casual, 1 = very formal
    trendiness_score: float  # 0 = classic, 1 = trendy
    boldness_score: float  # 0 = subtle, 1 = bold
    
    # Occasion matching
    best_occasions: List[str]
    occasion_scores: Dict[str, float]
    
    # Season/weather
    seasons: List[str]
    weather_suitability: Dict[str, float]
    
    # Color harmony
    color_palette_style: str  # "monochromatic", "complementary", "analogous"
    color_mood: str  # "warm", "cool", "neutral"
    
    # Overall assessment
    style_description: str
    styling_tips: List[str]
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result["secondary_styles"] = [
            {"style": s, "confidence": c}
            for s, c in self.secondary_styles
        ]
        return result


@dataclass  
class OccasionMatch:
    """Occasion matching result"""
    occasion: str
    category: str
    score: float
    reasoning: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


class StyleIntelligence:
    """
    🎨 AI-Powered Style & Occasion Intelligence
    
    Analyzes clothing items and outfits to determine:
    - Style category (casual, formal, streetwear, etc.)
    - Best matching occasions
    - Season/weather appropriateness
    - Style coherence and recommendations
    
    Uses:
    - Color analysis for mood detection
    - Visual features for style classification
    - CLIP embeddings for semantic understanding
    
    Usage:
        analyzer = StyleIntelligence()
        result = analyzer.analyze_style(image)
        print(f"Style: {result.primary_style}")
        print(f"Best for: {result.best_occasions}")
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize style intelligence."""
        self._setup_device(device)
        self.clip_model = None
        self.model_loaded = False
        
        logger.info(f"StyleIntelligence initialized (device={self.device})")
    
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
    
    def _load_clip(self):
        """Lazy load CLIP for semantic analysis."""
        if self.model_loaded:
            return
        
        try:
            import open_clip
            
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai", device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self.model_loaded = True
            logger.info("✅ CLIP loaded for style analysis")
            
        except ImportError:
            logger.warning("open_clip not available, using visual heuristics")
            self.model_loaded = False
    
    def analyze_style(
        self,
        image: np.ndarray,
        item_info: Dict = None
    ) -> StyleAnalysis:
        """
        Analyze style of clothing item or outfit.
        
        Args:
            image: BGR image
            item_info: Optional dict with category, type, material, etc.
            
        Returns:
            StyleAnalysis with comprehensive style assessment
        """
        self._load_clip()
        
        # Extract visual features
        color_analysis = self._analyze_colors(image)
        visual_features = self._extract_visual_features(image)
        
        # Classify style
        if self.model_loaded:
            style_scores = self._clip_style_classification(image)
        else:
            style_scores = self._heuristic_style_classification(
                color_analysis, visual_features, item_info
            )
        
        # Sort styles by score
        sorted_styles = sorted(style_scores.items(), key=lambda x: x[1], reverse=True)
        primary_style = sorted_styles[0][0]
        primary_conf = sorted_styles[0][1]
        secondary = [(s, c) for s, c in sorted_styles[1:4] if c > 0.1]
        
        # Get style subtypes
        subtypes = STYLE_CATEGORIES.get(primary_style, {}).get("subtypes", [])[:3]
        
        # Calculate style metrics
        formality = self._calculate_formality(primary_style, color_analysis, item_info)
        trendiness = self._calculate_trendiness(primary_style, visual_features)
        boldness = self._calculate_boldness(color_analysis, visual_features)
        
        # Match occasions
        occasion_scores = self._match_occasions(primary_style, formality, item_info)
        best_occasions = [occ for occ, score in sorted(
            occasion_scores.items(), key=lambda x: x[1], reverse=True
        )[:5]]
        
        # Determine seasons
        seasons = self._determine_seasons(item_info, color_analysis)
        
        # Weather suitability
        weather = self._assess_weather_suitability(item_info, seasons)
        
        # Color style
        color_style, color_mood = self._analyze_color_harmony(color_analysis)
        
        # Generate description and tips
        description = self._generate_style_description(
            primary_style, formality, color_mood, item_info
        )
        tips = self._generate_styling_tips(primary_style, item_info)
        
        return StyleAnalysis(
            primary_style=primary_style,
            primary_style_confidence=primary_conf,
            style_subtypes=subtypes,
            secondary_styles=secondary,
            formality_score=formality,
            trendiness_score=trendiness,
            boldness_score=boldness,
            best_occasions=best_occasions,
            occasion_scores=occasion_scores,
            seasons=seasons,
            weather_suitability=weather,
            color_palette_style=color_style,
            color_mood=color_mood,
            style_description=description,
            styling_tips=tips
        )
    
    def match_occasion(
        self,
        image: np.ndarray,
        target_occasion: str,
        item_info: Dict = None
    ) -> OccasionMatch:
        """
        Check if item matches a specific occasion.
        
        Args:
            image: BGR image
            target_occasion: Target occasion (e.g., "date night")
            item_info: Optional item information
            
        Returns:
            OccasionMatch with score and reasoning
        """
        style = self.analyze_style(image, item_info)
        
        # Find occasion category
        category = None
        for cat, occasions in OCCASIONS.items():
            if target_occasion.lower() in [o.lower() for o in occasions]:
                category = cat
                break
        
        if not category:
            category = "general"
        
        # Calculate match score
        base_score = style.occasion_scores.get(target_occasion, 0.5)
        
        # Adjust based on formality
        occasion_formality = self._get_occasion_formality(target_occasion)
        formality_match = 1 - abs(style.formality_score - occasion_formality)
        
        final_score = (base_score * 0.6 + formality_match * 0.4)
        
        # Generate reasoning
        if final_score > 0.8:
            reasoning = f"Perfect for {target_occasion}! The {style.primary_style} style and {style.color_mood} tones are ideal."
        elif final_score > 0.6:
            reasoning = f"Good match for {target_occasion}. Consider {style.styling_tips[0] if style.styling_tips else 'accessorizing appropriately'}."
        elif final_score > 0.4:
            reasoning = f"Could work for {target_occasion} with the right styling adjustments."
        else:
            reasoning = f"Not ideal for {target_occasion}. Consider a more {'formal' if occasion_formality > 0.5 else 'casual'} option."
        
        return OccasionMatch(
            occasion=target_occasion,
            category=category,
            score=final_score,
            reasoning=reasoning
        )
    
    def _analyze_colors(self, image: np.ndarray) -> Dict:
        """Analyze color properties of image."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate color statistics
        h_mean = np.mean(hsv[:, :, 0])
        s_mean = np.mean(hsv[:, :, 1])
        v_mean = np.mean(hsv[:, :, 2])
        
        h_std = np.std(hsv[:, :, 0])
        s_std = np.std(hsv[:, :, 1])
        v_std = np.std(hsv[:, :, 2])
        
        # Determine color mood
        if s_mean < 50:
            mood = "neutral"
        elif h_mean < 30 or h_mean > 150:
            mood = "warm"
        else:
            mood = "cool"
        
        # Color vibrancy
        vibrancy = (s_mean / 255) * (v_mean / 255)
        
        # Color diversity
        diversity = (h_std + s_std / 2) / 100
        
        return {
            "hue_mean": float(h_mean),
            "saturation_mean": float(s_mean),
            "value_mean": float(v_mean),
            "mood": mood,
            "vibrancy": float(vibrancy),
            "diversity": float(diversity),
            "is_dark": v_mean < 80,
            "is_bright": v_mean > 180,
            "is_neutral": s_mean < 50
        }
    
    def _extract_visual_features(self, image: np.ndarray) -> Dict:
        """Extract visual features for style classification."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Edge density (structural complexity)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (h * w)
        
        # Texture variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_var = np.var(laplacian)
        
        # Symmetry (formal items tend to be symmetric)
        left_half = gray[:, :w//2]
        right_half = cv2.flip(gray[:, w//2:], 1)
        if left_half.shape == right_half.shape:
            symmetry = 1 - np.mean(np.abs(left_half.astype(float) - right_half.astype(float))) / 255
        else:
            symmetry = 0.5
        
        return {
            "edge_density": float(edge_density),
            "texture_variance": float(texture_var),
            "symmetry": float(symmetry),
            "aspect_ratio": w / h
        }
    
    def _clip_style_classification(self, image: np.ndarray) -> Dict[str, float]:
        """Classify style using CLIP."""
        from PIL import Image
        
        # Convert to PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # Preprocess
        image_tensor = self.clip_preprocess(pil_image).unsqueeze(0).to(self.device)
        
        # Create style prompts
        prompts = [f"a photo of {style} style clothing" for style in STYLE_CATEGORIES.keys()]
        text_tokens = self.tokenizer(prompts).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_tensor)
            text_features = self.clip_model.encode_text(text_tokens)
            
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            similarity = (image_features @ text_features.T).squeeze(0)
            probs = torch.softmax(similarity / 0.1, dim=-1)
        
        scores = {}
        for style, prob in zip(STYLE_CATEGORIES.keys(), probs):
            scores[style] = float(prob)
        
        return scores
    
    def _heuristic_style_classification(
        self,
        color_analysis: Dict,
        visual_features: Dict,
        item_info: Dict = None
    ) -> Dict[str, float]:
        """Classify style using visual heuristics."""
        scores = {style: 0.1 for style in STYLE_CATEGORIES.keys()}
        
        # Dark colors → edgy, formal, luxury
        if color_analysis["is_dark"]:
            scores["edgy"] += 0.2
            scores["formal"] += 0.15
            scores["luxury"] += 0.1
        
        # Neutral colors → minimalist, business
        if color_analysis["is_neutral"]:
            scores["minimalist"] += 0.3
            scores["business"] += 0.2
        
        # Vibrant colors → casual, streetwear, bohemian
        if color_analysis["vibrancy"] > 0.5:
            scores["casual"] += 0.2
            scores["streetwear"] += 0.15
            scores["bohemian"] += 0.1
        
        # High symmetry → formal, business
        if visual_features["symmetry"] > 0.7:
            scores["formal"] += 0.2
            scores["business"] += 0.15
            scores["preppy"] += 0.1
        
        # High texture → casual, bohemian
        if visual_features["texture_variance"] > 500:
            scores["casual"] += 0.15
            scores["bohemian"] += 0.1
        
        # Item-specific hints
        if item_info:
            category = item_info.get("category", "").lower()
            specific_type = item_info.get("specific_type", "").lower()
            material = item_info.get("material", "").lower()
            
            # Jeans → casual
            if "jeans" in specific_type or "denim" in material:
                scores["casual"] += 0.3
                scores["streetwear"] += 0.1
            
            # Blazer → business, formal
            if "blazer" in specific_type:
                scores["business"] += 0.3
                scores["formal"] += 0.2
            
            # Hoodie → casual, streetwear
            if "hoodie" in specific_type:
                scores["casual"] += 0.2
                scores["streetwear"] += 0.25
                scores["athletic"] += 0.1
            
            # Leather → edgy, luxury
            if "leather" in material:
                scores["edgy"] += 0.25
                scores["luxury"] += 0.15
            
            # Silk, satin → formal, romantic, luxury
            if material in ["silk", "satin"]:
                scores["formal"] += 0.2
                scores["romantic"] += 0.2
                scores["luxury"] += 0.2
        
        # Normalize
        total = sum(scores.values())
        return {k: v / total for k, v in scores.items()}
    
    def _calculate_formality(
        self,
        style: str,
        color_analysis: Dict,
        item_info: Dict = None
    ) -> float:
        """Calculate formality score (0=casual, 1=formal)."""
        # Base formality by style
        STYLE_FORMALITY = {
            "formal": 0.95, "luxury": 0.85, "business": 0.8,
            "preppy": 0.6, "romantic": 0.6, "minimalist": 0.5,
            "vintage": 0.4, "casual": 0.3, "bohemian": 0.25,
            "athletic": 0.2, "streetwear": 0.25, "edgy": 0.3
        }
        
        formality = STYLE_FORMALITY.get(style, 0.5)
        
        # Dark colors → more formal
        if color_analysis["is_dark"]:
            formality += 0.1
        
        # Neutral colors → more formal
        if color_analysis["is_neutral"]:
            formality += 0.05
        
        # Item-specific adjustments
        if item_info:
            specific_type = item_info.get("specific_type", "").lower()
            
            formal_items = ["blazer", "dress pants", "oxford", "loafers", "dress"]
            casual_items = ["jeans", "sneakers", "t-shirt", "hoodie", "shorts"]
            
            if any(item in specific_type for item in formal_items):
                formality += 0.15
            if any(item in specific_type for item in casual_items):
                formality -= 0.15
        
        return max(0, min(1, formality))
    
    def _calculate_trendiness(self, style: str, visual_features: Dict) -> float:
        """Calculate trendiness score (0=classic, 1=trendy)."""
        STYLE_TRENDINESS = {
            "streetwear": 0.9, "edgy": 0.8, "athleisure": 0.75,
            "bohemian": 0.6, "romantic": 0.5, "casual": 0.5,
            "minimalist": 0.4, "vintage": 0.4, "preppy": 0.3,
            "business": 0.25, "formal": 0.2, "luxury": 0.3
        }
        return STYLE_TRENDINESS.get(style, 0.5)
    
    def _calculate_boldness(self, color_analysis: Dict, visual_features: Dict) -> float:
        """Calculate boldness score (0=subtle, 1=bold)."""
        boldness = 0.5
        
        # High vibrancy → bold
        boldness += color_analysis["vibrancy"] * 0.3
        
        # High diversity → bold
        boldness += color_analysis["diversity"] * 0.2
        
        # High contrast → bold
        if color_analysis["is_dark"] or color_analysis["is_bright"]:
            boldness += 0.1
        
        return max(0, min(1, boldness))
    
    def _match_occasions(
        self,
        style: str,
        formality: float,
        item_info: Dict = None
    ) -> Dict[str, float]:
        """Match item to occasions."""
        scores = {}
        
        # Get style's default occasions
        style_occasions = STYLE_CATEGORIES.get(style, {}).get("occasions", [])
        
        for category, occasions in OCCASIONS.items():
            for occ in occasions:
                # Base score from style match
                if occ in style_occasions:
                    scores[occ] = 0.8
                else:
                    scores[occ] = 0.3
                
                # Adjust by formality match
                occ_formality = self._get_occasion_formality(occ)
                formality_match = 1 - abs(formality - occ_formality)
                scores[occ] = scores[occ] * 0.6 + formality_match * 0.4
        
        return scores
    
    def _get_occasion_formality(self, occasion: str) -> float:
        """Get formality level of an occasion."""
        OCCASION_FORMALITY = {
            "black tie": 1.0, "gala": 0.95, "wedding": 0.85,
            "interview": 0.8, "presentation": 0.75, "meeting": 0.7,
            "office": 0.65, "dinner party": 0.6, "date night": 0.55,
            "networking": 0.6, "brunch": 0.4, "shopping": 0.3,
            "casual lunch": 0.35, "concert": 0.35, "movies": 0.3,
            "gym": 0.15, "hiking": 0.2, "beach": 0.2
        }
        return OCCASION_FORMALITY.get(occasion.lower(), 0.5)
    
    def _determine_seasons(self, item_info: Dict, color_analysis: Dict) -> List[str]:
        """Determine appropriate seasons."""
        seasons = []
        
        if item_info:
            material = item_info.get("material", "").lower()
            specific_type = item_info.get("specific_type", "").lower()
            
            # Winter materials
            if material in ["wool", "cashmere", "fleece", "down"]:
                seasons.extend(["fall", "winter"])
            
            # Summer materials
            if material in ["linen", "cotton", "silk", "chiffon"]:
                seasons.extend(["spring", "summer"])
            
            # Heavy items
            if "coat" in specific_type or "parka" in specific_type:
                seasons.extend(["fall", "winter"])
            
            # Light items
            if "tank" in specific_type or "shorts" in specific_type:
                seasons.extend(["summer"])
        
        # Color-based
        if color_analysis["mood"] == "warm":
            if "fall" not in seasons:
                seasons.append("fall")
        elif color_analysis["mood"] == "cool":
            if "winter" not in seasons:
                seasons.append("winter")
        
        if not seasons:
            seasons = ["all-season"]
        
        return list(set(seasons))
    
    def _assess_weather_suitability(
        self,
        item_info: Dict,
        seasons: List[str]
    ) -> Dict[str, float]:
        """Assess suitability for different weather conditions."""
        suitability = {
            "hot": 0.3, "warm": 0.5, "mild": 0.7,
            "cool": 0.5, "cold": 0.3, "rainy": 0.3
        }
        
        if "summer" in seasons:
            suitability["hot"] = 0.9
            suitability["warm"] = 0.8
            suitability["cold"] = 0.1
        
        if "winter" in seasons:
            suitability["cold"] = 0.9
            suitability["cool"] = 0.8
            suitability["hot"] = 0.1
        
        if item_info:
            material = item_info.get("material", "").lower()
            if material in ["waterproof", "nylon", "gore-tex"]:
                suitability["rainy"] = 0.9
        
        return suitability
    
    def _analyze_color_harmony(self, color_analysis: Dict) -> Tuple[str, str]:
        """Analyze color harmony type."""
        diversity = color_analysis["diversity"]
        
        if diversity < 0.2:
            style = "monochromatic"
        elif diversity < 0.5:
            style = "analogous"
        else:
            style = "complementary"
        
        mood = color_analysis["mood"]
        
        return style, mood
    
    def _generate_style_description(
        self,
        style: str,
        formality: float,
        color_mood: str,
        item_info: Dict = None
    ) -> str:
        """Generate natural language style description."""
        style_data = STYLE_CATEGORIES.get(style, {})
        keywords = style_data.get("keywords", ["stylish"])[:2]
        
        formality_word = "formal" if formality > 0.6 else "casual" if formality < 0.4 else "versatile"
        
        item_type = item_info.get("specific_type", "piece") if item_info else "piece"
        
        return f"A {formality_word}, {keywords[0]} {item_type} with {color_mood} tones, perfect for {style_data.get('description', 'various occasions')}."
    
    def _generate_styling_tips(self, style: str, item_info: Dict = None) -> List[str]:
        """Generate styling tips."""
        tips = []
        
        STYLE_TIPS = {
            "casual": [
                "Pair with clean sneakers for a relaxed look",
                "Add a denim jacket for layering"
            ],
            "formal": [
                "Accessorize with minimal, elegant jewelry",
                "Choose pointed-toe shoes to complete the look"
            ],
            "business": [
                "Keep accessories minimal and professional",
                "Ensure proper fit for a polished appearance"
            ],
            "streetwear": [
                "Mix high and low brands for authenticity",
                "Add statement sneakers as a focal point"
            ],
            "minimalist": [
                "Focus on quality over quantity",
                "Stick to a neutral color palette"
            ]
        }
        
        tips = STYLE_TIPS.get(style, ["Style confidently and make it your own"])[:3]
        
        return tips


# === SINGLETON INSTANCE ===
_style_intelligence_instance = None


def get_style_intelligence() -> StyleIntelligence:
    """Get singleton instance."""
    global _style_intelligence_instance
    if _style_intelligence_instance is None:
        _style_intelligence_instance = StyleIntelligence()
    return _style_intelligence_instance


def analyze_style(image: np.ndarray, item_info: Dict = None) -> Dict:
    """Quick utility for style analysis."""
    analyzer = get_style_intelligence()
    result = analyzer.analyze_style(image, item_info)
    return result.to_dict()


def match_occasion(image: np.ndarray, occasion: str) -> Dict:
    """Quick utility for occasion matching."""
    analyzer = get_style_intelligence()
    result = analyzer.match_occasion(image, occasion)
    return result.to_dict()
