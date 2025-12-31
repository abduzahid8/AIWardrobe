"""
🌿 TextileNet Module - Neural Textile Fiber Recognition
Implements fiber composition detection for sustainability calculations

Key Features:
1. Visual texture analysis for fiber identification
2. Multimodal fusion (visual + text description)
3. Blend ratio estimation
4. Integration with LCA carbon footprint calculations
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import logging
import base64
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)


# ============================================
# 🧵 FIBER TAXONOMY
# ============================================

# Comprehensive fiber database with properties
FIBER_DATABASE = {
    # Natural fibers - Plant
    "cotton": {
        "category": "natural_plant",
        "co2e_per_kg": 10.5,  # Average conventional cotton
        "water_intensity": "high",
        "biodegradable": True,
        "visual_cues": ["matte finish", "soft texture", "absorbent look"],
        "common_in": ["t-shirts", "jeans", "underwear", "towels"]
    },
    "organic_cotton": {
        "category": "natural_plant",
        "co2e_per_kg": 6.4,
        "water_intensity": "medium",
        "biodegradable": True,
        "visual_cues": ["matte finish", "natural color variations"],
        "common_in": ["eco-friendly clothing"]
    },
    "linen": {
        "category": "natural_plant",
        "co2e_per_kg": 7.2,
        "water_intensity": "low",
        "biodegradable": True,
        "visual_cues": ["visible texture", "natural wrinkles", "matte"],
        "common_in": ["summer shirts", "dresses", "home textiles"]
    },
    "hemp": {
        "category": "natural_plant",
        "co2e_per_kg": 4.5,
        "water_intensity": "very_low",
        "biodegradable": True,
        "visual_cues": ["coarse texture", "natural beige color"],
        "common_in": ["casual wear", "accessories"]
    },
    
    # Natural fibers - Animal
    "wool": {
        "category": "natural_animal",
        "co2e_per_kg": 31.9,  # High due to methane
        "water_intensity": "medium",
        "biodegradable": True,
        "visual_cues": ["fuzzy texture", "warm appearance", "natural crimp"],
        "common_in": ["sweaters", "suits", "coats"]
    },
    "cashmere": {
        "category": "natural_animal",
        "co2e_per_kg": 45.0,
        "water_intensity": "medium",
        "biodegradable": True,
        "visual_cues": ["very soft", "fine fibers", "luxurious sheen"],
        "common_in": ["luxury sweaters", "scarves"]
    },
    "silk": {
        "category": "natural_animal",
        "co2e_per_kg": 87.5,  # Very energy intensive
        "water_intensity": "high",
        "biodegradable": True,
        "visual_cues": ["high sheen", "smooth", "drapes beautifully"],
        "common_in": ["dresses", "blouses", "ties", "lingerie"]
    },
    "leather": {
        "category": "natural_animal",
        "co2e_per_kg": 65.0,
        "water_intensity": "very_high",
        "biodegradable": True,
        "visual_cues": ["grain texture", "aged patina", "natural variations"],
        "common_in": ["jackets", "bags", "shoes", "belts"]
    },
    
    # Synthetic fibers
    "polyester": {
        "category": "synthetic",
        "co2e_per_kg": 12.5,  # Virgin polyester
        "water_intensity": "low",
        "biodegradable": False,
        "visual_cues": ["slight sheen", "smooth", "wrinkle-resistant"],
        "common_in": ["sportswear", "dresses", "blends"]
    },
    "recycled_polyester": {
        "category": "synthetic_recycled",
        "co2e_per_kg": 3.8,
        "water_intensity": "very_low",
        "biodegradable": False,
        "visual_cues": ["similar to virgin polyester"],
        "common_in": ["sustainable sportswear", "fleece"]
    },
    "nylon": {
        "category": "synthetic",
        "co2e_per_kg": 8.2,
        "water_intensity": "low",
        "biodegradable": False,
        "visual_cues": ["shiny", "smooth", "strong"],
        "common_in": ["activewear", "hosiery", "outerwear"]
    },
    "elastane": {
        "category": "synthetic",
        "co2e_per_kg": 15.0,
        "water_intensity": "medium",
        "biodegradable": False,
        "visual_cues": ["stretchy appearance"],
        "common_in": ["stretch jeans", "sportswear", "underwear"]
    },
    "acrylic": {
        "category": "synthetic",
        "co2e_per_kg": 11.5,
        "water_intensity": "low",
        "biodegradable": False,
        "visual_cues": ["wool-like", "lightweight", "fluffy"],
        "common_in": ["sweaters", "blankets"]
    },
    
    # Semi-synthetic / Regenerated
    "viscose": {
        "category": "regenerated",
        "co2e_per_kg": 20.1,  # Chemical processing
        "water_intensity": "high",
        "biodegradable": True,
        "visual_cues": ["silky drape", "soft", "breathable look"],
        "common_in": ["dresses", "blouses", "linings"]
    },
    "modal": {
        "category": "regenerated",
        "co2e_per_kg": 12.0,
        "water_intensity": "medium",
        "biodegradable": True,
        "visual_cues": ["very soft", "drapes well"],
        "common_in": ["underwear", "t-shirts"]
    },
    "tencel": {
        "category": "regenerated",
        "co2e_per_kg": 8.5,
        "water_intensity": "low",
        "biodegradable": True,
        "visual_cues": ["silky", "smooth", "breathable"],
        "common_in": ["eco-friendly fashion", "activewear"]
    },
    
    # Denim-specific
    "denim": {
        "category": "cotton_blend",
        "co2e_per_kg": 14.0,  # Cotton + processing
        "water_intensity": "very_high",
        "biodegradable": True,
        "visual_cues": ["twill weave", "indigo color", "visible texture"],
        "common_in": ["jeans", "jackets"]
    }
}

# All fiber names for classification
ALL_FIBERS = list(FIBER_DATABASE.keys())


# ============================================
# 🎯 DATA STRUCTURES
# ============================================

@dataclass
class FiberAnalysis:
    """Result of fiber composition analysis"""
    primary_fiber: str
    primary_confidence: float
    composition: Dict[str, float]  # fiber_name -> percentage
    category: str  # natural, synthetic, regenerated
    visual_indicators: List[str]
    text_indicators: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "primaryFiber": self.primary_fiber,
            "primaryConfidence": round(self.primary_confidence, 4),
            "composition": {k: round(v, 2) for k, v in self.composition.items()},
            "category": self.category,
            "visualIndicators": self.visual_indicators,
            "textIndicators": self.text_indicators
        }


@dataclass
class LCAResult:
    """Life Cycle Assessment result"""
    total_co2e_kg: float
    per_kg_co2e: float
    breakdown: Dict[str, float]  # fiber -> contribution
    water_impact: str  # low, medium, high, very_high
    biodegradability: float  # 0-1 percentage that's biodegradable
    sustainability_tips: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "totalCO2eKg": round(self.total_co2e_kg, 3),
            "perKgCO2e": round(self.per_kg_co2e, 2),
            "breakdown": {k: round(v, 3) for k, v in self.breakdown.items()},
            "waterImpact": self.water_impact,
            "biodegradabilityPercent": round(self.biodegradability * 100, 1),
            "sustainabilityTips": self.sustainability_tips
        }


@dataclass
class EcoScore:
    """Consumer-friendly eco-score"""
    grade: str  # A, B, C, D, E, F
    score: int  # 0-100
    explanation: str
    comparison: str  # vs category average
    improvements: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "grade": self.grade,
            "score": self.score,
            "explanation": self.explanation,
            "comparison": self.comparison,
            "improvements": self.improvements
        }


# ============================================
# 🧬 TEXTILE NET CLASSIFIER
# ============================================

class TextileNetClassifier:
    """
    TextileNet: Neural textile fiber recognition.
    
    Uses visual texture analysis and optional text fusion to identify
    fiber composition for sustainability calculations.
    """
    
    def __init__(self, device: str = "auto"):
        """Initialize TextileNet classifier."""
        self.device = self._setup_device(device)
        
        # Lazy-loaded models
        self._clip_model = None
        self._fiber_embeddings = None
        
        logger.info("TextileNetClassifier initialized")
    
    def _setup_device(self, device: str) -> str:
        """Setup compute device."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def _load_clip_for_fibers(self):
        """Load CLIP model and precompute fiber embeddings."""
        if self._clip_model is not None:
            return
        
        try:
            import torch
            import open_clip
            
            logger.info("Loading CLIP for fiber classification...")
            
            self._clip_model, _, self._preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self._clip_model = self._clip_model.to(self.device)
            self._clip_model.eval()
            
            # Precompute fiber embeddings
            fiber_prompts = [
                f"clothing made of {fiber}, {' '.join(FIBER_DATABASE[fiber]['visual_cues'])}"
                for fiber in ALL_FIBERS
            ]
            
            with torch.no_grad():
                tokens = self._tokenizer(fiber_prompts)
                tokens = tokens.to(self.device)
                self._fiber_embeddings = self._clip_model.encode_text(tokens)
                self._fiber_embeddings = self._fiber_embeddings / self._fiber_embeddings.norm(dim=-1, keepdim=True)
            
            logger.info("CLIP fiber embeddings ready")
            
        except Exception as e:
            logger.warning(f"CLIP not available for fibers: {e}")
    
    # ============================================
    # 🔍 CLASSIFICATION METHODS
    # ============================================
    
    def classify_fiber(
        self,
        image: np.ndarray,
        text_description: Optional[str] = None,
        category_hint: Optional[str] = None
    ) -> FiberAnalysis:
        """
        Classify fiber composition using multimodal analysis.
        
        Args:
            image: BGR image of clothing item
            text_description: Optional product description text
            category_hint: Optional category (e.g., "jeans", "sweater")
            
        Returns:
            FiberAnalysis with composition estimates
        """
        visual_scores = self._analyze_visual_texture(image)
        text_scores = self._analyze_text_description(text_description) if text_description else {}
        category_priors = self._get_category_priors(category_hint) if category_hint else {}
        
        # Combine scores with weights
        combined_scores = {}
        for fiber in ALL_FIBERS:
            visual = visual_scores.get(fiber, 0.0)
            text = text_scores.get(fiber, 0.0)
            prior = category_priors.get(fiber, 0.0)
            
            # Weight: visual 0.5, text 0.3, category_prior 0.2
            combined_scores[fiber] = visual * 0.5 + text * 0.3 + prior * 0.2
        
        # Normalize to probabilities
        total = sum(combined_scores.values()) + 1e-10
        probs = {k: v / total for k, v in combined_scores.items()}
        
        # Get top fibers
        sorted_fibers = sorted(probs.items(), key=lambda x: x[1], reverse=True)
        primary_fiber = sorted_fibers[0][0]
        primary_conf = sorted_fibers[0][1]
        
        # Estimate composition (top fibers that could be in blend)
        composition = {}
        remaining = 1.0
        
        for fiber, prob in sorted_fibers[:3]:  # Max 3 components
            if prob > 0.1 and remaining > 0:
                share = min(remaining, prob / (prob + 0.3))  # Dampened share
                composition[fiber] = share
                remaining -= share
        
        # Normalize composition to 100%
        comp_total = sum(composition.values())
        if comp_total > 0:
            composition = {k: v / comp_total for k, v in composition.items()}
        else:
            composition = {primary_fiber: 1.0}
        
        # Get category
        category = FIBER_DATABASE.get(primary_fiber, {}).get("category", "unknown")
        
        # Visual and text indicators
        visual_indicators = FIBER_DATABASE.get(primary_fiber, {}).get("visual_cues", [])
        text_indicators = list(text_scores.keys())[:3] if text_scores else []
        
        return FiberAnalysis(
            primary_fiber=primary_fiber,
            primary_confidence=primary_conf,
            composition=composition,
            category=category,
            visual_indicators=visual_indicators,
            text_indicators=text_indicators
        )
    
    def _analyze_visual_texture(self, image: np.ndarray) -> Dict[str, float]:
        """Analyze visual texture to score fiber likelihood."""
        scores = {}
        
        # Try CLIP-based classification first
        clip_scores = self._classify_with_clip(image)
        if clip_scores:
            return clip_scores
        
        # Fallback to texture analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            gray = image
            hsv = None
        
        # Texture features
        # 1. Sheen/gloss (high variance in highlights)
        sheen = self._calculate_sheen(gray)
        
        # 2. Texture coarseness
        coarseness = self._calculate_coarseness(gray)
        
        # 3. Color saturation
        saturation = hsv[:,:,1].mean() / 255.0 if hsv is not None else 0.5
        
        # Score fibers based on features
        for fiber, props in FIBER_DATABASE.items():
            cues = props.get("visual_cues", [])
            
            score = 0.5  # Base score
            
            # Sheen-based scoring
            if "high sheen" in cues or "shiny" in cues:
                score += sheen * 0.3
            elif "matte" in cues or "matte finish" in cues:
                score += (1 - sheen) * 0.2
            
            # Texture-based scoring
            if "coarse texture" in cues or "visible texture" in cues:
                score += coarseness * 0.2
            elif "smooth" in cues or "soft texture" in cues:
                score += (1 - coarseness) * 0.2
            
            # Fuzzy texture
            if "fuzzy" in cues or "fluffy" in cues:
                score += coarseness * 0.15
            
            scores[fiber] = min(1.0, score)
        
        return scores
    
    def _classify_with_clip(self, image: np.ndarray) -> Dict[str, float]:
        """Use CLIP for fiber classification."""
        self._load_clip_for_fibers()
        
        if self._clip_model is None or self._fiber_embeddings is None:
            return {}
        
        try:
            import torch
            
            # Convert to PIL
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Encode image
            image_input = self._preprocess(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self._clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarities
                similarities = (image_features @ self._fiber_embeddings.T).squeeze(0)
                
                # Softmax for probabilities
                probs = torch.softmax(similarities * 5, dim=0)  # Temperature scaling
                
                scores = {fiber: probs[i].item() for i, fiber in enumerate(ALL_FIBERS)}
            
            return scores
            
        except Exception as e:
            logger.warning(f"CLIP fiber classification failed: {e}")
            return {}
    
    def _calculate_sheen(self, gray: np.ndarray) -> float:
        """Calculate surface sheen/gloss from grayscale image."""
        # High sheen = high max brightness with localized highlights
        max_bright = gray.max()
        mean_bright = gray.mean()
        
        # Highlight ratio
        highlights = (gray > 0.9 * max_bright).sum()
        total_pixels = gray.size
        highlight_ratio = highlights / total_pixels
        
        # Sheen combines brightness difference and highlight presence
        brightness_diff = (max_bright - mean_bright) / 255.0
        sheen = brightness_diff * 0.5 + min(highlight_ratio * 10, 0.5)
        
        return min(1.0, sheen)
    
    def _calculate_coarseness(self, gray: np.ndarray) -> float:
        """Calculate texture coarseness using Laplacian variance."""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize (empirically, high variance indicates coarse texture)
        coarseness = min(1.0, variance / 1000.0)
        
        return coarseness
    
    def _analyze_text_description(self, text: str) -> Dict[str, float]:
        """Extract fiber hints from product description."""
        text_lower = text.lower()
        scores = {}
        
        # Direct fiber mentions
        for fiber in ALL_FIBERS:
            if fiber.replace("_", " ") in text_lower:
                scores[fiber] = 0.9
            elif fiber in text_lower:
                scores[fiber] = 0.8
        
        # Keyword associations
        keyword_fiber_map = {
            "sustainable": ["organic_cotton", "tencel", "hemp", "recycled_polyester"],
            "stretch": ["elastane", "polyester"],
            "breathable": ["cotton", "linen", "tencel"],
            "warm": ["wool", "cashmere", "fleece"],
            "luxurious": ["silk", "cashmere"],
            "wrinkle-free": ["polyester", "nylon"],
            "athletic": ["polyester", "nylon", "elastane"],
            "eco-friendly": ["organic_cotton", "tencel", "hemp", "recycled_polyester"],
            "vegan": ["cotton", "polyester", "tencel", "modal"]
        }
        
        for keyword, fibers in keyword_fiber_map.items():
            if keyword in text_lower:
                for fiber in fibers:
                    scores[fiber] = max(scores.get(fiber, 0), 0.6)
        
        return scores
    
    def _get_category_priors(self, category: str) -> Dict[str, float]:
        """Get fiber priors based on garment category."""
        category_lower = category.lower()
        
        # Category to likely fibers
        category_priors = {
            "jeans": {"denim": 0.8, "cotton": 0.7, "elastane": 0.4},
            "denim": {"denim": 0.9, "cotton": 0.6},
            "t-shirt": {"cotton": 0.8, "polyester": 0.3},
            "sweater": {"wool": 0.5, "cotton": 0.4, "cashmere": 0.3, "acrylic": 0.4},
            "dress": {"polyester": 0.4, "cotton": 0.4, "silk": 0.3, "viscose": 0.4},
            "sportswear": {"polyester": 0.7, "nylon": 0.5, "elastane": 0.5},
            "suit": {"wool": 0.7, "polyester": 0.4, "cotton": 0.3},
            "jacket": {"polyester": 0.4, "nylon": 0.4, "leather": 0.3, "cotton": 0.3},
            "underwear": {"cotton": 0.7, "modal": 0.4, "elastane": 0.4},
            "socks": {"cotton": 0.6, "wool": 0.3, "polyester": 0.3}
        }
        
        for cat, priors in category_priors.items():
            if cat in category_lower:
                return priors
        
        return {}


# ============================================
# 📊 LCA CALCULATOR
# ============================================

class LCACalculator:
    """
    Automated Life Cycle Assessment for fashion items.
    
    Calculates carbon footprint based on material composition.
    """
    
    # Average garment weights by category (kg)
    CATEGORY_WEIGHTS = {
        "t-shirt": 0.15,
        "shirt": 0.25,
        "blouse": 0.20,
        "sweater": 0.40,
        "hoodie": 0.50,
        "jacket": 0.80,
        "coat": 1.20,
        "jeans": 0.60,
        "pants": 0.45,
        "shorts": 0.25,
        "dress": 0.35,
        "skirt": 0.25,
        "suit": 1.50,
        "underwear": 0.05,
        "socks": 0.05,
        "shoes": 0.80,
        "bag": 0.50,
        "scarf": 0.15,
        "default": 0.35
    }
    
    def __init__(self):
        """Initialize LCA calculator."""
        pass
    
    def calculate_footprint(
        self,
        fiber_composition: Dict[str, float],
        estimated_weight_kg: Optional[float] = None,
        category: Optional[str] = None,
        is_recycled: bool = False
    ) -> LCAResult:
        """
        Calculate total carbon footprint.
        
        Args:
            fiber_composition: Dict of fiber -> percentage (0-1)
            estimated_weight_kg: Optional weight override
            category: Garment category for weight estimation
            is_recycled: Whether materials are recycled
            
        Returns:
            LCAResult with footprint details
        """
        # Estimate weight if not provided
        if estimated_weight_kg is None:
            if category:
                category_lower = category.lower()
                for cat, weight in self.CATEGORY_WEIGHTS.items():
                    if cat in category_lower:
                        estimated_weight_kg = weight
                        break
                else:
                    estimated_weight_kg = self.CATEGORY_WEIGHTS["default"]
            else:
                estimated_weight_kg = self.CATEGORY_WEIGHTS["default"]
        
        # Calculate footprint per fiber
        breakdown = {}
        total_co2e = 0.0
        water_impacts = []
        biodegradable_share = 0.0
        
        for fiber, percentage in fiber_composition.items():
            props = FIBER_DATABASE.get(fiber, {})
            co2e_per_kg = props.get("co2e_per_kg", 15.0)  # Default if unknown
            
            # Apply recycled discount
            if is_recycled and fiber in ["polyester", "nylon", "cotton"]:
                co2e_per_kg *= 0.3  # 70% reduction for recycled
            
            # Calculate contribution
            contribution = co2e_per_kg * percentage * estimated_weight_kg
            breakdown[fiber] = contribution
            total_co2e += contribution
            
            # Track other impacts
            water_impacts.append((props.get("water_intensity", "medium"), percentage))
            if props.get("biodegradable", False):
                biodegradable_share += percentage
        
        # Aggregate water impact
        water_scores = {"very_low": 0, "low": 1, "medium": 2, "high": 3, "very_high": 4}
        reverse_water = {v: k for k, v in water_scores.items()}
        
        avg_water = sum(water_scores.get(w, 2) * p for w, p in water_impacts)
        water_impact = reverse_water.get(round(avg_water), "medium")
        
        # Per-kg calculation
        per_kg_co2e = total_co2e / estimated_weight_kg if estimated_weight_kg > 0 else 0
        
        # Generate tips
        tips = self._generate_sustainability_tips(fiber_composition, is_recycled)
        
        return LCAResult(
            total_co2e_kg=total_co2e,
            per_kg_co2e=per_kg_co2e,
            breakdown=breakdown,
            water_impact=water_impact,
            biodegradability=biodegradable_share,
            sustainability_tips=tips
        )
    
    def _generate_sustainability_tips(
        self,
        composition: Dict[str, float],
        is_recycled: bool
    ) -> List[str]:
        """Generate actionable sustainability tips."""
        tips = []
        
        # Check for high-impact materials
        for fiber, percentage in composition.items():
            if percentage < 0.2:
                continue
            
            props = FIBER_DATABASE.get(fiber, {})
            co2e = props.get("co2e_per_kg", 0)
            
            if co2e > 50:
                tips.append(f"Consider alternatives to {fiber} (high carbon footprint)")
            
            if props.get("water_intensity") == "very_high":
                tips.append(f"{fiber.title()} is water-intensive; look for certified organic")
            
            if not props.get("biodegradable", False):
                tips.append(f"{fiber.title()} is not biodegradable; ensure proper recycling")
        
        # Generic tips
        if not is_recycled:
            tips.append("Look for recycled material options to reduce impact by up to 70%")
        
        if not tips:
            tips.append("This is a relatively sustainable choice! Extend lifespan through proper care.")
        
        return tips[:4]  # Max 4 tips
    
    def generate_eco_score(
        self,
        footprint: LCAResult,
        category_average: Optional[float] = None
    ) -> EcoScore:
        """
        Generate consumer-friendly eco-score.
        
        Args:
            footprint: LCAResult from calculate_footprint
            category_average: Optional average for comparison
            
        Returns:
            EcoScore with grade A-F
        """
        # Default category averages (kg CO2e for typical item)
        if category_average is None:
            category_average = 5.0  # Rough average for clothing item
        
        per_kg = footprint.per_kg_co2e
        
        # Score based on per-kg footprint
        # Lower is better: A < 5, B < 10, C < 15, D < 25, E < 40, F >= 40
        if per_kg < 5:
            grade = "A"
            score = 90 + int((5 - per_kg) * 2)
        elif per_kg < 10:
            grade = "B"
            score = 70 + int((10 - per_kg) * 4)
        elif per_kg < 15:
            grade = "C"
            score = 50 + int((15 - per_kg) * 4)
        elif per_kg < 25:
            grade = "D"
            score = 30 + int((25 - per_kg) * 2)
        elif per_kg < 40:
            grade = "E"
            score = 10 + int((40 - per_kg) / 1.5)
        else:
            grade = "F"
            score = max(0, 10 - int((per_kg - 40) / 5))
        
        score = max(0, min(100, score))
        
        # Comparison to average
        if footprint.total_co2e_kg < category_average * 0.7:
            comparison = "30% better than average"
        elif footprint.total_co2e_kg < category_average:
            comparison = "Better than average"
        elif footprint.total_co2e_kg < category_average * 1.3:
            comparison = "About average"
        else:
            comparison = "Above average impact"
        
        # Explanation
        grade_explanations = {
            "A": "Excellent sustainability profile with minimal environmental impact.",
            "B": "Good sustainability choice with below-average impact.",
            "C": "Average environmental impact. Consider alternatives for improvement.",
            "D": "Above-average environmental impact. Look for sustainable alternatives.",
            "E": "High environmental impact. Consider recycled or natural alternatives.",
            "F": "Very high environmental impact. Strongly consider alternatives."
        }
        
        # Improvements
        improvements = []
        if not footprint.biodegradability > 0.8:
            improvements.append("Choose more natural, biodegradable fibers")
        if footprint.water_impact in ["high", "very_high"]:
            improvements.append("Look for low water impact materials like hemp or tencel")
        if "recycled" not in str(footprint.breakdown.keys()).lower():
            improvements.append("Consider recycled material alternatives")
        
        return EcoScore(
            grade=grade,
            score=score,
            explanation=grade_explanations[grade],
            comparison=comparison,
            improvements=improvements[:3]
        )


# ============================================
# 🔧 UTILITY FUNCTIONS
# ============================================

def analyze_sustainability(
    image_b64: str,
    description: Optional[str] = None,
    category: Optional[str] = None
) -> Dict:
    """
    Complete sustainability analysis from image.
    
    Returns fiber composition, LCA footprint, and eco-score.
    """
    # Decode image
    if ',' in image_b64:
        image_b64 = image_b64.split(',')[1]
    
    img_bytes = base64.b64decode(image_b64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "Could not decode image"}
    
    # Classify fibers
    classifier = TextileNetClassifier()
    fiber_analysis = classifier.classify_fiber(image, description, category)
    
    # Calculate LCA
    calculator = LCACalculator()
    lca_result = calculator.calculate_footprint(
        fiber_analysis.composition,
        category=category
    )
    
    # Generate eco-score
    eco_score = calculator.generate_eco_score(lca_result)
    
    return {
        "fiberAnalysis": fiber_analysis.to_dict(),
        "lcaResult": lca_result.to_dict(),
        "ecoScore": eco_score.to_dict()
    }


# Singleton instances
_textile_net_instance = None
_lca_calculator_instance = None

def get_textile_net_classifier() -> TextileNetClassifier:
    """Get singleton TextileNetClassifier."""
    global _textile_net_instance
    if _textile_net_instance is None:
        _textile_net_instance = TextileNetClassifier()
    return _textile_net_instance

def get_lca_calculator() -> LCACalculator:
    """Get singleton LCACalculator."""
    global _lca_calculator_instance
    if _lca_calculator_instance is None:
        _lca_calculator_instance = LCACalculator()
    return _lca_calculator_instance
