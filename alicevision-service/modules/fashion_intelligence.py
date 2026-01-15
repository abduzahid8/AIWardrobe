"""
🧠 FASHION INTELLIGENCE ENGINE
Deep Clothing Understanding System

This module provides comprehensive fashion analysis using multi-pass VLM processing:
- Pass 1: Item Detection (identify all clothing items)
- Pass 2: Deep Attribute Analysis (30+ attributes per item)
- Pass 3: Style & Aesthetic Classification
- Pass 4: Outfit Intelligence (compatibility, suggestions)

This goes FAR beyond simple detection - it provides fashion expert-level understanding.
"""

import cv2
import numpy as np
import base64
import logging
import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================
# 📦 DATA STRUCTURES
# ============================================

@dataclass
class ColorInfo:
    """Complete color information"""
    primary: str
    secondary: Optional[str] = None
    hex: str = "#000000"
    temperature: str = "neutral"  # warm, cool, neutral
    saturation: str = "medium"  # vibrant, muted, pastel


@dataclass
class MaterialInfo:
    """Material and texture information"""
    outer: str
    lining: Optional[str] = None
    texture: str = "smooth"
    weight: str = "midweight"  # lightweight, midweight, heavy
    stretch: str = "no stretch"
    sheerness: str = "opaque"


@dataclass 
class ConstructionInfo:
    """Garment construction details"""
    closure: str = "none"  # buttons, zipper, pullover
    neckline: Optional[str] = None
    sleeves: Optional[str] = None
    hemline: Optional[str] = None
    pockets: Optional[str] = None
    hardware: Optional[str] = None
    details: List[str] = field(default_factory=list)


@dataclass
class FitInfo:
    """Fit and silhouette information"""
    fit: str = "regular"  # slim, regular, relaxed, oversized
    rise: Optional[str] = None  # low, mid, high (for pants)
    length: str = "regular"  # cropped, regular, longline
    silhouette: str = "structured"


@dataclass
class StyleInfo:
    """Style classification and context"""
    aesthetics: List[str] = field(default_factory=list)  # streetwear, minimalist, preppy
    formality: str = "casual"  # formal, smart casual, casual, athleisure
    occasions: List[str] = field(default_factory=list)
    seasons: List[str] = field(default_factory=list)
    gender: str = "unisex"
    age_group: str = "all ages"
    trends: List[str] = field(default_factory=list)


@dataclass
class QualityInfo:
    """Quality and condition assessment"""
    condition: str = "new"  # new, gently used, vintage, distressed
    quality_level: str = "mid-range"  # luxury, premium, mid-range, budget
    craftsmanship: str = "mass-produced"
    estimated_price_range: Optional[str] = None


@dataclass
class FashionItem:
    """Complete fashion item with deep attributes"""
    # Identity
    type: str
    sub_type: str = ""
    category: str = "clothing"
    brand_guess: Optional[str] = None
    
    # Deep attributes
    color: ColorInfo = None
    material: MaterialInfo = None
    construction: ConstructionInfo = None
    fit: FitInfo = None
    style: StyleInfo = None
    quality: QualityInfo = None
    
    # Confidence
    confidence: float = 0.95
    analysis_depth: str = "comprehensive"
    
    # Image data
    bbox: Optional[List[int]] = None
    cutout_image: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "identity": {
                "type": self.type,
                "subType": self.sub_type,
                "category": self.category,
                "brandGuess": self.brand_guess
            },
            "color": {
                "primary": self.color.primary if self.color else "unknown",
                "secondary": self.color.secondary if self.color else None,
                "hex": self.color.hex if self.color else "#000000",
                "temperature": self.color.temperature if self.color else "neutral"
            },
            "material": {
                "outer": self.material.outer if self.material else "unknown",
                "lining": self.material.lining if self.material else None,
                "texture": self.material.texture if self.material else "smooth",
                "weight": self.material.weight if self.material else "midweight"
            },
            "construction": {
                "closure": self.construction.closure if self.construction else "none",
                "neckline": self.construction.neckline if self.construction else None,
                "sleeves": self.construction.sleeves if self.construction else None,
                "pockets": self.construction.pockets if self.construction else None,
                "details": self.construction.details if self.construction else []
            },
            "fit": {
                "fit": self.fit.fit if self.fit else "regular",
                "length": self.fit.length if self.fit else "regular",
                "silhouette": self.fit.silhouette if self.fit else "structured"
            },
            "style": {
                "aesthetics": self.style.aesthetics if self.style else [],
                "formality": self.style.formality if self.style else "casual",
                "occasions": self.style.occasions if self.style else [],
                "seasons": self.style.seasons if self.style else [],
                "gender": self.style.gender if self.style else "unisex",
                "trends": self.style.trends if self.style else []
            },
            "quality": {
                "condition": self.quality.condition if self.quality else "new",
                "level": self.quality.quality_level if self.quality else "mid-range",
                "priceRange": self.quality.estimated_price_range if self.quality else None
            },
            "confidence": self.confidence,
            "bbox": self.bbox,
            "cutoutImage": self.cutout_image
        }


@dataclass
class OutfitIntelligence:
    """Complete outfit analysis and recommendations"""
    overall_aesthetic: str = ""
    style_coherence: float = 0.0  # 0-10
    fashion_influences: List[str] = field(default_factory=list)
    trend_alignment: List[str] = field(default_factory=list)
    signature_elements: str = ""
    
    # Recommendations
    suggestions: List[str] = field(default_factory=list)
    compatible_additions: List[str] = field(default_factory=list)
    alternative_styling: List[str] = field(default_factory=list)
    
    # Occasion ratings (0-10)
    occasion_ratings: Dict[str, float] = field(default_factory=dict)
    weather_suitability: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "overallAesthetic": self.overall_aesthetic,
            "styleCoherence": self.style_coherence,
            "fashionInfluences": self.fashion_influences,
            "trendAlignment": self.trend_alignment,
            "signatureElements": self.signature_elements,
            "suggestions": self.suggestions,
            "compatibleAdditions": self.compatible_additions,
            "alternativeStyling": self.alternative_styling,
            "occasionRatings": self.occasion_ratings,
            "weatherSuitability": self.weather_suitability
        }


@dataclass
class FashionIntelligenceResult:
    """Complete result from Fashion Intelligence Engine"""
    success: bool
    items: List[FashionItem] = field(default_factory=list)
    outfit_intelligence: Optional[OutfitIntelligence] = None
    total_items: int = 0
    processing_time_ms: float = 0
    analysis_passes: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "items": [item.to_dict() for item in self.items],
            "outfitIntelligence": self.outfit_intelligence.to_dict() if self.outfit_intelligence else None,
            "totalItems": self.total_items,
            "processingTimeMs": self.processing_time_ms,
            "analysisPasses": self.analysis_passes
        }


# ============================================
# 🧠 VLM PROMPTS FOR MULTI-PASS ANALYSIS
# ============================================

PASS1_DETECTION_PROMPT = """You are a fashion expert. Analyze this image and identify EVERY clothing item visible.

For each item provide:
1. type: Be VERY specific (e.g., "MA-1 bomber jacket" not just "jacket")
2. category: tops/bottoms/outerwear/footwear/accessories/full-body
3. visibility: fully/partially visible

Return ONLY valid JSON:
{
    "items": [
        {"type": "specific type", "category": "category", "visibility": "full/partial"}
    ]
}"""


PASS2_DEEP_ANALYSIS_PROMPT = """You are a fashion industry expert with deep knowledge of garments.
Analyze this {garment_type} in EXTREME detail.

Extract ALL of these attributes:

1. IDENTITY
   - type: exact garment type
   - subType: specific variant (e.g., "MA-1" for bomber, "501" for jeans)
   - brandGuess: if it resembles a known brand's style

2. COLOR
   - primary: main color (be specific - "olive green" not "green")
   - secondary: accent/trim color if any
   - hex: approximate hex code
   - temperature: warm/cool/neutral

3. MATERIAL
   - outer: main fabric (cotton, denim, leather, wool, nylon, polyester, etc.)
   - lining: lining fabric if visible
   - texture: smooth/ribbed/quilted/distressed/matte/shiny
   - weight: lightweight/midweight/heavy

4. CONSTRUCTION
   - closure: buttons/zipper/pullover/snap/tie
   - neckline: crew/v-neck/mock/turtleneck/collar (for tops)
   - sleeves: long/short/3-4/sleeveless/raglan
   - pockets: patch/welt/cargo/hidden/none
   - hardware: metal zippers/buttons/grommets (describe)
   - details: any special features (stitching, embroidery, patches)

5. FIT
   - fit: slim/regular/relaxed/oversized
   - rise: low/mid/high (for bottoms only)
   - length: cropped/regular/longline
   - silhouette: structured/flowing/bodycon/boxy

6. STYLE
   - aesthetics: list style categories (streetwear, minimalist, preppy, bohemian, etc.)
   - formality: formal/smart-casual/casual/athleisure
   - occasions: list suitable occasions
   - seasons: spring/summer/fall/winter/all-season
   - gender: men's/women's/unisex
   - trends: relevant fashion trends

7. QUALITY
   - condition: new/gently-used/vintage/distressed
   - level: luxury/premium/mid-range/budget
   - priceRange: estimated price range

Return ONLY valid JSON with ALL these fields."""


PASS3_STYLE_CLASSIFICATION_PROMPT = """As a fashion stylist, analyze the COMPLETE outfit in this image.

Provide:

1. OVERALL_AESTHETIC: Primary style category
   (e.g., "streetwear", "minimalist", "preppy", "bohemian", "athleisure", "smart casual")

2. STYLE_COHERENCE: Rate 1-10 how well items work together. Explain.

3. FASHION_INFLUENCES: What trends, eras, or subcultures does this reference?
   (e.g., "90s grunge", "quiet luxury", "Y2K revival", "workwear heritage", "Scandinavian minimalism")

4. SIGNATURE_ELEMENTS: What makes this outfit distinctive?

5. TREND_ALIGNMENT: Current fashion trends this outfit reflects

Return ONLY valid JSON:
{
    "overallAesthetic": "style",
    "styleCoherence": {"score": 8, "explanation": "..."},
    "fashionInfluences": ["influence1", "influence2"],
    "signatureElements": "description",
    "trendAlignment": ["trend1", "trend2"]
}"""


PASS4_OUTFIT_INTELLIGENCE_PROMPT = """As a personal stylist, provide outfit intelligence for this look:

1. COMPATIBLE_ADDITIONS: What items would enhance this outfit?
   Be specific (e.g., "camel wool overcoat", "white leather sneakers")

2. ALTERNATIVE_STYLING: How could these same items be styled differently?
   Give 2-3 alternative outfit ideas.

3. OCCASION_RATINGS: Rate suitability 1-10 for each:
   - Work/Office
   - Casual Weekend  
   - Date Night
   - Party/Event
   - Travel
   - Outdoor Activity

4. WEATHER_SUITABILITY: Temperature range and weather conditions this works for.

5. IMPROVEMENT_SUGGESTIONS: What would make this outfit stronger?
   Give 2-3 specific, actionable suggestions.

Return ONLY valid JSON:
{
    "compatibleAdditions": ["item1", "item2"],
    "alternativeStyling": ["idea1", "idea2"],
    "occasionRatings": {"work": 7, "casual": 9, "dateNight": 6, "party": 5, "travel": 8, "outdoor": 7},
    "weatherSuitability": "description",
    "suggestions": ["suggestion1", "suggestion2"]
}"""


# ============================================
# 🧠 FASHION INTELLIGENCE ENGINE
# ============================================

class FashionIntelligenceEngine:
    """
    🧠 COMPREHENSIVE FASHION UNDERSTANDING
    
    Multi-pass VLM analysis for deep clothing comprehension.
    """
    
    def __init__(self, provider: str = "replicate"):
        self.provider = provider
        self._qwen = None
    
    def _get_qwen(self):
        """Get Qwen VLM instance"""
        if self._qwen is None:
            from modules.qwen_vl_reasoning import get_qwen_reasoning
            self._qwen = get_qwen_reasoning(provider=self.provider)
        return self._qwen
    
    def _extract_json(self, text: str) -> Dict:
        """Extract JSON from VLM response"""
        try:
            # Try direct parse
            return json.loads(text)
        except:
            pass
        
        # Try to find JSON block
        import re
        json_match = re.search(r'\{[\s\S]*\}', text)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        # Try array
        array_match = re.search(r'\[[\s\S]*\]', text)
        if array_match:
            try:
                return {"items": json.loads(array_match.group())}
            except:
                pass
        
        return {}
    
    async def analyze(self, image_b64: str) -> FashionIntelligenceResult:
        """
        🧠 COMPLETE FASHION ANALYSIS
        
        Runs all 4 passes for comprehensive understanding.
        """
        start_time = time.time()
        logger.info("=" * 70)
        logger.info("🧠 FASHION INTELLIGENCE ENGINE: Starting deep analysis...")
        
        passes_completed = 0
        
        try:
            qwen = self._get_qwen()
            
            # ========== PASS 1: Detection ==========
            logger.info("📍 PASS 1: Detecting clothing items...")
            
            detect_result = qwen.query(
                image_b64,
                prompt=PASS1_DETECTION_PROMPT,
                json_output=True,
                max_tokens=1024
            )
            
            if not detect_result.success:
                logger.error(f"Pass 1 failed: {detect_result.answer}")
                return FashionIntelligenceResult(success=False)
            
            detected_items = detect_result.structured_data.get("items", [])
            logger.info(f"   Found {len(detected_items)} items")
            passes_completed = 1
            
            if not detected_items:
                return FashionIntelligenceResult(success=False)
            
            # ========== PASS 2: Deep Analysis ==========
            logger.info("🔍 PASS 2: Deep attribute analysis...")
            
            fashion_items = []
            for i, item in enumerate(detected_items):
                item_type = item.get("type", "clothing")
                logger.info(f"   Analyzing {i+1}/{len(detected_items)}: {item_type}")
                
                # Get deep attributes for this item
                deep_prompt = PASS2_DEEP_ANALYSIS_PROMPT.replace("{garment_type}", item_type)
                
                deep_result = qwen.query(
                    image_b64,
                    prompt=deep_prompt,
                    json_output=True,
                    max_tokens=2048
                )
                
                if deep_result.success and deep_result.structured_data:
                    data = deep_result.structured_data
                    
                    # Build FashionItem from deep analysis
                    fashion_item = self._build_fashion_item(item_type, item.get("category", "clothing"), data)
                    fashion_items.append(fashion_item)
                else:
                    # Basic item if deep analysis fails
                    fashion_items.append(FashionItem(
                        type=item_type,
                        category=item.get("category", "clothing")
                    ))
            
            passes_completed = 2
            logger.info(f"   Analyzed {len(fashion_items)} items deeply")
            
            # ========== PASS 3: Style Classification ==========
            logger.info("🎨 PASS 3: Style classification...")
            
            style_result = qwen.query(
                image_b64,
                prompt=PASS3_STYLE_CLASSIFICATION_PROMPT,
                json_output=True,
                max_tokens=1024
            )
            
            style_data = style_result.structured_data if style_result.success else {}
            passes_completed = 3
            
            # ========== PASS 4: Outfit Intelligence ==========
            logger.info("💡 PASS 4: Outfit intelligence...")
            
            intel_result = qwen.query(
                image_b64,
                prompt=PASS4_OUTFIT_INTELLIGENCE_PROMPT,
                json_output=True,
                max_tokens=1024
            )
            
            intel_data = intel_result.structured_data if intel_result.success else {}
            passes_completed = 4
            
            # Build outfit intelligence
            outfit_intel = self._build_outfit_intelligence(style_data, intel_data)
            
            # Generate cutouts for each item
            await self._generate_cutouts(image_b64, fashion_items)
            
            processing_time = (time.time() - start_time) * 1000
            
            logger.info("=" * 70)
            logger.info(f"🧠 FASHION INTELLIGENCE COMPLETE: {len(fashion_items)} items, {passes_completed} passes, {processing_time:.0f}ms")
            logger.info("=" * 70)
            
            return FashionIntelligenceResult(
                success=True,
                items=fashion_items,
                outfit_intelligence=outfit_intel,
                total_items=len(fashion_items),
                processing_time_ms=processing_time,
                analysis_passes=passes_completed
            )
            
        except Exception as e:
            logger.error(f"Fashion Intelligence error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return FashionIntelligenceResult(success=False)
    
    def _build_fashion_item(self, item_type: str, category: str, data: Dict) -> FashionItem:
        """Build FashionItem from deep analysis data"""
        
        # Extract color info
        color_data = data.get("color", data.get("COLOR", {}))
        if isinstance(color_data, dict):
            color = ColorInfo(
                primary=color_data.get("primary", "unknown"),
                secondary=color_data.get("secondary"),
                hex=color_data.get("hex", "#000000"),
                temperature=color_data.get("temperature", "neutral")
            )
        else:
            color = ColorInfo(primary=str(color_data) if color_data else "unknown")
        
        # Extract material info
        mat_data = data.get("material", data.get("MATERIAL", {}))
        if isinstance(mat_data, dict):
            material = MaterialInfo(
                outer=mat_data.get("outer", "unknown"),
                lining=mat_data.get("lining"),
                texture=mat_data.get("texture", "smooth"),
                weight=mat_data.get("weight", "midweight")
            )
        else:
            material = MaterialInfo(outer=str(mat_data) if mat_data else "unknown")
        
        # Extract construction info
        const_data = data.get("construction", data.get("CONSTRUCTION", {}))
        if isinstance(const_data, dict):
            construction = ConstructionInfo(
                closure=const_data.get("closure", "none"),
                neckline=const_data.get("neckline"),
                sleeves=const_data.get("sleeves"),
                pockets=const_data.get("pockets"),
                hardware=const_data.get("hardware"),
                details=const_data.get("details", [])
            )
        else:
            construction = ConstructionInfo()
        
        # Extract fit info
        fit_data = data.get("fit", data.get("FIT", {}))
        if isinstance(fit_data, dict):
            fit = FitInfo(
                fit=fit_data.get("fit", "regular"),
                rise=fit_data.get("rise"),
                length=fit_data.get("length", "regular"),
                silhouette=fit_data.get("silhouette", "structured")
            )
        else:
            fit = FitInfo()
        
        # Extract style info
        style_data = data.get("style", data.get("STYLE", {}))
        if isinstance(style_data, dict):
            style = StyleInfo(
                aesthetics=style_data.get("aesthetics", []),
                formality=style_data.get("formality", "casual"),
                occasions=style_data.get("occasions", []),
                seasons=style_data.get("seasons", []),
                gender=style_data.get("gender", "unisex"),
                trends=style_data.get("trends", [])
            )
        else:
            style = StyleInfo()
        
        # Extract quality info
        qual_data = data.get("quality", data.get("QUALITY", {}))
        if isinstance(qual_data, dict):
            quality = QualityInfo(
                condition=qual_data.get("condition", "new"),
                quality_level=qual_data.get("level", qual_data.get("qualityLevel", "mid-range")),
                estimated_price_range=qual_data.get("priceRange")
            )
        else:
            quality = QualityInfo()
        
        # Extract identity info
        identity_data = data.get("identity", data.get("IDENTITY", {}))
        sub_type = ""
        brand_guess = None
        if isinstance(identity_data, dict):
            sub_type = identity_data.get("subType", "")
            brand_guess = identity_data.get("brandGuess")
        
        return FashionItem(
            type=identity_data.get("type", item_type) if isinstance(identity_data, dict) else item_type,
            sub_type=sub_type,
            category=category,
            brand_guess=brand_guess,
            color=color,
            material=material,
            construction=construction,
            fit=fit,
            style=style,
            quality=quality,
            confidence=0.95
        )
    
    def _build_outfit_intelligence(self, style_data: Dict, intel_data: Dict) -> OutfitIntelligence:
        """Build OutfitIntelligence from analysis results"""
        
        coherence_data = style_data.get("styleCoherence", {})
        if isinstance(coherence_data, dict):
            coherence_score = coherence_data.get("score", 7)
        else:
            coherence_score = float(coherence_data) if coherence_data else 7
        
        return OutfitIntelligence(
            overall_aesthetic=style_data.get("overallAesthetic", ""),
            style_coherence=coherence_score,
            fashion_influences=style_data.get("fashionInfluences", []),
            trend_alignment=style_data.get("trendAlignment", []),
            signature_elements=style_data.get("signatureElements", ""),
            suggestions=intel_data.get("suggestions", []),
            compatible_additions=intel_data.get("compatibleAdditions", []),
            alternative_styling=intel_data.get("alternativeStyling", []),
            occasion_ratings=intel_data.get("occasionRatings", {}),
            weather_suitability=intel_data.get("weatherSuitability", "")
        )
    
    async def _generate_cutouts(self, image_b64: str, items: List[FashionItem]) -> None:
        """Generate cutout images for items using SegFormer"""
        try:
            # Decode image
            if ',' in image_b64:
                img_data = image_b64.split(',')[1]
            else:
                img_data = image_b64
            
            img_bytes = base64.b64decode(img_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return
            
            from modules.segmentation import get_advanced_segmentor
            segmentor = get_advanced_segmentor()
            seg_result = segmentor.segment(image, add_white_bg=False, return_items=True)
            
            h, w = image.shape[:2]
            
            # Map categories
            category_map = {
                "tops": ["upper_clothes"],
                "bottoms": ["pants", "shorts"],
                "outerwear": ["upper_clothes"],
                "footwear": ["left_shoe", "right_shoe", "shoes"],
                "accessories": ["hat", "bag", "scarf", "belt", "sunglasses"],
                "full-body": ["dress"]
            }
            
            for fashion_item in items:
                cat = fashion_item.category.lower()
                seg_cats = category_map.get(cat, [cat])
                
                for seg_item in seg_result.items:
                    if seg_item.category.lower() in seg_cats:
                        mask = seg_item.mask
                        bbox = seg_item.bbox
                        
                        if mask is not None:
                            white_bg = np.ones_like(image) * 255
                            mask_3ch = cv2.merge([mask, mask, mask])
                            cutout = np.where(mask_3ch > 127, image, white_bg).astype(np.uint8)
                            
                            if bbox:
                                x, y, bw, bh = bbox
                                pad = 20
                                x1 = max(0, x - pad)
                                y1 = max(0, y - pad)
                                x2 = min(w, x + bw + pad)
                                y2 = min(h, y + bh + pad)
                                cutout = cutout[y1:y2, x1:x2]
                                fashion_item.bbox = [x1, y1, x2 - x1, y2 - y1]
                            
                            _, buffer = cv2.imencode('.jpg', cutout, [cv2.IMWRITE_JPEG_QUALITY, 92])
                            fashion_item.cutout_image = f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"
                        break
                        
        except Exception as e:
            logger.warning(f"Cutout generation failed: {e}")


# ============================================
# 🔧 UTILITY FUNCTIONS
# ============================================

_engine = None

def get_fashion_intelligence_engine() -> FashionIntelligenceEngine:
    """Get singleton engine instance"""
    global _engine
    if _engine is None:
        _engine = FashionIntelligenceEngine()
    return _engine


async def analyze_fashion_deep(image_b64: str) -> FashionIntelligenceResult:
    """
    🧠 ANALYZE FASHION WITH DEEP UNDERSTANDING
    
    Entry point for Fashion Intelligence Engine.
    Returns comprehensive analysis with 30+ attributes per item.
    """
    engine = get_fashion_intelligence_engine()
    return await engine.analyze(image_b64)
