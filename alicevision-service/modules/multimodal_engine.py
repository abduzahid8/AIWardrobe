"""
Multimodal Fashion AI Engine
Combines Vision + Text understanding for intelligent outfit recommendations

Uses Google Gemini Vision for multimodal understanding combined with
Fashion-CLIP embeddings for semantic wardrobe search.
"""

import os
import logging
import base64
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from PIL import Image
import io
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class ClothingItem:
    """Represents a clothing item in the wardrobe."""
    id: str
    category: str  # upper_clothes, pants, dress, etc.
    specific_type: str  # denim jacket, skinny jeans, etc.
    primary_color: str
    color_hex: str
    pattern: Optional[str] = None
    material: Optional[str] = None
    style_tags: List[str] = field(default_factory=list)
    occasion_tags: List[str] = field(default_factory=list)
    image_base64: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "category": self.category,
            "specificType": self.specific_type,
            "primaryColor": self.primary_color,
            "colorHex": self.color_hex,
            "pattern": self.pattern,
            "material": self.material,
            "styleTags": self.style_tags,
            "occasionTags": self.occasion_tags,
            "hasImage": self.image_base64 is not None
        }


@dataclass
class Outfit:
    """Represents an outfit combination."""
    items: List[ClothingItem]
    confidence: float
    reasoning: str
    occasion: str
    style: str
    color_harmony: str
    
    def to_dict(self) -> Dict:
        return {
            "items": [item.to_dict() for item in self.items],
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "occasion": self.occasion,
            "style": self.style,
            "colorHarmony": self.color_harmony
        }


@dataclass
class WardrobeAnalysis:
    """Complete wardrobe analysis result."""
    items: List[ClothingItem]
    total_items: int
    categories: Dict[str, int]  # category -> count
    color_palette: List[str]  # dominant colors
    style_profile: Dict[str, float]  # style -> percentage
    completeness_score: float  # 0-1 how complete is wardrobe
    recommendations: List[str]  # what's missing
    
    def to_dict(self) -> Dict:
        return {
            "totalItems": self.total_items,
            "categories": self.categories,
            "colorPalette": self.color_palette,
            "styleProfile": self.style_profile,
            "completenessScore": self.completeness_score,
            "recommendations": self.recommendations,
            "items": [item.to_dict() for item in self.items]
        }


@dataclass
class UserPreferences:
    """User style preferences."""
    preferred_styles: List[str] = field(default_factory=list)
    avoid_colors: List[str] = field(default_factory=list)
    preferred_colors: List[str] = field(default_factory=list)
    body_type: Optional[str] = None
    skin_tone: Optional[str] = None


class MultimodalFashionAI:
    """
    🧠 Unified Vision + Language Model for Fashion Understanding
    
    The brain of our multimodal outfit recommendation system.
    Combines:
    - Google Gemini Vision for multimodal understanding
    - Fashion-CLIP embeddings for semantic search
    - Fashion knowledge for rule-based compatibility
    
    Capabilities:
    - Understand complete wardrobe from videos/photos
    - Answer natural language questions about outfits
    - Generate outfit recommendations with reasoning
    - Semantic search ("find my warm jackets")
    - Learn user style preferences
    """
    
    # Fashion knowledge base
    OCCASION_REQUIREMENTS = {
        "job interview": {
            "required_categories": ["upper_clothes", "pants"],  # or dress
            "styles": ["professional", "business", "smart_casual"],
            "avoid_patterns": ["loud", "busy", "bright_colors"],
            "color_preferences": ["navy", "black", "gray", "white", "beige"]
        },
        "casual dinner": {
            "required_categories": ["upper_clothes", "pants"],
            "styles": ["smart_casual", "casual", "elegant"],
            "avoid_patterns": [],
            "color_preferences": []
        },
        "beach": {
            "required_categories": ["upper_clothes", "shorts"],
            "styles": ["casual", "vacation", "relaxed"],
            "avoid_patterns": [],
            "color_preferences": ["bright", "light"]
        },
        "wedding": {
            "required_categories": ["dress"],  # or suit
            "styles": ["formal", "elegant", "dressy"],
            "avoid_patterns": ["casual"],
            "color_preferences": []
        },
        "casual": {
            "required_categories": ["upper_clothes", "pants"],
            "styles": ["casual", "streetwear", "relaxed"],
            "avoid_patterns": [],
            "color_preferences": []
        },
        "workout": {
            "required_categories": ["upper_clothes", "pants"],
            "styles": ["athletic", "sporty"],
            "avoid_patterns": [],
            "color_preferences": []
        }
    }
    
    # Color harmony rules
    COLOR_HARMONIES = {
        "complementary": {
            "blue": ["orange", "rust", "coral"],
            "red": ["green", "teal", "mint"],
            "yellow": ["purple", "violet", "lavender"],
            "navy": ["cream", "beige", "tan", "rust"],
            "black": ["white", "red", "gold", "any"],
            "white": ["any"],
            "gray": ["yellow", "pink", "blue", "any"],
            "brown": ["blue", "teal", "cream"],
            "green": ["pink", "red", "cream"],
        },
        "neutral_combos": ["black", "white", "gray", "navy", "beige", "brown", "khaki", "cream"],
        "always_safe": [
            ("navy", "white"),
            ("black", "white"),
            ("gray", "blue"),
            ("beige", "brown"),
            ("white", "denim_blue"),
            ("black", "gray"),
        ]
    }
    
    def __init__(
        self,
        gemini_api_key: str = None,
        use_local_llm: bool = False,
        device: str = "auto"
    ):
        """
        Initialize the Multimodal Fashion AI.
        
        Args:
            gemini_api_key: Google Gemini API key (defaults to env var)
            use_local_llm: Use local LLM instead of Gemini (future)
            device: "cuda", "cpu", or "auto"
        """
        self.gemini_api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")
        self.use_local_llm = use_local_llm
        self.device = self._setup_device(device)
        
        self.genai = None
        self.gemini_model = None
        self.clip_encoder = None
        self.wardrobe_memory = {}  # user_id -> WardrobeEmbeddingStore
        
        logger.info("MultimodalFashionAI initialized")
        
    def _setup_device(self, device: str) -> str:
        """Setup compute device."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def _load_gemini(self):
        """Lazy load Gemini Vision model."""
        if self.genai is None:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.gemini_api_key)
                self.genai = genai
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Gemini Vision model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load Gemini: {e}")
                raise
    
    def _load_clip(self):
        """Lazy load Fashion-CLIP encoder."""
        if self.clip_encoder is None:
            try:
                from .fashion_clip import get_fashion_clip
                self.clip_encoder = get_fashion_clip()
                logger.info("Fashion-CLIP loaded successfully")
            except Exception as e:
                logger.warning(f"Fashion-CLIP not available: {e}")
    
    def _decode_image(self, image_base64: str) -> Image.Image:
        """Decode base64 image to PIL Image."""
        if image_base64.startswith('data:'):
            image_base64 = image_base64.split(',')[1]
        image_data = base64.b64decode(image_base64)
        return Image.open(io.BytesIO(image_data))
    
    async def understand_wardrobe(
        self,
        images: List[str],
        existing_items: List[Dict] = None
    ) -> WardrobeAnalysis:
        """
        🔍 Extract complete wardrobe understanding from images/video frames.
        
        Uses Gemini Vision to analyze all clothing items and build
        a comprehensive wardrobe profile.
        
        Args:
            images: List of base64-encoded images
            existing_items: Previously detected items to augment
            
        Returns:
            WardrobeAnalysis with complete wardrobe understanding
        """
        self._load_gemini()
        
        # Build multimodal prompt
        prompt = """You are an expert fashion analyst. Analyze these wardrobe images and extract:

1. **All Clothing Items**: For each item, identify:
   - Category (upper_clothes, pants, dress, skirt, jacket, shoes, bag, hat, etc.)
   - Specific type (e.g., "denim jacket", "skinny jeans", "maxi dress")
   - Primary color (use specific color names like "navy blue", "forest green")
   - Pattern (solid, striped, plaid, floral, geometric, etc.)
   - Material (cotton, denim, silk, leather, wool, etc.)
   - Style tags (casual, formal, sporty, elegant, bohemian, etc.)
   - Occasion suitability (work, casual, formal events, sports)

2. **Wardrobe Analysis**:
   - Overall style profile (% casual, formal, sporty, etc.)
   - Color palette summary (dominant colors)
   - Wardrobe completeness (what categories are well-covered)
   - Missing items recommendations

Return as JSON with this structure:
{
    "items": [
        {
            "category": "...",
            "specificType": "...",
            "primaryColor": "...",
            "pattern": "...",
            "material": "...",
            "styleTags": ["...", "..."],
            "occasionTags": ["...", "..."]
        }
    ],
    "styleProfile": {"casual": 0.4, "formal": 0.3, ...},
    "colorPalette": ["navy", "white", "gray", ...],
    "completenessScore": 0.75,
    "recommendations": ["Add more formal shoes", "Consider a blazer"]
}"""

        try:
            # Prepare images for Gemini
            pil_images = [self._decode_image(img) for img in images[:5]]  # Limit to 5 images
            
            # Call Gemini Vision
            response = self.gemini_model.generate_content(
                [prompt] + pil_images
            )
            
            # Parse response
            response_text = response.text
            
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                result = json.loads(response_text[json_start:json_end])
            else:
                result = {"items": [], "styleProfile": {}, "colorPalette": [], 
                         "completenessScore": 0, "recommendations": []}
            
            # Convert to ClothingItem objects
            items = []
            for i, item_data in enumerate(result.get("items", [])):
                item = ClothingItem(
                    id=f"item_{i}",
                    category=item_data.get("category", "unknown"),
                    specific_type=item_data.get("specificType", ""),
                    primary_color=item_data.get("primaryColor", ""),
                    color_hex=self._color_name_to_hex(item_data.get("primaryColor", "")),
                    pattern=item_data.get("pattern"),
                    material=item_data.get("material"),
                    style_tags=item_data.get("styleTags", []),
                    occasion_tags=item_data.get("occasionTags", [])
                )
                items.append(item)
            
            # Calculate category counts
            categories = {}
            for item in items:
                cat = item.category
                categories[cat] = categories.get(cat, 0) + 1
            
            return WardrobeAnalysis(
                items=items,
                total_items=len(items),
                categories=categories,
                color_palette=result.get("colorPalette", []),
                style_profile=result.get("styleProfile", {}),
                completeness_score=result.get("completenessScore", 0),
                recommendations=result.get("recommendations", [])
            )
            
        except Exception as e:
            logger.error(f"Wardrobe analysis failed: {e}")
            return WardrobeAnalysis(
                items=[],
                total_items=0,
                categories={},
                color_palette=[],
                style_profile={},
                completeness_score=0,
                recommendations=[f"Error during analysis: {str(e)}"]
            )
    
    async def chat(
        self,
        message: str,
        wardrobe: WardrobeAnalysis = None,
        conversation_history: List[Dict] = None,
        context: Dict = None
    ) -> Dict:
        """
        💬 Conversational fashion AI - Answer questions about outfits.
        
        Args:
            message: User's question (e.g., "What should I wear tomorrow?")
            wardrobe: User's wardrobe analysis
            conversation_history: Previous chat messages
            context: Additional context (weather, occasion, etc.)
            
        Returns:
            Dict with response, suggested outfits, and follow-up questions
        """
        self._load_gemini()
        
        # Build context prompt
        wardrobe_context = ""
        if wardrobe:
            wardrobe_context = f"""
User's Wardrobe:
- Total items: {wardrobe.total_items}
- Categories: {json.dumps(wardrobe.categories)}
- Style profile: {json.dumps(wardrobe.style_profile)}
- Color palette: {', '.join(wardrobe.color_palette)}

Available items:
"""
            for item in wardrobe.items[:20]:  # Limit for context
                wardrobe_context += f"- {item.specific_type or item.category} ({item.primary_color})\n"
        
        weather_context = ""
        if context and "weather" in context:
            w = context["weather"]
            weather_context = f"\nWeather: {w.get('temp', 'unknown')}°C, {w.get('condition', 'unknown')}"
        
        history_context = ""
        if conversation_history:
            for msg in conversation_history[-5:]:  # Last 5 messages
                role = msg.get("role", "user")
                content = msg.get("content", "")
                history_context += f"\n{role}: {content}"
        
        prompt = f"""You are a personal fashion stylist AI assistant. Be friendly, helpful, and give specific outfit recommendations.

{wardrobe_context}
{weather_context}

Previous conversation:{history_context}

User: {message}

Respond with:
1. A helpful, friendly answer addressing the user's question
2. Specific outfit recommendations from their wardrobe if relevant
3. 2-3 follow-up questions they might want to ask

Format as JSON:
{{
    "response": "Your friendly response here...",
    "suggestedOutfits": [
        {{
            "items": [
                {{"category": "...", "description": "..."}},
                ...
            ],
            "reasoning": "Why this outfit works..."
        }}
    ],
    "followUpQuestions": ["Question 1?", "Question 2?"]
}}"""

        try:
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text
            
            # Parse JSON response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                result = json.loads(response_text[json_start:json_end])
            else:
                result = {
                    "response": response_text,
                    "suggestedOutfits": [],
                    "followUpQuestions": []
                }
            
            return {
                "success": True,
                "response": result.get("response", "I'm not sure how to help with that."),
                "suggestedOutfits": result.get("suggestedOutfits", []),
                "followUpQuestions": result.get("followUpQuestions", [])
            }
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            return {
                "success": False,
                "response": "I apologize, but I'm having trouble processing your request. Please try again.",
                "suggestedOutfits": [],
                "followUpQuestions": [],
                "error": str(e)
            }
    
    async def recommend_outfit(
        self,
        wardrobe: WardrobeAnalysis,
        occasion: str,
        weather: Dict = None,
        preferences: UserPreferences = None,
        max_outfits: int = 3
    ) -> List[Outfit]:
        """
        👔 Generate intelligent outfit recommendations.
        
        Combines AI understanding with fashion rules to create
        perfectly matched outfits.
        
        Args:
            wardrobe: User's wardrobe analysis
            occasion: Event/occasion (e.g., "job interview", "casual dinner")
            weather: Weather conditions
            preferences: User preferences
            max_outfits: Maximum outfits to return
            
        Returns:
            List of Outfit recommendations with reasoning
        """
        self._load_gemini()
        
        # Build the prompt
        items_description = ""
        for i, item in enumerate(wardrobe.items):
            items_description += f"{i+1}. {item.specific_type or item.category} - {item.primary_color}"
            if item.pattern:
                items_description += f", {item.pattern}"
            if item.material:
                items_description += f", {item.material}"
            items_description += f" (tags: {', '.join(item.style_tags)})\n"
        
        weather_info = ""
        if weather:
            weather_info = f"\nWeather: {weather.get('temp', 20)}°C, {weather.get('condition', 'clear')}"
        
        preference_info = ""
        if preferences:
            if preferences.preferred_styles:
                preference_info += f"\nPreferred styles: {', '.join(preferences.preferred_styles)}"
            if preferences.avoid_colors:
                preference_info += f"\nColors to avoid: {', '.join(preferences.avoid_colors)}"
        
        prompt = f"""You are an expert fashion stylist. Create {max_outfits} outfit recommendations for this occasion.

Occasion: {occasion}
{weather_info}
{preference_info}

Available wardrobe items:
{items_description}

For each outfit, consider:
1. Appropriateness for the occasion
2. Color harmony (complementary, analogous, or neutral combos)
3. Style consistency
4. Weather suitability
5. User preferences

Return as JSON array:
[
    {{
        "items": [1, 4, 7],  // item numbers from the list
        "confidence": 0.92,
        "reasoning": "Detailed explanation of why this outfit works...",
        "style": "smart casual",
        "colorHarmony": "complementary - navy and cream"
    }}
]"""

        try:
            response = self.gemini_model.generate_content(prompt)
            response_text = response.text
            
            # Parse JSON array
            json_start = response_text.find('[')
            json_end = response_text.rfind(']') + 1
            if json_start != -1 and json_end > json_start:
                outfit_data = json.loads(response_text[json_start:json_end])
            else:
                outfit_data = []
            
            # Convert to Outfit objects
            outfits = []
            for data in outfit_data[:max_outfits]:
                item_indices = data.get("items", [])
                outfit_items = []
                for idx in item_indices:
                    if isinstance(idx, int) and 1 <= idx <= len(wardrobe.items):
                        outfit_items.append(wardrobe.items[idx - 1])
                
                if outfit_items:
                    outfit = Outfit(
                        items=outfit_items,
                        confidence=data.get("confidence", 0.8),
                        reasoning=data.get("reasoning", ""),
                        occasion=occasion,
                        style=data.get("style", ""),
                        color_harmony=data.get("colorHarmony", "")
                    )
                    outfits.append(outfit)
            
            return outfits
            
        except Exception as e:
            logger.error(f"Outfit recommendation failed: {e}")
            return []
    
    def semantic_search(
        self,
        query: str,
        wardrobe: WardrobeAnalysis,
        top_k: int = 5
    ) -> List[ClothingItem]:
        """
        🔎 Find items matching natural language description.
        
        Uses Fashion-CLIP to semantically search the wardrobe.
        
        Args:
            query: Natural language query (e.g., "warm winter jacket")
            wardrobe: User's wardrobe
            top_k: Number of results to return
            
        Returns:
            List of matching ClothingItem objects
        """
        self._load_clip()
        
        if not self.clip_encoder:
            # Fallback to keyword matching if CLIP unavailable
            query_lower = query.lower()
            matches = []
            for item in wardrobe.items:
                score = 0
                if query_lower in item.specific_type.lower():
                    score += 2
                if query_lower in item.primary_color.lower():
                    score += 1
                if any(query_lower in tag.lower() for tag in item.style_tags):
                    score += 1
                if score > 0:
                    matches.append((item, score))
            
            matches.sort(key=lambda x: x[1], reverse=True)
            return [m[0] for m in matches[:top_k]]
        
        # TODO: Implement CLIP-based semantic search
        # This would require embeddings stored for each item
        return []
    
    def check_color_harmony(self, colors: List[str]) -> Tuple[bool, str]:
        """
        Check if colors work well together.
        
        Args:
            colors: List of color names
            
        Returns:
            Tuple of (is_harmonious, harmony_type)
        """
        colors_lower = [c.lower() for c in colors]
        
        # Check if all neutrals
        neutrals = self.COLOR_HARMONIES["neutral_combos"]
        if all(any(n in c for n in neutrals) for c in colors_lower):
            return True, "neutral_palette"
        
        # Check known safe combinations
        for safe_combo in self.COLOR_HARMONIES["always_safe"]:
            if all(any(s in c for s in safe_combo) for c in colors_lower):
                return True, f"classic_combo_{safe_combo[0]}_{safe_combo[1]}"
        
        # Check complementary
        for base, complements in self.COLOR_HARMONIES["complementary"].items():
            if any(base in c for c in colors_lower):
                if any(any(comp in c for c in colors_lower) for comp in complements):
                    return True, f"complementary_{base}"
        
        return True, "unknown"  # Default to allowing it
    
    def _color_name_to_hex(self, color_name: str) -> str:
        """Convert color name to hex code."""
        color_map = {
            "black": "#000000",
            "white": "#FFFFFF",
            "navy": "#1B3A57",
            "navy blue": "#1B3A57",
            "red": "#E53935",
            "blue": "#2196F3",
            "green": "#4CAF50",
            "yellow": "#FFC107",
            "gray": "#9E9E9E",
            "grey": "#9E9E9E",
            "brown": "#795548",
            "beige": "#F5F5DC",
            "cream": "#FFFDD0",
            "pink": "#E91E63",
            "purple": "#9C27B0",
            "orange": "#FF9800",
            "olive": "#808000",
            "denim": "#1560BD",
            "khaki": "#C3B091"
        }
        return color_map.get(color_name.lower(), "#808080")


# Singleton instance
_multimodal_ai_instance = None


def get_multimodal_ai() -> MultimodalFashionAI:
    """Get singleton instance of MultimodalFashionAI."""
    global _multimodal_ai_instance
    if _multimodal_ai_instance is None:
        _multimodal_ai_instance = MultimodalFashionAI()
    return _multimodal_ai_instance
