"""
📊 LangGraph Agent State
Shared state for the Digital Stylist conversation

State includes:
- User preferences (style, colors, sizes)
- Wardrobe inventory
- Current context (occasion, weather)
- Conversation history
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class StyleCategory(Enum):
    """User style preferences."""
    CASUAL = "casual"
    FORMAL = "formal"
    BUSINESS = "business"
    SPORTY = "sporty"
    BOHEMIAN = "bohemian"
    MINIMALIST = "minimalist"
    STREETWEAR = "streetwear"
    CLASSIC = "classic"
    EDGY = "edgy"
    ROMANTIC = "romantic"


class Occasion(Enum):
    """Event occasions."""
    EVERYDAY = "everyday"
    WORK = "work"
    DATE_NIGHT = "date_night"
    PARTY = "party"
    WEDDING = "wedding"
    INTERVIEW = "interview"
    CASUAL_OUTING = "casual_outing"
    WORKOUT = "workout"
    TRAVEL = "travel"
    FORMAL_EVENT = "formal_event"


@dataclass
class UserPreferences:
    """User style preferences and constraints."""
    user_id: str = ""
    
    # Style
    preferred_styles: List[StyleCategory] = field(default_factory=list)
    disliked_styles: List[StyleCategory] = field(default_factory=list)
    
    # Colors
    favorite_colors: List[str] = field(default_factory=list)
    colors_to_avoid: List[str] = field(default_factory=list)
    
    # Sizing
    top_size: str = ""
    bottom_size: str = ""
    shoe_size: str = ""
    
    # Body
    body_type: str = ""  # hourglass, pear, apple, rectangle, inverted_triangle
    height: str = ""  # short, average, tall
    
    # Budget
    budget_range: str = ""  # budget, mid_range, luxury
    
    # Sustainability
    prefer_sustainable: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "userId": self.user_id,
            "preferredStyles": [s.value for s in self.preferred_styles],
            "dislikedStyles": [s.value for s in self.disliked_styles],
            "favoriteColors": self.favorite_colors,
            "colorsToAvoid": self.colors_to_avoid,
            "topSize": self.top_size,
            "bottomSize": self.bottom_size,
            "shoeSize": self.shoe_size,
            "bodyType": self.body_type,
            "height": self.height,
            "budgetRange": self.budget_range,
            "preferSustainable": self.prefer_sustainable
        }


@dataclass
class WardrobeItem:
    """Single item in user's wardrobe."""
    item_id: str
    category: str  # tops, bottoms, shoes, accessories
    specific_type: str  # t-shirt, jeans, sneakers
    
    # Attributes
    primary_color: str = ""
    colors: List[str] = field(default_factory=list)
    pattern: str = ""
    material: str = ""
    
    # Style
    formality: int = 5  # 1-10 (1=very casual, 10=very formal)
    style_tags: List[str] = field(default_factory=list)
    
    # Sustainability
    eco_score: str = ""  # A-F
    
    # Images
    thumbnail_url: str = ""
    cutout_b64: str = ""
    scan_3d_url: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "itemId": self.item_id,
            "category": self.category,
            "specificType": self.specific_type,
            "primaryColor": self.primary_color,
            "colors": self.colors,
            "pattern": self.pattern,
            "material": self.material,
            "formality": self.formality,
            "styleTags": self.style_tags,
            "ecoScore": self.eco_score,
            "thumbnailUrl": self.thumbnail_url
        }


@dataclass
class WeatherContext:
    """Current weather for context-aware styling."""
    temperature: float = 20.0  # Celsius
    condition: str = "clear"  # clear, cloudy, rain, snow
    humidity: float = 50.0


@dataclass
class OutfitRecommendation:
    """Single outfit recommendation."""
    recommendation_id: str
    items: List[WardrobeItem]
    
    # Metadata
    occasion: str = ""
    style_description: str = ""
    confidence: float = 0.5
    
    # Visualization
    vton_image_b64: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "recommendationId": self.recommendation_id,
            "items": [item.to_dict() for item in self.items],
            "occasion": self.occasion,
            "styleDescription": self.style_description,
            "confidence": self.confidence,
            "vtonImage": self.vton_image_b64
        }


@dataclass
class StylistState:
    """
    Complete state for Digital Stylist agent.
    
    This state is persisted across conversation turns
    and passed between graph nodes.
    """
    # Session
    session_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    # User
    user_preferences: UserPreferences = field(default_factory=UserPreferences)
    wardrobe_inventory: List[WardrobeItem] = field(default_factory=list)
    
    # Context
    current_occasion: Optional[Occasion] = None
    weather: WeatherContext = field(default_factory=WeatherContext)
    special_requirements: List[str] = field(default_factory=list)
    
    # Conversation
    messages: List[Dict[str, str]] = field(default_factory=list)
    current_query: str = ""
    
    # Recommendations
    current_recommendations: List[OutfitRecommendation] = field(default_factory=list)
    accepted_recommendation: Optional[OutfitRecommendation] = None
    
    # Pending actions
    pending_uploads: List[str] = field(default_factory=list)  # Base64 images to process
    pending_vton_request: Optional[Dict] = None
    
    # Agent state
    current_node: str = "supervisor"
    iteration_count: int = 0
    should_end: bool = False
    
    def add_message(self, role: str, content: str):
        """Add message to conversation history."""
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_wardrobe_by_category(self, category: str) -> List[WardrobeItem]:
        """Get wardrobe items by category."""
        return [item for item in self.wardrobe_inventory if item.category == category]
    
    def get_conversation_context(self, max_messages: int = 10) -> str:
        """Get recent conversation as context string."""
        recent = self.messages[-max_messages:]
        return "\n".join([
            f"{m['role'].upper()}: {m['content']}"
            for m in recent
        ])
    
    def to_dict(self) -> Dict:
        return {
            "sessionId": self.session_id,
            "userPreferences": self.user_preferences.to_dict(),
            "wardrobeCount": len(self.wardrobe_inventory),
            "currentOccasion": self.current_occasion.value if self.current_occasion else None,
            "weather": {
                "temperature": self.weather.temperature,
                "condition": self.weather.condition
            },
            "messageCount": len(self.messages),
            "currentNode": self.current_node,
            "iterationCount": self.iteration_count
        }
