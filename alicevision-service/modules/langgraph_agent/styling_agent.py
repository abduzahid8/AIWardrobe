"""
👗 Styling Agent
AI-powered outfit recommendation engine

Uses:
- User preferences
- Wardrobe inventory
- Occasion context
- Style rules
"""

from typing import Dict, List
import logging
import random
import uuid

from .state import StylistState, WardrobeItem, OutfitRecommendation, Occasion

logger = logging.getLogger(__name__)


class StylingAgent:
    """
    Styling agent that generates outfit recommendations.
    
    Considers:
    - User style preferences
    - Current occasion
    - Weather conditions
    - Color harmony
    - Formality matching
    """
    
    # Color harmony rules
    COLOR_COMPLEMENTS = {
        "red": ["green", "navy", "white", "black"],
        "blue": ["orange", "white", "beige", "brown"],
        "green": ["red", "brown", "cream", "white"],
        "yellow": ["purple", "navy", "gray", "black"],
        "purple": ["yellow", "cream", "white", "gray"],
        "orange": ["blue", "navy", "brown", "white"],
        "pink": ["gray", "navy", "white", "black"],
        "black": ["white", "red", "pink", "gold"],
        "white": ["navy", "black", "red", "blue"],
        "navy": ["white", "cream", "coral", "gold"],
        "beige": ["navy", "brown", "white", "burgundy"],
        "brown": ["cream", "blue", "white", "green"]
    }
    
    # Formality levels by occasion
    OCCASION_FORMALITY = {
        Occasion.EVERYDAY: (1, 4),
        Occasion.WORK: (5, 7),
        Occasion.DATE_NIGHT: (5, 8),
        Occasion.PARTY: (4, 8),
        Occasion.WEDDING: (7, 10),
        Occasion.INTERVIEW: (7, 9),
        Occasion.CASUAL_OUTING: (2, 5),
        Occasion.WORKOUT: (1, 2),
        Occasion.TRAVEL: (2, 5),
        Occasion.FORMAL_EVENT: (8, 10)
    }
    
    def __init__(self):
        """Initialize styling agent."""
        self._qwen = None
    
    def _get_qwen(self):
        """Lazy load Qwen for advanced reasoning."""
        if self._qwen is None:
            try:
                from modules.qwen_vl_reasoning import get_qwen_reasoning
                self._qwen = get_qwen_reasoning()
            except:
                pass
        return self._qwen
    
    def process(self, state: StylistState) -> StylistState:
        """
        Generate outfit recommendations.
        
        Args:
            state: Current state with wardrobe and context
            
        Returns:
            Updated state with recommendations
        """
        if not state.wardrobe_inventory:
            state.add_message(
                "assistant",
                "Your wardrobe is empty! Let's add some items first. "
                "Upload photos of your clothes and I'll analyze them."
            )
            state.current_node = "supervisor"
            return state
        
        # Get occasion context
        occasion = state.current_occasion or Occasion.EVERYDAY
        formality_range = self.OCCASION_FORMALITY.get(occasion, (3, 6))
        
        # Generate recommendations
        recommendations = self._generate_outfits(
            state.wardrobe_inventory,
            state.user_preferences,
            formality_range,
            state.weather.temperature,
            num_outfits=3
        )
        
        state.current_recommendations = recommendations
        
        # Generate response
        if recommendations:
            response = self._format_recommendations(recommendations, occasion)
        else:
            response = (
                "I couldn't find suitable combinations in your wardrobe. "
                "You might need more items for this occasion!"
            )
        
        state.add_message("assistant", response)
        state.current_node = "supervisor"
        
        return state
    
    def _generate_outfits(
        self,
        wardrobe: List[WardrobeItem],
        preferences,
        formality_range,
        temperature: float,
        num_outfits: int = 3
    ) -> List[OutfitRecommendation]:
        """Generate outfit combinations."""
        recommendations = []
        
        # Categorize wardrobe
        tops = [item for item in wardrobe if item.category == "tops"]
        bottoms = [item for item in wardrobe if item.category == "bottoms"]
        footwear = [item for item in wardrobe if item.category == "footwear"]
        
        if not tops and not bottoms:
            return []
        
        # Generate combinations
        seen_combos = set()
        attempts = 0
        max_attempts = 50
        
        while len(recommendations) < num_outfits and attempts < max_attempts:
            attempts += 1
            
            # Select items
            outfit_items = []
            
            if tops:
                top = random.choice(tops)
                outfit_items.append(top)
            
            if bottoms:
                bottom = random.choice(bottoms)
                outfit_items.append(bottom)
            
            if footwear:
                shoes = random.choice(footwear)
                outfit_items.append(shoes)
            
            if not outfit_items:
                continue
            
            # Create combo key
            combo_key = tuple(sorted([item.item_id for item in outfit_items]))
            if combo_key in seen_combos:
                continue
            seen_combos.add(combo_key)
            
            # Score outfit
            score = self._score_outfit(
                outfit_items,
                preferences,
                formality_range,
                temperature
            )
            
            if score > 0.5:
                rec = OutfitRecommendation(
                    recommendation_id=str(uuid.uuid4())[:8],
                    items=outfit_items,
                    style_description=self._describe_outfit(outfit_items),
                    confidence=score
                )
                recommendations.append(rec)
        
        # Sort by confidence
        recommendations.sort(key=lambda r: r.confidence, reverse=True)
        
        return recommendations[:num_outfits]
    
    def _score_outfit(
        self,
        items: List[WardrobeItem],
        preferences,
        formality_range,
        temperature: float
    ) -> float:
        """Score an outfit combination."""
        score = 0.5  # Base score
        
        # Color harmony
        colors = [item.primary_color.lower() for item in items if item.primary_color]
        if len(colors) >= 2:
            main_color = colors[0]
            for other_color in colors[1:]:
                complements = self.COLOR_COMPLEMENTS.get(main_color, [])
                if other_color in complements or other_color == main_color:
                    score += 0.1
                elif other_color in ["black", "white", "gray", "navy"]:
                    score += 0.05  # Neutrals always work
        
        # Formality matching
        min_form, max_form = formality_range
        avg_formality = sum(item.formality for item in items) / len(items)
        if min_form <= avg_formality <= max_form:
            score += 0.15
        elif abs(avg_formality - (min_form + max_form) / 2) <= 2:
            score += 0.05
        
        # Temperature appropriateness
        heavy_materials = ["wool", "leather", "fleece", "denim"]
        light_materials = ["cotton", "linen", "silk", "chiffon"]
        
        materials = [item.material.lower() for item in items if item.material]
        
        if temperature < 15:  # Cold
            if any(m in heavy_materials for m in materials):
                score += 0.1
        elif temperature > 25:  # Hot
            if any(m in light_materials for m in materials):
                score += 0.1
        
        # User preferences
        if preferences:
            # Favorite colors
            for item in items:
                if item.primary_color and item.primary_color.lower() in [c.lower() for c in preferences.favorite_colors]:
                    score += 0.1
            
            # Avoid colors
            for item in items:
                if item.primary_color and item.primary_color.lower() in [c.lower() for c in preferences.colors_to_avoid]:
                    score -= 0.15
        
        return min(1.0, max(0.0, score))
    
    def _describe_outfit(self, items: List[WardrobeItem]) -> str:
        """Generate natural language description of outfit."""
        descriptions = []
        
        for item in items:
            color = item.primary_color or ""
            garment = item.specific_type or item.category
            
            if color and garment:
                descriptions.append(f"{color} {garment}")
            elif garment:
                descriptions.append(garment)
        
        if not descriptions:
            return "A stylish combination"
        
        if len(descriptions) == 1:
            return f"A {descriptions[0]}"
        elif len(descriptions) == 2:
            return f"{descriptions[0].title()} paired with {descriptions[1]}"
        else:
            return f"{descriptions[0].title()} with {descriptions[1]} and {descriptions[2]}"
    
    def _format_recommendations(
        self,
        recommendations: List[OutfitRecommendation],
        occasion: Occasion
    ) -> str:
        """Format recommendations as response."""
        occasion_name = occasion.value.replace("_", " ").title()
        
        response = f"Here are my top outfit suggestions for {occasion_name}:\n\n"
        
        for i, rec in enumerate(recommendations, 1):
            confidence_emoji = "⭐⭐⭐" if rec.confidence > 0.7 else "⭐⭐" if rec.confidence > 0.5 else "⭐"
            
            response += f"**Outfit {i}** {confidence_emoji}\n"
            response += f"{rec.style_description}\n"
            
            for item in rec.items:
                response += f"  • {item.specific_type} ({item.primary_color})\n"
            
            response += "\n"
        
        response += "Would you like to try any of these on virtually? Just say 'try on outfit 1' (or 2, 3)!"
        
        return response


def create_styling_node():
    """Create styling node for LangGraph."""
    agent = StylingAgent()
    
    def styling_node(state: Dict) -> Dict:
        from .state import StylistState
        stylist_state = StylistState(**state) if isinstance(state, dict) else state
        updated_state = agent.process(stylist_state)
        return updated_state.to_dict()
    
    return styling_node
