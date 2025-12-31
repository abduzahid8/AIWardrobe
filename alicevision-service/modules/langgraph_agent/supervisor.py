"""
🎯 Supervisor Node
Routes user requests to appropriate agent nodes

Responsibilities:
- Analyze user intent
- Route to perception, styling, or simulation
- Handle conversation flow
"""

from typing import Dict, Tuple, Optional
import logging
import re

from .state import StylistState, Occasion

logger = logging.getLogger(__name__)


class StylistSupervisor:
    """
    Supervisor node that routes user requests.
    
    Routes to:
    - perception_tool: When user uploads images
    - styling_agent: When user asks for recommendations
    - simulation_tool: When user wants to try on
    - END: When conversation should end
    """
    
    # Intent patterns
    UPLOAD_PATTERNS = [
        r"upload", r"add.*wardrobe", r"scan", r"take.*photo",
        r"here.*is", r"this.*is", r"new.*item"
    ]
    
    STYLING_PATTERNS = [
        r"what.*wear", r"outfit", r"suggest", r"recommend",
        r"style", r"help.*dress", r"match", r"pair.*with",
        r"occasion", r"event", r"date", r"interview", r"party"
    ]
    
    TRYON_PATTERNS = [
        r"try.*on", r"how.*look", r"see.*on.*me", r"visualize",
        r"show.*me.*wearing"
    ]
    
    END_PATTERNS = [
        r"bye", r"goodbye", r"thank", r"that.*all", r"done",
        r"perfect", r"love.*it", r"great"
    ]
    
    def __init__(self):
        """Initialize supervisor."""
        self.intent_weights = {}
    
    def route(self, state: StylistState) -> Tuple[str, StylistState]:
        """
        Route user request to appropriate node.
        
        Args:
            state: Current conversation state
            
        Returns:
            Tuple of (next_node_name, updated_state)
        """
        query = state.current_query.lower()
        
        # Check for pending uploads (images in request)
        if state.pending_uploads:
            logger.info("Routing to perception_tool (pending uploads)")
            state.current_node = "perception_tool"
            return "perception_tool", state
        
        # Check for VTON request
        if state.pending_vton_request:
            logger.info("Routing to simulation_tool (VTON request)")
            state.current_node = "simulation_tool"
            return "simulation_tool", state
        
        # Intent classification
        intent = self._classify_intent(query)
        
        if intent == "upload":
            state.current_node = "perception_tool"
            return "perception_tool", state
        
        elif intent == "styling":
            # Extract occasion if mentioned
            occasion = self._extract_occasion(query)
            if occasion:
                state.current_occasion = occasion
            
            state.current_node = "styling_agent"
            return "styling_agent", state
        
        elif intent == "tryon":
            state.current_node = "simulation_tool"
            return "simulation_tool", state
        
        elif intent == "end":
            state.should_end = True
            return "END", state
        
        else:
            # Default to styling for ambiguous queries
            state.current_node = "styling_agent"
            return "styling_agent", state
    
    def _classify_intent(self, query: str) -> str:
        """Classify user intent from query."""
        scores = {
            "upload": 0,
            "styling": 0,
            "tryon": 0,
            "end": 0
        }
        
        for pattern in self.UPLOAD_PATTERNS:
            if re.search(pattern, query):
                scores["upload"] += 1
        
        for pattern in self.STYLING_PATTERNS:
            if re.search(pattern, query):
                scores["styling"] += 1
        
        for pattern in self.TRYON_PATTERNS:
            if re.search(pattern, query):
                scores["tryon"] += 1
        
        for pattern in self.END_PATTERNS:
            if re.search(pattern, query):
                scores["end"] += 1
        
        # Return highest scoring intent
        max_intent = max(scores, key=scores.get)
        
        if scores[max_intent] == 0:
            return "styling"  # Default
        
        return max_intent
    
    def _extract_occasion(self, query: str) -> Optional[Occasion]:
        """Extract occasion from query."""
        occasion_map = {
            r"work|office|meeting|business": Occasion.WORK,
            r"date|romantic|dinner": Occasion.DATE_NIGHT,
            r"party|club|night.*out": Occasion.PARTY,
            r"wedding|ceremony": Occasion.WEDDING,
            r"interview|job": Occasion.INTERVIEW,
            r"casual|everyday|daily": Occasion.EVERYDAY,
            r"workout|gym|exercise": Occasion.WORKOUT,
            r"travel|trip|vacation": Occasion.TRAVEL,
            r"formal|gala|black.*tie": Occasion.FORMAL_EVENT
        }
        
        for pattern, occasion in occasion_map.items():
            if re.search(pattern, query.lower()):
                return occasion
        
        return None
    
    def generate_response(self, state: StylistState) -> str:
        """Generate supervisor response for conversation."""
        if state.should_end:
            return "Thank you for using the Digital Stylist! Have a fashionable day! 👗✨"
        
        if not state.current_query:
            return (
                "Hello! I'm your AI Digital Stylist. I can help you:\n"
                "• Build outfits from your wardrobe\n"
                "• Get style recommendations for any occasion\n"
                "• Virtually try on clothes\n"
                "• Scan and analyze new items\n\n"
                "What would you like help with today?"
            )
        
        return ""  # Other nodes will respond


def create_supervisor_node():
    """Create supervisor node for LangGraph."""
    supervisor = StylistSupervisor()
    
    def supervisor_node(state: Dict) -> Dict:
        """LangGraph node function."""
        stylist_state = StylistState(**state) if isinstance(state, dict) else state
        
        next_node, updated_state = supervisor.route(stylist_state)
        
        return {
            "next": next_node,
            **updated_state.to_dict()
        }
    
    return supervisor_node
