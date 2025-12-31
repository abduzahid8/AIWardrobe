"""
🎭 Simulation Tool
Virtual try-on and 3D visualization

Uses:
- CatVTON for virtual try-on
- 3DGS for 3D viewing
"""

from typing import Dict, Optional
import logging

from .state import StylistState, OutfitRecommendation

logger = logging.getLogger(__name__)


class SimulationTool:
    """
    Simulation tool for virtual try-on and visualization.
    
    Capabilities:
    - Virtual try-on (garment on user avatar)
    - 3D visualization of scanned items
    - Outfit composition rendering
    """
    
    def __init__(self):
        """Initialize simulation tool."""
        self._vton = None
    
    def _get_vton(self):
        """Lazy load VTON engine."""
        if self._vton is None:
            try:
                from modules.catvton_tryon import get_vton_engine
                self._vton = get_vton_engine()
            except Exception as e:
                logger.warning(f"VTON not available: {e}")
        return self._vton
    
    def process(self, state: StylistState) -> StylistState:
        """
        Process simulation request.
        
        Args:
            state: Current state with VTON request
            
        Returns:
            Updated state with result
        """
        vton_request = state.pending_vton_request
        
        if not vton_request:
            # Try to infer from message
            query = state.current_query.lower()
            
            if "try on" in query or "try" in query:
                # Parse outfit number
                outfit_num = self._extract_outfit_number(query)
                
                if outfit_num and outfit_num <= len(state.current_recommendations):
                    outfit = state.current_recommendations[outfit_num - 1]
                    result = self._try_on_outfit(outfit)
                    
                    if result:
                        outfit.vton_image_b64 = result
                        state.add_message(
                            "assistant",
                            f"Here's how Outfit {outfit_num} would look on you! "
                            "[Virtual try-on image generated]\n\n"
                            "Do you love it? Say 'perfect' to save this look, or "
                            "ask me for different suggestions!"
                        )
                    else:
                        state.add_message(
                            "assistant",
                            "I couldn't generate the try-on visualization. "
                            "Make sure you have a full-body photo uploaded as your avatar."
                        )
                else:
                    state.add_message(
                        "assistant",
                        "Please specify which outfit to try on (1, 2, or 3)."
                    )
            else:
                state.add_message(
                    "assistant",
                    "I can help you visualize outfits! Just say 'try on outfit 1' "
                    "after I give you recommendations."
                )
        else:
            # Process explicit VTON request
            person_image = vton_request.get("person_image")
            garment_image = vton_request.get("garment_image")
            garment_type = vton_request.get("garment_type", "upper_body")
            
            result = self._run_vton(person_image, garment_image, garment_type)
            
            if result:
                state.add_message(
                    "assistant",
                    "Virtual try-on complete! [Image generated]\n\n"
                    "What do you think? Would you like to try something else?"
                )
            else:
                state.add_message(
                    "assistant",
                    "Sorry, the virtual try-on encountered an issue. "
                    "Please try with a different image."
                )
            
            state.pending_vton_request = None
        
        state.current_node = "supervisor"
        return state
    
    def _extract_outfit_number(self, query: str) -> Optional[int]:
        """Extract outfit number from query."""
        import re
        
        # Match "outfit 1", "look 2", "#3", etc.
        match = re.search(r'(?:outfit|look|option|#)\s*(\d+)', query)
        if match:
            return int(match.group(1))
        
        # Match "first", "second", "third"
        ordinals = {"first": 1, "second": 2, "third": 3, "1st": 1, "2nd": 2, "3rd": 3}
        for word, num in ordinals.items():
            if word in query:
                return num
        
        return None
    
    def _try_on_outfit(self, outfit: OutfitRecommendation) -> Optional[str]:
        """Generate try-on for outfit."""
        # Get first garment with cutout
        garment_image = None
        for item in outfit.items:
            if item.cutout_b64:
                garment_image = item.cutout_b64
                break
        
        if not garment_image:
            logger.warning("No garment image available for try-on")
            return None
        
        # For now, return placeholder (actual VTON would need user avatar)
        # In production, user would have uploaded a photo of themselves
        return "try_on_placeholder"
    
    def _run_vton(
        self,
        person_image: str,
        garment_image: str,
        garment_type: str
    ) -> Optional[str]:
        """Run actual VTON inference."""
        vton = self._get_vton()
        
        if not vton:
            return None
        
        try:
            result = vton.try_on(
                person_image,
                garment_image,
                garment_type
            )
            
            if result.success:
                return result.result_image_b64
            return None
            
        except Exception as e:
            logger.error(f"VTON failed: {e}")
            return None


def create_simulation_node():
    """Create simulation node for LangGraph."""
    tool = SimulationTool()
    
    def simulation_node(state: Dict) -> Dict:
        from .state import StylistState
        stylist_state = StylistState(**state) if isinstance(state, dict) else state
        updated_state = tool.process(stylist_state)
        return updated_state.to_dict()
    
    return simulation_node
