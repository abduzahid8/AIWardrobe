"""
👁️ Perception Tool
Handle image uploads and garment analysis

Uses:
- Florence-2 for detection
- Qwen2.5-VL for reasoning
- SAM 2 for segmentation
"""

from typing import Dict, List
import logging
import uuid

from .state import StylistState, WardrobeItem

logger = logging.getLogger(__name__)


class PerceptionTool:
    """
    Perception tool for image analysis.
    
    Processes uploaded images to:
    - Detect clothing items
    - Extract attributes (color, material, pattern)
    - Create cutouts
    - Add to wardrobe inventory
    """
    
    def __init__(self):
        """Initialize perception tool."""
        self._florence = None
        self._qwen = None
    
    def _get_florence(self):
        """Lazy load Florence-2."""
        if self._florence is None:
            try:
                from modules.florence2_perception import get_florence2_perception
                self._florence = get_florence2_perception()
            except Exception as e:
                logger.warning(f"Florence-2 not available: {e}")
        return self._florence
    
    def _get_qwen(self):
        """Lazy load Qwen."""
        if self._qwen is None:
            try:
                from modules.qwen_vl_reasoning import get_qwen_reasoning
                self._qwen = get_qwen_reasoning()
            except Exception as e:
                logger.warning(f"Qwen not available: {e}")
        return self._qwen
    
    def process(self, state: StylistState) -> StylistState:
        """
        Process pending image uploads.
        
        Args:
            state: Current state with pending_uploads
            
        Returns:
            Updated state with new wardrobe items
        """
        if not state.pending_uploads:
            state.add_message("assistant", "No images to process.")
            return state
        
        new_items = []
        
        for image_b64 in state.pending_uploads:
            items = self._analyze_image(image_b64)
            new_items.extend(items)
        
        # Add to wardrobe
        state.wardrobe_inventory.extend(new_items)
        
        # Clear pending
        state.pending_uploads = []
        
        # Generate response
        if new_items:
            item_names = [f"{item.specific_type} ({item.primary_color})" for item in new_items]
            response = f"I've added {len(new_items)} items to your wardrobe:\n"
            response += "\n".join([f"• {name}" for name in item_names])
            response += "\n\nWould you like styling suggestions for any of these?"
        else:
            response = "I couldn't detect any clothing items in the image. Please try again with a clearer photo."
        
        state.add_message("assistant", response)
        state.current_node = "supervisor"
        
        return state
    
    def _analyze_image(self, image_b64: str) -> List[WardrobeItem]:
        """Analyze single image and extract items."""
        items = []
        
        # Try Florence-2 first
        florence = self._get_florence()
        if florence:
            try:
                result = florence.analyze_garment(image_b64)
                
                if result.success:
                    item = WardrobeItem(
                        item_id=str(uuid.uuid4())[:8],
                        category=self._get_category(result.garment_type),
                        specific_type=result.garment_type or "unknown",
                        primary_color=result.colors[0] if result.colors else "unknown",
                        colors=result.colors,
                        pattern=result.patterns[0] if result.patterns else "solid",
                        material=result.materials[0] if result.materials else "unknown",
                        style_tags=result.style_tags,
                        cutout_b64=image_b64  # Use original for now
                    )
                    items.append(item)
                    return items
                    
            except Exception as e:
                logger.warning(f"Florence-2 analysis failed: {e}")
        
        # Fallback to Qwen
        qwen = self._get_qwen()
        if qwen:
            try:
                result = qwen.extract_attributes(image_b64)
                
                if result.success and result.structured_data:
                    data = result.structured_data
                    item = WardrobeItem(
                        item_id=str(uuid.uuid4())[:8],
                        category=self._get_category(data.get("type", "")),
                        specific_type=data.get("type", "unknown"),
                        primary_color=data.get("primaryColor", "unknown"),
                        colors=data.get("secondaryColors", []),
                        pattern=data.get("pattern", {}).get("type", "solid") if isinstance(data.get("pattern"), dict) else "solid",
                        material=data.get("material", {}).get("type", "unknown") if isinstance(data.get("material"), dict) else "unknown",
                        cutout_b64=image_b64
                    )
                    items.append(item)
                    return items
                    
            except Exception as e:
                logger.warning(f"Qwen analysis failed: {e}")
        
        # Fallback: basic item
        items.append(WardrobeItem(
            item_id=str(uuid.uuid4())[:8],
            category="unknown",
            specific_type="clothing item",
            primary_color="unknown",
            cutout_b64=image_b64
        ))
        
        return items
    
    def _get_category(self, garment_type: str) -> str:
        """Map garment type to category."""
        garment_type = garment_type.lower()
        
        tops = ["shirt", "t-shirt", "blouse", "sweater", "hoodie", "jacket", "coat", "vest", "top"]
        bottoms = ["pants", "jeans", "shorts", "skirt", "trousers"]
        footwear = ["shoes", "sneakers", "boots", "sandals", "heels", "loafers"]
        accessories = ["bag", "hat", "scarf", "belt", "watch", "jewelry", "glasses"]
        
        for word in tops:
            if word in garment_type:
                return "tops"
        
        for word in bottoms:
            if word in garment_type:
                return "bottoms"
        
        for word in footwear:
            if word in garment_type:
                return "footwear"
        
        for word in accessories:
            if word in garment_type:
                return "accessories"
        
        return "other"


def create_perception_node():
    """Create perception node for LangGraph."""
    tool = PerceptionTool()
    
    def perception_node(state: Dict) -> Dict:
        from .state import StylistState
        stylist_state = StylistState(**state) if isinstance(state, dict) else state
        updated_state = tool.process(stylist_state)
        return updated_state.to_dict()
    
    return perception_node
