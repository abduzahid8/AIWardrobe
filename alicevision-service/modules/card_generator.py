"""
Card Generator Module - AI-to-AI Communication
Generates detailed prompts for product card photo generation
"""

import logging
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CardPrompt:
    """Generated prompt for card photo AI"""
    main_prompt: str
    negative_prompt: str
    style_tags: List[str]
    metadata: Dict[str, any]


class CardPromptGenerator:
    """
    Generates detailed prompts for AI card photo generation.
    
    Takes attributes from FashionCLIP and creates optimized prompts
    for Stable Diffusion, DALL-E, or other image generation models.
    """
    
    # Style presets for different aesthetics
    STYLE_PRESETS = {
        "massimo_dutti": {
            "lighting": "soft diffused studio lighting, even illumination",
            "background": "pure white seamless background",
            "quality": "professional fashion photography, high resolution, sharp focus",
            "camera": "shot with medium format camera, 85mm lens, f/2.8"
        },
        "zara": {
            "lighting": "natural window light, bright and airy",
            "background": "clean white background, minimalist",
            "quality": "professional lookbook style, editorial quality",
            "camera": "50mm lens, shallow depth of field"
        },
        "hm": {
            "lighting": "bright studio lighting, high key",
            "background": "white background, simple and clean",
            "quality": "catalog photography, crisp and clean",
           "camera": "standard photography setup"
        },
        "ecommerce": {
            "lighting": "even studio lighting, no harsh shadows",
            "background": "pure white background (#FFFFFF)",
            "quality": "product photography, maximum detail, color accurate",
            "camera": "straight-on angle, centered composition"
        }
    }
    
    def __init__(self, style: str = "ecommerce"):
        """
        Initialize card prompt generator.
        
        Args:
            style: Style preset (massimo_dutti, zara, hm, ecommerce)
        """
        self.style = style
        self.style_config = self.STYLE_PRESETS.get(
            style,
            self.STYLE_PRESETS["ecommerce"]
        )
    
    def generate_prompt(
        self,
        attributes: Dict,
        include_model: bool = False,
        front_facing: bool = True
    ) -> CardPrompt:
        """
        Generate prompt for product card photo.
        
        Args:
            attributes: Clothing attributes from FashionCLIP
            include_model: Whether to include human model
            front_facing: Whether garment should be front-facing
            
        Returns:
            CardPrompt with detailed generation instructions
        """
        # Extract attributes
        category = attributes.get("category", "clothing item")
        description = attributes.get("description", "")
        colors = attributes.get("colors", [])
        patterns = attributes.get("patterns", [])
        details = attributes.get("details", {})
        
        # Build main prompt parts
        parts = []
        
        # 1. The item itself
        if description:
            parts.append(description)
        else:
            # Fallback if no description
            if colors:
                primary_color = colors[0]["name"]
                parts.append(f"{primary_color} {category}")
            else:
                parts.append(category)
        
        # 2. Add positioning
        if front_facing:
            if include_model:
                parts.append("worn by model, front view")
            else:
                parts.append("front view, flat lay perspective")
        
        # 3. Add details
        if details:
            detail_str = ", ".join([f"{v}" for k, v in details.items()])
            if detail_str:
                parts.append(detail_str)
        
        # 4. Add style-specific elements
        parts.append(self.style_config["lighting"])
        parts.append(self.style_config["background"])
        parts.append(self.style_config["quality"])
        parts.append(self.style_config["camera"])
        
        # Combine into main prompt
        main_prompt = ", ".join(parts)
        
        # Generate negative prompt
        negative_prompt = self._generate_negative_prompt(include_model)
        
        # Extract style tags
        style_tags = self._extract_style_tags(attributes)
        
        # Build metadata
        metadata = {
            "category": category,
            "colors": [c["name"] for c in colors] if colors else [],
            "style": self.style,
            "include_model": include_model,
            "front_facing": front_facing
        }
        
        return CardPrompt(
            main_prompt=main_prompt,
            negative_prompt=negative_prompt,
            style_tags=style_tags,
            metadata=metadata
        )
    
    def _generate_negative_prompt(self, include_model: bool) -> str:
        """Generate negative prompt to avoid unwanted elements"""
        negative = [
            "blurry", "out of focus", "low quality", "pixelated",
            "watermark", "text", "logo", "brand name",
            "bad lighting", "shadows", "dark", "grainy",
            "distorted", "deformed", "cropped"
        ]
        
        if not include_model:
            negative.extend([
                "person", "human", "model", "face", "body",
                "mannequin", "hanger"
            ])
        
        return ", ".join(negative)
    
    def _extract_style_tags(self, attributes: Dict) -> List[str]:
        """Extract relevant style tags for categorization"""
        tags = []
        
        # Add category
        if "category" in attributes:
            tags.append(attributes["category"])
        
        # Add styles
        if "styles" in attributes:
            for style in attributes["styles"][:2]:  # Top 2 styles
                tags.append(style["name"])
        
        # Add primary color
        if "colors" in attributes and attributes["colors"]:
            tags.append(attributes["colors"][0]["name"])
        
        # Add pattern
        if "patterns" in attributes and attributes["patterns"]:
            pattern = attributes["patterns"][0]["name"]
            if "solid" not in pattern.lower():
                tags.append(pattern)
        
        return tags
    
    def generate_for_batch(
        self,
        detections: List[Dict],
        style: Optional[str] = None
    ) -> List[CardPrompt]:
        """
        Generate prompts for multiple detected items.
        
        Args:
            detections: List of detection results with attributes
            style: Optional style override
            
        Returns:
            List of CardPrompts, one per detection
        """
        if style:
            original_style = self.style
            self.style = style
            self.style_config = self.STYLE_PRESETS.get(style, self.style_config)
        
        prompts = []
        for detection in detections:
            if "attributes" in detection:
                prompt = self.generate_prompt(detection["attributes"])
                prompts.append(prompt)
        
        if style:
            self.style = original_style
            self.style_config = self.STYLE_PRESETS.get(original_style, self.STYLE_PRESETS["ecommerce"])
        
        return prompts
    
    def to_api_format(self, prompt: CardPrompt) -> Dict:
        """
        Convert CardPrompt to API-friendly format.
        
        Args:
            prompt: CardPrompt object
            
        Returns:
            Dictionary for API response
        """
        return {
            "prompt": prompt.main_prompt,
            "negative_prompt": prompt.negative_prompt,
            "tags": prompt.style_tags,
            "metadata": prompt.metadata
        }


# Convenience function
def generate_card_prompt(
    attributes: Dict,
    style: str = "ecommerce",
    include_model: bool = False
) -> Dict:
    """
    Quick function to generate card prompt.
    
    Args:
        attributes: Clothing attributes from FashionCLIP
        style: Visual style preset
        include_model: Include human model
        
    Returns:
        Dictionary with prompt information
    """
    generator = CardPromptGenerator(style=style)
    prompt = generator.generate_prompt(attributes, include_model=include_model)
    return generator.to_api_format(prompt)
