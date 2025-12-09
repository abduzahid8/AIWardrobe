"""
Card Styling Module
Transforms cutout clothing images into professional e-commerce product cards
"""

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import base64
import io
import logging

logger = logging.getLogger(__name__)


class CardStylist:
    """Create professional product card from cutout clothing"""
    
    def __init__(self):
        self.card_width = 1024
        self.card_height = 1024
        self.padding = 80
        
    def create_product_card(
        self,
        image: np.ndarray,
        add_shadow: bool = True,
        add_border: bool = False,
        background_color: tuple = (255, 255, 255)
    ) -> np.ndarray:
        """
        Transform cutout clothing into professional product card
        
        Args:
            image: Input image (RGBA or RGB)
            add_shadow: Add professional drop shadow
            add_border: Add subtle border frame
            background_color: Background color (default white)
            
        Returns:
            Professional product card image (RGB)
        """
        logger.info("Creating professional product card")
        
        # Convert to PIL for better image processing
        if len(image.shape) == 3 and image.shape[2] == 3:
            # RGB to RGBA
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        
        # Convert OpenCV to PIL
        img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
        
        # Create card canvas
        card = Image.new('RGBA', (self.card_width, self.card_height), (*background_color, 255))
        
        # Resize and center clothing
        clothing = self._resize_and_center(img_pil)
        
        # Add shadow if requested
        if add_shadow:
            card = self._add_professional_shadow(card, clothing)
        
        # Paste clothing on card
        card.paste(clothing, (0, 0), clothing)
        
        # Add border frame if requested
        if add_border:
            card = self._add_border_frame(card)
        
        # Enhance overall quality
        card = self._enhance_quality(card)
        
        # Convert back to OpenCV format (RGB)
        result = cv2.cvtColor(np.array(card.convert('RGB')), cv2.COLOR_RGB2BGR)
        
        logger.info("Product card created successfully")
        return result
    
    def _resize_and_center(self, img: Image.Image) -> Image.Image:
        """Resize clothing to fit card with padding and center it"""
        # Get transparent bounds
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        
        # Calculate resize to fit with padding
        max_width = self.card_width - (2 * self.padding)
        max_height = self.card_height - (2 * self.padding)
        
        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        
        # Center on canvas
        centered = Image.new('RGBA', (self.card_width, self.card_height), (0, 0, 0, 0))
        x = (self.card_width - img.width) // 2
        y = (self.card_height - img.height) // 2
        centered.paste(img, (x, y), img)
        
        return centered
    
    def _add_professional_shadow(self, card: Image.Image, clothing: Image.Image) -> Image.Image:
        """Add realistic drop shadow like professional product photos"""
        # Create shadow layer
        shadow = Image.new('RGBA', (self.card_width, self.card_height), (0, 0, 0, 0))
        
        # Get clothing bounds
        bbox = clothing.getbbox()
        if not bbox:
            return card
        
        # Create shadow mask from clothing alpha
        shadow_mask = clothing.split()[3]
        
        # Offset shadow (down and right)
        shadow_offset_x = 12
        shadow_offset_y = 20
        
        # Create shadow with offset
        shadow_layer = Image.new('RGBA', (self.card_width, self.card_height), (0, 0, 0, 0))
        shadow_layer.paste((40, 40, 40, 80), (shadow_offset_x, shadow_offset_y), shadow_mask)
        
        # Blur shadow for softness
        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=15))
        
        # Composite: card -> shadow -> clothing
        result = Image.alpha_composite(card, shadow_layer)
        
        return result
    
    def _add_border_frame(self, card: Image.Image) -> Image.Image:
        """Add subtle border frame for premium look"""
        draw = ImageDraw.Draw(card)
        
        # Outer border (very subtle)
        border_color = (230, 230, 230, 255)
        border_width = 2
        
        draw.rectangle(
            [border_width, border_width, 
             self.card_width - border_width, self.card_height - border_width],
            outline=border_color,
            width=border_width
        )
        
        return card
    
    def _enhance_quality(self, card: Image.Image) -> Image.Image:
        """Enhance overall image quality for catalog look"""
        # Convert to RGB for enhancement
        card_rgb = card.convert('RGB')
        
        # Slight sharpness increase
        enhancer = ImageEnhance.Sharpness(card_rgb)
        card_rgb = enhancer.enhance(1.1)
        
        # Slight contrast increase
        enhancer = ImageEnhance.Contrast(card_rgb)
        card_rgb = enhancer.enhance(1.05)
        
        # Slight color saturation
        enhancer = ImageEnhance.Color(card_rgb)
        card_rgb = enhancer.enhance(1.08)
        
        return card_rgb.convert('RGBA')


def create_product_card_from_base64(
    image_b64: str,
    add_shadow: bool = True,
    add_border: bool = False
) -> str:
    """
    Create product card from base64 image
    
    Args:
        image_b64: Base64-encoded image (with or without data URI prefix)
        add_shadow: Add professional drop shadow
        add_border: Add subtle border frame
        
    Returns:
        Base64-encoded product card image
    """
    # Decode base64
    if ',' in image_b64:
        image_b64 = image_b64.split(',')[1]
    
    img_bytes = base64.b64decode(image_b64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    
    # Create card
    stylist = CardStylist()
    card = stylist.create_product_card(image, add_shadow, add_border)
    
    # Encode back to base64
    _, buffer = cv2.imencode('.jpg', card, [cv2.IMWRITE_JPEG_QUALITY, 95])
    card_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/jpeg;base64,{card_b64}"
