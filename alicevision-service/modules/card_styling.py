"""
Professional Product Card Styling
Creates e-commerce quality product cards from clothing cutouts
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import logging
import base64
import io

logger = logging.getLogger(__name__)


@dataclass
class CardTemplate:
    """Product card template configuration"""
    name: str
    canvas_size: Tuple[int, int]
    padding: int
    background_color: Tuple[int, int, int]
    shadow_enabled: bool
    shadow_offset: Tuple[int, int]
    shadow_blur: int
    shadow_opacity: float
    border_enabled: bool
    border_width: int
    border_color: Tuple[int, int, int]
    enhancement: Dict[str, float]


# Pre-defined professional templates
TEMPLATES = {
    "catalog": CardTemplate(
        name="Catalog",
        canvas_size=(800, 1000),
        padding=60,
        background_color=(255, 255, 255),
        shadow_enabled=True,
        shadow_offset=(8, 12),
        shadow_blur=25,
        shadow_opacity=0.15,
        border_enabled=False,
        border_width=0,
        border_color=(0, 0, 0),
        enhancement={"contrast": 1.05, "sharpness": 1.1, "brightness": 1.02}
    ),
    "minimal": CardTemplate(
        name="Minimal",
        canvas_size=(800, 800),
        padding=80,
        background_color=(250, 250, 250),
        shadow_enabled=True,
        shadow_offset=(4, 6),
        shadow_blur=15,
        shadow_opacity=0.08,
        border_enabled=False,
        border_width=0,
        border_color=(0, 0, 0),
        enhancement={"contrast": 1.0, "sharpness": 1.05, "brightness": 1.0}
    ),
    "lifestyle": CardTemplate(
        name="Lifestyle",
        canvas_size=(800, 1200),
        padding=40,
        background_color=(248, 248, 245),
        shadow_enabled=True,
        shadow_offset=(12, 18),
        shadow_blur=35,
        shadow_opacity=0.2,
        border_enabled=True,
        border_width=1,
        border_color=(230, 230, 230),
        enhancement={"contrast": 1.08, "sharpness": 1.15, "brightness": 1.03}
    ),
    "ecommerce": CardTemplate(
        name="E-Commerce",
        canvas_size=(1000, 1000),
        padding=100,
        background_color=(255, 255, 255),
        shadow_enabled=True,
        shadow_offset=(6, 10),
        shadow_blur=20,
        shadow_opacity=0.12,
        border_enabled=False,
        border_width=0,
        border_color=(0, 0, 0),
        enhancement={"contrast": 1.1, "sharpness": 1.2, "brightness": 1.05}
    ),
    # 🚀 NEW: Premium luxury template for maximum quality
    "luxury": CardTemplate(
        name="Luxury",
        canvas_size=(1200, 1400),
        padding=120,
        background_color=(255, 255, 255),
        shadow_enabled=True,
        shadow_offset=(10, 15),
        shadow_blur=40,
        shadow_opacity=0.18,
        border_enabled=False,
        border_width=0,
        border_color=(0, 0, 0),
        enhancement={"contrast": 1.12, "sharpness": 1.25, "brightness": 1.04}
    ),
    # 🏆 Massimo Dutti inspired - professional catalog style (ENHANCED)
    "massimo": CardTemplate(
        name="Massimo Dutti Premium",
        canvas_size=(800, 1000),  # Taller for clothing
        padding=40,  # Less padding = larger item
        background_color=(245, 245, 243),  # Light gray like reference #F5F5F3
        shadow_enabled=True,
        shadow_offset=(0, 20),  # Shadow directly below (ground shadow)
        shadow_blur=35,
        shadow_opacity=0.12,  # Subtle shadow
        border_enabled=False,
        border_width=0,
        border_color=(0, 0, 0),
        enhancement={"contrast": 1.06, "sharpness": 1.12, "brightness": 1.02}
    )
}


class ProductCardStylist:
    """
    Creates professional e-commerce quality product cards.
    
    Features:
    - Multiple professional templates
    - Natural drop shadows with blur
    - Smart resizing with aspect ratio preservation
    - Image quality enhancements
    - Clean borders and backgrounds
    """
    
    def __init__(self, template: str = "catalog"):
        if template not in TEMPLATES:
            template = "catalog"
        self.template = TEMPLATES[template]
    
    def _resize_and_center(
        self, 
        image: Image.Image, 
        canvas_size: Tuple[int, int],
        padding: int
    ) -> Image.Image:
        """Resize image to fit canvas with padding, preserving aspect ratio"""
        
        # Calculate available space
        max_width = canvas_size[0] - 2 * padding
        max_height = canvas_size[1] - 2 * padding
        
        # Get original size
        orig_width, orig_height = image.size
        
        # Calculate scale factor
        scale = min(max_width / orig_width, max_height / orig_height)
        
        # New dimensions
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        # Resize with high quality
        resized = image.resize(
            (new_width, new_height), 
            Image.Resampling.LANCZOS
        )
        
        # Create canvas
        canvas = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
        
        # Calculate position (centered)
        x = (canvas_size[0] - new_width) // 2
        y = (canvas_size[1] - new_height) // 2
        
        # Paste resized image
        canvas.paste(resized, (x, y), resized if resized.mode == 'RGBA' else None)
        
        return canvas
    
    def _create_shadow(
        self,
        image: Image.Image,
        offset: Tuple[int, int],
        blur_radius: int,
        opacity: float
    ) -> Image.Image:
        """Create natural-looking drop shadow"""
        
        # Get alpha channel
        if image.mode == 'RGBA':
            alpha = image.split()[3]
        else:
            alpha = Image.new('L', image.size, 255)
        
        # Create shadow from alpha channel
        shadow = Image.new('L', image.size, 0)
        shadow.paste(alpha, (0, 0))
        
        # Apply blur
        shadow = shadow.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # Create shadow layer
        shadow_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
        shadow_array = np.array(shadow).astype(np.float32)
        shadow_array = (shadow_array * opacity).astype(np.uint8)
        
        # Convert to RGBA with black color
        shadow_rgba = Image.new('RGBA', image.size, (0, 0, 0, 0))
        shadow_rgb = Image.new('RGB', image.size, (30, 30, 30))  # Slightly dark shadow
        shadow_rgba.paste(shadow_rgb, (0, 0), Image.fromarray(shadow_array))
        
        # Create offset shadow
        result_size = (
            image.size[0] + abs(offset[0]) * 2,
            image.size[1] + abs(offset[1]) * 2
        )
        result = Image.new('RGBA', result_size, (0, 0, 0, 0))
        
        # Paste shadow with offset
        shadow_x = max(0, offset[0]) + abs(offset[0])
        shadow_y = max(0, offset[1]) + abs(offset[1])
        result.paste(shadow_rgba, (shadow_x, shadow_y), shadow_rgba)
        
        # Paste original on top
        orig_x = abs(offset[0]) + (-min(0, offset[0]))
        orig_y = abs(offset[1]) + (-min(0, offset[1]))
        result.paste(image, (orig_x, orig_y), image if image.mode == 'RGBA' else None)
        
        # Crop back to original size (centered)
        crop_x = abs(offset[0])
        crop_y = abs(offset[1])
        result = result.crop((crop_x, crop_y, crop_x + image.size[0], crop_y + image.size[1]))
        
        return result
    
    def _add_professional_shadow(
        self,
        card: Image.Image,
        clothing: Image.Image
    ) -> Image.Image:
        """Add professional soft drop shadow (Massimo Dutti style) - SIMPLIFIED"""
        
        if not self.template.shadow_enabled:
            return card
        
        # Get alpha mask from clothing
        if clothing.mode == 'RGBA':
            alpha = clothing.split()[3]
        else:
            # No alpha, return card as-is
            return card
        
        # Create shadow from alpha channel - simple blur approach
        # Convert alpha to numpy for processing
        alpha_np = np.array(alpha).astype(np.float32)
        
        # Apply opacity to shadow
        shadow_alpha_np = (alpha_np * self.template.shadow_opacity).astype(np.uint8)
        shadow_alpha = Image.fromarray(shadow_alpha_np)
        
        # Blur the shadow for soft edges
        shadow_alpha = shadow_alpha.filter(ImageFilter.GaussianBlur(self.template.shadow_blur))
        
        # Create shadow color layer (dark gray)
        shadow_rgb = Image.new('RGB', clothing.size, (30, 30, 35))
        
        # Combine into RGBA shadow
        shadow_rgba = Image.new('RGBA', clothing.size, (0, 0, 0, 0))
        shadow_rgba.paste(shadow_rgb, (0, 0), shadow_alpha)
        
        # Create result with background color
        result = Image.new('RGB', card.size, self.template.background_color)
        
        # Calculate clothing position (centered)
        cloth_x = (card.size[0] - clothing.size[0]) // 2
        cloth_y = (card.size[1] - clothing.size[1]) // 2
        
        # Shadow position (slightly offset down and right for natural look)
        shadow_x = cloth_x + self.template.shadow_offset[0]
        shadow_y = cloth_y + self.template.shadow_offset[1]
        
        # Paste shadow first
        result.paste(shadow_rgba, (shadow_x, shadow_y), shadow_rgba)
        
        return result
    
    def _enhance_quality(self, image: Image.Image) -> Image.Image:
        """Apply quality enhancements"""
        
        # Convert to RGB for enhancements
        if image.mode == 'RGBA':
            # Preserve alpha
            alpha = image.split()[3]
            rgb = image.convert('RGB')
        else:
            alpha = None
            rgb = image
        
        # Apply enhancements
        enhancements = self.template.enhancement
        
        # Contrast
        if enhancements.get("contrast", 1.0) != 1.0:
            enhancer = ImageEnhance.Contrast(rgb)
            rgb = enhancer.enhance(enhancements["contrast"])
        
        # Sharpness
        if enhancements.get("sharpness", 1.0) != 1.0:
            enhancer = ImageEnhance.Sharpness(rgb)
            rgb = enhancer.enhance(enhancements["sharpness"])
        
        # Brightness
        if enhancements.get("brightness", 1.0) != 1.0:
            enhancer = ImageEnhance.Brightness(rgb)
            rgb = enhancer.enhance(enhancements["brightness"])
        
        # Restore alpha if needed
        if alpha is not None:
            result = rgb.convert('RGBA')
            result.putalpha(alpha)
            return result
        
        return rgb
    
    def _add_border(self, image: Image.Image) -> Image.Image:
        """Add subtle border frame"""
        
        if not self.template.border_enabled:
            return image
        
        # Create border
        bordered = Image.new('RGB', image.size, self.template.border_color)
        
        # Calculate inner size
        inner_x = self.template.border_width
        inner_y = self.template.border_width
        inner_w = image.size[0] - 2 * self.template.border_width
        inner_h = image.size[1] - 2 * self.template.border_width
        
        # Resize and paste inner image
        inner = image.resize((inner_w, inner_h), Image.Resampling.LANCZOS)
        
        if inner.mode == 'RGBA':
            bordered.paste(inner, (inner_x, inner_y), inner)
        else:
            bordered.paste(inner, (inner_x, inner_y))
        
        return bordered
    
    def create_product_card(
        self,
        image: np.ndarray,
        add_shadow: bool = True,
        add_border: bool = False,
        clothing_type: str = None,
        color_name: str = None
    ) -> np.ndarray:
        """
        Create professional product card from clothing cutout.
        
        Args:
            image: Input image (BGR or BGRA)
            add_shadow: Add drop shadow
            add_border: Add border frame
            clothing_type: Optional clothing type to display (e.g., "Denim Jacket")
            color_name: Optional color name to display (e.g., "Navy Blue")
            
        Returns:
            Product card image (BGR)
        """
        # Convert to PIL
        if image.shape[2] == 4:
            # BGRA to RGBA
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
        else:
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Resize and center on canvas
        clothing = self._resize_and_center(
            pil_img, 
            self.template.canvas_size,
            self.template.padding
        )
        
        # Create background
        card = Image.new('RGB', self.template.canvas_size, self.template.background_color)
        
        # Add shadow
        if add_shadow and self.template.shadow_enabled:
            card = self._add_professional_shadow(card, clothing)
        
        # Paste clothing
        x = (card.size[0] - clothing.size[0]) // 2
        y = (card.size[1] - clothing.size[1]) // 2
        card.paste(clothing, (x, y), clothing if clothing.mode == 'RGBA' else None)
        
        # Add border
        if add_border:
            card = self._add_border(card)
        
        # Enhance quality
        card = self._enhance_quality(card)
        
        # 🚀 Add text label if type/color provided
        if clothing_type or color_name:
            card = self._add_text_label(card, clothing_type, color_name)
        
        # Convert back to OpenCV
        result = cv2.cvtColor(np.array(card), cv2.COLOR_RGB2BGR)
        
        return result
    
    def _add_text_label(
        self, 
        image: Image.Image, 
        clothing_type: str = None,
        color_name: str = None
    ) -> Image.Image:
        """Add a subtle text label at the bottom of the card"""
        try:
            from PIL import ImageFont
            
            draw = ImageDraw.Draw(image)
            
            # Build label text
            label_parts = []
            if color_name:
                label_parts.append(color_name.title())
            if clothing_type:
                label_parts.append(clothing_type.title())
            
            if not label_parts:
                return image
            
            label = " • ".join(label_parts)
            
            # Use default font (or try to load a nice one)
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            except:
                font = ImageFont.load_default()
            
            # Calculate text position (centered at bottom)
            bbox = draw.textbbox((0, 0), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = (image.size[0] - text_width) // 2
            y = image.size[1] - 50 - text_height
            
            # Draw subtle text
            draw.text((x, y), label, fill=(80, 80, 80), font=font)
            
            return image
            
        except Exception as e:
            logger.debug(f"Could not add text label: {e}")
            return image


def create_product_card_from_base64(
    image_base64: str,
    add_shadow: bool = True,
    add_border: bool = False,
    template: str = "catalog"
) -> str:
    """
    Create product card from base64 image.
    
    Args:
        image_base64: Base64-encoded image
        add_shadow: Add drop shadow
        add_border: Add border
        template: Template name (catalog, minimal, lifestyle, ecommerce)
        
    Returns:
        Base64-encoded product card image
    """
    # Remove data URL prefix
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]
    
    # Decode
    img_bytes = base64.b64decode(image_base64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
    
    if image is None:
        raise ValueError("Could not decode image")
    
    # Create card
    stylist = ProductCardStylist(template=template)
    card = stylist.create_product_card(image, add_shadow, add_border)
    
    # Encode to base64
    _, buffer = cv2.imencode('.png', card)
    result_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return f"data:image/png;base64,{result_base64}"
