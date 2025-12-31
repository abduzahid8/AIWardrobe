"""
🏷️ PROFESSIONAL PRODUCT CARD GENERATOR

Creates Massimo Dutti-style professional product cards:
- AI-enhanced studio lighting
- Ghost mannequin effect
- Professional fabric texture
- Clean gradient background
"""

import cv2
import numpy as np
import base64
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import io
import logging
import os
import requests
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def create_professional_product_card(
    cutout_b64: str,
    item_type: str,
    color: str = "",
    material: str = "",
    card_size: Tuple[int, int] = (800, 1000)
) -> str:
    """
    Transform a cutout into a professional Massimo Dutti-style product card.
    
    Uses AI to enhance the clothing with:
    - Professional studio lighting
    - Ghost mannequin effect
    - Enhanced fabric texture
    - Clean studio background
    
    Args:
        cutout_b64: Base64 encoded cutout image
        item_type: Type of clothing (e.g., "Crew neck sweater")
        color: Color of the item
        material: Material of the item
        card_size: Output card dimensions (width, height)
        
    Returns:
        Base64 encoded professional product card
    """
    try:
        # Try AI enhancement first
        ai_result = create_ai_enhanced_product_photo(cutout_b64, item_type, color, material)
        if ai_result:
            logger.info(f"✅ Created AI-enhanced product card for {item_type}")
            return ai_result
        
        # Fallback to local enhancement
        logger.info(f"⚠️ AI enhancement unavailable, using local enhancement for {item_type}")
        return create_local_enhanced_card(cutout_b64, item_type, card_size)
        
    except Exception as e:
        logger.error(f"Product card generation failed: {e}")
        return cutout_b64


def create_ai_enhanced_product_photo(
    cutout_b64: str,
    item_type: str,
    color: str = "",
    material: str = ""
) -> Optional[str]:
    """
    Create Massimo Dutti-style product card using REFERENCE-GUIDED GENERATION.
    
    Uses ControlNet with Massimo Dutti catalog images as layout references:
    1. Analyze clothing with GPT-4V
    2. Select appropriate Massimo Dutti reference image for this category
    3. Use ControlNet Canny to generate with EXACT layout from reference
    4. Apply professional post-processing
    
    This ensures sweaters look like MD sweaters, jackets like MD jackets, etc.
    """
    api_token = os.environ.get("REPLICATE_API_TOKEN")
    
    if not api_token:
        logger.warning("REPLICATE_API_TOKEN not set, using local enhancement")
        return None
    
    try:
        import replicate
        
        # Prepare image
        if ',' in cutout_b64:
            img_data = cutout_b64.split(',')[1]
        else:
            img_data = cutout_b64
        
        # STEP 1: Analyze clothing with GPT-4V
        logger.info("   📝 Analyzing clothing details...")
        detailed_description = analyze_clothing_details(img_data, item_type, color, material)
        
        if not detailed_description:
            detailed_description = f"{color} {material} {item_type}".strip() if material else f"{color} {item_type}".strip()
        
        logger.info(f"   Description: {detailed_description[:80]}...")
        
        # STEP 2: Use the actual CUTOUT as ControlNet reference
        # This ensures the generated image matches the ACTUAL clothing shape!
        # (NOT external reference files which may be mislabeled)
        
        # Determine garment category for prompt customization
        item_lower = item_type.lower()
        is_shoes = any(word in item_lower for word in ["shoe", "sneaker", "boot", "loafer", "oxford", "footwear", "heel", "sandal"])
        is_pants = any(word in item_lower for word in ["pant", "trouser", "jean", "chino", "short", "slack"])
        is_bag = any(word in item_lower for word in ["bag", "backpack", "crossbody", "tote", "handbag"])
        
        # Build item-specific prompt
        if is_shoes:
            category_style = "luxury leather shoes with polished finish, angled pair display"
            aspect = "1:1"
        elif is_pants:
            category_style = "luxury pants with perfect drape on invisible ghost mannequin form, natural fabric folds"
            aspect = "4:5"
        elif is_bag:
            category_style = "luxury bag displayed front-facing, leather texture visible"
            aspect = "1:1"
        else:
            category_style = "luxury garment on invisible ghost mannequin, 3D body form, natural drape"
            aspect = "4:5"
        
        prompt = f"""MDSTYLE, professional Massimo Dutti e-commerce product photograph:
{detailed_description}

IMPORTANT: This is a {item_type} - generate ONLY this type of {item_type}, NOT any other garment!

Style: {category_style}, clean off-white gradient studio background,
professional studio lighting, sharp fabric texture visible,
high-end fashion catalog quality, 8K photorealistic"""
        
        logger.info(f"   🎨 Generating {item_type} with ControlNet (using actual cutout as reference)...")
        
        # Use the ACTUAL CUTOUT as the ControlNet reference
        # This ensures the edges match the real clothing, not an external reference
        cutout_uri = f"data:image/png;base64,{img_data}"
        
        try:
            output = replicate.run(
                "black-forest-labs/flux-canny-pro",
                input={
                    "control_image": cutout_uri,  # The ACTUAL clothing as reference
                    "prompt": prompt,
                    "control_strength": 0.5,  # Lower to allow style improvement while keeping shape
                    "num_outputs": 1,
                    "aspect_ratio": aspect,
                    "output_format": "jpg",
                    "output_quality": 95
                }
            )
            
            if output:
                result_url = output[0] if isinstance(output, list) else output
                response = requests.get(result_url, timeout=60)
                if response.status_code == 200:
                    result_b64 = base64.b64encode(response.content).decode()
                    logger.info(f"   ✅ Created Massimo Dutti-style {item_type} card!")
                    return f"data:image/jpeg;base64,{result_b64}"
                    
        except Exception as cn_error:
            logger.warning(f"   ControlNet failed ({cn_error}), falling back to photo processing...")
        
        # FALLBACK: Use professional photo processing (no AI regeneration)
        logger.info("   🔄 Using professional photo processing fallback...")
        return create_photo_processed_card(cutout_b64, item_type)
        
    except Exception as e:
        logger.warning(f"AI generation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def get_massimo_dutti_reference(item_type: str) -> Optional[str]:
    """Get the appropriate Massimo Dutti reference image for this item type."""
    item_lower = item_type.lower()
    
    # Reference image directory
    ref_dir = os.path.join(os.path.dirname(__file__), "..", "references", "massimo_dutti")
    
    # Log incoming item type for debugging
    logger.info(f"   🔍 Finding reference for: '{item_type}'")
    
    # Map item types to reference images (ORDER MATTERS - most specific first)
    # Check shoes/boots first (they contain "boot" which could match "combat boots")
    if any(word in item_lower for word in ["shoe", "sneaker", "boot", "loafer", "oxford", "footwear", "heel", "sandal"]):
        ref_path = os.path.join(ref_dir, "shoes.png")
        logger.info(f"   ✅ Matched SHOES: {os.path.basename(ref_path)}")
        return ref_path
    
    # Check pants/trousers before other categories
    elif any(word in item_lower for word in ["pant", "trouser", "jean", "chino", "short", "slack"]):
        ref_path = os.path.join(ref_dir, "pants.png")
        logger.info(f"   ✅ Matched PANTS: {os.path.basename(ref_path)}")
        return ref_path
    
    # Bags and accessories
    elif any(word in item_lower for word in ["bag", "backpack", "crossbody", "tote", "handbag", "purse"]):
        ref_path = os.path.join(ref_dir, "bag.png")
        if os.path.exists(ref_path):
            logger.info(f"   ✅ Matched BAG: {os.path.basename(ref_path)}")
            return ref_path
        else:
            logger.warning(f"   ⚠️ bag.png not found, using sweater fallback")
            return os.path.join(ref_dir, "sweater.png")
    
    # Jackets and outerwear
    elif any(word in item_lower for word in ["jacket", "coat", "blazer", "outerwear", "overcoat"]):
        ref_path = os.path.join(ref_dir, "jacket.png")
        logger.info(f"   ✅ Matched JACKET: {os.path.basename(ref_path)}")
        return ref_path
    
    # Sweaters and knitwear
    elif any(word in item_lower for word in ["sweater", "knit", "pullover", "cardigan", "hoodie"]):
        ref_path = os.path.join(ref_dir, "sweater.png")
        logger.info(f"   ✅ Matched SWEATER: {os.path.basename(ref_path)}")
        return ref_path
    
    # Shirts and tops (check after sweaters)
    elif any(word in item_lower for word in ["shirt", "blouse", "top", "t-shirt", "tee", "polo"]):
        ref_path = os.path.join(ref_dir, "shirt.png")
        logger.info(f"   ✅ Matched SHIRT: {os.path.basename(ref_path)}")
        return ref_path
    
    # Dresses and skirts
    elif any(word in item_lower for word in ["dress", "skirt", "gown"]):
        ref_path = os.path.join(ref_dir, "dress.png")
        if os.path.exists(ref_path):
            logger.info(f"   ✅ Matched DRESS: {os.path.basename(ref_path)}")
            return ref_path
        else:
            logger.warning(f"   ⚠️ dress.png not found, using sweater fallback")
            return os.path.join(ref_dir, "sweater.png")
    
    # Default to sweater for unknown items
    ref_path = os.path.join(ref_dir, "sweater.png")
    logger.warning(f"   ⚠️ No match for '{item_type}', using default: {os.path.basename(ref_path)}")
    return ref_path


def create_photo_processed_card(cutout_b64: str, item_type: str) -> Optional[str]:
    """Create a professional card using photo processing (background removal + studio bg)."""
    try:
        import replicate
        
        # Prepare image
        if ',' in cutout_b64:
            img_data = cutout_b64.split(',')[1]
        else:
            img_data = cutout_b64
        
        image_uri = f"data:image/png;base64,{img_data}"
        
        # Remove background with BiRefNet
        logger.info("   🔪 Removing background with BiRefNet...")
        transparent_cutout = None
        
        try:
            output = replicate.run(
                "cjwbw/birefnet",
                input={"image": image_uri}
            )
            
            if output:
                result_url = output if isinstance(output, str) else output[0]
                response = requests.get(result_url, timeout=60)
                if response.status_code == 200:
                    transparent_cutout = Image.open(io.BytesIO(response.content)).convert("RGBA")
                    logger.info("   ✅ Background removed with BiRefNet")
        except Exception as biref_error:
            logger.warning(f"   BiRefNet failed ({biref_error}), using rembg...")
        
        # Fallback to rembg
        if transparent_cutout is None:
            try:
                from rembg import remove
                img_bytes = base64.b64decode(img_data)
                original_img = Image.open(io.BytesIO(img_bytes))
                transparent_cutout = remove(original_img).convert("RGBA")
                logger.info("   ✅ Background removed with rembg")
            except Exception:
                img_bytes = base64.b64decode(img_data)
                transparent_cutout = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
        
        # Create studio background and composite
        card_width, card_height = 800, 1000
        studio_bg = create_massimo_dutti_background(card_width, card_height)
        
        # Resize and position
        max_width = int(card_width * 0.80)
        max_height = int(card_height * 0.80)
        item_w, item_h = transparent_cutout.size
        scale = min(max_width / item_w, max_height / item_h)
        new_w, new_h = int(item_w * scale), int(item_h * scale)
        
        resized_item = transparent_cutout.resize((new_w, new_h), Image.Resampling.LANCZOS)
        pos_x = (card_width - new_w) // 2
        pos_y = int((card_height - new_h) * 0.45)
        
        # Add shadow and composite
        shadow = create_drop_shadow(resized_item, offset=(8, 12), blur_radius=20, opacity=0.25)
        studio_bg.paste(shadow, (pos_x + 8, pos_y + 12), shadow)
        studio_bg.paste(resized_item, (pos_x, pos_y), resized_item)
        
        # Apply color grading
        final_image = apply_massimo_dutti_grading(studio_bg)
        
        # Convert to base64
        buffer = io.BytesIO()
        final_image.save(buffer, format="JPEG", quality=95)
        result_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        logger.info(f"   ✅ Created professional product card for {item_type}")
        return f"data:image/jpeg;base64,{result_b64}"
        
    except Exception as e:
        logger.warning(f"Photo processing failed: {e}")
        return None


def create_massimo_dutti_background(width: int, height: int) -> Image.Image:
    """Create the signature Massimo Dutti off-white gradient background."""
    from PIL import ImageDraw
    
    # Create gradient from top (lighter) to bottom (slightly darker)
    bg = Image.new('RGB', (width, height), (248, 248, 246))
    draw = ImageDraw.Draw(bg)
    
    # Subtle vertical gradient
    top_color = (250, 250, 248)  # Very light off-white
    bottom_color = (242, 241, 238)  # Slightly warmer off-white
    
    for y in range(height):
        ratio = y / height
        r = int(top_color[0] + (bottom_color[0] - top_color[0]) * ratio)
        g = int(top_color[1] + (bottom_color[1] - top_color[1]) * ratio)
        b = int(top_color[2] + (bottom_color[2] - top_color[2]) * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    
    # Add subtle vignette effect
    vignette = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    vignette_draw = ImageDraw.Draw(vignette)
    
    # Draw subtle darkening in corners
    for i in range(50):
        alpha = int((50 - i) * 0.3)  # Very subtle
        margin = i * 3
        vignette_draw.rectangle(
            [margin, margin, width - margin, height - margin],
            outline=(0, 0, 0, alpha)
        )
    
    # Apply vignette
    bg = bg.convert('RGBA')
    bg = Image.alpha_composite(bg, vignette)
    
    return bg.convert('RGB')


def create_drop_shadow(image: Image.Image, offset: tuple = (5, 10), 
                       blur_radius: int = 15, opacity: float = 0.3) -> Image.Image:
    """Create a realistic drop shadow for the clothing item."""
    # Create shadow from alpha channel
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    # Extract alpha channel and create shadow
    alpha = image.split()[3]
    
    # Create black shadow with alpha
    shadow = Image.new('RGBA', image.size, (0, 0, 0, 0))
    shadow_color = (30, 25, 20, int(255 * opacity))  # Warm dark shadow
    
    # Fill shadow where clothing exists
    shadow_pixels = shadow.load()
    alpha_pixels = alpha.load()
    
    for y in range(image.size[1]):
        for x in range(image.size[0]):
            if alpha_pixels[x, y] > 50:  # Where clothing exists
                shadow_pixels[x, y] = shadow_color
    
    # Blur the shadow
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    return shadow


def apply_massimo_dutti_grading(image: Image.Image) -> Image.Image:
    """Apply professional Massimo Dutti color grading."""
    # Ensure RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Slight contrast enhancement (subtle)
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(1.05)
    
    # Gentle saturation boost (very subtle for that luxury muted look)
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(1.02)
    
    # Slight sharpening for fabric detail
    enhancer = ImageEnhance.Sharpness(image)
    image = enhancer.enhance(1.1)
    
    # Slight brightness adjustment
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(1.02)
    
    return image


def analyze_clothing_details(img_b64: str, item_type: str, color: str, material: str) -> Optional[str]:
    """
    Use GPT-4V to analyze EVERY detail of the clothing item.
    
    Returns detailed description including:
    - Exact color shades
    - Fabric texture and pattern
    - Buttons, zippers, collar style
    - Design details
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        logger.warning("OPENAI_API_KEY not set, using basic description")
        return None
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_key}"
        }
        
        payload = {
            "model": "gpt-4o",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Analyze this {item_type} in EXTREME detail for product photography recreation.

Describe EVERY visible detail:
1. EXACT color (shade, tone, warmth)
2. Fabric texture (ribbed, smooth, knit pattern, weave)
3. Design elements (collar style, button type, zipper, pockets)
4. Construction details (seams, stitching, hem style)
5. Overall silhouette and fit style

Output a single paragraph description that could be used to recreate this EXACT item as a professional product photo. Be extremely specific."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_b64}",
                                "detail": "high"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }
        
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            description = result["choices"][0]["message"]["content"].strip()
            logger.info(f"   🔍 GPT-4V analyzed: {description[:80]}...")
            return description
        else:
            logger.warning(f"GPT-4V analysis failed: {response.status_code}")
            return None
            
    except Exception as e:
        logger.warning(f"Clothing analysis failed: {e}")
        return None


def create_local_enhanced_card(
    cutout_b64: str,
    item_type: str,
    card_size: Tuple[int, int] = (800, 1000)
) -> str:
    """
    Create enhanced product card using local image processing.
    Fallback when AI is unavailable.
    """
    try:
        # Decode cutout
        if ',' in cutout_b64:
            cutout_b64 = cutout_b64.split(',')[1]
        
        img_bytes = base64.b64decode(cutout_b64)
        cutout = Image.open(io.BytesIO(img_bytes))
        
        # Convert to RGBA if needed
        if cutout.mode != 'RGBA':
            cutout = cutout.convert('RGBA')
        
        # Create professional gradient background
        card_w, card_h = card_size
        card = create_gradient_background(card_w, card_h)
        
        # Resize cutout to fit nicely in card (80% of card width max)
        max_item_w = int(card_w * 0.8)
        max_item_h = int(card_h * 0.85)
        
        cutout = resize_to_fit(cutout, max_item_w, max_item_h)
        
        # Enhance the clothing image
        cutout = enhance_clothing(cutout)
        
        # Create soft shadow
        shadow = create_soft_shadow(cutout)
        
        # Calculate position (centered, slightly above center)
        item_x = (card_w - cutout.width) // 2
        item_y = (card_h - cutout.height) // 2 - int(card_h * 0.02)
        
        # Place shadow (offset down and right slightly)
        shadow_x = item_x + 5
        shadow_y = item_y + 8
        card.paste(shadow, (shadow_x, shadow_y), shadow)
        
        # Place the cutout
        card.paste(cutout, (item_x, item_y), cutout)
        
        # Encode to base64
        buffer = io.BytesIO()
        card.convert('RGB').save(buffer, format='JPEG', quality=95)
        
        logger.info(f"✅ Created locally enhanced product card for {item_type}")
        return f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode()}"
        
    except Exception as e:
        logger.error(f"Local card generation failed: {e}")
        return cutout_b64


def enhance_clothing(image: Image.Image) -> Image.Image:
    """Enhance the clothing item for professional look."""
    try:
        # Work on RGB copy for enhancement
        if image.mode == 'RGBA':
            alpha = image.split()[3]
            rgb = image.convert('RGB')
        else:
            alpha = None
            rgb = image
        
        # Increase contrast slightly
        enhancer = ImageEnhance.Contrast(rgb)
        rgb = enhancer.enhance(1.15)
        
        # Boost color saturation slightly
        enhancer = ImageEnhance.Color(rgb)
        rgb = enhancer.enhance(1.1)
        
        # Sharpen for fabric detail
        enhancer = ImageEnhance.Sharpness(rgb)
        rgb = enhancer.enhance(1.3)
        
        # Adjust brightness for professional look
        enhancer = ImageEnhance.Brightness(rgb)
        rgb = enhancer.enhance(1.05)
        
        # Restore alpha
        if alpha:
            rgb = rgb.convert('RGBA')
            rgb.putalpha(alpha)
        
        return rgb
        
    except Exception as e:
        logger.warning(f"Clothing enhancement failed: {e}")
        return image


def create_gradient_background(width: int, height: int) -> Image.Image:
    """Create a professional Massimo Dutti-style gradient background."""
    # Create gradient from light gray at top to slightly lighter at bottom
    gradient = Image.new('RGBA', (width, height))
    
    for y in range(height):
        # Subtle gradient from #f0f0f0 to #fafafa
        ratio = y / height
        # Add slight warm tint like Massimo Dutti
        r = int(240 + (250 - 240) * ratio)
        g = int(238 + (248 - 238) * ratio)
        b = int(235 + (245 - 235) * ratio)
        
        for x in range(width):
            gradient.putpixel((x, y), (r, g, b, 255))
    
    return gradient


def resize_to_fit(image: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Resize image to fit within max dimensions while maintaining aspect ratio."""
    ratio = min(max_w / image.width, max_h / image.height)
    if ratio < 1:
        new_size = (int(image.width * ratio), int(image.height * ratio))
        return image.resize(new_size, Image.LANCZOS)
    return image


def create_soft_shadow(image: Image.Image) -> Image.Image:
    """Create a soft, professional drop shadow."""
    # Create shadow mask from alpha channel
    if image.mode != 'RGBA':
        return Image.new('RGBA', image.size, (0, 0, 0, 0))
    
    # Extract alpha channel
    alpha = image.split()[3]
    
    # Create shadow (dark gray, semi-transparent)
    shadow = Image.new('RGBA', image.size, (0, 0, 0, 0))
    shadow_color = Image.new('RGBA', image.size, (60, 55, 50, 40))
    shadow.paste(shadow_color, mask=alpha)
    
    # Blur the shadow
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=15))
    
    return shadow


def format_display_name(item_type: str, color: str = "") -> str:
    """Format a nice display name for the product."""
    name_parts = []
    if color:
        name_parts.append(color.title())
    name_parts.append(item_type.title())
    return " ".join(name_parts)
