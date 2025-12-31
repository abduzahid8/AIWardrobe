"""
AI-Powered Product Card Generation using Replicate FLUX
Generates professional ghost mannequin product photos
"""

import os
import base64
import logging
import requests
from typing import Optional
from PIL import Image
import io

logger = logging.getLogger(__name__)

# Replicate API Token
REPLICATE_API_TOKEN = os.environ.get("REPLICATE_API_TOKEN")

# Category-specific prompt templates - EXACT Massimo Dutti catalog style
PROMPT_TEMPLATES = {
    # TOPS - Ghost mannequin, floating as if worn
    "tops": """Massimo Dutti luxury catalog product photograph. A {gender}'s {color} {material} {garment_type} displayed on an invisible ghost mannequin, front view facing camera directly. The garment floats naturally as if worn by an invisible person, showing three-dimensional form. {details} Arms relaxed at sides. Soft diffused studio lighting from above and front, creating minimal soft shadows. Background is solid warm cream off-white color (hex #F5F3F0), seamless and clean. No wrinkles, perfectly pressed. Ultra high resolution 4K, sharp fabric texture details visible. Professional luxury e-commerce photography.""",
    
    # PANTS - Ghost mannequin standing (NOT flat-lay!), 3D form as if worn
    "pants": """Massimo Dutti luxury catalog product photograph. {gender}'s {color} {material} {garment_type} displayed on an invisible ghost mannequin, standing upright as if worn, front view facing camera. The trousers show natural three-dimensional form, NOT flat. Inner waistband lining slightly visible at top edge. {details} Natural slight crease down center of each leg. Legs straight and parallel, hem at bottom clean. Soft diffused studio lighting, minimal shadows. Background is solid warm cream off-white color (hex #F5F3F0), seamless. Ultra high resolution 4K, fabric texture visible. Professional luxury e-commerce photography.""",
    
    # SHOES - Pair positioned at angle, one slightly behind the other
    "shoes": """Massimo Dutti luxury catalog product photograph. A pair of {gender}'s {color} {material} {garment_type}, shown from a 3/4 side angle. Left shoe positioned slightly behind and to the left, right shoe in front and to the right, both facing the same direction toward right side of frame. {details} Shoes appear naturally placed as if someone just stepped out of them. Soft diffused studio lighting, very minimal soft shadow beneath. Background is solid warm cream off-white color (hex #F5F3F0), seamless. Ultra high resolution 4K, material texture and stitching clearly visible. Professional luxury e-commerce photography.""",
    
    # ACCESSORIES - Clean centered product shot
    "accessories": """Massimo Dutti luxury catalog product photograph. A {gender}'s {color} {material} {garment_type}, front view facing camera, perfectly centered in frame. {details} Soft diffused studio lighting from above. Background is solid warm cream off-white color (hex #F5F3F0), seamless. Ultra high resolution 4K. Professional luxury e-commerce photography.""",
    
    # DRESSES - Ghost mannequin full length, flowing naturally
    "dresses": """Massimo Dutti luxury catalog product photograph. A {gender}'s {color} {material} {garment_type} displayed on an invisible ghost mannequin, full length front view facing camera. The dress flows naturally as if worn, showing three-dimensional form and fabric drape. {details} Soft diffused studio lighting, minimal shadows. Background is solid warm cream off-white color (hex #F5F3F0), seamless. Ultra high resolution 4K, fabric texture visible. Professional luxury e-commerce photography.""",
}

# Garment to category mapping
GARMENT_CATEGORIES = {
    # Tops
    "jacket": "tops", "bomber jacket": "tops", "leather jacket": "tops", "denim jacket": "tops",
    "blazer": "tops", "coat": "tops", "hoodie": "tops", "cardigan": "tops", "sweater": "tops",
    "shirt": "tops", "dress shirt": "tops", "polo shirt": "tops", "polo": "tops",
    "t-shirt": "tops", "tank top": "tops", "blouse": "tops", "top": "tops", "vest": "tops",
    "pullover": "tops", "zip polo": "tops", "knit polo": "tops",
    
    # Pants
    "jeans": "pants", "dress pants": "pants", "trousers": "pants", "chinos": "pants",
    "shorts": "pants", "joggers": "pants", "sweatpants": "pants", "cargo pants": "pants",
    "corduroy pants": "pants", "wool trousers": "pants", "slim pants": "pants",
    
    # Shoes
    "sneakers": "shoes", "boots": "shoes", "dress shoes": "shoes", "loafers": "shoes",
    "heels": "shoes", "sandals": "shoes", "oxfords": "shoes", "brogues": "shoes",
    "moccasins": "shoes", "slip-ons": "shoes", "espadrilles": "shoes", "derby shoes": "shoes",
    
    # Accessories
    "bag": "accessories", "tote bag": "accessories", "backpack": "accessories",
    "crossbody bag": "accessories", "belt": "accessories", "hat": "accessories",
    "scarf": "accessories", "watch": "accessories", "wallet": "accessories",
    "sunglasses": "accessories", "tie": "accessories", "gloves": "accessories",
    
    # Dresses & Skirts
    "dress": "dresses", "skirt": "dresses", "maxi dress": "dresses", "midi dress": "dresses",
}

# Garment-specific detail mappings - HYPER DETAILED for Massimo Dutti style
GARMENT_DETAILS = {
    # Tops with collars and zippers
    "jacket": "Collar stands naturally with visible inner lining at neck. Zipper half-open with metal pull tab visible. Fabric texture crisp and defined.",
    "bomber jacket": "Ribbed collar, cuffs, and hem perfectly defined. Zipper with metal pull tab visible. Fabric smooth and clean.",
    "leather jacket": "Rich leather grain texture visible throughout. Collar perfectly shaped. Metal zipper and hardware prominent.",
    "denim jacket": "Denim weave texture visible. Collar stands crisp. Metal buttons clearly shown. Stitching details visible.",
    "blazer": "Lapels perfectly pressed with sharp edges. Buttons positioned symmetrically. Inner lining visible at collar.",
    "coat": "Collar perfectly shaped with inner lining slightly visible. Buttons aligned. Fabric drape natural.",
    "hoodie": "Hood naturally shaped at back of neck. Drawstrings visible. Kangaroo pocket defined.",
    "cardigan": "Neckline forms perfect V-shape. Buttons evenly spaced and visible. Ribbed hem and cuffs.",
    "sweater": "Ribbed neckline and cuffs perfectly defined. Knit texture pattern visible throughout body.",
    "pullover": "Neckline perfectly round. Ribbed hem and cuffs visible. Knit texture prominent.",
    
    # Polo and knit tops - MATCHING THE CAMEL POLO REFERENCE EXACTLY
    "polo shirt": "Collar stands naturally with visible inner lining pattern at neck. Placket with buttons/zip visible. Ribbed cuffs.",
    "polo": "Collar stands naturally with visible inner patterned lining at neck. Vertical ribbed knit texture on torso body, smooth knit on sleeves. Ribbed cuffs and hem band. Half-zip with metal zipper pull visible.",
    "zip polo": "Collar stands naturally with inner patterned lining visible at neck. Half-zip with metal zipper pull. Vertical ribbed knit texture on body, smooth sleeves. Ribbed cuffs and hem.",
    "knit polo": "Collar stands naturally with inner lining visible. Vertical ribbed knit texture on torso, smooth knit sleeves. Perfectly ribbed cuffs and hem band.",
    
    # Shirts
    "shirt": "Collar perfectly pressed and stands naturally. Buttons aligned down center placket. Fabric crisp.",
    "dress shirt": "Collar crisp and perfectly shaped. Button placket immaculate. French cuffs visible.",
    "t-shirt": "Crew neckline perfectly round. Fabric drapes naturally showing cotton texture.",
    "tank top": "Neckline and armholes cleanly finished. Fabric hangs naturally.",
    "blouse": "Neckline elegantly shaped. Fabric flows naturally showing delicate texture.",
    
    # Pants - MATCHING THE CREAM PANTS REFERENCE EXACTLY
    "jeans": "Waistband structured with button, fly, and belt loops visible. Inner waistband lining slightly visible at top. Denim texture throughout.",
    "dress pants": "Waistband perfectly pressed with button and belt loops. Inner waistband lining visible at top edge. Natural crease running down center of each leg from hip to hem.",
    "trousers": "Waistband sits flat with button and belt loops. Inner lining visible at top edge. Fabric falls in clean lines with natural center crease.",
    "chinos": "Waistband clean with button visible. Inner lining at top edge. Subtle cotton twill texture.",
    "shorts": "Waistband structured with button visible. Hem cleanly finished.",
    "joggers": "Elastic waistband with drawstring visible. Ribbed cuffs at ankles.",
    "corduroy pants": "Waistband shows button and belt loops. Inner lining visible. Vertical corduroy ribbing texture throughout.",
    
    # Dresses & Skirts
    "dress": "Neckline elegantly shaped. Waist defined. Fabric flows naturally to hem.",
    "skirt": "Waistband perfectly finished. Fabric falls in clean lines to even hem.",
    
    # Shoes - MATCHING THE BURGUNDY LOAFERS REFERENCE EXACTLY
    "sneakers": "Laces neatly tied. Side profile visible. Stitching details sharp. Sole clearly shown.",
    "boots": "Leather perfectly conditioned. Laces or zipper visible. Sole and heel profile shown.",
    "dress shoes": "Leather polished to shine. Stitching along welt visible. Laces neatly tied.",
    "loafers": "Rich suede texture visible. Hand-stitching around toe box (moccasin construction). Clean white rubber sole visible. Heel counter defined.",
    "oxfords": "Leather polished. Closed lacing visible. Brogue detailing if present.",
    "moccasins": "Soft suede texture visible. Hand-stitching around toe. Clean flexible sole.",
    "slip-ons": "Material texture visible. Clean vamp. Sole profile shown.",
    "heels": "Heel shape elegant. Material texture visible. Sole and heel tip shown.",
    "sandals": "Straps perfectly arranged. Buckle details visible. Sole profile shown.",
    
    # Accessories
    "bag": "Handles and hardware perfectly positioned. Closure mechanism visible. Leather grain texture shown.",
    "tote bag": "Handles symmetrical. Stitching visible along edges. Interior slightly visible.",
    "backpack": "Straps symmetrical. Zippers and pulls visible. Fabric texture shown.",
    "crossbody bag": "Strap hardware visible. Closure and front details shown. Leather texture prominent.",
    "belt": "Buckle centered with visible prong. Leather grain texture full length. Stitching visible.",
    "hat": "Shape perfectly maintained. Material texture visible. Any hardware or trim shown.",
    "scarf": "Fabric elegantly draped showing weave pattern and fringe details.",
    "watch": "Face clearly visible. Band material and clasp shown. Polished finish.",
}


def generate_product_card(
    garment_type: str,
    color: str,
    material: str = "fabric",
    gender: str = "men"
) -> Optional[str]:
    """
    Generate a professional product card image using Replicate FLUX.
    
    Args:
        garment_type: Type of garment (e.g., "t-shirt", "jacket", "dress pants")
        color: Color name (e.g., "navy blue", "black", "taupe")
        material: Material type (e.g., "cotton", "denim", "leather")
        gender: "men" or "women"
        
    Returns:
        Base64 encoded image data URL, or None if generation fails
    """
    try:
        # Clean up garment type
        garment_clean = garment_type.replace("_", " ").lower()
        
        # Material mapping
        material_map = {
            "denim": "denim",
            "cotton": "cotton fabric",
            "leather": "leather",
            "wool": "wool",
            "silk": "silk",
            "polyester": "synthetic fabric",
            "linen": "linen",
            "suede": "suede",
            "canvas": "canvas",
            "knit": "knit fabric",
            "corduroy": "corduroy",
            "velvet": "velvet",
            "cashmere": "cashmere",
            "fleece": "fleece",
        }
        material_clean = material_map.get(material.lower(), "fabric")
        
        # Get garment-specific details
        details = GARMENT_DETAILS.get(garment_clean, "The fabric texture and construction details are visible.")
        
        # Determine category for appropriate prompt template
        category = GARMENT_CATEGORIES.get(garment_clean, "tops")  # Default to tops template
        
        # Get the category-specific prompt template
        prompt_template = PROMPT_TEMPLATES.get(category, PROMPT_TEMPLATES["tops"])
        
        # Generate prompt with the category-specific template
        prompt = prompt_template.format(
            gender=gender,
            color=color.lower(),
            material=material_clean,
            garment_type=garment_clean,
            details=details
        )
        
        logger.info(f"🎨 Generating product card with FLUX: {garment_clean} ({color})")
        logger.info(f"   Prompt: {prompt[:100]}...")
        
        # Use HTTP API directly for more control
        headers = {
            "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Create prediction
        create_url = "https://api.replicate.com/v1/predictions"
        payload = {
            "version": "5599ed30703defd1d160a25a63321b4dec97101d98b4674bcc56e41f62f35637",  # FLUX schnell
            "input": {
                "prompt": prompt,
                "aspect_ratio": "4:5",
                "num_outputs": 1,
                "output_format": "webp",
                "output_quality": 90
            }
        }
        
        logger.info(f"   📡 Calling Replicate API...")
        
        # Retry logic for rate limiting (429 errors)
        max_retries = 3
        retry_delay = 12  # seconds between retries (6 requests/min = 10s minimum)
        response = None # Initialize response outside the loop
        
        for attempt in range(max_retries):
            # Create the prediction
            response = requests.post(create_url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 201:
                break  # Success!
            elif response.status_code == 429:
                # Rate limited - wait and retry
                if attempt < max_retries - 1:
                    logger.warning(f"   ⚠️ Rate limited (429), waiting {retry_delay}s before retry {attempt + 2}/{max_retries}...")
                    import time
                    time.sleep(retry_delay)
                else:
                    logger.error(f"   ❌ Rate limit exceeded after {max_retries} attempts")
                    return None
            else:
                logger.error(f"   ❌ Replicate API error: {response.status_code} - {response.text[:200]}")
                return None
        
        # If the loop completed without a successful response (e.g., all retries failed or non-429 error on first attempt)
        if response is None or response.status_code != 201:
            return None

        prediction = response.json()
        prediction_id = prediction.get("id")
        
        if not prediction_id:
            logger.error("   ❌ No prediction ID returned")
            return None
        
        logger.info(f"   🔄 Prediction created: {prediction_id}")
        
        # Poll for completion
        get_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
        
        import time
        max_attempts = 60  # 60 seconds max wait
        for attempt in range(max_attempts):
            time.sleep(1)
            
            status_response = requests.get(get_url, headers=headers, timeout=10)
            if status_response.status_code != 200:
                continue
            
            result = status_response.json()
            status = result.get("status")
            
            if status == "succeeded":
                output = result.get("output")
                if output and len(output) > 0:
                    image_url = output[0] if isinstance(output, list) else output
                    logger.info(f"   ✅ Generated! Downloading image...")
                    
                    # Download the image
                    img_response = requests.get(image_url, timeout=30)
                    if img_response.status_code == 200:
                        img_b64 = base64.b64encode(img_response.content).decode()
                        logger.info(f"   ✅ Success! Image size: {len(img_b64)} bytes")
                        return f"data:image/webp;base64,{img_b64}"
                    else:
                        logger.error(f"   ❌ Failed to download image: {img_response.status_code}")
                        return None
                else:
                    logger.error("   ❌ No output in result")
                    return None
                    
            elif status == "failed":
                error = result.get("error", "Unknown error")
                logger.error(f"   ❌ Generation failed: {error}")
                return None
            
            # Still processing, continue polling
            if attempt % 5 == 0:
                logger.info(f"   ⏳ Still generating... ({attempt}s)")
        
        logger.error("   ❌ Timeout waiting for generation")
        return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"   ❌ Network error: {e}")
        return None
    except Exception as e:
        logger.error(f"   ❌ Product card generation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def generate_product_card_with_fallback(
    garment_type: str,
    color: str,
    original_cutout: Optional[Image.Image] = None,
    material: str = "fabric",
    gender: str = "men"
) -> Optional[str]:
    """
    Generate product card with AI, falling back to styled original if AI fails.
    """
    # Try AI generation first
    ai_result = generate_product_card(garment_type, color, material, gender)
    if ai_result:
        return ai_result
    
    # Fallback to styling the original cutout
    if original_cutout:
        try:
            from PIL import ImageFilter, ImageEnhance
            
            CARD_SIZE = (800, 1000)
            BG_COLOR = (245, 245, 243)
            PADDING = 50
            
            # Create canvas
            canvas = Image.new('RGB', CARD_SIZE, BG_COLOR)
            
            # Resize clothing
            max_w = CARD_SIZE[0] - 2 * PADDING
            max_h = CARD_SIZE[1] - 2 * PADDING
            scale = min(max_w / original_cutout.width, max_h / original_cutout.height)
            new_w = int(original_cutout.width * scale)
            new_h = int(original_cutout.height * scale)
            
            resized = original_cutout.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
            # Center
            pos_x = (CARD_SIZE[0] - new_w) // 2
            pos_y = (CARD_SIZE[1] - new_h) // 2
            
            # Paste
            if resized.mode == 'RGBA':
                canvas.paste(resized, (pos_x, pos_y), resized)
            else:
                canvas.paste(resized, (pos_x, pos_y))
            
            # Encode
            buffer = io.BytesIO()
            canvas.save(buffer, format='PNG', quality=95)
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            
            logger.info("   ⚠️ Using fallback styled cutout")
            return f"data:image/png;base64,{img_b64}"
            
        except Exception as e:
            logger.error(f"Fallback styling failed: {e}")
            return None
    
    return None


# ============================================
# 🆕 BIREFNET - SOTA Background Removal
# ============================================

def remove_background_birefnet(image_b64: str) -> Optional[str]:
    """
    🎯 SOTA Background Removal using BIREFNET via Replicate.
    
    BIREFNET provides cleaner edges than rembg, especially for:
    - Fine details (hair, fur, lace)
    - Transparent materials
    - Complex clothing edges
    
    Args:
        image_b64: Base64 encoded image (with or without data URI prefix)
        
    Returns:
        Base64 encoded RGBA image with transparent background, or None if failed
    """
    try:
        import time
        
        # Clean the base64 string
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        
        logger.info("🔪 BIREFNET: Removing background with SOTA model...")
        
        headers = {
            "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # BIREFNET model on Replicate
        create_url = "https://api.replicate.com/v1/predictions"
        payload = {
            "version": "1c0ce3c6870ee7feee2c4c2e3c1a5ecc2bbdf93d97b72a8e6ce8eb8bb1b83f13",  # BIREFNET
            "input": {
                "image": f"data:image/png;base64,{image_b64}"
            }
        }
        
        # Retry logic for rate limiting
        max_retries = 3
        retry_delay = 12
        
        for attempt in range(max_retries):
            response = requests.post(create_url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 201:
                break
            elif response.status_code == 429:
                if attempt < max_retries - 1:
                    logger.warning(f"   ⚠️ BIREFNET rate limited, waiting {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error("   ❌ BIREFNET rate limit exceeded")
                    return None
            else:
                logger.error(f"   ❌ BIREFNET API error: {response.status_code}")
                return None
        
        if response.status_code != 201:
            return None
            
        prediction = response.json()
        prediction_id = prediction.get("id")
        
        if not prediction_id:
            return None
        
        # Poll for completion
        get_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
        
        for attempt in range(60):
            time.sleep(1)
            status_response = requests.get(get_url, headers=headers, timeout=10)
            
            if status_response.status_code != 200:
                continue
            
            result = status_response.json()
            status = result.get("status")
            
            if status == "succeeded":
                output = result.get("output")
                if output:
                    image_url = output if isinstance(output, str) else output[0]
                    img_response = requests.get(image_url, timeout=30)
                    if img_response.status_code == 200:
                        img_b64 = base64.b64encode(img_response.content).decode()
                        logger.info("   ✅ BIREFNET: Background removed successfully!")
                        return f"data:image/png;base64,{img_b64}"
                return None
                
            elif status == "failed":
                logger.error(f"   ❌ BIREFNET failed: {result.get('error')}")
                return None
        
        logger.error("   ❌ BIREFNET timeout")
        return None
        
    except Exception as e:
        logger.error(f"   ❌ BIREFNET error: {e}")
        return None


# ============================================
# 🆕 IMG2IMG Product Card Generation
# ============================================

def generate_product_card_img2img(
    original_cutout_b64: str,
    garment_type: str,
    color: str,
    material: str = "fabric",
    gender: str = "men"
) -> Optional[str]:
    """
    🎯 Generate product card using IMG2IMG - preserves original clothing!
    
    Instead of generating a new image from text, this uses the original
    clothing cutout and enhances it with a professional background.
    
    Args:
        original_cutout_b64: Base64 encoded cutout image (RGBA with transparency)
        garment_type: Type of garment
        color: Color of the garment
        material: Material type
        gender: "men" or "women"
        
    Returns:
        Base64 encoded product card image, or None if failed
    """
    try:
        import time
        
        # Clean the base64 string
        if ',' in original_cutout_b64:
            original_cutout_b64 = original_cutout_b64.split(',')[1]
        
        garment_clean = garment_type.replace("_", " ").lower()
        category = GARMENT_CATEGORIES.get(garment_clean, "tops")
        details = GARMENT_DETAILS.get(garment_clean, "Fabric texture visible.")
        
        # Create enhancement prompt (not full generation - just enhance!)
        prompt = f"""Professional Massimo Dutti catalog photo. {color} {material} {garment_clean} on solid warm cream off-white background (#F5F3F0). Soft diffused studio lighting. {details} Ultra high resolution 4K, sharp details. Keep the exact original garment, only enhance lighting and background."""
        
        logger.info(f"🖼️ IMG2IMG: Enhancing {garment_clean} with professional background...")
        
        headers = {
            "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Use FLUX img2img / fill model for background enhancement
        create_url = "https://api.replicate.com/v1/predictions"
        payload = {
            "version": "5599ed30703defd1d160a25a63321b4dec97101d98b4674bcc56e41f62f35637",  # FLUX schnell
            "input": {
                "prompt": prompt,
                "image": f"data:image/png;base64,{original_cutout_b64}",
                "prompt_strength": 0.35,  # Low = keep more of original image
                "num_outputs": 1,
                "aspect_ratio": "4:5",
                "output_format": "webp",
                "output_quality": 90
            }
        }
        
        # Retry logic
        max_retries = 3
        retry_delay = 12
        
        for attempt in range(max_retries):
            response = requests.post(create_url, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 201:
                break
            elif response.status_code == 429:
                if attempt < max_retries - 1:
                    logger.warning(f"   ⚠️ Rate limited, waiting {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    logger.error("   ❌ Rate limit exceeded")
                    return None
            else:
                logger.error(f"   ❌ API error: {response.status_code} - {response.text[:100]}")
                return None
        
        if response.status_code != 201:
            return None
            
        prediction = response.json()
        prediction_id = prediction.get("id")
        
        if not prediction_id:
            return None
        
        logger.info(f"   🔄 Prediction: {prediction_id}")
        
        # Poll for completion
        get_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
        
        for attempt in range(60):
            time.sleep(1)
            status_response = requests.get(get_url, headers=headers, timeout=10)
            
            if status_response.status_code != 200:
                continue
            
            result = status_response.json()
            status = result.get("status")
            
            if status == "succeeded":
                output = result.get("output")
                if output:
                    image_url = output[0] if isinstance(output, list) else output
                    img_response = requests.get(image_url, timeout=30)
                    if img_response.status_code == 200:
                        img_b64 = base64.b64encode(img_response.content).decode()
                        logger.info("   ✅ IMG2IMG: Product card created!")
                        return f"data:image/webp;base64,{img_b64}"
                return None
                
            elif status == "failed":
                logger.error(f"   ❌ Generation failed: {result.get('error')}")
                return None
            
            if attempt % 10 == 0:
                logger.info(f"   ⏳ Enhancing... ({attempt}s)")
        
        logger.error("   ❌ Timeout")
        return None
        
    except Exception as e:
        logger.error(f"   ❌ IMG2IMG error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


# ============================================
# 🆕 Ultimate Product Card - Best Quality
# ============================================

def generate_ultimate_product_card(
    original_image_b64: str,
    garment_type: str,
    color: str,
    material: str = "fabric",
    gender: str = "men",
    use_birefnet: bool = True,
    use_img2img: bool = True
) -> Optional[str]:
    """
    🏆 ULTIMATE PRODUCT CARD GENERATION
    
    Combines all improvements for best quality:
    1. BIREFNET for SOTA background removal
    2. IMG2IMG to enhance while preserving original
    3. Fallback to text generation if needed
    
    Args:
        original_image_b64: Base64 encoded original image (can have background)
        Other args same as generate_product_card
        
    Returns:
        Base64 encoded professional product card
    """
    logger.info(f"🏆 Ultimate Product Card: {garment_type} ({color})")
    
    cutout_b64 = original_image_b64
    
    # Step 1: BIREFNET background removal (if enabled)
    if use_birefnet:
        birefnet_result = remove_background_birefnet(original_image_b64)
        if birefnet_result:
            cutout_b64 = birefnet_result
            logger.info("   ✅ Step 1: BIREFNET cutout complete")
        else:
            logger.warning("   ⚠️ Step 1: BIREFNET failed, using original")
    
    # Step 2: IMG2IMG enhancement (if enabled)
    if use_img2img:
        img2img_result = generate_product_card_img2img(
            cutout_b64, garment_type, color, material, gender
        )
        if img2img_result:
            logger.info("   ✅ Step 2: IMG2IMG enhancement complete")
            return img2img_result
        else:
            logger.warning("   ⚠️ Step 2: IMG2IMG failed, trying text generation")
    
    # Step 3: Fallback to text-only generation
    text_result = generate_product_card(garment_type, color, material, gender)
    if text_result:
        logger.info("   ✅ Step 3: Text generation fallback complete")
        return text_result
    
    # Step 4: Ultimate fallback - styled cutout
    logger.warning("   ⚠️ All AI methods failed, using styled cutout")
    try:
        # Decode the cutout
        if ',' in cutout_b64:
            cutout_b64 = cutout_b64.split(',')[1]
        
        img_data = base64.b64decode(cutout_b64)
        cutout_img = Image.open(io.BytesIO(img_data))
        
        # Style it
        CARD_SIZE = (800, 1000)
        BG_COLOR = (245, 245, 243)
        PADDING = 50
        
        canvas = Image.new('RGB', CARD_SIZE, BG_COLOR)
        
        max_w = CARD_SIZE[0] - 2 * PADDING
        max_h = CARD_SIZE[1] - 2 * PADDING
        scale = min(max_w / cutout_img.width, max_h / cutout_img.height)
        new_w = int(cutout_img.width * scale)
        new_h = int(cutout_img.height * scale)
        
        resized = cutout_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        pos_x = (CARD_SIZE[0] - new_w) // 2
        pos_y = (CARD_SIZE[1] - new_h) // 2
        
        if resized.mode == 'RGBA':
            canvas.paste(resized, (pos_x, pos_y), resized)
        else:
            canvas.paste(resized, (pos_x, pos_y))
        
        buffer = io.BytesIO()
        canvas.save(buffer, format='PNG', quality=95)
        img_b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{img_b64}"
        
    except Exception as e:
        logger.error(f"   ❌ Ultimate fallback failed: {e}")
        return None

