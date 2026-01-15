"""
Brand Detection Module for AIWardrobe
Uses AI to identify clothing brands from images
"""

import logging
import re
from typing import Dict, List, Optional, Tuple
import os

logger = logging.getLogger(__name__)

# Known fashion brands database
FASHION_BRANDS = {
    # Luxury
    "gucci": {"tier": "luxury", "price_range": "$$$$$"},
    "louis vuitton": {"tier": "luxury", "price_range": "$$$$$"},
    "prada": {"tier": "luxury", "price_range": "$$$$$"},
    "chanel": {"tier": "luxury", "price_range": "$$$$$"},
    "dior": {"tier": "luxury", "price_range": "$$$$$"},
    "balenciaga": {"tier": "luxury", "price_range": "$$$$$"},
    "versace": {"tier": "luxury", "price_range": "$$$$$"},
    "burberry": {"tier": "luxury", "price_range": "$$$$"},
    "fendi": {"tier": "luxury", "price_range": "$$$$$"},
    "valentino": {"tier": "luxury", "price_range": "$$$$$"},
    
    # Premium
    "ralph lauren": {"tier": "premium", "price_range": "$$$$"},
    "calvin klein": {"tier": "premium", "price_range": "$$$"},
    "tommy hilfiger": {"tier": "premium", "price_range": "$$$"},
    "hugo boss": {"tier": "premium", "price_range": "$$$$"},
    "michael kors": {"tier": "premium", "price_range": "$$$"},
    "coach": {"tier": "premium", "price_range": "$$$"},
    "lacoste": {"tier": "premium", "price_range": "$$$"},
    "massimo dutti": {"tier": "premium", "price_range": "$$$"},
    "theory": {"tier": "premium", "price_range": "$$$"},
    "ted baker": {"tier": "premium", "price_range": "$$$"},
    
    # Mid-range
    "zara": {"tier": "mid", "price_range": "$$"},
    "h&m": {"tier": "mid", "price_range": "$$"},
    "mango": {"tier": "mid", "price_range": "$$"},
    "uniqlo": {"tier": "mid", "price_range": "$$"},
    "cos": {"tier": "mid", "price_range": "$$"},
    "& other stories": {"tier": "mid", "price_range": "$$"},
    "arket": {"tier": "mid", "price_range": "$$"},
    "banana republic": {"tier": "mid", "price_range": "$$"},
    "gap": {"tier": "mid", "price_range": "$$"},
    "j.crew": {"tier": "mid", "price_range": "$$"},
    
    # Sportswear
    "nike": {"tier": "sport", "price_range": "$$"},
    "adidas": {"tier": "sport", "price_range": "$$"},
    "puma": {"tier": "sport", "price_range": "$$"},
    "under armour": {"tier": "sport", "price_range": "$$"},
    "new balance": {"tier": "sport", "price_range": "$$"},
    "reebok": {"tier": "sport", "price_range": "$$"},
    "lululemon": {"tier": "sport", "price_range": "$$$"},
    "gymshark": {"tier": "sport", "price_range": "$$"},
    
    # Fast fashion
    "shein": {"tier": "fast", "price_range": "$"},
    "forever 21": {"tier": "fast", "price_range": "$"},
    "primark": {"tier": "fast", "price_range": "$"},
    "boohoo": {"tier": "fast", "price_range": "$"},
    "asos": {"tier": "fast", "price_range": "$"},
}

# Brand logo visual patterns (simplified)
BRAND_PATTERNS = {
    "gucci": ["interlocking g", "green red stripe", "gg pattern"],
    "louis vuitton": ["lv monogram", "damier pattern", "fleur de lis"],
    "nike": ["swoosh", "checkmark logo"],
    "adidas": ["three stripes", "trefoil", "mountain logo"],
    "ralph lauren": ["polo player", "horse logo"],
    "tommy hilfiger": ["flag logo", "red white blue"],
    "calvin klein": ["ck logo", "minimalist text"],
    "chanel": ["double c", "interlocking c"],
}


class BrandDetector:
    """AI-powered brand detection for clothing items"""
    
    def __init__(self):
        self.brands_db = FASHION_BRANDS
        self.brand_patterns = BRAND_PATTERNS
        self._vision_model = None
        logger.info("BrandDetector initialized")
    
    def detect_brand_from_image(self, image_base64: str) -> Dict:
        """
        Detect brand from clothing image using AI vision
        
        Args:
            image_base64: Base64 encoded image
            
        Returns:
            Dict with detected brand info
        """
        try:
            # Try to use OpenAI Vision or similar for brand detection
            detected = self._analyze_with_vision(image_base64)
            if detected:
                return detected
            
            # Fallback: Return unknown brand
            return {
                "detected": False,
                "brand": None,
                "confidence": 0,
                "tier": "unknown",
                "price_range": "unknown",
                "method": "fallback"
            }
            
        except Exception as e:
            logger.error(f"Brand detection error: {e}")
            return {
                "detected": False,
                "brand": None,
                "confidence": 0,
                "error": str(e)
            }
    
    def _analyze_with_vision(self, image_base64: str) -> Optional[Dict]:
        """Use vision model to detect brand"""
        try:
            import google.generativeai as genai
            
            api_key = os.environ.get("GOOGLE_API_KEY")
            if not api_key:
                logger.warning("GOOGLE_API_KEY not found, using pattern matching")
                return None
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            # Prepare image
            import base64
            image_bytes = base64.b64decode(image_base64)
            
            prompt = """Analyze this clothing item image and identify:
1. The brand (if visible from logo, tag, or distinctive patterns)
2. Confidence level (high/medium/low)
3. What visual elements helped identify the brand

Respond in this exact JSON format:
{
    "brand": "Brand Name or null",
    "confidence": "high/medium/low",
    "visual_clues": ["list", "of", "clues"],
    "brand_tier": "luxury/premium/mid/fast/sport/unknown"
}

If you cannot identify the brand, return:
{"brand": null, "confidence": "none", "visual_clues": [], "brand_tier": "unknown"}
"""
            
            response = model.generate_content([
                prompt,
                {"mime_type": "image/jpeg", "data": image_bytes}
            ])
            
            # Parse response
            import json
            result_text = response.text
            
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                brand_name = result.get("brand")
                if brand_name:
                    brand_lower = brand_name.lower()
                    brand_info = self.brands_db.get(brand_lower, {})
                    
                    confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.4, "none": 0}
                    
                    return {
                        "detected": True,
                        "brand": brand_name,
                        "confidence": confidence_map.get(result.get("confidence", "low"), 0.5),
                        "tier": brand_info.get("tier", result.get("brand_tier", "unknown")),
                        "price_range": brand_info.get("price_range", "unknown"),
                        "visual_clues": result.get("visual_clues", []),
                        "method": "gemini_vision"
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            return None
    
    def get_brand_info(self, brand_name: str) -> Dict:
        """Get detailed information about a brand"""
        brand_lower = brand_name.lower()
        
        if brand_lower in self.brands_db:
            info = self.brands_db[brand_lower]
            return {
                "name": brand_name.title(),
                "tier": info["tier"],
                "price_range": info["price_range"],
                "found": True
            }
        
        return {
            "name": brand_name.title(),
            "tier": "unknown",
            "price_range": "unknown",
            "found": False
        }
    
    def search_similar_brands(self, brand_name: str, tier: Optional[str] = None) -> List[Dict]:
        """Find similar brands based on tier"""
        brand_lower = brand_name.lower()
        brand_info = self.brands_db.get(brand_lower)
        
        if not brand_info and not tier:
            return []
        
        target_tier = tier or brand_info["tier"]
        
        similar = []
        for name, info in self.brands_db.items():
            if info["tier"] == target_tier and name != brand_lower:
                similar.append({
                    "name": name.title(),
                    "tier": info["tier"],
                    "price_range": info["price_range"]
                })
        
        return similar[:10]  # Return top 10


# Singleton instance
_brand_detector = None

def get_brand_detector() -> BrandDetector:
    global _brand_detector
    if _brand_detector is None:
        _brand_detector = BrandDetector()
    return _brand_detector
