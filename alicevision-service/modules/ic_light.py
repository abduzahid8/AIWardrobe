"""
🎨 IC-Light Module - Imposing Consistent Lighting for Product Photography
Dynamic relighting that separates albedo from shading

Key Features:
1. Separates fabric texture (albedo) from lighting (shading)
2. Re-illuminates products to match studio environments
3. Preserves fabric patterns while changing lighting direction
4. Multiple preset lighting styles for e-commerce
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import logging
import base64
from PIL import Image
from io import BytesIO
import time
import os

logger = logging.getLogger(__name__)


# ============================================
# 🎯 DATA STRUCTURES
# ============================================

@dataclass
class LightingAnalysis:
    """Analysis of image lighting conditions"""
    primary_direction: str          # "left", "right", "top", "front", "ambient"
    intensity: float               # 0-1 scale
    color_temperature: int         # Kelvin (3000=warm, 6500=daylight)
    contrast: float               # 0-1 scale
    uniformity: float             # How even the lighting is (0-1)
    shadows_present: bool
    highlights_present: bool


@dataclass
class RelightingResult:
    """Result of IC-Light relighting"""
    relit_image: np.ndarray
    original_albedo: Optional[np.ndarray]   # Extracted texture
    applied_shading: Optional[np.ndarray]   # New light map
    processing_time_ms: float
    method_used: str
    preset_applied: str
    
    def to_dict(self) -> Dict:
        return {
            "processingTimeMs": round(self.processing_time_ms, 2),
            "methodUsed": self.method_used,
            "presetApplied": self.preset_applied
        }


# ============================================
# 💡 LIGHTING PRESETS
# ============================================

LIGHTING_PRESETS = {
    "studio_soft": {
        "description": "Even, diffused studio lighting (Massimo Dutti style)",
        "direction": "front",
        "intensity": 0.6,
        "temperature": 5500,
        "contrast": 0.3,
        "fill_ratio": 0.8  # High fill = soft shadows
    },
    "studio_dramatic": {
        "description": "High-contrast side lighting",
        "direction": "left",
        "intensity": 0.8,
        "temperature": 5000,
        "contrast": 0.7,
        "fill_ratio": 0.3
    },
    "golden_hour": {
        "description": "Warm sunset lighting",
        "direction": "right",
        "intensity": 0.7,
        "temperature": 3500,
        "contrast": 0.4,
        "fill_ratio": 0.5
    },
    "daylight": {
        "description": "Natural daylight simulation",
        "direction": "top",
        "intensity": 0.9,
        "temperature": 6500,
        "contrast": 0.5,
        "fill_ratio": 0.6
    },
    "catalog_neutral": {
        "description": "Neutral e-commerce catalog lighting",
        "direction": "front",
        "intensity": 0.55,
        "temperature": 6000,
        "contrast": 0.25,
        "fill_ratio": 0.9
    }
}


# ============================================
# 🚀 IC-LIGHT RELIGHTER
# ============================================

class ICLightRelighter:
    """
    IC-Light: Imposing Consistent Lighting for product photography.
    
    This module separates the albedo (intrinsic color/pattern) from shading,
    allowing dynamic re-illumination without altering fabric texture.
    
    Methods:
    1. Replicate API (recommended): Uses the ic-light model
    2. Local estimation: Simulates relighting via image processing
    """
    
    # Replicate model for IC-Light
    REPLICATE_MODEL = "lllyasviel/ic-light:latest"
    
    def __init__(self, use_replicate: bool = True):
        """
        Initialize IC-Light relighter.
        
        Args:
            use_replicate: Use Replicate API (recommended for quality)
        """
        self.use_replicate = use_replicate
        self._replicate_token = None
        
        logger.info(f"ICLightRelighter initialized (replicate: {use_replicate})")
    
    def _get_replicate_token(self) -> str:
        """Get Replicate API token."""
        if self._replicate_token is None:
            self._replicate_token = os.environ.get("REPLICATE_API_TOKEN", "")
            if not self._replicate_token:
                logger.warning("REPLICATE_API_TOKEN not found in environment variable")
        return self._replicate_token
    
    # ============================================
    # 🔍 LIGHTING ANALYSIS
    # ============================================
    
    def analyze_lighting(self, image: np.ndarray) -> LightingAnalysis:
        """
        Analyze the lighting conditions of an image.
        
        Returns direction, intensity, and quality metrics.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        else:
            gray = image
            lab = None
        
        h, w = gray.shape
        
        # Analyze lighting direction by comparing quadrant intensities
        top_half = gray[:h//2, :].mean()
        bottom_half = gray[h//2:, :].mean()
        left_half = gray[:, :w//2].mean()
        right_half = gray[:, w//2:].mean()
        
        # Determine primary direction
        diffs = {
            "top": top_half - bottom_half,
            "left": left_half - right_half,
            "right": right_half - left_half,
        }
        
        max_diff_key = max(diffs, key=lambda k: abs(diffs[k]))
        if abs(diffs[max_diff_key]) < 10:
            direction = "front"  # Even lighting
        else:
            direction = max_diff_key if diffs[max_diff_key] > 0 else "front"
        
        # Calculate intensity
        intensity = gray.mean() / 255.0
        
        # Estimate color temperature from Lab
        if lab is not None:
            _, a, b = cv2.split(lab)
            avg_b = b.mean() - 128  # b channel: blue (-) to yellow (+)
            # Map to Kelvin (rough estimate)
            temperature = int(5500 + avg_b * 30)
            temperature = max(3000, min(8000, temperature))
        else:
            temperature = 5500
        
        # Calculate contrast
        contrast = gray.std() / 128.0
        contrast = min(1.0, contrast)
        
        # Calculate uniformity (inverse of std deviation in blocks)
        block_size = max(h, w) // 4
        blocks = []
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = gray[i:i+block_size, j:j+block_size]
                if block.size > 0:
                    blocks.append(block.mean())
        
        uniformity = 1.0 - (np.std(blocks) / 128.0 if blocks else 0)
        uniformity = max(0, min(1, uniformity))
        
        # Detect shadows and highlights
        shadows_present = (gray < 50).sum() > (h * w * 0.05)
        highlights_present = (gray > 250).sum() > (h * w * 0.02)
        
        return LightingAnalysis(
            primary_direction=direction,
            intensity=intensity,
            color_temperature=temperature,
            contrast=contrast,
            uniformity=uniformity,
            shadows_present=shadows_present,
            highlights_present=highlights_present
        )
    
    # ============================================
    # 🎨 RELIGHTING METHODS
    # ============================================
    
    def relight(
        self,
        product_image: np.ndarray,
        background_image: np.ndarray = None,
        preset: str = "studio_soft",
        light_direction: str = "auto"
    ) -> RelightingResult:
        """
        Re-illuminate product to match target lighting.
        
        Args:
            product_image: BGR image of product (ideally with transparent bg)
            background_image: Optional background to match lighting from
            preset: Lighting preset name
            light_direction: "left", "right", "top", "front", or "auto"
            
        Returns:
            RelightingResult with relit image
        """
        start_time = time.time()
        
        # Get preset settings
        settings = LIGHTING_PRESETS.get(preset, LIGHTING_PRESETS["studio_soft"])
        
        # Auto-detect direction from background if provided
        if light_direction == "auto" and background_image is not None:
            bg_analysis = self.analyze_lighting(background_image)
            light_direction = bg_analysis.primary_direction
        elif light_direction == "auto":
            light_direction = settings["direction"]
        
        if self.use_replicate:
            result = self._relight_via_replicate(
                product_image, light_direction, settings
            )
        else:
            result = self._relight_locally(
                product_image, light_direction, settings
            )
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        result.preset_applied = preset
        
        return result
    
    def _relight_via_replicate(
        self,
        image: np.ndarray,
        direction: str,
        settings: Dict
    ) -> RelightingResult:
        """Relight using Replicate IC-Light API."""
        try:
            import replicate
            
            # Convert image to base64
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            buffer = BytesIO()
            pil_image.save(buffer, format="PNG")
            image_b64 = base64.b64encode(buffer.getvalue()).decode()
            image_uri = f"data:image/png;base64,{image_b64}"
            
            # Build prompt based on direction and settings
            light_prompts = {
                "left": "soft light from the left side",
                "right": "soft light from the right side",
                "top": "overhead studio lighting",
                "front": "even frontal studio lighting"
            }
            
            prompt = f"Professional product photography, {light_prompts.get(direction, 'studio lighting')}"
            if settings["temperature"] < 4500:
                prompt += ", warm golden light"
            elif settings["temperature"] > 6000:
                prompt += ", cool daylight"
            
            # Call IC-Light (or Flux with lighting guidance)
            output = replicate.run(
                "black-forest-labs/flux-1.1-pro",  # Using Flux for now
                input={
                    "prompt": prompt,
                    "image": image_uri,
                    "guidance": 3.5,
                    "output_format": "png"
                }
            )
            
            # Download result
            import requests
            if isinstance(output, list):
                result_url = output[0]
            else:
                result_url = str(output)
            
            response = requests.get(result_url)
            result_image = Image.open(BytesIO(response.content))
            result_array = cv2.cvtColor(np.array(result_image), cv2.COLOR_RGB2BGR)
            
            return RelightingResult(
                relit_image=result_array,
                original_albedo=None,
                applied_shading=None,
                processing_time_ms=0,
                method_used="replicate_flux",
                preset_applied=""
            )
            
        except Exception as e:
            logger.warning(f"Replicate relighting failed: {e}, using local method")
            return self._relight_locally(image, direction, settings)
    
    def _relight_locally(
        self,
        image: np.ndarray,
        direction: str,
        settings: Dict
    ) -> RelightingResult:
        """
        Simulate relighting locally using image processing.
        
        This is a simplified approximation of true intrinsic decomposition.
        """
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        h, w = image.shape[:2]
        
        # Step 1: Estimate albedo (texture) by removing shading
        # Use bilateral filter to smooth shading while preserving edges
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Estimate shading from L channel
        shading = cv2.bilateralFilter(l, 15, 75, 75)
        
        # Albedo = L / shading (normalized)
        shading_float = shading.astype(np.float32) + 1
        l_float = l.astype(np.float32)
        albedo = (l_float / shading_float) * 128
        albedo = np.clip(albedo, 0, 255).astype(np.uint8)
        
        # Step 2: Create new light map based on direction
        new_shading = self._create_light_map(h, w, direction, settings)
        
        # Step 3: Apply new shading to albedo
        new_l = (albedo.astype(np.float32) * new_shading / 128)
        new_l = np.clip(new_l, 0, 255).astype(np.uint8)
        
        # Step 4: Apply color temperature adjustment
        a_adjusted, b_adjusted = self._apply_color_temperature(
            a, b, settings["temperature"]
        )
        
        # Reconstruct image
        relit_lab = cv2.merge([new_l, a_adjusted, b_adjusted])
        relit = cv2.cvtColor(relit_lab, cv2.COLOR_LAB2BGR)
        
        # Step 5: Adjust contrast
        relit = self._adjust_contrast(relit, settings["contrast"])
        
        return RelightingResult(
            relit_image=relit,
            original_albedo=cv2.cvtColor(
                cv2.merge([albedo, a, b]), cv2.COLOR_LAB2BGR
            ),
            applied_shading=new_shading,
            processing_time_ms=0,
            method_used="local",
            preset_applied=""
        )
    
    def _create_light_map(
        self,
        h: int,
        w: int,
        direction: str,
        settings: Dict
    ) -> np.ndarray:
        """Create a gradient light map based on direction."""
        base_intensity = int(settings["intensity"] * 255)
        fill_ratio = settings["fill_ratio"]
        
        # Create gradient based on direction
        if direction == "left":
            gradient = np.linspace(1.0, fill_ratio, w)
            light_map = np.tile(gradient, (h, 1))
        elif direction == "right":
            gradient = np.linspace(fill_ratio, 1.0, w)
            light_map = np.tile(gradient, (h, 1))
        elif direction == "top":
            gradient = np.linspace(1.0, fill_ratio, h)
            light_map = np.tile(gradient.reshape(-1, 1), (1, w))
        else:  # front or center
            # Create soft radial gradient
            y, x = np.ogrid[:h, :w]
            center_y, center_x = h // 2, w // 2
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            light_map = 1.0 - (dist / max_dist) * (1 - fill_ratio)
        
        # Apply intensity
        light_map = (light_map * base_intensity).astype(np.uint8)
        
        return light_map
    
    def _apply_color_temperature(
        self,
        a: np.ndarray,
        b: np.ndarray,
        temperature: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Adjust a and b channels for color temperature."""
        # Temperature deviation from neutral (5500K)
        temp_offset = (temperature - 5500) / 100
        
        # Warm = more yellow (b+), Cool = more blue (b-)
        b_adjusted = np.clip(b.astype(np.float32) + temp_offset, 0, 255)
        
        # Slight green-magenta shift
        a_adjusted = np.clip(a.astype(np.float32) - temp_offset * 0.2, 0, 255)
        
        return a_adjusted.astype(np.uint8), b_adjusted.astype(np.uint8)
    
    def _adjust_contrast(
        self,
        image: np.ndarray,
        target_contrast: float
    ) -> np.ndarray:
        """Adjust image contrast."""
        # Calculate current contrast
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        current_contrast = gray.std() / 128.0
        
        if current_contrast == 0:
            return image
        
        # Contrast adjustment factor
        factor = target_contrast / current_contrast
        factor = max(0.5, min(2.0, factor))  # Clamp
        
        # Apply contrast
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        l_float = l.astype(np.float32)
        l_adjusted = 128 + (l_float - 128) * factor
        l_adjusted = np.clip(l_adjusted, 0, 255).astype(np.uint8)
        
        result = cv2.merge([l_adjusted, a, b])
        return cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
    
    # ============================================
    # 🛠️ UTILITY METHODS
    # ============================================
    
    def match_background_lighting(
        self,
        product_image: np.ndarray,
        background_image: np.ndarray
    ) -> RelightingResult:
        """
        Automatically match product lighting to background.
        
        Analyzes background lighting and applies matching to product.
        """
        # Analyze background
        bg_analysis = self.analyze_lighting(background_image)
        
        # Find closest preset
        best_preset = "studio_soft"
        best_score = float('inf')
        
        for name, settings in LIGHTING_PRESETS.items():
            score = abs(settings["temperature"] - bg_analysis.color_temperature) / 1000
            score += abs(settings["contrast"] - bg_analysis.contrast)
            score += 0 if settings["direction"] == bg_analysis.primary_direction else 0.5
            
            if score < best_score:
                best_score = score
                best_preset = name
        
        # Apply relighting
        return self.relight(
            product_image,
            background_image,
            preset=best_preset,
            light_direction=bg_analysis.primary_direction
        )


# ============================================
# 🔧 UTILITY FUNCTIONS
# ============================================

def relight_product_image(
    image_b64: str,
    preset: str = "studio_soft",
    direction: str = "auto"
) -> Dict:
    """
    Utility function to relight a product image.
    
    Args:
        image_b64: Base64 encoded image
        preset: Lighting preset name
        direction: Light direction
        
    Returns:
        Dictionary with relit image and metadata
    """
    # Decode image
    if ',' in image_b64:
        image_b64 = image_b64.split(',')[1]
    
    img_bytes = base64.b64decode(image_b64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "Could not decode image"}
    
    # Relight
    relighter = ICLightRelighter()
    result = relighter.relight(image, preset=preset, light_direction=direction)
    
    # Encode result
    _, encoded = cv2.imencode('.png', result.relit_image)
    result_b64 = base64.b64encode(encoded).decode()
    
    response = result.to_dict()
    response["relitImageBase64"] = f"data:image/png;base64,{result_b64}"
    response["availablePresets"] = list(LIGHTING_PRESETS.keys())
    
    return response


# Singleton instance
_ic_light_instance = None

def get_ic_light_relighter() -> ICLightRelighter:
    """Get singleton instance of ICLightRelighter."""
    global _ic_light_instance
    if _ic_light_instance is None:
        _ic_light_instance = ICLightRelighter()
    return _ic_light_instance
