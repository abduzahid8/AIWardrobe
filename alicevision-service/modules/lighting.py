"""
AliceVision Lighting Normalization Module
Consistent catalog-quality lighting for product photos
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import logging
import base64

logger = logging.getLogger(__name__)


@dataclass
class LightingAnalysis:
    """Analysis of image lighting conditions"""
    brightness: float  # 0-1, 0.5 is ideal
    contrast: float    # 0-1, higher is better
    white_balance_temp: float  # Color temperature in Kelvin
    exposure: float    # -1 to 1, 0 is ideal
    uniformity: float  # 0-1, how uniform the lighting is


@dataclass
class NormalizationResult:
    """Result of lighting normalization"""
    normalized_image: np.ndarray
    original_analysis: LightingAnalysis
    applied_corrections: Dict[str, float]


class LightingNormalizer:
    """
    Normalize lighting conditions for consistent catalog-style photos.
    
    Aims for Massimo Dutti / premium e-commerce aesthetic:
    - Soft, even lighting
    - Neutral white balance (5500-6500K)
    - Slightly high key (bright but not blown out)
    - Minimal harsh shadows
    """
    
    def __init__(
        self,
        target_brightness: float = 0.55,
        target_contrast: float = 0.7,
        target_temperature: float = 6000,  # Kelvin
        target_exposure: float = 0.1  # Slightly bright
    ):
        self.target_brightness = target_brightness
        self.target_contrast = target_contrast
        self.target_temperature = target_temperature
        self.target_exposure = target_exposure
    
    def analyze_lighting(self, image: np.ndarray) -> LightingAnalysis:
        """
        Analyze the lighting conditions of an image.
        """
        # Convert to different color spaces
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Extract L channel (luminance)
        l_channel = lab[:, :, 0]
        
        # Calculate brightness (normalized 0-1)
        brightness = l_channel.mean() / 255.0
        
        # Calculate contrast
        contrast = l_channel.std() / 128.0  # Normalize to 0-1 range
        
        # Estimate color temperature from B/R ratio
        b, g, r = cv2.split(image)
        r_mean = r.mean()
        b_mean = b.mean()
        
        if r_mean > 0:
            br_ratio = b_mean / r_mean
            # Approximate temperature (rough mapping)
            if br_ratio > 1.2:
                temperature = 7500  # Cool
            elif br_ratio < 0.8:
                temperature = 4500  # Warm
            else:
                temperature = 5500 + (br_ratio - 1.0) * 2000
        else:
            temperature = 5500
        
        # Calculate exposure (based on histogram)
        hist = cv2.calcHist([l_channel], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Weight by position
        weights = np.linspace(-1, 1, 256)
        exposure = np.sum(hist * weights)
        
        # Calculate uniformity (inverse of standard deviation)
        uniformity = 1.0 - min(l_channel.std() / 100.0, 1.0)
        
        return LightingAnalysis(
            brightness=brightness,
            contrast=min(contrast, 1.0),
            white_balance_temp=temperature,
            exposure=exposure,
            uniformity=uniformity
        )
    
    def adjust_brightness(
        self, 
        image: np.ndarray, 
        current: float,
        target: float
    ) -> Tuple[np.ndarray, float]:
        """
        Adjust image brightness.
        """
        if abs(current - target) < 0.05:
            return image, 0.0
        
        # Calculate adjustment factor
        adjustment = (target - current) * 255
        
        # Apply in LAB space for better results
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] += adjustment
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
        
        result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        
        return result, adjustment
    
    def adjust_contrast(
        self, 
        image: np.ndarray,
        current: float,
        target: float
    ) -> Tuple[np.ndarray, float]:
        """
        Adjust image contrast using CLAHE.
        """
        if abs(current - target) < 0.1:
            return image, 0.0
        
        # Calculate clip limit based on how much we need to adjust
        adjustment = target - current
        clip_limit = 2.0 + adjustment * 2.0
        clip_limit = max(1.0, min(clip_limit, 4.0))
        
        # Apply CLAHE to L channel
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return result, adjustment
    
    def adjust_white_balance(
        self, 
        image: np.ndarray,
        current_temp: float,
        target_temp: float
    ) -> Tuple[np.ndarray, float]:
        """
        Adjust white balance to target temperature.
        """
        if abs(current_temp - target_temp) < 200:
            return image, 0.0
        
        # Calculate adjustment
        temp_diff = target_temp - current_temp
        
        # Convert to float
        result = image.astype(np.float32)
        
        # Adjust blue/red channels based on temperature
        # Warmer (lower K) = more red, less blue
        # Cooler (higher K) = more blue, less red
        
        adjustment_factor = temp_diff / 3000.0  # Normalize
        adjustment_factor = max(-0.3, min(adjustment_factor, 0.3))
        
        # Apply adjustment
        result[:, :, 0] *= (1 + adjustment_factor)  # Blue
        result[:, :, 2] *= (1 - adjustment_factor)  # Red
        
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result, adjustment_factor
    
    def adjust_exposure(
        self,
        image: np.ndarray,
        current: float,
        target: float
    ) -> Tuple[np.ndarray, float]:
        """
        Adjust exposure using gamma correction.
        """
        if abs(current - target) < 0.1:
            return image, 0.0
        
        # Calculate gamma
        adjustment = target - current
        gamma = 1.0 - (adjustment * 0.5)
        gamma = max(0.5, min(gamma, 2.0))
        
        # Build lookup table
        inv_gamma = 1.0 / gamma
        table = np.array([
            ((i / 255.0) ** inv_gamma) * 255
            for i in np.arange(0, 256)
        ]).astype(np.uint8)
        
        result = cv2.LUT(image, table)
        
        return result, adjustment
    
    def add_soft_vignette(
        self, 
        image: np.ndarray,
        strength: float = 0.2
    ) -> np.ndarray:
        """
        Add subtle vignette for professional look.
        """
        h, w = image.shape[:2]
        
        # Create gradient mask
        X = cv2.getGaussianKernel(w, w * 0.7)
        Y = cv2.getGaussianKernel(h, h * 0.7)
        mask = Y * X.T
        mask = mask / mask.max()
        
        # Invert and scale
        mask = 1.0 - ((1.0 - mask) * strength)
        
        # Apply to image
        result = image.astype(np.float32)
        for i in range(3):
            result[:, :, i] *= mask
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def normalize(
        self, 
        image: np.ndarray,
        add_vignette: bool = False
    ) -> NormalizationResult:
        """
        Full lighting normalization pipeline.
        
        Args:
            image: Input image (BGR)
            add_vignette: Whether to add subtle vignette
            
        Returns:
            NormalizationResult with normalized image and metadata
        """
        # Analyze original
        original_analysis = self.analyze_lighting(image)
        
        logger.info(f"Original lighting: brightness={original_analysis.brightness:.2f}, "
                   f"contrast={original_analysis.contrast:.2f}, "
                   f"temp={original_analysis.white_balance_temp:.0f}K")
        
        corrections = {}
        result = image.copy()
        
        # Apply corrections in order
        
        # 1. White balance first
        result, wb_adj = self.adjust_white_balance(
            result, 
            original_analysis.white_balance_temp,
            self.target_temperature
        )
        corrections['white_balance'] = wb_adj
        
        # 2. Exposure
        result, exp_adj = self.adjust_exposure(
            result,
            original_analysis.exposure,
            self.target_exposure
        )
        corrections['exposure'] = exp_adj
        
        # 3. Brightness
        result, bright_adj = self.adjust_brightness(
            result,
            original_analysis.brightness,
            self.target_brightness
        )
        corrections['brightness'] = bright_adj
        
        # 4. Contrast
        result, contrast_adj = self.adjust_contrast(
            result,
            original_analysis.contrast,
            self.target_contrast
        )
        corrections['contrast'] = contrast_adj
        
        # 5. Optional vignette
        if add_vignette:
            result = self.add_soft_vignette(result, strength=0.15)
            corrections['vignette'] = 0.15
        
        logger.info(f"Applied corrections: {corrections}")
        
        return NormalizationResult(
            normalized_image=result,
            original_analysis=original_analysis,
            applied_corrections=corrections
        )


def normalize_lighting_from_base64(
    image_base64: str,
    target_brightness: float = 0.55,
    target_temperature: float = 6000,
    add_vignette: bool = False
) -> Dict:
    """
    Utility function to normalize lighting from base64 image.
    
    Args:
        image_base64: Base64-encoded image string
        
    Returns:
        Dictionary with normalized image and metadata
    """
    # Remove data URL prefix if present
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]
    
    # Decode base64 to image
    img_bytes = base64.b64decode(image_base64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "Could not decode image"}
    
    # Normalize
    normalizer = LightingNormalizer(
        target_brightness=target_brightness,
        target_temperature=target_temperature
    )
    result = normalizer.normalize(image, add_vignette=add_vignette)
    
    # Encode result to base64
    _, buffer = cv2.imencode('.jpg', result.normalized_image, 
                             [cv2.IMWRITE_JPEG_QUALITY, 95])
    result_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "normalizedImage": f"data:image/jpeg;base64,{result_base64}",
        "originalAnalysis": {
            "brightness": round(result.original_analysis.brightness, 4),
            "contrast": round(result.original_analysis.contrast, 4),
            "colorTemperature": round(result.original_analysis.white_balance_temp),
            "exposure": round(result.original_analysis.exposure, 4),
            "uniformity": round(result.original_analysis.uniformity, 4)
        },
        "appliedCorrections": {
            k: round(v, 4) if isinstance(v, float) else v 
            for k, v in result.applied_corrections.items()
        }
    }
