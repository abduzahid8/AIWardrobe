"""
Professional Photo Quality Assessment
Image quality scoring for e-commerce product photos
"""

import cv2
import numpy as np
from typing import Dict, Tuple
import logging
from dataclasses import dataclass
import base64

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality assessment scores"""
    overall: float  # 0-100
    sharpness: float  # 0-100
    lighting: float  # 0-100
    composition: float  # 0-100
    background: float  # 0-100
    ecommerce_ready: bool
    issues: list
    recommendations: list


class QualityAssessor:
    """
    Professional photo quality assessment for e-commerce
    
    Evaluates:
    - Image sharpness (blur detection)
    - Lighting quality
    - Composition (centering, framing)
    - Background quality
    - Overall e-commerce readiness
    """
    
    def __init__(self):
        self.min_resolution = (800, 800)
        self.ideal_resolution = (2000, 2000)
    
    def assess_sharpness(self, image: np.ndarray) -> Tuple[float, Dict]:
        """
        Assess image sharpness using Laplacian variance
        
        Args:
            image: BGR image
            
        Returns:
            Sharpness score (0-100) and details
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Laplacian variance method
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # FFT-based sharpness
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # High frequency content
        h, w = magnitude.shape
        center_h, center_w = h // 2, w // 2
        radius = min(h, w) // 4
        
        y, x = np.ogrid[:h, :w]
        mask = (x - center_w)**2 + (y - center_h)**2 > radius**2
        high_freq_energy = np.sum(magnitude[mask]) / np.sum(magnitude)
        
        # Combine metrics
        # Laplacian typically ranges 0-1000+, normalize
        laplacian_score = min(laplacian_var / 500 * 100, 100)
        fft_score = high_freq_energy * 500  # Scale appropriately
        
        # Weighted combination
        sharpness_score = (laplacian_score * 0.7 + fft_score * 0.3)
        sharpness_score = max(0, min(100, sharpness_score))
        
        details = {
            "laplacianVariance": float(laplacian_var),
            "highFreqEnergy": float(high_freq_energy),
            "isBlurry": laplacian_var < 100,
            "blurLevel": "sharp" if laplacian_var > 500 else "moderate" if laplacian_var > 100 else "blurry"
        }
        
        return float(sharpness_score), details
    
    def assess_lighting(self, image: np.ndarray) -> Tuple[float, Dict]:
        """
        Assess lighting quality
        
        Args:
            image: BGR image
            
        Returns:
            Lighting score (0-100) and details
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Calculate lighting metrics
        mean_brightness = np.mean(l_channel)
        std_brightness = np.std(l_channel)
        
        # Check for over/underexposure
        overexposed = np.sum(l_channel > 250) / l_channel.size
        underexposed = np.sum(l_channel < 20) / l_channel.size
        
        # Ideal brightness is around 127-170
        brightness_score = 100 - abs(mean_brightness - 148) / 148 * 100
        brightness_score = max(0, brightness_score)
        
        # Penalize over/underexposure
        exposure_penalty = (overexposed + underexposed) * 200
        
        # Contrast score (std dev should be moderate)
        # Ideal std is around 30-60
        contrast_score = 100 - abs(std_brightness - 45) / 45 * 100
        contrast_score = max(0, contrast_score)
        
        # Combined lighting score
        lighting_score = (brightness_score * 0.5 + contrast_score * 0.5) - exposure_penalty
        lighting_score = max(0, min(100, lighting_score))
        
        details = {
            "meanBrightness": float(mean_brightness),
            "stdBrightness": float(std_brightness),
            "overexposedPercent": float(overexposed * 100),
            "underexposedPercent": float(underexposed * 100),
            "quality": "good" if lighting_score > 70 else "moderate" if lighting_score > 40 else "poor"
        }
        
        return float(lighting_score), details
    
    def assess_composition(self, image: np.ndarray) -> Tuple[float, Dict]:
        """
        Assess composition (centering, framing, aspect ratio)
        
        Args:
            image: BGR image
            
        Returns:
            Composition score (0-100) and details
        """
        h, w = image.shape[:2]
        
        # Aspect ratio score (1:1 is ideal for e-commerce)
        aspect_ratio = w / h
        ideal_aspect = 1.0
        aspect_score = 100 - abs(aspect_ratio - ideal_aspect) / ideal_aspect * 100
        aspect_score = max(0, aspect_score)
        
        # Resolution score
        total_pixels = w * h
        min_pixels = self.min_resolution[0] * self.min_resolution[1]
        ideal_pixels = self.ideal_resolution[0] * self.ideal_resolution[1]
        
        if total_pixels < min_pixels:
            resolution_score = (total_pixels / min_pixels) * 50
        elif total_pixels < ideal_pixels:
            resolution_score = 50 + (total_pixels - min_pixels) / (ideal_pixels - min_pixels) * 50
        else:
            resolution_score = 100
        
        # Edge detection for subject detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Find subject center of mass
        y_indices, x_indices = np.where(edges > 0)
        if len(x_indices) > 0:
            center_x = np.mean(x_indices)
            center_y = np.mean(y_indices)
            
            # Calculate centering score
            x_offset = abs(center_x - w/2) / (w/2)
            y_offset = abs(center_y - h/2) / (h/2)
            
            centering_score = 100 - ((x_offset + y_offset) / 2 * 100)
            centering_score = max(0, centering_score)
        else:
            centering_score = 50
        
        # Combined composition score
        composition_score = (
            aspect_score * 0.3 +
            resolution_score * 0.4 +
            centering_score * 0.3
        )
        
        details = {
            "resolution": f"{w}x{h}",
            "aspectRatio": round(aspect_ratio, 2),
            "totalPixels": total_pixels,
            "resolutionQuality": "high" if total_pixels > ideal_pixels else "medium" if total_pixels > min_pixels else "low",
            "centeringScore": round(centering_score, 1)
        }
        
        return float(composition_score), details
    
    def assess_background(self, image: np.ndarray) -> Tuple[float, Dict]:
        """
        Assess background quality (cleanness, distractions)
        
        Args:
            image: BGR image
            
        Returns:
            Background score (0-100) and details
        """
        h, w = image.shape[:2]
        
        # Sample edges (likely background)
        border_size = min(h, w) // 10
        
        top = image[:border_size, :]
        bottom = image[-border_size:, :]
        left = image[:, :border_size]
        right = image[:, -border_size:]
        
        edges = np.concatenate([
            top.reshape(-1, 3),
            bottom.reshape(-1, 3),
            left.reshape(-1, 3),
            right.reshape(-1, 3)
        ])
        
        # Calculate background uniformity
        bg_std = np.std(edges, axis=0).mean()
        bg_mean = np.mean(edges, axis=0)
        
        # Check if background is close to white (ideal for e-commerce)
        whiteness = np.mean(bg_mean) / 255
        is_white_bg = whiteness > 0.85 and bg_std < 20
        
        # Uniformity score (low std = uniform background)
        uniformity_score = max(0, 100 - bg_std * 2)
        
        # White background bonus
        if is_white_bg:
            background_score = 100
        else:
            # Score based on uniformity
            background_score = uniformity_score * 0.8
        
        details = {
            "stdDev": float(bg_std),
            "meanColor": [int(x) for x in bg_mean],
            "whiteness": round(whiteness * 100, 1),
            "isWhiteBackground": is_white_bg,
            "uniformity": round(uniformity_score, 1),
            "quality": "excellent" if is_white_bg else "good" if uniformity_score > 70 else "moderate"
        }
        
        return float(background_score), details
    
    def assess_quality(self, image: np.ndarray) -> QualityScore:
        """
        Complete quality assessment
        
        Args:
            image: BGR image
            
        Returns:
            QualityScore with all metrics
        """
        # Run all assessments
        sharpness, sharp_details = self.assess_sharpness(image)
        lighting, light_details = self.assess_lighting(image)
        composition, comp_details = self.assess_composition(image)
        background, bg_details = self.assess_background(image)
        
        # Calculate overall score (weighted)
        overall = (
            sharpness * 0.30 +
            lighting * 0.25 +
            composition * 0.25 +
            background * 0.20
        )
        
        # Identify issues
        issues = []
        recommendations = []
        
        if sharpness < 50:
            issues.append("Image is blurry or out of focus")
            recommendations.append("Use better lighting and ensure camera is stable")
        
        if lighting < 50:
            issues.append("Poor lighting quality")
            recommendations.append("Use studio lighting or natural daylight")
        
        if light_details["overexposedPercent"] > 5:
            issues.append("Image is overexposed")
            recommendations.append("Reduce exposure or use softer lighting")
        
        if light_details["underexposedPercent"] > 5:
            issues.append("Image is underexposed")
            recommendations.append("Increase lighting or adjust exposure")
        
        if composition < 60:
            if comp_details["resolutionQuality"] == "low":
                issues.append("Resolution too low for e-commerce")
                recommendations.append(f"Use at least {self.min_resolution[0]}x{self.min_resolution[1]} resolution")
            
            if comp_details["centeringScore"] < 60:
                issues.append("Subject is off-center")
                recommendations.append("Center the product in the frame")
        
        if not bg_details["isWhiteBackground"]:
            issues.append("Background is not white")
            recommendations.append("Use white background for professional e-commerce photos")
        
        # E-commerce readiness
        ecommerce_ready = (
            overall >= 70 and
            sharpness >= 60 and
            lighting >= 50 and
            bg_details["isWhiteBackground"]
        )
        
        return QualityScore(
            overall=round(overall, 1),
            sharpness=round(sharpness, 1),
            lighting=round(lighting, 1),
            composition=round(composition, 1),
            background=round(background, 1),
            ecommerce_ready=ecommerce_ready,
            issues=issues,
            recommendations=recommendations
        )


def assess_photo_quality_from_base64(image_base64: str) -> Dict:
    """
    Utility function for base64 image quality assessment
    
    Args:
        image_base64: Base64-encoded image
        
    Returns:
        Quality assessment dictionary
    """
    # Remove data URL prefix
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]
    
    # Decode
    img_bytes = base64.b64decode(image_base64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "Could not decode image"}
    
    # Assess quality
    assessor = QualityAssessor()
    result = assessor.assess_quality(image)
    
    return {
        "overall": result.overall,
        "scores": {
            "sharpness": result.sharpness,
            "lighting": result.lighting,
            "composition": result.composition,
            "background": result.background
        },
        "ecommerceReady": result.ecommerce_ready,
        "issues": result.issues,
        "recommendations": result.recommendations,
        "grade": "A" if result.overall >= 90 else "B" if result.overall >= 75 else "C" if result.overall >= 60 else "D"
    }
