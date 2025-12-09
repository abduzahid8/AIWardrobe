"""
AliceVision Enhanced Segmentation Module
AI-powered clothing segmentation with edge refinement
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
from PIL import Image
from dataclasses import dataclass
import logging
import io
import base64

logger = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    """Result of clothing segmentation"""
    mask: np.ndarray
    segmented_image: np.ndarray
    confidence: float
    clothing_bbox: Optional[Tuple[int, int, int, int]] = None
    

class ClothingSegmentor:
    """
    Enhanced clothing segmentation with edge refinement.
    
    Uses multiple techniques for precise clothing cutouts:
    1. GrabCut for initial segmentation
    2. Edge-aware refinement
    3. Alpha matting for smooth edges
    """
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self._sam_model = None
        self._rembg_session = None
    
    def _get_rembg_session(self):
        """Lazy load rembg session"""
        if self._rembg_session is None:
            try:
                from rembg import new_session
                self._rembg_session = new_session("u2net_cloth_seg")
                logger.info("Loaded rembg clothing segmentation model")
            except ImportError:
                logger.warning("rembg not available, using fallback")
        return self._rembg_session
    
    def segment_with_rembg(self, image: np.ndarray) -> np.ndarray:
        """
        Use rembg's clothing-specific model for segmentation.
        """
        try:
            from rembg import remove
            
            # Convert numpy to PIL
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Remove background with clothing model
            session = self._get_rembg_session()
            if session:
                result = remove(pil_image, session=session)
            else:
                result = remove(pil_image)
            
            # Convert back to numpy with alpha
            result_array = np.array(result)
            
            return result_array
            
        except Exception as e:
            logger.error(f"rembg segmentation failed: {e}")
            return None
    
    def segment_with_grabcut(
        self, 
        image: np.ndarray,
        iterations: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        GrabCut-based segmentation as fallback.
        
        Returns:
            mask: Binary mask of segmented region
            segmented: Image with background removed
        """
        h, w = image.shape[:2]
        
        # Initial rectangle for GrabCut (assume clothing is center)
        margin_x = int(w * 0.1)
        margin_y = int(h * 0.05)
        rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
        
        # Initialize mask
        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Run GrabCut
        cv2.grabCut(
            image, mask, rect,
            bgd_model, fgd_model,
            iterations, cv2.GC_INIT_WITH_RECT
        )
        
        # Create binary mask
        binary_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255, 0
        ).astype(np.uint8)
        
        # Apply mask to image
        segmented = cv2.bitwise_and(image, image, mask=binary_mask)
        
        # Add alpha channel
        b, g, r = cv2.split(segmented)
        segmented_rgba = cv2.merge([r, g, b, binary_mask])
        
        return binary_mask, segmented_rgba
    
    def refine_edges(
        self, 
        image: np.ndarray, 
        mask: np.ndarray,
        feather_radius: int = 3
    ) -> np.ndarray:
        """
        Refine mask edges for smooth transitions.
        """
        # Gaussian blur for soft edges
        mask_float = mask.astype(np.float32) / 255.0
        
        blurred = cv2.GaussianBlur(
            mask_float, 
            (feather_radius*2+1, feather_radius*2+1), 
            0
        )
        
        # Edge-aware refinement using bilateral filter
        refined = cv2.bilateralFilter(
            (blurred * 255).astype(np.uint8),
            d=9, sigmaColor=75, sigmaSpace=75
        )
        
        return refined
    
    def detect_clothing_bbox(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect bounding box of clothing region.
        """
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if not contours:
            return None
        
        # Get largest contour
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        
        return (x, y, w, h)
    
    def add_white_background(
        self, 
        segmented_rgba: np.ndarray
    ) -> np.ndarray:
        """
        Add pure white background behind segmented clothing.
        """
        if segmented_rgba.shape[2] != 4:
            return segmented_rgba
        
        # Create white background
        h, w = segmented_rgba.shape[:2]
        white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Extract alpha channel
        alpha = segmented_rgba[:, :, 3:4] / 255.0
        rgb = segmented_rgba[:, :, :3]
        
        # Composite
        result = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        
        return result
    
    def segment(
        self, 
        image: np.ndarray,
        add_white_bg: bool = True,
        refine: bool = True
    ) -> SegmentationResult:
        """
        Full segmentation pipeline.
        
        Args:
            image: Input image (BGR)
            add_white_bg: Whether to add white background
            refine: Whether to apply edge refinement
            
        Returns:
            SegmentationResult with mask and segmented image
        """
        # Try rembg first (best quality)
        result = self.segment_with_rembg(image)
        
        if result is not None:
            # Extract mask from alpha channel
            if result.shape[2] == 4:
                mask = result[:, :, 3]
            else:
                mask = np.ones(result.shape[:2], dtype=np.uint8) * 255
            
            segmented = result
            confidence = 0.9  # rembg is generally reliable
        else:
            # Fallback to GrabCut
            logger.info("Using GrabCut fallback")
            mask, segmented = self.segment_with_grabcut(image)
            confidence = 0.7
        
        # Refine edges
        if refine and mask is not None:
            mask = self.refine_edges(image, mask)
        
        # Add white background
        if add_white_bg:
            final_image = self.add_white_background(segmented)
        else:
            final_image = segmented
        
        # Detect bounding box
        bbox = self.detect_clothing_bbox(mask)
        
        return SegmentationResult(
            mask=mask,
            segmented_image=final_image,
            confidence=confidence,
            clothing_bbox=bbox
        )


def segment_clothing_from_base64(
    image_base64: str,
    add_white_bg: bool = True
) -> Dict:
    """
    Utility function to segment clothing from base64 image.
    
    Args:
        image_base64: Base64-encoded image string
        
    Returns:
        Dictionary with segmented image and metadata
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
    
    # Segment
    segmentor = ClothingSegmentor()
    result = segmentor.segment(image, add_white_bg=add_white_bg)
    
    # Encode result to base64
    if add_white_bg:
        # RGB output
        _, buffer = cv2.imencode('.png', cv2.cvtColor(
            result.segmented_image, cv2.COLOR_RGB2BGR
        ))
    else:
        # RGBA output (PNG with transparency)
        _, buffer = cv2.imencode('.png', result.segmented_image)
    
    result_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return {
        "segmentedImage": f"data:image/png;base64,{result_base64}",
        "confidence": round(result.confidence, 4),
        "boundingBox": result.clothing_bbox,
        "hasTransparency": not add_white_bg
    }
