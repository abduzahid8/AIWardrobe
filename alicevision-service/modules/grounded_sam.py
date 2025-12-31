"""
Grounded SAM2 Module - Text-Prompted Clothing Segmentation
Most powerful option for precise clothing detection and segmentation
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
import base64
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result from clothing detection"""
    category: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: Optional[np.ndarray] = None
    attributes: Dict[str, any] = None


@dataclass
class SegmentationResult:
    """Result from Grounded SAM2 segmentation"""
    detections: List[DetectionResult]
    combined_mask: np.ndarray
    processing_time: float
    model_used: str


class GroundedSAM2:
    """
    Grounded SAM2 for text-prompted clothing segmentation.
    
    Combines:
    - Grounding DINO: Text-prompted object detection
    - SAM2: High-quality segmentation
    
    Usage:
        detector = GroundedSAM2()
        result = detector.segment_clothing(image, prompts=["shirt", "pants"])
    """
    
    def __init__(
        self,
        device: str = "auto",
        confidence_threshold: float = 0.25,
        model_size: str = "base"  # base, large
    ):
        """
        Initialize Grounded SAM2.
        
        Args:
            device: "cuda", "cpu", or "auto"
            confidence_threshold: Minimum confidence for detections
            model_size: Model size variant
        """
        self.device = self._setup_device(device)
        self.confidence_threshold = confidence_threshold
        self.model_size = model_size
        
        logger.info(f"Initializing Grounded SAM2 on {self.device}")
        
        # Lazy loading - models only loaded on first use
        self._grounding_dino = None
        self._sam2_predictor = None
        self._models_loaded = False
        
    def _setup_device(self, device: str) -> str:
        """Setup compute device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_models(self):
        """Lazy load models on first use"""
        if self._models_loaded:
            return
            
        try:
            import time
            start = time.time()
            
            # Load Grounding DINO
            logger.info("Loading Grounding DINO...")
            from groundingdino.util.inference import load_model as load_grounding_model
            from groundingdino.util.inference import predict as grounding_predict
            
            # Model paths - download if needed
            grounding_config = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
            grounding_checkpoint = "weights/groundingdino_swint_ogc.pth"
            
            self._grounding_dino = load_grounding_model(
                grounding_config, 
                grounding_checkpoint,
                device=self.device
            )
            self._grounding_predict = grounding_predict
            
            # Load SAM2
            logger.info("Loading SAM2...")
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
            
            # Map model sizes to actual config names
            model_map = {
                "base": "b+",  # base maps to b+ (base plus)
                "large": "l",  # large maps to l
                "small": "s",
                "tiny": "t"
            }
            actual_size = model_map.get(self.model_size, "b+")
            
            sam2_checkpoint = f"weights/sam2_hiera_{actual_size}.pt"
            sam2_config = f"sam2/sam2_hiera_{actual_size}.yaml"
            
            sam2_model = build_sam2(sam2_config, sam2_checkpoint, device=self.device)
            self._sam2_predictor = SAM2ImagePredictor(sam2_model)
            
            self._models_loaded = True
            elapsed = time.time() - start
            logger.info(f"Models loaded successfully in {elapsed:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def segment_clothing(
        self,
        image: np.ndarray,
        prompts: List[str] = None,
        return_masks: bool = True
    ) -> SegmentationResult:
        """
        Segment clothing items using text prompts.
        
        Args:
            image: Input image (BGR format)
            prompts: List of clothing items to detect, e.g., ["shirt", "pants"]
                    If None, uses default clothing categories
            return_masks: Whether to return segmentation masks
            
        Returns:
            SegmentationResult with detections and masks
        """
        import time
        start_time = time.time()
        
        # Load models if needed
        self._load_models()
        
        # Default clothing prompts
        if prompts is None:
            prompts = [
                "shirt", "t-shirt", "blouse", "top",
                "pants", "jeans", "trousers",
                "dress", "skirt",
                "jacket", "coat", "hoodie",
                "shoes", "sneakers", "boots",
                "bag", "handbag", "backpack",
                "hat", "cap"
            ]
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Step 1: Detect with Grounding DINO
        detections = self._detect_with_grounding_dino(image_pil, prompts)
        
        if not detections:
            logger.warning("No clothing items detected")
            return SegmentationResult(
                detections=[],
                combined_mask=np.zeros(image.shape[:2], dtype=np.uint8),
                processing_time=time.time() - start_time,
                model_used="grounded_sam2"
            )
        
        # Step 2: Segment with SAM2
        if return_masks:
            detections = self._segment_with_sam2(image_rgb, detections)
        
        # Step 3: Combine masks
        combined_mask = self._combine_masks([d.mask for d in detections if d.mask is not None])
        
        processing_time = time.time() - start_time
        logger.info(f"Segmented {len(detections)} items in {processing_time:.2f}s")
        
        return SegmentationResult(
            detections=detections,
            combined_mask=combined_mask,
            processing_time=processing_time,
            model_used="grounded_sam2"
        )
    
    def _detect_with_grounding_dino(
        self,
        image: Image.Image,
        prompts: List[str]
    ) -> List[DetectionResult]:
        """Detect objects using Grounding DINO"""
        try:
            # Create text prompt
            text_prompt = " . ".join(prompts) + " ."
            
            # Run detection
            boxes, logits, phrases = self._grounding_predict(
                model=self._grounding_dino,
                image=image,
                caption=text_prompt,
                box_threshold=self.confidence_threshold,
                text_threshold=self.confidence_threshold,
                device=self.device
            )
            
            # Convert to DetectionResult objects
            detections = []
            h, w = image.size[1], image.size[0]
            
            for box, confidence, phrase in zip(boxes, logits, phrases):
                # Convert normalized coords to pixels
                x1, y1, x2, y2 = box.cpu().numpy()
                x1, x2 = int(x1 * w), int(x2 * w)
                y1, y2 = int(y1 * h), int(y2 * h)
                
                detections.append(DetectionResult(
                    category=phrase.strip(),
                    confidence=float(confidence),
                    bbox=(x1, y1, x2, y2),
                    mask=None
                ))
            
            logger.info(f"Detected {len(detections)} items: {[d.category for d in detections]}")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return []
    
    def _segment_with_sam2(
        self,
        image: np.ndarray,
        detections: List[DetectionResult]
    ) -> List[DetectionResult]:
        """Generate masks using SAM2"""
        try:
            # Set image for SAM2
            self._sam2_predictor.set_image(image)
            
            # Process each detection
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                
                # Use bbox as prompt for SAM2
                input_box = np.array([x1, y1, x2, y2])
                
                masks, scores, _ = self._sam2_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False
                )
                
                # Use the best mask
                detection.mask = masks[0].astype(np.uint8) * 255
            
            return detections
            
        except Exception as e:
            logger.error(f"Segmentation failed: {e}")
            return detections
    
    def _combine_masks(self, masks: List[np.ndarray]) -> np.ndarray:
        """Combine multiple masks into one"""
        if not masks:
            return np.zeros((1, 1), dtype=np.uint8)
        
        combined = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            combined = np.maximum(combined, mask)
        
        return combined
    
    def segment_from_base64(
        self,
        image_base64: str,
        prompts: List[str] = None
    ) -> Dict:
        """
        Segment clothing from base64 image.
        
        Args:
            image_base64: Base64-encoded image
            prompts: Clothing categories to detect
            
        Returns:
            Dictionary with results and base64-encoded mask
        """
        try:
            # Decode image
            image_bytes = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Segment
            result = self.segment_clothing(image, prompts=prompts)
            
            # Encode mask
            _, mask_encoded = cv2.imencode('.png', result.combined_mask)
            mask_base64 = base64.b64encode(mask_encoded).decode('utf-8')
            
            return {
                "success": True,
                "detections": [
                    {
                        "category": d.category,
                        "confidence": d.confidence,
                        "bbox": d.bbox
                    }
                    for d in result.detections
                ],
                "mask_base64": mask_base64,
                "processing_time": result.processing_time,
                "model": result.model_used
            }
            
        except Exception as e:
            logger.error(f"Segmentation from base64 failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Singleton instance
_grounded_sam_instance = None

def get_grounded_sam() -> GroundedSAM2:
    """Get singleton instance of Grounded SAM2"""
    global _grounded_sam_instance
    if _grounded_sam_instance is None:
        _grounded_sam_instance = GroundedSAM2()
    return _grounded_sam_instance
