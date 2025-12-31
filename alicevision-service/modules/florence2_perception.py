"""
🎯 Florence-2 Unified Perception Engine
State-of-the-art Vision-Language Model for fashion understanding

Replaces the entire ensemble (YOLO + SegFormer + CLIP) with ONE model:
- Object Detection (<OD>)
- Dense Region Captioning (<DENSE_REGION_CAPTION>)
- OCR for Care Labels (<OCR>)
- Detailed Captioning (<DETAILED_CAPTION>)
- Visual Grounding (<CAPTION_TO_PHRASE_GROUNDING>)
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
import base64
from PIL import Image
from io import BytesIO
import re

logger = logging.getLogger(__name__)


# ============================================
# 🎯 FLORENCE-2 TASK DEFINITIONS
# ============================================

class Florence2Tasks:
    """Supported task prompts for Florence-2"""
    # Detection
    OBJECT_DETECTION = "<OD>"
    DENSE_REGION_CAPTION = "<DENSE_REGION_CAPTION>"
    REGION_PROPOSAL = "<REGION_PROPOSAL>"
    
    # Captioning
    CAPTION = "<CAPTION>"
    DETAILED_CAPTION = "<DETAILED_CAPTION>"
    MORE_DETAILED_CAPTION = "<MORE_DETAILED_CAPTION>"
    
    # OCR
    OCR = "<OCR>"
    OCR_WITH_REGION = "<OCR_WITH_REGION>"
    
    # Grounding
    CAPTION_TO_PHRASE_GROUNDING = "<CAPTION_TO_PHRASE_GROUNDING>"
    REFERRING_EXPRESSION_SEGMENTATION = "<REFERRING_EXPRESSION_SEGMENTATION>"
    OPEN_VOCABULARY_DETECTION = "<OPEN_VOCABULARY_DETECTION>"
    
    # Region-specific
    REGION_TO_CATEGORY = "<REGION_TO_CATEGORY>"
    REGION_TO_DESCRIPTION = "<REGION_TO_DESCRIPTION>"


# ============================================
# 🎯 DATA STRUCTURES
# ============================================

@dataclass
class DetectedObject:
    """Single detected object from Florence-2"""
    label: str
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 normalized
    confidence: float = 1.0
    caption: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "caption": self.caption
        }


@dataclass
class PerceptionResult:
    """Complete perception result from Florence-2"""
    success: bool
    task: str
    
    # Detection results
    objects: List[DetectedObject] = field(default_factory=list)
    
    # Caption results
    caption: str = ""
    detailed_caption: str = ""
    
    # OCR results
    ocr_text: str = ""
    ocr_regions: List[Dict] = field(default_factory=list)
    
    # Fashion-specific
    garment_type: str = ""
    colors: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    materials: List[str] = field(default_factory=list)
    style_tags: List[str] = field(default_factory=list)
    
    # Processing info
    processing_time_ms: float = 0
    model_used: str = "florence-2-large"
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "task": self.task,
            "objects": [obj.to_dict() for obj in self.objects],
            "caption": self.caption,
            "detailedCaption": self.detailed_caption,
            "ocrText": self.ocr_text,
            "ocrRegions": self.ocr_regions,
            "garmentType": self.garment_type,
            "colors": self.colors,
            "patterns": self.patterns,
            "materials": self.materials,
            "styleTags": self.style_tags,
            "processingTimeMs": self.processing_time_ms,
            "modelUsed": self.model_used
        }


# ============================================
# 🚀 FLORENCE-2 PERCEPTION ENGINE
# ============================================

class Florence2Perception:
    """
    🎯 UNIFIED PERCEPTION ENGINE
    
    Florence-2 is a generative VLM that can perform ANY vision task
    through natural language prompting. This replaces:
    - YOLO (object detection)
    - SegFormer (segmentation categories)
    - CLIP (classification)
    - Tesseract (OCR)
    
    With ONE model that understands fashion context.
    """
    
    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-large",
        device: str = "auto",
        use_flash_attention: bool = True
    ):
        """
        Initialize Florence-2.
        
        Args:
            model_id: HuggingFace model ID
            device: "auto", "cuda", "mps", or "cpu"
            use_flash_attention: Use Flash Attention 2 if available
        """
        self.model_id = model_id
        self.device = self._setup_device(device)
        self.use_flash_attention = use_flash_attention
        
        # Lazy-loaded model
        self._model = None
        self._processor = None
        
        # Fashion-specific knowledge
        self._fashion_prompts = self._build_fashion_prompts()
        
        logger.info(f"Florence2Perception initialized (device={self.device})")
    
    def _setup_device(self, device: str) -> str:
        """Setup compute device."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return device
    
    def _load_model(self):
        """Lazy load Florence-2 model."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            
            logger.info(f"Loading Florence-2 from {self.model_id}...")
            
            # Load processor
            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True
            )
            
            # Load model with appropriate settings
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
            }
            
            if self.use_flash_attention and self.device == "cuda":
                model_kwargs["attn_implementation"] = "flash_attention_2"
            
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs
            ).to(self.device)
            
            self._model.eval()
            
            logger.info(f"Florence-2 loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Florence-2: {e}")
            raise
    
    def _build_fashion_prompts(self) -> Dict[str, str]:
        """Build fashion-specific prompt templates."""
        return {
            "detect_clothing": "Detect all clothing items and accessories",
            "describe_outfit": "Describe this outfit in detail including style, colors, and materials",
            "identify_garment": "What type of garment is this?",
            "extract_colors": "What colors are present in this clothing?",
            "identify_pattern": "Does this clothing have a pattern? If so, what type?",
            "identify_material": "What material does this clothing appear to be made of?",
            "read_label": "Read any text, labels, or brand names visible",
            "style_aesthetic": "What fashion aesthetic or style does this represent?"
        }
    
    # ============================================
    # 🎯 CORE INFERENCE METHODS
    # ============================================
    
    def run_task(
        self,
        image: Union[Image.Image, np.ndarray, str],
        task: str,
        text_input: str = None
    ) -> Dict[str, Any]:
        """
        Run a Florence-2 task on an image.
        
        Args:
            image: PIL Image, numpy array, or base64 string
            task: Task prompt (e.g., "<OD>", "<DETAILED_CAPTION>")
            text_input: Optional additional text for grounding tasks
            
        Returns:
            Raw Florence-2 output
        """
        import time
        start_time = time.time()
        
        self._load_model()
        
        # Convert to PIL Image
        pil_image = self._to_pil(image)
        
        # Build prompt
        if text_input:
            prompt = f"{task}{text_input}"
        else:
            prompt = task
        
        # Process inputs
        inputs = self._processor(
            text=prompt,
            images=pil_image,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
        
        # Decode
        generated_text = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]
        
        # Post-process
        parsed = self._processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(pil_image.width, pil_image.height)
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "raw": parsed,
            "processing_time_ms": processing_time
        }
    
    def _to_pil(self, image: Union[Image.Image, np.ndarray, str]) -> Image.Image:
        """Convert various image formats to PIL."""
        if isinstance(image, Image.Image):
            return image
        
        if isinstance(image, np.ndarray):
            # OpenCV BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                import cv2
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image)
        
        if isinstance(image, str):
            # Base64 string
            if ',' in image:
                image = image.split(',')[1]
            img_bytes = base64.b64decode(image)
            return Image.open(BytesIO(img_bytes))
        
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # ============================================
    # 🎨 FASHION-SPECIFIC METHODS
    # ============================================
    
    def detect_clothing(self, image: Union[Image.Image, np.ndarray, str]) -> PerceptionResult:
        """
        Detect all clothing items in image.
        
        Returns:
            PerceptionResult with detected objects
        """
        result = self.run_task(image, Florence2Tasks.OBJECT_DETECTION)
        
        objects = []
        raw_data = result["raw"].get(Florence2Tasks.OBJECT_DETECTION, {})
        
        bboxes = raw_data.get("bboxes", [])
        labels = raw_data.get("labels", [])
        
        for bbox, label in zip(bboxes, labels):
            # Filter for clothing-related labels
            if self._is_clothing_label(label):
                objects.append(DetectedObject(
                    label=label,
                    bbox=tuple(bbox),
                    confidence=0.9
                ))
        
        return PerceptionResult(
            success=True,
            task="detect_clothing",
            objects=objects,
            processing_time_ms=result["processing_time_ms"]
        )
    
    def analyze_garment(self, image: Union[Image.Image, np.ndarray, str]) -> PerceptionResult:
        """
        Complete garment analysis: type, colors, patterns, materials.
        
        Returns:
            PerceptionResult with full analysis
        """
        import time
        start_time = time.time()
        
        self._load_model()
        pil_image = self._to_pil(image)
        
        # Run multiple tasks
        # 1. Detailed caption
        caption_result = self.run_task(
            pil_image, 
            Florence2Tasks.MORE_DETAILED_CAPTION
        )
        detailed_caption = caption_result["raw"].get(
            Florence2Tasks.MORE_DETAILED_CAPTION, ""
        )
        
        # 2. Object detection
        detection_result = self.run_task(pil_image, Florence2Tasks.OBJECT_DETECTION)
        raw_det = detection_result["raw"].get(Florence2Tasks.OBJECT_DETECTION, {})
        
        objects = []
        for bbox, label in zip(
            raw_det.get("bboxes", []),
            raw_det.get("labels", [])
        ):
            objects.append(DetectedObject(
                label=label,
                bbox=tuple(bbox),
                confidence=0.9
            ))
        
        # 3. OCR for labels
        ocr_result = self.run_task(pil_image, Florence2Tasks.OCR)
        ocr_text = ocr_result["raw"].get(Florence2Tasks.OCR, "")
        
        # Parse fashion attributes from caption
        garment_type, colors, patterns, materials, styles = self._parse_fashion_attributes(
            detailed_caption
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return PerceptionResult(
            success=True,
            task="analyze_garment",
            objects=objects,
            caption=detailed_caption[:200] if detailed_caption else "",
            detailed_caption=detailed_caption,
            ocr_text=ocr_text,
            garment_type=garment_type,
            colors=colors,
            patterns=patterns,
            materials=materials,
            style_tags=styles,
            processing_time_ms=processing_time
        )
    
    def ground_text(
        self, 
        image: Union[Image.Image, np.ndarray, str],
        text: str
    ) -> PerceptionResult:
        """
        Find specific item by text description.
        
        Example: ground_text(image, "the red leather jacket")
        
        Returns:
            PerceptionResult with grounded object
        """
        result = self.run_task(
            image,
            Florence2Tasks.CAPTION_TO_PHRASE_GROUNDING,
            text_input=text
        )
        
        raw_data = result["raw"].get(Florence2Tasks.CAPTION_TO_PHRASE_GROUNDING, {})
        
        objects = []
        for bbox, label in zip(
            raw_data.get("bboxes", []),
            raw_data.get("labels", [])
        ):
            objects.append(DetectedObject(
                label=label,
                bbox=tuple(bbox),
                confidence=0.95
            ))
        
        return PerceptionResult(
            success=True,
            task="ground_text",
            objects=objects,
            caption=text,
            processing_time_ms=result["processing_time_ms"]
        )
    
    def _is_clothing_label(self, label: str) -> bool:
        """Check if label is clothing-related."""
        clothing_keywords = [
            "shirt", "pants", "dress", "jacket", "coat", "shoe", "boot",
            "hat", "cap", "bag", "belt", "tie", "skirt", "jeans", "sweater",
            "hoodie", "vest", "blazer", "suit", "shorts", "scarf", "gloves",
            "watch", "glasses", "sunglasses", "sock", "underwear", "bra",
            "clothing", "garment", "apparel", "outfit", "wear", "top",
            "bottom", "accessory", "jewelry", "necklace", "bracelet", "ring"
        ]
        label_lower = label.lower()
        return any(kw in label_lower for kw in clothing_keywords)
    
    def _parse_fashion_attributes(
        self, 
        caption: str
    ) -> Tuple[str, List[str], List[str], List[str], List[str]]:
        """Parse fashion attributes from detailed caption."""
        caption_lower = caption.lower()
        
        # Garment types
        garment_types = [
            "t-shirt", "shirt", "blouse", "sweater", "hoodie", "jacket",
            "coat", "blazer", "vest", "dress", "skirt", "pants", "jeans",
            "shorts", "suit", "jumpsuit", "romper", "cardigan", "top"
        ]
        garment_type = ""
        for gt in garment_types:
            if gt in caption_lower:
                garment_type = gt
                break
        
        # Colors
        color_list = [
            "red", "blue", "green", "yellow", "orange", "purple", "pink",
            "black", "white", "gray", "grey", "brown", "beige", "navy",
            "burgundy", "maroon", "teal", "turquoise", "olive", "cream",
            "gold", "silver", "coral", "salmon", "lavender", "mint"
        ]
        colors = [c for c in color_list if c in caption_lower]
        
        # Patterns
        pattern_list = [
            "striped", "plaid", "checkered", "polka dot", "floral",
            "geometric", "solid", "print", "abstract", "paisley",
            "camouflage", "animal print", "leopard", "zebra", "houndstooth"
        ]
        patterns = [p for p in pattern_list if p in caption_lower]
        
        # Materials
        material_list = [
            "cotton", "denim", "leather", "silk", "wool", "linen",
            "polyester", "nylon", "velvet", "satin", "cashmere",
            "suede", "corduroy", "fleece", "jersey", "chiffon", "lace"
        ]
        materials = [m for m in material_list if m in caption_lower]
        
        # Style tags
        style_list = [
            "casual", "formal", "business", "sporty", "elegant", "vintage",
            "modern", "classic", "bohemian", "minimalist", "streetwear",
            "preppy", "grunge", "romantic", "edgy", "professional"
        ]
        styles = [s for s in style_list if s in caption_lower]
        
        return garment_type, colors, patterns, materials, styles


# ============================================
# 🔧 UTILITY FUNCTIONS
# ============================================

def analyze_image_florence(image_b64: str) -> Dict:
    """
    Utility function for complete Florence-2 analysis.
    
    Args:
        image_b64: Base64 encoded image
        
    Returns:
        Complete perception result
    """
    perception = get_florence2_perception()
    result = perception.analyze_garment(image_b64)
    return result.to_dict()


def detect_clothing_florence(image_b64: str) -> Dict:
    """Detect clothing items using Florence-2."""
    perception = get_florence2_perception()
    result = perception.detect_clothing(image_b64)
    return result.to_dict()


def ground_text_florence(image_b64: str, text: str) -> Dict:
    """Find specific item by text using Florence-2."""
    perception = get_florence2_perception()
    result = perception.ground_text(image_b64, text)
    return result.to_dict()


# Singleton instance
_florence2_instance = None

def get_florence2_perception() -> Florence2Perception:
    """Get singleton Florence-2 instance."""
    global _florence2_instance
    if _florence2_instance is None:
        _florence2_instance = Florence2Perception()
    return _florence2_instance
