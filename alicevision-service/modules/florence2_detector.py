"""
🚀 FLORENCE-2 DETECTOR - Microsoft's Most Powerful Vision Model
The BEST foundation model for object detection and segmentation

Features:
- Zero-shot detection of ANY object
- Text-prompted detection ("find the denim jacket")
- High accuracy captioning
- Dense region proposals
- Open-vocabulary detection

This is STATE-OF-THE-ART 2024 technology from Microsoft Research.
"""

import cv2
import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
import logging
import time
import base64
import io

logger = logging.getLogger(__name__)


# Comprehensive clothing detection prompts
FLORENCE_CLOTHING_PROMPTS = [
    # Tops
    "t-shirt", "polo shirt", "dress shirt", "blouse", "tank top", "crop top",
    "henley", "turtleneck", "hoodie", "sweatshirt", "sweater", "cardigan",
    
    # Jackets
    "jacket", "denim jacket", "leather jacket", "bomber jacket", "blazer",
    "sport coat", "pea coat", "trench coat", "parka", "windbreaker",
    "puffer jacket", "down jacket", "trucker jacket",
    
    # Bottoms
    "pants", "jeans", "denim jeans", "skinny jeans", "straight jeans",
    "chinos", "dress pants", "cargo pants", "joggers", "sweatpants",
    "shorts", "skirt", "mini skirt", "maxi skirt",
    
    # Full body
    "dress", "maxi dress", "jumpsuit", "romper", "suit",
    
    # Footwear
    "shoes", "sneakers", "boots", "dress shoes", "loafers", "sandals",
    "heels", "flats", "ankle boots",
    
    # Accessories
    "bag", "handbag", "backpack", "hat", "cap", "scarf", "belt", "watch"
]


@dataclass
class FlorenceDetection:
    """Single detection from Florence-2"""
    label: str
    bbox: List[int]  # [x1, y1, x2, y2]
    confidence: float
    
    # Optional mask
    mask: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result.pop('mask', None)
        return result


@dataclass
class FlorenceResult:
    """Complete Florence-2 detection result"""
    detections: List[FlorenceDetection]
    caption: str
    processing_time_ms: float
    task_used: str
    
    def to_dict(self) -> Dict:
        return {
            "success": True,
            "detections": [d.to_dict() for d in self.detections],
            "caption": self.caption,
            "processingTimeMs": self.processing_time_ms,
            "taskUsed": self.task_used
        }


class Florence2Detector:
    """
    🚀 Microsoft Florence-2 - State-of-the-Art Vision Model
    
    Florence-2 is a foundation vision model that excels at:
    - Object detection (open-vocabulary)
    - Image captioning
    - Dense region proposals
    - Visual grounding (text-to-region)
    
    This is the MOST POWERFUL publicly available vision model from Microsoft.
    
    Usage:
        detector = Florence2Detector()
        result = detector.detect(image, prompt="denim jacket")
        for det in result.detections:
            print(f"{det.label}: {det.bbox}")
    """
    
    # Florence-2 task types
    TASKS = {
        "caption": "<CAPTION>",
        "detailed_caption": "<DETAILED_CAPTION>",
        "more_detailed_caption": "<MORE_DETAILED_CAPTION>",
        "object_detection": "<OD>",
        "dense_region_caption": "<DENSE_REGION_CAPTION>",
        "region_proposal": "<REGION_PROPOSAL>",
        "caption_to_phrase_grounding": "<CAPTION_TO_PHRASE_GROUNDING>",
        "referring_expression_segmentation": "<REFERRING_EXPRESSION_SEGMENTATION>",
        "open_vocabulary_detection": "<OPEN_VOCABULARY_DETECTION>",
    }
    
    def __init__(self, model_size: str = "base", device: str = "auto"):
        """
        Initialize Florence-2.
        
        Args:
            model_size: "base" or "large"
            device: "auto", "cuda", "mps", or "cpu"
        """
        self.model_size = model_size
        self._setup_device(device)
        
        self.model = None
        self.processor = None
        self.model_loaded = False
        
        logger.info(f"Florence2Detector initialized (size={model_size}, device={self.device})")
    
    def _setup_device(self, device: str):
        """Setup compute device."""
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "cpu"  # Florence-2 may not fully support MPS
            else:
                self.device = "cpu"
        else:
            self.device = device
    
    def _load_model(self):
        """Load Florence-2 model (lazy loading)."""
        if self.model_loaded:
            return
        
        logger.info(f"📥 Loading Florence-2-{self.model_size} (this may take a moment)...")
        
        try:
            from transformers import AutoProcessor, AutoModelForCausalLM
            
            model_id = f"microsoft/Florence-2-{self.model_size}"
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            
            # Load model
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=dtype
            )
            
            if self.device == "cuda":
                self.model = self.model.cuda()
            
            self.model.eval()
            self.model_loaded = True
            
            logger.info(f"✅ Florence-2-{self.model_size} loaded successfully!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Florence-2: {e}")
            raise
    
    def detect(
        self,
        image: np.ndarray,
        task: str = "object_detection",
        prompt: str = None
    ) -> FlorenceResult:
        """
        Run Florence-2 detection.
        
        Args:
            image: BGR image
            task: Detection task type
            prompt: Optional text prompt for grounding
            
        Returns:
            FlorenceResult with detections
        """
        start_time = time.time()
        
        # Load model if needed
        self._load_model()
        
        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Get task prompt
        task_prompt = self.TASKS.get(task, "<OD>")
        
        # For grounding tasks, add the text prompt
        if prompt and task in ["caption_to_phrase_grounding", "open_vocabulary_detection"]:
            task_prompt = f"{task_prompt}{prompt}"
        
        # Process
        inputs = self.processor(
            text=task_prompt,
            images=pil_image,
            return_tensors="pt"
        )
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )
        
        # Decode
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]
        
        # Post-process based on task
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt.split(">")[0] + ">",  # Get base task
            image_size=(pil_image.width, pil_image.height)
        )
        
        # Extract detections
        detections = self._extract_detections(parsed, task_prompt)
        
        # Get caption if available
        caption = ""
        if "CAPTION" in task_prompt.upper():
            caption = parsed.get(task_prompt, "")
        
        processing_time = (time.time() - start_time) * 1000
        
        return FlorenceResult(
            detections=detections,
            caption=caption,
            processing_time_ms=processing_time,
            task_used=task
        )
    
    def detect_clothing(self, image: np.ndarray) -> FlorenceResult:
        """
        Detect clothing items specifically.
        Uses object detection + clothing filtering.
        """
        start_time = time.time()
        
        # Load model
        self._load_model()
        
        # Convert to PIL
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Use object detection
        task_prompt = "<OD>"
        
        inputs = self.processor(
            text=task_prompt,
            images=pil_image,
            return_tensors="pt"
        )
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
        
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]
        
        parsed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(pil_image.width, pil_image.height)
        )
        
        # Filter for clothing
        detections = []
        if task_prompt in parsed:
            bboxes = parsed[task_prompt].get('bboxes', [])
            labels = parsed[task_prompt].get('labels', [])
            
            for bbox, label in zip(bboxes, labels):
                label_lower = label.lower()
                
                # Check if it's clothing
                is_clothing = any(
                    p in label_lower 
                    for p in FLORENCE_CLOTHING_PROMPTS
                ) or any(
                    kw in label_lower 
                    for kw in ['shirt', 'pants', 'jacket', 'dress', 'shoe', 
                              'coat', 'sweater', 'jeans', 'skirt', 'top',
                              'blouse', 'hoodie', 'blazer', 'boots']
                )
                
                if is_clothing:
                    x1, y1, x2, y2 = bbox
                    detections.append(FlorenceDetection(
                        label=label,
                        bbox=[int(x1), int(y1), int(x2), int(y2)],
                        confidence=0.85
                    ))
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"Florence-2 detected {len(detections)} clothing items")
        
        return FlorenceResult(
            detections=detections,
            caption="",
            processing_time_ms=processing_time,
            task_used="clothing_detection"
        )
    
    def ground_prompt(
        self,
        image: np.ndarray,
        prompt: str
    ) -> FlorenceResult:
        """
        Find regions matching a text prompt.
        
        Example: ground_prompt(image, "blue denim jacket")
        """
        start_time = time.time()
        
        self._load_model()
        
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Caption to phrase grounding
        task_prompt = f"<CAPTION_TO_PHRASE_GROUNDING>{prompt}"
        
        inputs = self.processor(
            text=task_prompt,
            images=pil_image,
            return_tensors="pt"
        )
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )
        
        generated_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]
        
        parsed = self.processor.post_process_generation(
            generated_text,
            task="<CAPTION_TO_PHRASE_GROUNDING>",
            image_size=(pil_image.width, pil_image.height)
        )
        
        detections = self._extract_detections(parsed, "<CAPTION_TO_PHRASE_GROUNDING>")
        
        processing_time = (time.time() - start_time) * 1000
        
        return FlorenceResult(
            detections=detections,
            caption=prompt,
            processing_time_ms=processing_time,
            task_used="phrase_grounding"
        )
    
    def _extract_detections(
        self,
        parsed: Dict,
        task_prompt: str
    ) -> List[FlorenceDetection]:
        """Extract detections from parsed output."""
        detections = []
        
        # Try to find bboxes in parsed output
        for key, value in parsed.items():
            if isinstance(value, dict):
                bboxes = value.get('bboxes', [])
                labels = value.get('labels', [])
                
                for i, bbox in enumerate(bboxes):
                    label = labels[i] if i < len(labels) else "object"
                    x1, y1, x2, y2 = bbox
                    
                    detections.append(FlorenceDetection(
                        label=label,
                        bbox=[int(x1), int(y1), int(x2), int(y2)],
                        confidence=0.85
                    ))
        
        return detections
    
    def describe_image(self, image: np.ndarray, detail_level: str = "detailed") -> str:
        """
        Get detailed description of image.
        
        Args:
            image: BGR image
            detail_level: "basic", "detailed", or "more_detailed"
        """
        self._load_model()
        
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if detail_level == "basic":
            task = "<CAPTION>"
        elif detail_level == "more_detailed":
            task = "<MORE_DETAILED_CAPTION>"
        else:
            task = "<DETAILED_CAPTION>"
        
        inputs = self.processor(
            text=task,
            images=pil_image,
            return_tensors="pt"
        )
        
        if self.device == "cuda":
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512
            )
        
        caption = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )[0]
        
        return caption


# === SINGLETON INSTANCE ===
_florence2_instance = None


def get_florence2_detector(model_size: str = "base") -> Florence2Detector:
    """Get singleton Florence-2 detector."""
    global _florence2_instance
    if _florence2_instance is None:
        _florence2_instance = Florence2Detector(model_size=model_size)
    return _florence2_instance


def detect_with_florence2(image: np.ndarray, clothing_only: bool = True) -> Dict:
    """
    Quick detection utility.
    
    Args:
        image: BGR image
        clothing_only: Filter for clothing items
        
    Returns:
        Detection result dictionary
    """
    detector = get_florence2_detector()
    
    if clothing_only:
        result = detector.detect_clothing(image)
    else:
        result = detector.detect(image)
    
    return result.to_dict()
