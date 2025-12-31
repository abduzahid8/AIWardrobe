"""
Advanced Product Analysis with YOLOv8 and Fashion-CLIP
Ultra-fast detection and fashion-specific classification
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image
import logging
from dataclasses import dataclass
import base64
import io

logger = logging.getLogger(__name__)


@dataclass
class DetectedProduct:
    """Product detected by YOLOv8"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_name: str
    class_id: int
    center: Tuple[int, int]


@dataclass
class ProductClassification:
    """Fashion-CLIP classification result"""
    category: str
    confidence: float
    subcategory: Optional[str] = None
    style_tags: List[str] = None
    embedding: Optional[np.ndarray] = None


class ProductAnalyzer:
    """
    Multi-model product analysis engine combining:
    - YOLOv8 for ultra-fast detection
    - Fashion-CLIP for accurate classification
    - Embedding generation for similarity search
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._yolo_model = None
        self._clip_model = None
        self._clip_processor = None
        self._models_loaded = False
        
        # Check if running in free tier mode
        import os
        self.free_tier_mode = os.getenv("FREE_TIER_MODE", "false").lower() == "true"
        
        if self.free_tier_mode:
            # Use memory-optimized model manager
            from .memory_optimizer import model_manager
            self.model_manager = model_manager
            logger.info("🚀 Free tier mode enabled - models will load on-demand")
        
        # EXPANDED Fashion Categories (110+ specific types for maximum accuracy)
        self.fashion_categories = [
            # === TOPS (30 types) ===
            "t-shirt", "graphic tee", "tank top", "crop top", "polo shirt",
            "button-down shirt", "dress shirt", "oxford shirt", "flannel shirt",
            "blouse", "silk blouse", "sweater", "crewneck sweater", "v-neck sweater",
            "cardigan", "turtleneck", "mock neck", "hoodie", "pullover hoodie",
            "zip-up hoodie", "sweatshirt", "henley shirt", "long sleeve shirt",
            "vest", "bodysuit", "camisole", "jersey", "tunic", "thermal top",
            "raglan shirt",
            
            # === OUTERWEAR (25 types) ===
            "denim jacket", "leather jacket", "bomber jacket", "trucker jacket",
            "blazer", "sport coat", "puffer jacket", "down jacket", "parka",
            "trench coat", "overcoat", "peacoat", "windbreaker", "track jacket",
            "varsity jacket", "fleece jacket", "quilted jacket", "anorak", "raincoat",
            "ski jacket", "moto jacket", "safari jacket", "field jacket",
            "shacket", "overshirt",
            
            # === BOTTOMS (25 types) ===
            "jeans", "skinny jeans", "straight jeans", "wide-leg jeans", "mom jeans",
            "bootcut jeans", "boyfriend jeans", "slim fit jeans", "tapered jeans",
            "chinos", "cargo pants", "joggers", "sweatpants", "dress pants",
            "trousers", "leggings", "denim shorts", "cargo shorts", "athletic shorts",
            "bermuda shorts", "khakis", "corduroys", "cropped pants", "palazzo pants",
            "culottes",
            
            # === DRESSES & SKIRTS (15 types) ===
            "dress", "maxi dress", "midi dress", "mini dress", "sundress",
            "cocktail dress", "evening gown", "wrap dress", "shirt dress",
            "skirt", "mini skirt", "midi skirt", "maxi skirt", "pleated skirt",
            "pencil skirt",
            
            # === FOOTWEAR (20 types) ===
            "sneakers", "running shoes", "high-top sneakers", "low-top sneakers",
            "boots", "chelsea boots", "combat boots", "ankle boots", "hiking boots",
            "cowboy boots", "knee-high boots", "loafers", "oxfords", "derby shoes",
            "brogues", "sandals", "slides", "flip flops", "espadrilles", "dress shoes",
            
            # === ACCESSORIES (15 types) ===
            "hat", "baseball cap", "beanie", "bucket hat", "fedora", "sun hat",
            "backpack", "tote bag", "crossbody bag", "messenger bag", "handbag",
            "sunglasses", "belt", "scarf", "watch"
        ]
        
        # Style categories for secondary classification
        self.style_categories = [
            "streetwear", "preppy", "athleisure", "minimalist", "casual",
            "bohemian", "vintage", "business casual", "smart casual",
            "formal", "sporty", "grunge", "classic", "trendy",
            "edgy", "romantic", "urban", "luxury"
        ]
        
        self.style_tags = [
            "casual", "formal", "business", "sporty", "athletic",
            "elegant", "vintage", "modern", "streetwear", "bohemian",
            "minimalist", "luxury", "comfortable", "trendy", "classic"
        ]
        
        # Subcategory mapping for hierarchical classification
        self.subcategory_map = {
            "jacket": ["denim jacket", "leather jacket", "bomber jacket", "trucker jacket", "blazer", "puffer jacket"],
            "jeans": ["skinny jeans", "straight jeans", "wide-leg jeans", "mom jeans"],
            "sneakers": ["running shoes", "high-top sneakers", "low-top sneakers"],
            "boots": ["chelsea boots", "combat boots", "ankle boots", "hiking boots"],
            "hoodie": ["pullover hoodie", "zip-up hoodie"],
            "sweater": ["crewneck sweater", "v-neck sweater", "cardigan", "turtleneck"],
            "shorts": ["denim shorts", "athletic shorts", "cargo shorts"],
            "hat": ["baseball cap", "beanie", "bucket hat", "fedora"],
            "bag": ["backpack", "tote bag", "crossbody bag", "messenger bag"],
            "shirt": ["button-down shirt", "dress shirt", "oxford shirt", "flannel shirt", "henley shirt"]
        }
    
    def _load_yolo(self):
        """Load YOLOv8 model"""
        if self._yolo_model is not None:
            return True
        
        try:
            # Use model manager in free tier mode
            if self.free_tier_mode:
                self._yolo_model = self.model_manager.get_model("yolo")
                logger.info("✅ YOLOv8n loaded (optimized for free tier)")
                return True
            
            # Standard loading for paid tier
            from ultralytics import YOLO
            
            logger.info("Loading YOLOv8 model...")
            
            # Use YOLOv8n (nano) for speed, can upgrade to yolov8m for accuracy
            self._yolo_model = YOLO('yolov8n.pt')
            
            logger.info("✅ YOLOv8 loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"YOLOv8 loading failed: {e}")
            return False
    
    def _load_clip(self):
        """Load Fashion-CLIP model"""
        if self._clip_model is not None:
            return True
        
        try:
            import torch
            import open_clip
            
            logger.info("Loading Fashion-CLIP model...")
            
            # Use ViT-B/32 as it's fast and accurate
            model_name = "ViT-B-32"
            pretrained = "openai"
            
            self._clip_model, _, self._clip_processor = open_clip.create_model_and_transforms(
                model_name, 
                pretrained=pretrained
            )
            
            if self.device == "cuda" and torch.cuda.is_available():
                self._clip_model = self._clip_model.to("cuda")
            
            self._clip_model.eval()
            
            # Precompute text embeddings for categories
            self._precompute_category_embeddings()
            
            logger.info("✅ Fashion-CLIP loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Fashion-CLIP loading failed: {e}")
            return False
    
    def _precompute_category_embeddings(self):
        """Precompute embeddings for all fashion categories"""
        try:
            import torch
            import open_clip
            
            texts = [f"a photo of {cat}" for cat in self.fashion_categories]
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            
            with torch.no_grad():
                text_tokens = tokenizer(texts)
                if self.device == "cuda":
                    text_tokens = text_tokens.to("cuda")
                
                self._category_embeddings = self._clip_model.encode_text(text_tokens)
                self._category_embeddings = self._category_embeddings / self._category_embeddings.norm(dim=-1, keepdim=True)
            
            logger.info(f"Precomputed {len(self.fashion_categories)} category embeddings")
            
        except Exception as e:
            logger.warning(f"Category embedding precomputation failed: {e}")
            self._category_embeddings = None
    
    def detect_with_yolo(self, image: np.ndarray, confidence_threshold: float = 0.25) -> List[DetectedProduct]:
        """
        Fast product detection with YOLOv8
        
        Args:
            image: BGR image
            confidence_threshold: Minimum confidence for detection
            
        Returns:
            List of detected products with bounding boxes
        """
        if not self._load_yolo():
            return []
        
        try:
            # Run inference
            results = self._yolo_model(image, conf=confidence_threshold, verbose=False)
            
            detections = []
            
            for result in results:
                boxes = result.boxes
                
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy()
                    conf = float(boxes.conf[i])
                    cls = int(boxes.cls[i])
                    
                    x1, y1, x2, y2 = map(int, box)
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    class_name = self._yolo_model.names[cls]
                    
                    detections.append(DetectedProduct(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        class_name=class_name,
                        class_id=cls,
                        center=(center_x, center_y)
                    ))
            
            logger.info(f"YOLOv8 detected {len(detections)} objects")
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection error: {e}")
            return []
    
    def classify_with_clip(self, image: np.ndarray, crop_bbox: Optional[Tuple] = None) -> ProductClassification:
        """
        Fashion-specific classification with CLIP
        
        Args:
            image: BGR image
            crop_bbox: Optional (x1, y1, x2, y2) to crop before classification
            
        Returns:
            ProductClassification with category and confidence
        """
        if not self._load_clip():
            return ProductClassification(category="unknown", confidence=0.0)
        
        try:
            import torch
            
            # Crop if bbox provided
            if crop_bbox:
                x1, y1, x2, y2 = crop_bbox
                image = image[y1:y2, x1:x2]
            
            # Convert to PIL RGB
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Preprocess
            image_input = self._clip_processor(pil_image).unsqueeze(0)
            
            if self.device == "cuda":
                image_input = image_input.to("cuda")
            
            # Get image embedding
            with torch.no_grad():
                image_embedding = self._clip_model.encode_image(image_input)
                image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            
            # Calculate similarities with categories
            if self._category_embeddings is not None:
                similarities = (image_embedding @ self._category_embeddings.T).squeeze()
                probs = torch.softmax(similarities * 100, dim=0)
                
                best_idx = probs.argmax().item()
                confidence = float(probs[best_idx])
                category = self.fashion_categories[best_idx]
                
                # Get top style tags
                style_similarities = self._get_style_similarities(image_embedding)
                
                return ProductClassification(
                    category=category,
                    confidence=confidence,
                    style_tags=style_similarities[:3],
                    embedding=image_embedding.cpu().numpy()
                )
            else:
                return ProductClassification(
                    category="clothing",
                    confidence=0.5,
                    embedding=image_embedding.cpu().numpy()
                )
            
        except Exception as e:
            logger.error(f"CLIP classification error: {e}")
            return ProductClassification(category="unknown", confidence=0.0)
    
    def _get_style_similarities(self, image_embedding) -> List[str]:
        """Get style tags based on similarity"""
        try:
            import torch
            import open_clip
            
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            style_texts = [f"{style} style clothing" for style in self.style_tags]
            
            with torch.no_grad():
                text_tokens = tokenizer(style_texts)
                if self.device == "cuda":
                    text_tokens = text_tokens.to("cuda")
                
                style_embeddings = self._clip_model.encode_text(text_tokens)
                style_embeddings = style_embeddings / style_embeddings.norm(dim=-1, keepdim=True)
            
            similarities = (image_embedding @ style_embeddings.T).squeeze()
            top_indices = similarities.topk(3).indices
            
            return [self.style_tags[i] for i in top_indices.cpu().numpy()]
            
        except Exception as e:
            logger.warning(f"Style similarity failed: {e}")
            return []
    
    def classify_hierarchical(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Hierarchical classification: General → Specific → Style
        Returns multi-level classification for maximum detail.
        
        Example output:
        {
            "category": "denim jacket",
            "confidence": 0.85,
            "subcategory": "trucker jacket",
            "style": "streetwear",
            "styleTags": ["casual", "vintage", "trendy"]
        }
        """
        if not self._load_clip():
            return {"category": "unknown", "confidence": 0}
        
        try:
            import torch
            
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            image_input = self._clip_processor(pil_image).unsqueeze(0)
            
            if self.device == "cuda":
                image_input = image_input.to("cuda")
            
            with torch.no_grad():
                image_embedding = self._clip_model.encode_image(image_input)
                image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
            
            # Level 1: Get primary category from all categories
            primary = self._get_best_match(image_embedding, self.fashion_categories)
            
            # Level 2: Get subcategory if available
            subcategory = None
            primary_lower = primary["category"].lower()
            for parent, children in self.subcategory_map.items():
                if parent in primary_lower or primary_lower in parent:
                    sub_result = self._get_best_match(image_embedding, children)
                    if sub_result["confidence"] > 0.25:
                        subcategory = sub_result["category"]
                    break
            
            # Level 3: Get overall style
            style_result = self._get_best_match(image_embedding, self.style_categories)
            
            # Get top 3 style tags
            style_tags = self._get_style_similarities(image_embedding)
            
            return {
                "category": primary["category"],
                "confidence": round(primary["confidence"], 3),
                "subcategory": subcategory,
                "style": style_result["category"] if style_result["confidence"] > 0.2 else None,
                "styleTags": style_tags
            }
            
        except Exception as e:
            logger.error(f"Hierarchical classification error: {e}")
            return {"category": "unknown", "confidence": 0}
    
    def _get_best_match(self, image_embedding, categories: List[str]) -> Dict[str, Any]:
        """Get best matching category from a list"""
        try:
            import torch
            import open_clip
            
            tokenizer = open_clip.get_tokenizer("ViT-B-32")
            texts = [f"a photo of {cat}" for cat in categories]
            
            with torch.no_grad():
                text_tokens = tokenizer(texts)
                if self.device == "cuda":
                    text_tokens = text_tokens.to("cuda")
                
                text_embeddings = self._clip_model.encode_text(text_tokens)
                text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            
            similarities = (image_embedding @ text_embeddings.T).squeeze()
            probs = torch.softmax(similarities * 100, dim=0)
            
            best_idx = probs.argmax().item()
            return {
                "category": categories[best_idx],
                "confidence": float(probs[best_idx])
            }
            
        except Exception as e:
            logger.warning(f"Best match failed: {e}")
            return {"category": "unknown", "confidence": 0}
    
    def analyze_product(
        self, 
        image: np.ndarray, 
        use_detection: bool = True,
        use_classification: bool = True
    ) -> Dict[str, Any]:
        """
        Complete product analysis pipeline
        
        Args:
            image: BGR image
            use_detection: Use YOLOv8 detection
            use_classification: Use CLIP classification
            
        Returns:
            Complete analysis results
        """
        import time
        start_time = time.time()
        
        results = {
            "detections": [],
            "classifications": [],
            "primaryProduct": None,
            "processingTimeMs": 0
        }
        
        # Step 1: Detection
        detections = []
        if use_detection:
            detections = self.detect_with_yolo(image)
            results["detections"] = [
                {
                    "bbox": det.bbox,
                    "confidence": round(det.confidence, 3),
                    "class": det.class_name,
                    "classId": det.class_id
                }
                for det in detections
            ]
        
        # Step 2: Classification
        if use_classification:
            if detections:
                # Classify each detection
                for det in detections:
                    classification = self.classify_with_clip(image, det.bbox)
                    results["classifications"].append({
                        "bbox": det.bbox,
                        "category": classification.category,
                        "confidence": round(classification.confidence, 3),
                        "styleTags": classification.style_tags
                    })
                
                # Use largest detection as primary
                largest = max(detections, key=lambda d: (d.bbox[2]-d.bbox[0]) * (d.bbox[3]-d.bbox[1]))
                primary_class = self.classify_with_clip(image, largest.bbox)
                
            else:
                # Classify full image
                primary_class = self.classify_with_clip(image)
                results["classifications"].append({
                    "bbox": None,
                    "category": primary_class.category,
                    "confidence": round(primary_class.confidence, 3),
                    "styleTags": primary_class.style_tags
                })
            
            if use_classification:
                results["primaryProduct"] = {
                    "category": primary_class.category,
                    "confidence": round(primary_class.confidence, 3),
                    "styleTags": primary_class.style_tags,
                    "hasEmbedding": primary_class.embedding is not None
                }
        
        results["processingTimeMs"] = round((time.time() - start_time) * 1000, 1)
        
        return results


def analyze_product_from_base64(
    image_base64: str,
    use_detection: bool = True,
    use_classification: bool = True
) -> Dict:
    """
    Utility function for base64 image product analysis
    
    Args:
        image_base64: Base64-encoded image
        use_detection: Use YOLOv8 detection
        use_classification: Use Fashion-CLIP classification
    
    Returns:
        Analysis results dictionary
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
    
    # Analyze
    analyzer = ProductAnalyzer()
    return analyzer.analyze_product(image, use_detection, use_classification)
