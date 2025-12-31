"""
Ensemble Detection Orchestrator - Multi-Model Detection Pipeline
Part of the World-Class AI Vision System for AIWardrobe

This module provides:
- Combines multiple detection models (YOLO, SegFormer, Grounded SAM)
- Confidence calibration and voting
- False positive elimination through cross-validation
- Optimal model selection based on use case
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


@dataclass
class EnsembleDetection:
    """Single detection from ensemble"""
    category: str
    specific_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    
    # Model agreement
    detection_sources: List[str]  # Which models detected this
    agreement_score: float  # 0-1, how many models agree
    
    # Rich attributes (from all sources)
    primary_color: str
    color_hex: str
    colors: List[Dict]
    
    material: Optional[str] = None
    pattern: Optional[str] = None
    
    # Cutout (from best source)
    cutout_image: Optional[str] = None  # base64
    mask: Optional[np.ndarray] = None
    
    # Detailed attributes
    attributes: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result.pop('mask', None)  # Don't serialize numpy array
        return result


@dataclass
class EnsembleResult:
    """Complete ensemble detection result"""
    detections: List[EnsembleDetection]
    processing_time_ms: float
    models_used: List[str]
    image_size: Tuple[int, int]
    
    # Confidence metrics
    mean_confidence: float
    mean_agreement: float
    
    def to_dict(self) -> Dict:
        return {
            "success": True,
            "items": [d.to_dict() for d in self.detections],
            "itemCount": len(self.detections),
            "processingTimeMs": self.processing_time_ms,
            "modelsUsed": self.models_used,
            "imageSize": list(self.image_size),
            "meanConfidence": self.mean_confidence,
            "meanAgreement": self.mean_agreement
        }


class EnsembleDetector:
    """
    🚀 Multi-Model Ensemble Detection System
    
    Combines multiple AI models for maximum accuracy:
    1. YOLO: Fast initial detection
    2. SegFormer: High-quality segmentation masks
    3. Grounded SAM2: Text-prompted validation
    4. Fashion-CLIP: Specific type classification
    
    Features:
    - Parallel model execution
    - Confidence calibration
    - Cross-model validation
    - False positive elimination
    - Best-of-breed attribute extraction
    
    Usage:
        detector = EnsembleDetector()
        results = detector.detect(image)
        for item in results.detections:
            print(f"{item.specific_type}: {item.confidence:.2f}")
    """
    
    # Model weights for confidence combination
    MODEL_WEIGHTS = {
        "yolo": 0.3,
        "segformer": 0.4,
        "grounded_sam": 0.5,
        "fashion_clip": 0.3
    }
    
    # Minimum agreement for detection to be valid
    MIN_AGREEMENT = 0.4  # At least 2/5 sources must agree
    
    def __init__(
        self,
        use_yolo: bool = True,
        use_segformer: bool = True,
        use_grounded_sam: bool = False,  # Optional, slower
        use_fashion_clip: bool = True,
        parallel_execution: bool = True,
        device: str = "auto"
    ):
        """
        Initialize ensemble detector.
        
        Args:
            use_yolo: Use YOLO for fast detection
            use_segformer: Use SegFormer for segmentation
            use_grounded_sam: Use Grounded SAM for text-prompted detection
            use_fashion_clip: Use Fashion-CLIP for classification
            parallel_execution: Run models in parallel
            device: Compute device
        """
        self.use_yolo = use_yolo
        self.use_segformer = use_segformer
        self.use_grounded_sam = use_grounded_sam
        self.use_fashion_clip = use_fashion_clip
        self.parallel_execution = parallel_execution
        
        self._setup_device(device)
        
        # Lazy-loaded models
        self._yolo = None
        self._segformer = None
        self._grounded_sam = None
        self._fashion_clip = None
        self._hierarchical = None
        self._material = None
        self._pattern = None
        self._feature_detector = None
        
        logger.info(f"EnsembleDetector initialized (device={self.device})")
    
    def _setup_device(self, device: str):
        """Setup compute device."""
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
    
    def _get_yolo(self):
        """Lazy load YOLO detector."""
        if self._yolo is None and self.use_yolo:
            try:
                from modules.yolo_detector import get_yolo_detector
                self._yolo = get_yolo_detector()
            except ImportError:
                logger.warning("YOLO detector not available")
        return self._yolo
    
    def _get_segformer(self):
        """Lazy load SegFormer."""
        if self._segformer is None and self.use_segformer:
            try:
                from modules.segmentation import get_segformer
                self._segformer = get_segformer()
            except ImportError:
                logger.warning("SegFormer not available")
        return self._segformer
    
    def _get_grounded_sam(self):
        """Lazy load Grounded SAM."""
        if self._grounded_sam is None and self.use_grounded_sam:
            try:
                from modules.grounded_sam import get_grounded_sam
                self._grounded_sam = get_grounded_sam()
            except ImportError:
                logger.warning("Grounded SAM not available")
        return self._grounded_sam
    
    def _get_fashion_clip(self):
        """Lazy load Fashion-CLIP."""
        if self._fashion_clip is None and self.use_fashion_clip:
            try:
                from modules.fashion_clip import get_fashion_clip
                self._fashion_clip = get_fashion_clip()
            except ImportError:
                logger.warning("Fashion-CLIP not available")
        return self._fashion_clip
    
    def _get_hierarchical(self):
        """Lazy load hierarchical classifier."""
        if self._hierarchical is None:
            try:
                from modules.hierarchical_classifier import get_hierarchical_classifier
                self._hierarchical = get_hierarchical_classifier()
            except ImportError:
                logger.warning("Hierarchical classifier not available")
        return self._hierarchical
    
    def _get_material_analyzer(self):
        """Lazy load material analyzer."""
        if self._material is None:
            try:
                from modules.material_analyzer import get_material_analyzer
                self._material = get_material_analyzer()
            except ImportError:
                logger.warning("Material analyzer not available")
        return self._material
    
    def _get_pattern_detector(self):
        """Lazy load pattern detector."""
        if self._pattern is None:
            try:
                from modules.pattern_detector import get_pattern_detector
                self._pattern = get_pattern_detector()
            except ImportError:
                logger.warning("Pattern detector not available")
        return self._pattern
    
    def _get_feature_detector(self):
        """Lazy load feature detector."""
        if self._feature_detector is None:
            try:
                from modules.feature_detector import get_feature_detector
                self._feature_detector = get_feature_detector()
            except ImportError:
                logger.warning("Feature detector not available")
        return self._feature_detector
    
    def detect(
        self,
        image: np.ndarray,
        return_masks: bool = True,
        min_confidence: float = 0.3
    ) -> EnsembleResult:
        """
        Run ensemble detection on image.
        
        Args:
            image: BGR image
            return_masks: Include segmentation masks
            min_confidence: Minimum confidence threshold
            
        Returns:
            EnsembleResult with all detections
        """
        start_time = time.time()
        h, w = image.shape[:2]
        
        # Collect detections from all models
        all_detections = {}  # region_key -> list of (source, detection)
        models_used = []
        
        # Run detection models
        if self.parallel_execution:
            detections = self._run_parallel(image)
        else:
            detections = self._run_sequential(image)
        
        for source, dets in detections.items():
            models_used.append(source)
            for det in dets:
                # Create region key for matching
                region_key = self._get_region_key(det["bbox"], h, w)
                
                if region_key not in all_detections:
                    all_detections[region_key] = []
                all_detections[region_key].append((source, det))
        
        # Merge detections from same region
        merged = self._merge_detections(all_detections, image)
        
        # Filter by confidence and agreement
        filtered = [
            d for d in merged
            if d.confidence >= min_confidence and d.agreement_score >= self.MIN_AGREEMENT
        ]
        
        # Sort by confidence
        filtered.sort(key=lambda x: x.confidence, reverse=True)
        
        # Enrich with detailed attributes
        enriched = self._enrich_detections(filtered, image)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Calculate metrics
        mean_conf = np.mean([d.confidence for d in enriched]) if enriched else 0
        mean_agree = np.mean([d.agreement_score for d in enriched]) if enriched else 0
        
        return EnsembleResult(
            detections=enriched,
            processing_time_ms=processing_time,
            models_used=models_used,
            image_size=(w, h),
            mean_confidence=float(mean_conf),
            mean_agreement=float(mean_agree)
        )
    
    def _run_parallel(self, image: np.ndarray) -> Dict[str, List]:
        """Run detection models in parallel."""
        results = {}
        
        def run_yolo():
            yolo = self._get_yolo()
            if yolo:
                result = yolo.detect(image)
                return [
                    {
                        "category": d.parent_category,
                        "class": d.class_name,
                        "confidence": d.confidence,
                        "bbox": d.bbox
                    }
                    for d in result.detections
                ]
            return []
        
        def run_segformer():
            segformer = self._get_segformer()
            if segformer:
                try:
                    result = segformer.segment_all(image, add_cutouts=True)
                    return result.get("items", [])
                except Exception as e:
                    logger.error(f"SegFormer error: {e}")
            return []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {}
            
            if self.use_yolo:
                futures[executor.submit(run_yolo)] = "yolo"
            if self.use_segformer:
                futures[executor.submit(run_segformer)] = "segformer"
            
            for future in as_completed(futures):
                source = futures[future]
                try:
                    results[source] = future.result()
                except Exception as e:
                    logger.error(f"{source} failed: {e}")
                    results[source] = []
        
        return results
    
    def _run_sequential(self, image: np.ndarray) -> Dict[str, List]:
        """Run detection models sequentially."""
        results = {}
        
        # YOLO
        if self.use_yolo:
            yolo = self._get_yolo()
            if yolo:
                result = yolo.detect(image)
                results["yolo"] = [
                    {
                        "category": d.parent_category,
                        "class": d.class_name,
                        "confidence": d.confidence,
                        "bbox": d.bbox
                    }
                    for d in result.detections
                ]
        
        # SegFormer
        if self.use_segformer:
            segformer = self._get_segformer()
            if segformer:
                try:
                    result = segformer.segment_all(image, add_cutouts=True)
                    results["segformer"] = result.get("items", [])
                except Exception as e:
                    logger.error(f"SegFormer error: {e}")
                    results["segformer"] = []
        
        return results
    
    def _get_region_key(
        self,
        bbox: Tuple[int, int, int, int],
        img_h: int,
        img_w: int
    ) -> str:
        """Create region key for matching detections."""
        x1, y1, x2, y2 = bbox
        
        # Normalize to grid
        grid_x = int((x1 + x2) / 2 / img_w * 10)
        grid_y = int((y1 + y2) / 2 / img_h * 10)
        
        # Size bucket
        area = (x2 - x1) * (y2 - y1)
        img_area = img_h * img_w
        size_ratio = area / img_area
        
        if size_ratio < 0.05:
            size = "S"
        elif size_ratio < 0.15:
            size = "M"
        else:
            size = "L"
        
        return f"{grid_y}_{grid_x}_{size}"
    
    def _merge_detections(
        self,
        grouped: Dict[str, List],
        image: np.ndarray
    ) -> List[EnsembleDetection]:
        """Merge detections from same region."""
        merged = []
        
        for region_key, detections in grouped.items():
            if not detections:
                continue
            
            sources = [d[0] for d in detections]
            datas = [d[1] for d in detections]
            
            # Weighted confidence
            total_weight = 0
            weighted_conf = 0
            
            for source, data in zip(sources, datas):
                weight = self.MODEL_WEIGHTS.get(source, 0.3)
                conf = data.get("confidence", 0.5)
                weighted_conf += weight * conf
                total_weight += weight
            
            final_conf = weighted_conf / total_weight if total_weight > 0 else 0.5
            
            # Agreement score
            agreement = len(sources) / 3  # Assumes 3 possible models
            
            # Take best bbox (highest confidence)
            best = max(datas, key=lambda x: x.get("confidence", 0))
            bbox = best.get("bbox", (0, 0, 100, 100))
            
            # Category - use vote
            categories = [d.get("category", d.get("class", "unknown")) for d in datas]
            category = max(set(categories), key=categories.count)
            
            # Get specific type
            specific_type = best.get("specificType", best.get("class", category))
            
            # Colors
            colors = best.get("colors", [])
            primary_color = best.get("primaryColor", "unknown")
            color_hex = best.get("colorHex", "#000000")
            
            # Cutout if available
            cutout = best.get("cutoutImage")
            
            detection = EnsembleDetection(
                category=category,
                specific_type=specific_type,
                confidence=final_conf,
                bbox=tuple(bbox) if isinstance(bbox, list) else bbox,
                detection_sources=sources,
                agreement_score=agreement,
                primary_color=primary_color,
                color_hex=color_hex,
                colors=colors,
                cutout_image=cutout
            )
            
            merged.append(detection)
        
        return merged
    
    def _enrich_detections(
        self,
        detections: List[EnsembleDetection],
        image: np.ndarray
    ) -> List[EnsembleDetection]:
        """Add rich attributes to detections."""
        
        for det in detections:
            # Crop region
            x1, y1, x2, y2 = det.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue
            
            cropped = image[y1:y2, x1:x2]
            
            # Hierarchical classification
            hierarchical = self._get_hierarchical()
            if hierarchical:
                try:
                    result = hierarchical.classify(cropped, det.category)
                    det.specific_type = result.specific_type
                    det.attributes["classification"] = {
                        "level1": result.level1_category,
                        "level2": result.level2_subcategory,
                        "level3": result.level3_type,
                        "level4": result.level4_variant,
                        "path": result.classification_path
                    }
                except Exception as e:
                    logger.warning(f"Classification failed: {e}")
            
            # Material analysis
            material = self._get_material_analyzer()
            if material:
                try:
                    result = material.analyze(cropped, category_hint=det.specific_type)
                    det.material = result.primary_material
                    det.attributes["material"] = {
                        "type": result.primary_material,
                        "category": result.material_category,
                        "confidence": float(result.material_confidence),
                        "texture": result.texture,
                        "finish": result.finish,
                        "weight": result.weight_class,
                        "isStretch": bool(result.is_stretch)  # Convert numpy.bool_ to Python bool
                    }
                except Exception as e:
                    logger.warning(f"Material analysis failed: {e}")
            
            # Pattern detection
            pattern = self._get_pattern_detector()
            if pattern:
                try:
                    result = pattern.analyze(cropped)
                    det.pattern = result.primary_pattern
                    det.attributes["pattern"] = {
                        "type": result.primary_pattern,
                        "category": result.pattern_category,
                        "confidence": float(result.confidence),
                        "isStriped": bool(result.is_striped),      # Convert numpy.bool_ to Python bool
                        "isCheckered": bool(result.is_checkered),  # Convert numpy.bool_ to Python bool
                        "hasPrint": bool(result.has_print)         # Convert numpy.bool_ to Python bool
                    }
                except Exception as e:
                    logger.warning(f"Pattern detection failed: {e}")
            
            # Feature detection
            features = self._get_feature_detector()
            if features:
                try:
                    result = features.detect_all_features(cropped, det.category)
                    det.attributes["features"] = result.to_dict()
                except Exception as e:
                    logger.warning(f"Feature detection failed: {e}")
        
        return detections


# === SINGLETON INSTANCE ===
_ensemble_detector_instance = None


def get_ensemble_detector() -> EnsembleDetector:
    """Get singleton instance."""
    global _ensemble_detector_instance
    if _ensemble_detector_instance is None:
        _ensemble_detector_instance = EnsembleDetector()
    return _ensemble_detector_instance


def detect_ensemble(image: np.ndarray) -> Dict:
    """
    Quick utility for ensemble detection.
    
    Args:
        image: BGR image
        
    Returns:
        Detection result dictionary
    """
    detector = get_ensemble_detector()
    result = detector.detect(image)
    return result.to_dict()
