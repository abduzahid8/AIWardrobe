"""
🚀 Visual Intelligence Engine - The Most Powerful AI Vision System
Master orchestrator for next-generation fashion analysis

Combines all components into a unified pipeline:
1. Advanced Frame Selection (FFT + Optical Flow + CLIP)
2. Zero-Shot Segmentation (SAM 2 + Grounding DINO)
3. Fine-Grained Classification (FashionCLIP 2.0)
4. Material & Sustainability Analysis (TextileNet + LCA)
5. Professional Asset Generation (IC-Light + ControlNet)
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
import base64
from PIL import Image
from io import BytesIO
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


# ============================================
# 🎯 DATA STRUCTURES
# ============================================

@dataclass
class EngineConfig:
    """Configuration for Visual Intelligence Engine"""
    # Frame selection
    use_clip_keyframes: bool = True
    use_optical_flow: bool = True
    keyframe_count: int = 5
    
    # Segmentation
    segmentation_model: str = "sam2"  # "sam2", "segformer", "grounded_sam"
    segment_all_items: bool = True
    
    # Classification
    use_fashion_clip: bool = True
    classify_aesthetics: bool = True
    
    # Sustainability
    include_sustainability: bool = True
    include_eco_score: bool = True
    
    # Generation
    generate_product_cards: bool = False
    use_ic_light: bool = True
    product_card_preset: str = "catalog_neutral"
    
    # Performance
    max_parallel_items: int = 5
    enable_caching: bool = True


@dataclass
class DetectedItem:
    """A single detected clothing item with full analysis"""
    # Basic info
    category: str
    specific_type: str
    confidence: float
    
    # Spatial
    bbox: Tuple[int, int, int, int]
    mask: Optional[np.ndarray] = None
    cutout_base64: Optional[str] = None
    
    # Colors
    primary_color: str = ""
    color_hex: str = ""
    colors: List[Dict] = field(default_factory=list)
    
    # Classification
    pattern: str = ""
    material: str = ""
    style_tags: List[str] = field(default_factory=list)
    aesthetic: str = ""
    
    # Features
    features: Dict[str, Any] = field(default_factory=dict)
    
    # Sustainability
    fiber_composition: Dict[str, float] = field(default_factory=dict)
    carbon_footprint_kg: float = 0.0
    eco_score: str = ""
    eco_score_details: Dict = field(default_factory=dict)
    
    # Generated assets
    product_card_base64: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "category": self.category,
            "specificType": self.specific_type,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "cutoutBase64": self.cutout_base64,
            "primaryColor": self.primary_color,
            "colorHex": self.color_hex,
            "colors": self.colors,
            "pattern": self.pattern,
            "material": self.material,
            "styleTags": self.style_tags,
            "aesthetic": self.aesthetic,
            "features": self.features,
            "fiberComposition": {k: round(v * 100, 1) for k, v in self.fiber_composition.items()},
            "carbonFootprintKg": round(self.carbon_footprint_kg, 3),
            "ecoScore": self.eco_score,
            "ecoScoreDetails": self.eco_score_details,
            "productCardBase64": self.product_card_base64
        }


@dataclass
class VisualIntelligenceResult:
    """Complete result from Visual Intelligence Engine"""
    success: bool
    items: List[DetectedItem]
    
    # Metadata
    processing_time_ms: float
    frames_analyzed: int
    method_used: str
    
    # Frame selection info
    keyframe_scores: List[Dict] = field(default_factory=list)
    
    # Overall outfit info
    outfit_style: str = ""
    outfit_aesthetic: str = ""
    color_harmony: str = ""
    
    # Embeddings for semantic search
    semantic_embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "itemsCount": len(self.items),
            "items": [item.to_dict() for item in self.items],
            "processingTimeMs": round(self.processing_time_ms, 2),
            "framesAnalyzed": self.frames_analyzed,
            "methodUsed": self.method_used,
            "keyframeScores": self.keyframe_scores,
            "outfitStyle": self.outfit_style,
            "outfitAesthetic": self.outfit_aesthetic,
            "colorHarmony": self.color_harmony,
            "hasSemanticEmbedding": self.semantic_embedding is not None
        }


# ============================================
# 🚀 VISUAL INTELLIGENCE ENGINE
# ============================================

class VisualIntelligenceEngine:
    """
    🚀 THE ULTIMATE VISUAL AI ENGINE
    
    Orchestrates all components into a unified pipeline for complete
    fashion analysis from video or image input.
    
    Pipeline:
    1. Intelligent frame selection (FFT + Flow + CLIP)
    2. Zero-shot segmentation (Grounded SAM 2)
    3. Fine-grained classification (FashionCLIP 2.0)
    4. Material & sustainability analysis (TextileNet + LCA)
    5. Professional asset generation (IC-Light + ControlNet)
    """
    
    def __init__(self, config: EngineConfig = None):
        """Initialize Visual Intelligence Engine."""
        self.config = config or EngineConfig()
        
        # Lazy-loaded components
        self._frame_selector = None
        self._sam2_segmenter = None
        self._fashion_clip = None
        self._textile_net = None
        self._lca_calculator = None
        self._ic_light = None
        self._segformer = None
        
        # Thread pool for parallel processing
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_parallel_items)
        
        logger.info(f"VisualIntelligenceEngine initialized with config: {self.config}")
    
    # ============================================
    # 📦 COMPONENT LOADING
    # ============================================
    
    def _get_frame_selector(self):
        """Get advanced frame selector."""
        if self._frame_selector is None:
            try:
                from modules.advanced_frame_selector import AdvancedFrameSelector
                self._frame_selector = AdvancedFrameSelector(
                    use_clip=self.config.use_clip_keyframes,
                    use_optical_flow=self.config.use_optical_flow
                )
            except ImportError:
                logger.warning("AdvancedFrameSelector not available")
        return self._frame_selector
    
    def _get_sam2_segmenter(self):
        """Get SAM 2 segmenter."""
        if self._sam2_segmenter is None:
            try:
                from modules.sam2_segmentation import SAM2Segmenter
                self._sam2_segmenter = SAM2Segmenter(model_size="base")
            except ImportError:
                logger.warning("SAM2Segmenter not available")
        return self._sam2_segmenter
    
    def _get_fashion_clip(self):
        """Get FashionCLIP classifier."""
        if self._fashion_clip is None:
            try:
                from modules.fashion_clip import get_fashion_clip
                self._fashion_clip = get_fashion_clip()
            except ImportError:
                logger.warning("FashionCLIP not available")
        return self._fashion_clip
    
    def _get_textile_net(self):
        """Get TextileNet classifier."""
        if self._textile_net is None:
            try:
                from modules.textile_net import TextileNetClassifier, LCACalculator
                self._textile_net = TextileNetClassifier()
                self._lca_calculator = LCACalculator()
            except ImportError:
                logger.warning("TextileNet not available")
        return self._textile_net
    
    def _get_ic_light(self):
        """Get IC-Light relighter."""
        if self._ic_light is None:
            try:
                from modules.ic_light import ICLightRelighter
                self._ic_light = ICLightRelighter()
            except ImportError:
                logger.warning("ICLightRelighter not available")
        return self._ic_light
    
    def _get_segformer(self):
        """Get SegFormer for semantic segmentation."""
        if self._segformer is None:
            try:
                from modules.segmentation import process_with_segformer
                self._segformer = process_with_segformer
            except ImportError:
                logger.warning("SegFormer not available")
        return self._segformer
    
    # ============================================
    # 🎯 MAIN ANALYSIS METHODS
    # ============================================
    
    def analyze_complete(
        self,
        input_source: Union[str, np.ndarray, List[np.ndarray]],
        semantic_query: str = None
    ) -> VisualIntelligenceResult:
        """
        Complete end-to-end analysis.
        
        Args:
            input_source: Video path, image (BGR), or list of frames
            semantic_query: Optional semantic filter (e.g., "red dress")
            
        Returns:
            VisualIntelligenceResult with complete analysis
        """
        start_time = time.time()
        
        # Determine input type and extract frames
        frames, frame_info = self._prepare_input(input_source)
        
        if not frames:
            return VisualIntelligenceResult(
                success=False,
                items=[],
                processing_time_ms=0,
                frames_analyzed=0,
                method_used="none"
            )
        
        # Step 1: Select optimal frames
        selected_frames, keyframe_scores = self._select_keyframes(
            frames, semantic_query
        )
        
        # Step 2: Segment and detect items
        detected_items = self._detect_and_segment(selected_frames[0])
        
        # Step 3: Classify each item
        classified_items = self._classify_items(detected_items, selected_frames[0])
        
        # Step 4: Sustainability analysis
        if self.config.include_sustainability:
            classified_items = self._add_sustainability(classified_items)
        
        # Step 5: Generate product cards
        if self.config.generate_product_cards:
            classified_items = self._generate_product_cards(
                classified_items, selected_frames[0]
            )
        
        # Step 6: Analyze overall outfit
        outfit_style, outfit_aesthetic, color_harmony = self._analyze_outfit(classified_items)
        
        # Step 7: Generate semantic embedding
        embedding = self._generate_semantic_embedding(selected_frames[0])
        
        processing_time = (time.time() - start_time) * 1000
        
        return VisualIntelligenceResult(
            success=True,
            items=classified_items,
            processing_time_ms=processing_time,
            frames_analyzed=len(selected_frames),
            method_used=f"unified_{self.config.segmentation_model}",
            keyframe_scores=keyframe_scores,
            outfit_style=outfit_style,
            outfit_aesthetic=outfit_aesthetic,
            color_harmony=color_harmony,
            semantic_embedding=embedding
        )
    
    def _prepare_input(
        self,
        input_source: Union[str, np.ndarray, List[np.ndarray]]
    ) -> Tuple[List[np.ndarray], Dict]:
        """Prepare input and extract frames."""
        info = {"type": "unknown", "count": 0}
        
        if isinstance(input_source, str):
            # Video path
            if input_source.endswith(('.mp4', '.avi', '.mov', '.webm')):
                frames = self._extract_video_frames(input_source)
                info = {"type": "video", "path": input_source, "count": len(frames)}
            # Base64 image
            elif input_source.startswith("data:") or len(input_source) > 1000:
                frame = self._decode_base64_image(input_source)
                frames = [frame] if frame is not None else []
                info = {"type": "base64", "count": len(frames)}
            else:
                frames = []
        
        elif isinstance(input_source, np.ndarray):
            # Single image
            frames = [input_source]
            info = {"type": "image", "count": 1}
        
        elif isinstance(input_source, list):
            # List of frames
            frames = input_source
            info = {"type": "frame_list", "count": len(frames)}
        
        else:
            frames = []
        
        return frames, info
    
    def _extract_video_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract frames from video."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        sample_rate = 3  # Every 3rd frame
        
        while cap.isOpened() and len(frames) < 100:  # Max 100 frames
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                frames.append(frame)
            frame_count += 1
        
        cap.release()
        return frames
    
    def _decode_base64_image(self, b64: str) -> Optional[np.ndarray]:
        """Decode base64 image."""
        try:
            if ',' in b64:
                b64 = b64.split(',')[1]
            
            img_bytes = base64.b64decode(b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            return None
    
    def _select_keyframes(
        self,
        frames: List[np.ndarray],
        semantic_query: str = None
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        """Select optimal keyframes."""
        if len(frames) <= 1:
            return frames, []
        
        selector = self._get_frame_selector()
        
        if selector:
            result = selector.select_optimal_frames(
                frames=frames,
                top_n=self.config.keyframe_count,
                semantic_query=semantic_query
            )
            
            selected = [frames[s.original_index] for s in result.best_frames]
            scores = [s.to_dict() for s in result.best_frames]
            
            return selected if selected else frames[:1], scores
        
        # Fallback: use first frame
        return frames[:1], []
    
    def _detect_and_segment(self, image: np.ndarray) -> List[DetectedItem]:
        """Detect and segment clothing items."""
        items = []
        
        # Try SAM 2 first
        if self.config.segmentation_model == "sam2":
            sam2 = self._get_sam2_segmenter()
            if sam2:
                result = sam2.segment_all_clothing(image)
                for mask_obj in result.masks:
                    items.append(DetectedItem(
                        category="clothing",
                        specific_type="unknown",
                        confidence=mask_obj.confidence,
                        bbox=mask_obj.bbox,
                        mask=mask_obj.mask
                    ))
        
        # Fallback to SegFormer
        if not items:
            segformer = self._get_segformer()
            if segformer:
                try:
                    result = segformer(image)
                    if result and "items" in result:
                        for item_data in result["items"]:
                            items.append(DetectedItem(
                                category=item_data.get("category", "clothing"),
                                specific_type=item_data.get("type", "unknown"),
                                confidence=item_data.get("confidence", 0.8),
                                bbox=tuple(item_data.get("bbox", [0, 0, 0, 0])),
                                mask=item_data.get("mask"),
                                primary_color=item_data.get("primaryColor", ""),
                                color_hex=item_data.get("colorHex", ""),
                                colors=item_data.get("colors", [])
                            ))
                except Exception as e:
                    logger.error(f"SegFormer failed: {e}")
        
        # Generate cutouts
        for item in items:
            if item.mask is not None:
                cutout = self._create_cutout(image, item.mask)
                item.cutout_base64 = self._encode_image_base64(cutout)
        
        return items
    
    def _classify_items(
        self,
        items: List[DetectedItem],
        image: np.ndarray
    ) -> List[DetectedItem]:
        """Classify each detected item with FashionCLIP."""
        fashion_clip = self._get_fashion_clip()
        
        for item in items:
            # Extract item region
            x1, y1, x2, y2 = item.bbox
            if x2 > x1 and y2 > y1:
                item_region = image[y1:y2, x1:x2]
            else:
                item_region = image
            
            if fashion_clip and item_region.size > 0:
                try:
                    # Classify specific type
                    specific_type, confidence = fashion_clip.classify_specific_type(
                        item_region,
                        category_hint=item.category
                    )
                    item.specific_type = specific_type
                    item.confidence = max(item.confidence, confidence)
                    
                    # Extract attributes
                    attrs = fashion_clip.extract_attributes(item_region)
                    if attrs:
                        item.pattern = attrs.patterns[0][0] if attrs.patterns else ""
                        item.material = attrs.fabric or ""
                        item.style_tags = [s[0] for s in (attrs.style or [])[:3]]
                        
                        if not item.primary_color and attrs.colors:
                            item.primary_color = attrs.colors[0][0]
                        
                        # Use description as aesthetic hint
                        if attrs.description:
                            item.aesthetic = self._extract_aesthetic(attrs.description)
                            
                except Exception as e:
                    logger.warning(f"FashionCLIP classification failed: {e}")
            
            # Classify aesthetic
            if self.config.classify_aesthetics and fashion_clip:
                try:
                    aesthetics = fashion_clip.classify_aesthetic(item_region)
                    if aesthetics:
                        item.aesthetic = aesthetics[0][0]
                except Exception:
                    pass
        
        return items
    
    def _add_sustainability(self, items: List[DetectedItem]) -> List[DetectedItem]:
        """Add sustainability analysis to items."""
        textile_net = self._get_textile_net()
        
        if not textile_net or not self._lca_calculator:
            return items
        
        for item in items:
            try:
                # Get item image from cutout
                if item.cutout_base64:
                    item_image = self._decode_base64_image(item.cutout_base64)
                else:
                    continue
                
                if item_image is None:
                    continue
                
                # Classify fibers
                fiber_result = textile_net.classify_fiber(
                    item_image,
                    category_hint=item.specific_type
                )
                item.fiber_composition = fiber_result.composition
                item.material = fiber_result.primary_fiber
                
                # Calculate footprint
                lca_result = self._lca_calculator.calculate_footprint(
                    fiber_result.composition,
                    category=item.specific_type
                )
                item.carbon_footprint_kg = lca_result.total_co2e_kg
                
                # Generate eco-score
                if self.config.include_eco_score:
                    eco_score = self._lca_calculator.generate_eco_score(lca_result)
                    item.eco_score = eco_score.grade
                    item.eco_score_details = eco_score.to_dict()
                    
            except Exception as e:
                logger.warning(f"Sustainability analysis failed for {item.specific_type}: {e}")
        
        return items
    
    def _generate_product_cards(
        self,
        items: List[DetectedItem],
        original_image: np.ndarray
    ) -> List[DetectedItem]:
        """Generate professional product cards with IC-Light."""
        ic_light = self._get_ic_light()
        
        if not ic_light:
            return items
        
        for item in items:
            if not item.cutout_base64:
                continue
            
            try:
                item_image = self._decode_base64_image(item.cutout_base64)
                if item_image is None:
                    continue
                
                # Apply IC-Light relighting
                result = ic_light.relight(
                    item_image,
                    preset=self.config.product_card_preset
                )
                
                item.product_card_base64 = self._encode_image_base64(
                    result.relit_image
                )
                
            except Exception as e:
                logger.warning(f"Product card generation failed: {e}")
        
        return items
    
    def _analyze_outfit(
        self,
        items: List[DetectedItem]
    ) -> Tuple[str, str, str]:
        """Analyze overall outfit style and harmony."""
        if not items:
            return "", "", ""
        
        # Aggregate styles
        all_styles = []
        all_aesthetics = []
        colors = []
        
        for item in items:
            all_styles.extend(item.style_tags)
            if item.aesthetic:
                all_aesthetics.append(item.aesthetic)
            if item.primary_color:
                colors.append(item.primary_color)
        
        # Determine dominant style
        from collections import Counter
        style_counts = Counter(all_styles)
        dominant_style = style_counts.most_common(1)[0][0] if style_counts else ""
        
        aesthetic_counts = Counter(all_aesthetics)
        dominant_aesthetic = aesthetic_counts.most_common(1)[0][0] if aesthetic_counts else ""
        
        # Color harmony (simplified)
        if len(colors) >= 2:
            unique_colors = list(set(colors))
            if len(unique_colors) == 1:
                harmony = "Monochromatic"
            elif len(unique_colors) == 2:
                harmony = "Complementary"
            else:
                harmony = "Mixed palette"
        else:
            harmony = "Minimal"
        
        return dominant_style, dominant_aesthetic, harmony
    
    def _generate_semantic_embedding(
        self,
        image: np.ndarray
    ) -> Optional[List[float]]:
        """Generate semantic embedding for search."""
        fashion_clip = self._get_fashion_clip()
        
        if not fashion_clip:
            return None
        
        try:
            # Use FashionCLIP to generate embedding
            # This would be the image encoder output
            return None  # Placeholder - would return actual embedding
        except Exception:
            return None
    
    def _create_cutout(
        self,
        image: np.ndarray,
        mask: np.ndarray
    ) -> np.ndarray:
        """Create transparent cutout from mask."""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Ensure mask is same size as image
        if mask.shape[:2] != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Create RGBA image
        rgba = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        
        # Apply mask to alpha channel
        if len(mask.shape) == 2:
            rgba[:, :, 3] = mask
        else:
            rgba[:, :, 3] = mask[:, :, 0] if mask.shape[2] > 0 else 255
        
        return rgba
    
    def _encode_image_base64(self, image: np.ndarray) -> str:
        """Encode image to base64."""
        if image is None:
            return ""
        
        # Handle RGBA
        if len(image.shape) == 3 and image.shape[2] == 4:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA))
        elif len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        
        buffer = BytesIO()
        pil_image.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode()
        
        return f"data:image/png;base64,{b64}"
    
    def _extract_aesthetic(self, description: str) -> str:
        """Extract aesthetic keywords from description."""
        aesthetics = [
            "casual", "formal", "streetwear", "minimalist", "bohemian",
            "classic", "modern", "vintage", "elegant", "sporty"
        ]
        
        desc_lower = description.lower()
        for aesthetic in aesthetics:
            if aesthetic in desc_lower:
                return aesthetic
        
        return ""


# ============================================
# 🔧 UTILITY FUNCTIONS
# ============================================

def analyze_with_visual_intelligence(
    image_b64: str,
    config: Dict = None
) -> Dict:
    """
    Utility function for complete visual intelligence analysis.
    
    Args:
        image_b64: Base64 encoded image
        config: Optional configuration dict
        
    Returns:
        Complete analysis result
    """
    engine_config = EngineConfig()
    
    if config:
        for key, value in config.items():
            if hasattr(engine_config, key):
                setattr(engine_config, key, value)
    
    engine = VisualIntelligenceEngine(engine_config)
    result = engine.analyze_complete(image_b64)
    
    return result.to_dict()


# Singleton instance
_engine_instance = None

def get_visual_intelligence_engine(config: EngineConfig = None) -> VisualIntelligenceEngine:
    """Get singleton instance of VisualIntelligenceEngine."""
    global _engine_instance
    if _engine_instance is None or config is not None:
        _engine_instance = VisualIntelligenceEngine(config or EngineConfig())
    return _engine_instance
