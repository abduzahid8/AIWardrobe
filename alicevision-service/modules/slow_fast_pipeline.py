"""
🚀 Slow-Fast Architecture for Real-Time Fashion Video Analysis

This module implements the AIWardrobe 2.0 Slow-Fast pipeline:

Fast Path (Every Frame ~1.5ms):
 - YOLOv8-Seg: Detection + Segmentation
 - FeatureSORT: ReID-based tracking
 
Slow Path (Keyframes Only ~100-200ms):
 - Florence-2: Dense captioning
 - FashionFAE: Fine-grained attributes
 
State Fusion:
 - Propagate rich metadata to subsequent frames
 - Update on new TrackID or scheduled keyframes
"""

import numpy as np
import cv2
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================
# Data Structures
# ============================================

@dataclass
class TrackState:
    """State for a tracked clothing item."""
    track_id: int
    category: str
    bbox: Tuple[float, float, float, float]
    mask: Optional[np.ndarray] = None
    outfit_id: int = 1
    is_new: bool = False
    
    # Slow path metadata (populated on keyframes)
    caption: str = ""
    specific_type: str = ""
    primary_color: str = ""
    color_hex: str = "#000000"
    material: str = ""
    pattern: str = ""
    neckline: str = ""
    sleeve_type: str = ""
    style_tags: List[str] = field(default_factory=list)
    
    # Cutout image
    cutout_base64: str = ""
    
    # Tracking metadata
    last_slow_update: int = -1
    confidence: float = 0.5
    frames_seen: int = 0


@dataclass
class PersonState:
    """State for a tracked person (outfit anchor)."""
    person_id: int
    bbox: Tuple[float, float, float, float]
    mask: Optional[np.ndarray] = None
    reid_embedding: Optional[np.ndarray] = None
    clothing_track_ids: List[int] = field(default_factory=list)


@dataclass
class FrameResult:
    """Result from processing a single frame."""
    frame_idx: int
    tracks: List[TrackState]
    persons: List[PersonState]
    processing_time_ms: float
    used_slow_path: bool


# ============================================
# Slow-Fast Pipeline
# ============================================

class SlowFastPipeline:
    """
    🚀 Slow-Fast Architecture for Real-Time Fashion Video
    
    Orchestrates fast detection (every frame) with deep analysis (keyframes only).
    
    Fast Path (~1.5ms per frame):
        - YOLOv8-Seg for detection + instance segmentation
        - FeatureSORT for ReID-based tracking
        
    Slow Path (~100-200ms on keyframes):
        - Florence-2 for dense captioning
        - FashionFAE for fine-grained attributes
        - Only triggered for new tracks or scheduled keyframes
    """
    
    def __init__(
        self,
        keyframe_interval: int = 30,
        quality_threshold: float = 0.7,
        enable_slow_path: bool = True,
        use_reid: bool = True
    ):
        """
        Initialize Slow-Fast Pipeline.
        
        Args:
            keyframe_interval: Frames between slow-path analysis
            quality_threshold: Minimum frame quality for slow-path
            enable_slow_path: Enable/disable slow path (for benchmarking)
            use_reid: Use ReID for person tracking
        """
        self.keyframe_interval = keyframe_interval
        self.quality_threshold = quality_threshold
        self.enable_slow_path = enable_slow_path
        self.use_reid = use_reid
        
        # Track states
        self.track_states: Dict[int, TrackState] = {}
        self.person_states: Dict[int, PersonState] = {}
        
        # 🚀 NEW: Per-outfit item storage to prevent cross-outfit merging
        # Key: (outfit_id, category) → best item for that outfit+category combo
        self.outfit_items: Dict[Tuple[int, str], Dict[str, Any]] = {}
        
        # Frame counter
        self.frame_count = 0
        
        # 🎬 SCENE CHANGE DETECTION (NEW)
        self._prev_frame_hist = None
        self._current_outfit_id = 1
        self._outfit_boundaries = [0]
        self._scene_change_threshold = 0.35  # Tuned for rapid outfit changes
        
        # Lazy-loaded components
        self._yolo_seg = None
        self._feature_sort = None
        self._florence = None
        self._fashion_fae = None
        self._keyframe_scheduler = None
        
        logger.info(f"SlowFastPipeline initialized (keyframe_interval={keyframe_interval})")
    
    # ----------------------------------------
    # Lazy Loading
    # ----------------------------------------
    
    @property
    def yolo_seg(self):
        """Lazy load YOLOv8-Seg detector."""
        if self._yolo_seg is None:
            from modules.yolo_seg_detector import get_yolo_seg_detector
            self._yolo_seg = get_yolo_seg_detector()
            logger.info("✅ YOLOv8-Seg loaded for fast path")
        return self._yolo_seg
    
    @property
    def feature_sort(self):
        """Lazy load FeatureSORT tracker."""
        if self._feature_sort is None:
            from modules.feature_sort_tracker import get_feature_sort_tracker
            self._feature_sort = get_feature_sort_tracker()
            logger.info("✅ FeatureSORT loaded for tracking")
        return self._feature_sort
    
    @property
    def florence(self):
        """Lazy load Florence-2 detector."""
        if self._florence is None:
            from modules.florence_detector import get_florence_detector
            self._florence = get_florence_detector()
            logger.info("✅ Florence-2 loaded for slow path")
        return self._florence
    
    @property
    def fashion_fae(self):
        """Lazy load FashionFAE classifier."""
        if self._fashion_fae is None:
            from modules.fashion_fae import get_fashion_fae
            self._fashion_fae = get_fashion_fae()
            logger.info("✅ FashionFAE loaded for attributes")
        return self._fashion_fae
    
    @property
    def keyframe_scheduler(self):
        """Lazy load keyframe scheduler."""
        if self._keyframe_scheduler is None:
            from modules.keyframe_scheduler import KeyframeScheduler
            self._keyframe_scheduler = KeyframeScheduler(
                interval=self.keyframe_interval,
                quality_threshold=self.quality_threshold
            )
            logger.info("✅ KeyframeScheduler loaded")
        return self._keyframe_scheduler
    
    # ----------------------------------------
    # Main Processing
    # ----------------------------------------
    
    def process_frame(self, frame: np.ndarray, frame_idx: int = None) -> FrameResult:
        """
        Process a single video frame through the Slow-Fast pipeline.
        
        Args:
            frame: BGR image (numpy array)
            frame_idx: Frame index (auto-incremented if None)
            
        Returns:
            FrameResult with tracked items and metadata
        """
        start_time = time.time()
        
        if frame_idx is None:
            frame_idx = self.frame_count
        self.frame_count = frame_idx + 1
        
        # ========================================
        # 1. FAST PATH - Runs every frame (~1.5ms)
        # ========================================
        
        # 🎬 SCENE CHANGE DETECTION (NEW)
        scene_changed = self._detect_scene_change(frame, frame_idx)
        
        # Initialize variables
        clothing_dets = []
        person_dets = []
        masks = None
        
        # 1a. FIXED: Use SegFormer (18 clothing categories) instead of YOLO (only person+accessories)
        # YOLO only detects: person, tie, backpack, handbag, suitcase - NOT actual clothing!
        # SegFormer detects: upper_clothes, pants, dress, skirt, shoes, jacket, sweater, etc.
        try:
            from modules.segmentation import AdvancedClothingSegmentor
            segmentor = AdvancedClothingSegmentor(use_segformer=True)
            seg_result = segmentor.segment(frame, add_white_bg=False, return_items=True)
            
            # Convert segmentor items to YOLOSegDetection format
            from modules.yolo_seg_detector import YOLOSegDetection
            for item in seg_result.items:
                mask_data = getattr(item, 'mask', None)
                bbox_data = getattr(item, 'bbox', [0, 0, 100, 100])
                det = YOLOSegDetection(
                    class_name=item.category,
                    category=item.category,
                    confidence=item.confidence,
                    bbox=tuple(bbox_data) if bbox_data else (0, 0, 100, 100),
                    mask=mask_data
                )
                clothing_dets.append(det)
            
            # Collect masks array
            item_masks = [getattr(item, 'mask', None) for item in seg_result.items if getattr(item, 'mask', None) is not None]
            if item_masks:
                masks = np.stack(item_masks)
            
            # Use YOLO for person detection only
            try:
                yolo_dets, _ = self.yolo_seg.detect_with_masks(frame)
                person_dets = [d for d in yolo_dets if d.class_name.lower() == "person"]
            except Exception:
                pass
                
            logger.info(f"  Frame {frame_idx}: SegFormer detected {len(clothing_dets)} clothing items")
            
            # 🚀 NEW: Store items per-outfit to prevent cross-outfit merging
            for item in seg_result.items:
                key = (self._current_outfit_id, item.category)
                item_dict = {
                    "trackId": hash(key) % 100000,  # Unique ID per outfit+category
                    "category": item.category,
                    "specificType": getattr(item, 'specific_type', item.category),
                    "primaryColor": getattr(item, 'primary_color', 'Unknown'),
                    "colorHex": getattr(item, 'color_hex', '#000000'),
                    "material": "",
                    "pattern": "",
                    "confidence": item.confidence,
                    "outfitId": self._current_outfit_id,
                    "bbox": list(item.bbox) if hasattr(item, 'bbox') else [0, 0, 100, 100],
                    "framesDetected": 1,
                    "cutoutImage": ""
                }
                # Keep highest confidence item for each (outfit, category) combo
                if key not in self.outfit_items or item.confidence > self.outfit_items[key].get("confidence", 0):
                    self.outfit_items[key] = item_dict
            
        except Exception as e:
            logger.warning(f"SegFormer fallback to YOLO: {e}")
            # Fallback to YOLO
            detections, masks = self.yolo_seg.detect_with_masks(frame)
            person_dets = [d for d in detections if d.class_name.lower() == "person"]
            clothing_dets = [d for d in detections if d.class_name.lower() != "person"]
        
        # 1c. FeatureSORT: Track persons and clothing
        tracked_persons, tracked_clothing = self.feature_sort.update(
            person_dets, clothing_dets, masks, frame
        )
        
        # 1d. Assign outfit IDs - use scene-based if no persons detected
        if not tracked_persons:
            # No persons detected - use scene change for outfit boundaries
            for track in tracked_clothing:
                track.outfit_id = self._current_outfit_id
        else:
            self._assign_outfit_ids(tracked_clothing, tracked_persons)
        
        # ========================================
        # 2. UPDATE TRACK STATES
        # ========================================
        
        current_track_ids = set()
        tracks_for_slow = []
        
        for track in tracked_clothing:
            track_id = track.track_id
            current_track_ids.add(track_id)
            
            # Create or update track state
            if track_id not in self.track_states:
                # New track
                self.track_states[track_id] = TrackState(
                    track_id=track_id,
                    category=track.category,
                    bbox=track.bbox,
                    mask=track.mask,
                    outfit_id=track.outfit_id,
                    is_new=True,
                    confidence=track.confidence
                )
                tracks_for_slow.append(self.track_states[track_id])
            else:
                # Update existing track
                state = self.track_states[track_id]
                state.bbox = track.bbox
                state.mask = track.mask
                state.outfit_id = track.outfit_id
                state.is_new = False
                state.confidence = track.confidence
                state.frames_seen += 1
        
        # ========================================
        # 3. SLOW PATH - Keyframes only (~100-200ms)
        # ========================================
        
        used_slow_path = False
        
        if self.enable_slow_path:
            # Check which tracks need slow-path analysis
            for track_id in current_track_ids:
                state = self.track_states[track_id]
                
                should_analyze = (
                    state.is_new or
                    (frame_idx - state.last_slow_update) >= self.keyframe_interval
                )
                
                if should_analyze and track_id not in [t.track_id for t in tracks_for_slow]:
                    tracks_for_slow.append(state)
            
            # Run slow path on selected tracks
            if tracks_for_slow:
                used_slow_path = True
                self._run_slow_path(frame, tracks_for_slow, frame_idx)
        
        # ========================================
        # 4. BUILD RESULT
        # ========================================
        
        # Update person states
        person_states = []
        for person in tracked_persons:
            if person.track_id not in self.person_states:
                self.person_states[person.track_id] = PersonState(
                    person_id=person.track_id,
                    bbox=person.bbox,
                    mask=person.mask,
                    reid_embedding=person.reid_embedding
                )
            ps = self.person_states[person.track_id]
            ps.bbox = person.bbox
            ps.clothing_track_ids = [t.track_id for t in tracked_clothing if t.outfit_id == person.track_id]
            person_states.append(ps)
        
        # Get current tracks
        result_tracks = [
            self.track_states[tid] for tid in current_track_ids
            if tid in self.track_states
        ]
        
        processing_time = (time.time() - start_time) * 1000
        
        return FrameResult(
            frame_idx=frame_idx,
            tracks=result_tracks,
            persons=person_states,
            processing_time_ms=processing_time,
            used_slow_path=used_slow_path
        )
    
    def _run_slow_path(
        self, 
        frame: np.ndarray, 
        tracks: List[TrackState], 
        frame_idx: int
    ):
        """
        Run slow-path analysis on selected tracks.
        
        Extracts:
        - Dense caption from Florence-2
        - Fine-grained attributes from FashionFAE
        """
        logger.info(f"🔬 Slow path: Analyzing {len(tracks)} tracks at frame {frame_idx}")
        
        for state in tracks:
            try:
                # Extract crop
                x1, y1, x2, y2 = [int(v) for v in state.bbox]
                crop = frame[y1:y2, x1:x2]
                
                if crop.size == 0:
                    continue
                
                # Florence-2: Dense caption
                try:
                    florence_result = self.florence.detect_clothing(crop)
                    if florence_result:
                        det = florence_result[0]
                        state.caption = det.description
                        state.specific_type = det.specific_type
                        if det.colors:
                            state.primary_color = det.colors[0]
                except Exception as e:
                    logger.warning(f"Florence-2 failed: {e}")
                
                # FashionFAE: Fine-grained attributes
                try:
                    fae_result = self.fashion_fae.extract(crop)
                    state.material = fae_result.get("material", "")
                    state.pattern = fae_result.get("pattern", "")
                    state.neckline = fae_result.get("neckline", "")
                    state.sleeve_type = fae_result.get("sleeve_type", "")
                    state.style_tags = fae_result.get("style_tags", [])
                except Exception as e:
                    logger.warning(f"FashionFAE failed: {e}")
                
                # Create cutout
                state.cutout_base64 = self._create_cutout(crop, state.mask)
                
                state.last_slow_update = frame_idx
                state.is_new = False
                
                logger.info(f"  ✅ Track {state.track_id}: {state.specific_type or state.category}")
                
            except Exception as e:
                logger.error(f"Slow path error for track {state.track_id}: {e}")
    
    def _assign_outfit_ids(
        self, 
        clothing_tracks: List, 
        person_tracks: List
    ):
        """
        Assign outfit IDs based on person-clothing overlap.
        
        Each clothing item gets the ID of the person whose mask
        it most overlaps with.
        """
        if not person_tracks:
            # No persons detected - all items get outfit_id = 1
            for track in clothing_tracks:
                track.outfit_id = 1
            return
        
        for cloth in clothing_tracks:
            best_overlap = 0
            best_person_id = 1
            
            for person in person_tracks:
                if cloth.mask is not None and person.mask is not None:
                    overlap = self._compute_mask_overlap(cloth.mask, person.mask)
                else:
                    overlap = self._compute_bbox_overlap(cloth.bbox, person.bbox)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_person_id = person.track_id
            
            cloth.outfit_id = best_person_id if best_overlap > 0.1 else 1
    
    def _compute_mask_overlap(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two masks."""
        if mask1.shape != mask2.shape:
            return 0.0
        
        intersection = np.logical_and(mask1 > 0, mask2 > 0).sum()
        union = np.logical_or(mask1 > 0, mask2 > 0).sum()
        
        return intersection / union if union > 0 else 0.0
    
    def _compute_bbox_overlap(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Compute IoU between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0
    
    def _detect_scene_change(self, frame: np.ndarray, frame_idx: int) -> bool:
        """
        🎬 Detect scene change using color histogram comparison.
        
        When scenes change significantly (different outfit visible),
        increment the outfit ID so new tracks get new outfit numbers.
        
        Returns:
            True if scene change detected
        """
        try:
            # Compute HSV histogram for current frame
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            
            if self._prev_frame_hist is None:
                self._prev_frame_hist = hist
                return False
            
            # Compare with previous frame
            correlation = cv2.compareHist(self._prev_frame_hist, hist, cv2.HISTCMP_CORREL)
            change_score = max(0.0, 1.0 - correlation)
            
            # Update previous histogram
            self._prev_frame_hist = hist
            
            # Check for scene change
            if change_score > self._scene_change_threshold:
                # Only trigger if enough gap from last boundary
                if frame_idx - self._outfit_boundaries[-1] >= 2:
                    self._current_outfit_id += 1
                    self._outfit_boundaries.append(frame_idx)
                    logger.info(f"🎬 Scene change detected at frame {frame_idx}, "
                               f"now outfit #{self._current_outfit_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.debug(f"Scene change detection error: {e}")
            return False
    
    def _create_cutout(self, crop: np.ndarray, mask: np.ndarray = None) -> str:
        """Create base64 cutout image with white background."""
        import base64
        
        try:
            if mask is not None:
                # Apply mask
                crop = crop.copy()
                if mask.shape[:2] != crop.shape[:2]:
                    mask = cv2.resize(mask.astype(np.uint8), (crop.shape[1], crop.shape[0]))
                crop[mask == 0] = 255
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return base64.b64encode(buffer).decode('utf-8')
        except:
            return ""
    
    # ----------------------------------------
    # Finalization
    # ----------------------------------------
    
    def finalize(self) -> List[Dict[str, Any]]:
        """
        Finalize analysis and return deduplicated items.
        
        Returns:
            List of item dictionaries ready for API response
        """
        # 🚀 FIXED: Use outfit_items instead of track_states
        # outfit_items stores items by (outfit_id, category) to prevent cross-outfit merging
        items = list(self.outfit_items.values())
        
        # If no outfit_items, fall back to track_states (for non-SegFormer paths)
        if not items:
            for track_id, state in self.track_states.items():
                items.append({
                    "trackId": track_id,
                    "category": state.category,
                    "specificType": state.specific_type or state.category,
                    "primaryColor": state.primary_color,
                    "colorHex": state.color_hex,
                    "material": state.material,
                    "pattern": state.pattern,
                    "neckline": state.neckline,
                    "sleeveType": state.sleeve_type,
                    "styleTags": state.style_tags,
                    "caption": state.caption,
                    "confidence": round(state.confidence, 3),
                    "outfitId": state.outfit_id,
                    "bbox": list(state.bbox),
                    "cutoutImage": state.cutout_base64,
                    "framesDetected": state.frames_seen
                })
        
        # Sort by confidence
        items.sort(key=lambda x: x["confidence"], reverse=True)
        
        return items
    
    def get_outfit_groups(self) -> Dict[int, List[int]]:
        """
        Get outfit groupings.
        
        Returns:
            Dictionary mapping outfit_id to list of track_ids
        """
        groups = defaultdict(list)
        
        for track_id, state in self.track_states.items():
            groups[state.outfit_id].append(track_id)
        
        return dict(groups)
    
    def reset(self):
        """Reset pipeline state for new video."""
        self.track_states.clear()
        self.person_states.clear()
        self.outfit_items.clear()  # Clear per-outfit items
        self.frame_count = 0
        
        # Reset scene change tracking
        self._prev_frame_hist = None
        self._current_outfit_id = 1
        self._outfit_boundaries = [0]
        
        if self._feature_sort:
            self._feature_sort.reset()
        
        logger.info("SlowFastPipeline reset")


# ============================================
# Singleton
# ============================================

_pipeline_instance = None


def get_slow_fast_pipeline(**kwargs) -> SlowFastPipeline:
    """Get singleton SlowFastPipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = SlowFastPipeline(**kwargs)
    return _pipeline_instance
