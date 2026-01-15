"""
🎬 Outfit Timeline Analyzer - Precise Video Outfit Detection with Timestamps

This module provides frame-by-frame outfit detection with exact time ranges.

Output Format:
  jacket(zip black "cotton") - pants(gurkha white "wool")(0-2)

Key Features:
1. Precise timestamp tracking (frame → seconds)
2. Per-outfit item grouping with time ranges
3. Enhanced attribute extraction (type, color, material)
4. Formatted string output matching user specification
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging
import base64
import time

logger = logging.getLogger(__name__)


@dataclass
class TimelinedItem:
    """Clothing item with precise time range."""
    category: str  # e.g., "jacket", "pants", "shoes"
    specific_type: str  # e.g., "zip", "gurkha", "boots"
    color: str  # e.g., "black", "white", "dark brown"
    color_hex: str
    material: str  # e.g., "cotton", "wool", "leather", "suede"
    pattern: str  # e.g., "solid", "striped"
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    confidence: float
    cutout_image: str = ""
    bbox: List[int] = field(default_factory=lambda: [0, 0, 100, 100])
    
    def to_formatted_string(self) -> str:
        """
        Format as: category(type color "material")
        
        Examples:
        - jacket(zip black "cotton")
        - pants(gurkha white "wool")
        - shoes(boots dark brown "leather")
        - sweaters(half zip dark blue "wool")
        """
        # Map internal categories to user-friendly names
        CATEGORY_DISPLAY = {
            "upper_clothes": "jacket",
            "upper clothes": "jacket",
            "pants": "pants",
            "shoes": "shoes",
            "left_shoe": "shoes",
            "right_shoe": "shoes",
            "dress": "dress",
            "skirt": "skirt",
            "scarf": "scarf",
            "hat": "hat",
            "bag": "bag",
        }
        
        # Get display category
        cat_lower = self.category.lower()
        display_category = CATEGORY_DISPLAY.get(cat_lower, cat_lower)
        
        # 🚀 FIX: Map vague types to better names
        spec_type = (self.specific_type or "").lower()
        
        # Replace vague "clothing item" with better type based on category
        if "clothing item" in spec_type or spec_type == "" or spec_type == "unknown":
            if "upper" in cat_lower or cat_lower == "jacket":
                spec_type = "jacket"  # Default upper to jacket
            elif "pants" in cat_lower:
                spec_type = "trousers"
            elif "shoe" in cat_lower:
                spec_type = "shoes"
        
        # Replace vague "top" with better type
        if spec_type == "top":
            spec_type = "sweater"  # Most "tops" are sweaters or similar
        
        # Special handling: if specific_type contains "sweater", use "sweaters"
        if "sweater" in spec_type:
            display_category = "sweaters"
        elif "jacket" in spec_type:
            display_category = "jacket"
        elif "coat" in spec_type:
            display_category = "jacket"
        
        parts = []
        
        # Extract short type descriptor
        if self.specific_type and self.specific_type.lower() != self.category.lower():
            type_clean = self.specific_type.replace("-", " ").replace("_", " ").lower()
            # Remove category words to get just the type modifier
            for remove_word in ["jacket", "pants", "shoes", "sweater", "sweaters", "coat", "dress", "skirt"]:
                type_clean = type_clean.replace(remove_word, "").strip()
            
            # Common type mappings for cleaner output
            TYPE_MAPPINGS = {
                "zip up": "zip",
                "half zip": "half zip",
                "quarter zip": "quarter zip",
                "down": "down",
                "suit": "suit",
                "bomber": "bomber",
                "denim": "denim",
                "leather": "leather",
            }
            
            # Get mapped type or use cleaned type
            for key, value in TYPE_MAPPINGS.items():
                if key in type_clean:
                    type_clean = value
                    break
            
            if type_clean:
                parts.append(type_clean)
        
        # Add color
        if self.color:
            parts.append(self.color.lower())
        
        # Add material in quotes
        if self.material:
            # Normalize material
            mat = self.material.lower().replace("-", " ").strip()
            if mat and mat not in ["unknown", "none", ""]:
                parts.append(f'"{mat}"')
        
        inner = " ".join(parts) if parts else display_category
        return f"{display_category}({inner})"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "specificType": self.specific_type,
            "color": self.color,
            "colorHex": self.color_hex,
            "material": self.material,
            "pattern": self.pattern,
            "startFrame": self.start_frame,
            "endFrame": self.end_frame,
            "startTimeSec": round(self.start_time_sec, 2),
            "endTimeSec": round(self.end_time_sec, 2),
            "confidence": round(self.confidence, 3),
            "cutoutImage": self.cutout_image,
            "bbox": self.bbox,
            "formatted": self.to_formatted_string()
        }


@dataclass
class OutfitTimeline:
    """Complete outfit with all items and time range."""
    outfit_id: int
    start_frame: int
    end_frame: int
    start_time_sec: float
    end_time_sec: float
    items: List[TimelinedItem] = field(default_factory=list)
    
    @property
    def formatted_string(self) -> str:
        """Format as: item - item - item(start-end)"""
        if not self.items:
            return ""
        
        # Define item ordering priority (jacket first, then pants, then shoes, then accessories)
        CATEGORY_ORDER = {
            "jacket": 0,
            "sweaters": 1,
            "coat": 2,
            "upper_clothes": 3,
            "pants": 4,
            "skirt": 5,
            "dress": 6,
            "shoes": 7,
            "scarf": 8,
            "hat": 9,
            "bag": 10,
        }
        
        # Sort items by category order
        sorted_items = sorted(
            self.items,
            key=lambda x: CATEGORY_ORDER.get(x.category.lower(), 99)
        )
        
        item_strs = [item.to_formatted_string() for item in sorted_items]
        
        # Better time range display - use actual seconds or frame-based timing
        start_sec = round(self.start_time_sec, 1)
        end_sec = round(self.end_time_sec, 1)
        # If times are very close, use frame-based approximation
        if end_sec - start_sec < 0.5:
            time_range = f"({int(self.start_frame)}-{int(self.end_frame)})"
        else:
            time_range = f"({int(start_sec)}-{int(end_sec)})"
        
        return " - ".join(item_strs) + time_range
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "outfitId": self.outfit_id,
            "startFrame": self.start_frame,
            "endFrame": self.end_frame,
            "startTimeSec": round(self.start_time_sec, 2),
            "endTimeSec": round(self.end_time_sec, 2),
            "items": [item.to_dict() for item in self.items],
            "formatted": self.formatted_string
        }


@dataclass
class FrameDetection:
    """Detection in a single frame."""
    frame_idx: int
    category: str
    specific_type: str
    color: str
    color_hex: str
    material: str
    pattern: str
    confidence: float
    bbox: List[int]
    cutout_image: str = ""


class OutfitTimelineAnalyzer:
    """
    🎬 Precise Video Outfit Timeline Detection
    
    Analyzes video frames and produces:
    - Outfit segments with exact time ranges
    - Per-item tracking with timestamps
    - Formatted output strings
    
    Usage:
        analyzer = OutfitTimelineAnalyzer(fps=30.0)
        for frame, detections in video_data:
            analyzer.add_frame(frame, detections, frame_idx)
        result = analyzer.finalize()
    """
    
    def __init__(
        self,
        fps: float = 30.0,
        scene_change_threshold: float = 0.35,
        min_item_frames: int = 2,  # Minimum frames to count an item
        material_detector_enabled: bool = True
    ):
        """
        Initialize Timeline Analyzer.
        
        Args:
            fps: Video frames per second (for time calculation)
            scene_change_threshold: Histogram correlation for outfit change
            min_item_frames: Minimum frames an item must appear to be counted
            material_detector_enabled: Whether to run material detection
        """
        self.fps = fps
        self.scene_change_threshold = scene_change_threshold
        self.min_item_frames = min_item_frames
        self.material_detector_enabled = material_detector_enabled
        
        # State
        self.frame_detections: Dict[int, List[FrameDetection]] = {}
        self.outfit_boundaries: List[int] = [0]
        self.frame_count = 0
        
        # Previous frame for scene detection
        self._prev_frame_hist: Optional[np.ndarray] = None
        
        # Lazy-loaded detectors
        self._material_analyzer = None
        self._ensemble_detector = None
        
        logger.info(f"OutfitTimelineAnalyzer initialized (fps={fps})")
    
    @property
    def material_analyzer(self):
        """Lazy load material analyzer."""
        if self._material_analyzer is None and self.material_detector_enabled:
            try:
                from modules.material_analyzer import get_material_analyzer
                self._material_analyzer = get_material_analyzer()
                logger.info("✅ Material analyzer loaded")
            except Exception as e:
                logger.warning(f"Material analyzer unavailable: {e}")
        return self._material_analyzer
    
    @property
    def ensemble_detector(self):
        """Lazy load ensemble detector."""
        if self._ensemble_detector is None:
            try:
                from modules.ensemble_detector import get_ensemble_detector
                self._ensemble_detector = get_ensemble_detector()
                logger.info("✅ Ensemble detector loaded")
            except Exception as e:
                logger.warning(f"Ensemble detector unavailable: {e}")
        return self._ensemble_detector
    
    def add_frame(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
        frame_idx: int = None
    ):
        """
        Add a video frame with its detections.
        
        Args:
            frame: BGR image (numpy array)
            detections: List of detection dictionaries
            frame_idx: Frame index (auto-incremented if None)
        """
        if frame_idx is None:
            frame_idx = self.frame_count
        
        self.frame_count = max(self.frame_count, frame_idx + 1)
        
        # Detect scene change (outfit boundary)
        if self._detect_scene_change(frame, frame_idx):
            self.outfit_boundaries.append(frame_idx)
            logger.info(f"🎬 Outfit change detected at frame {frame_idx}")
        
        # Process detections
        frame_dets = []
        
        for det in detections:
            category = det.get("category", "unknown")
            specific_type = det.get("specificType", category)
            color = det.get("primaryColor", "")
            color_hex = det.get("colorHex", "")
            confidence = det.get("confidence", 0.5)
            bbox = det.get("bbox", [0, 0, 100, 100])
            cutout = det.get("cutoutImage", "")
            pattern = det.get("pattern", "solid")
            
            # Get material (already detected or try to detect)
            material = det.get("material", "")
            if not material and self.material_analyzer and cutout:
                try:
                    material = self._detect_material(frame, bbox)
                except Exception:
                    material = ""
            
            frame_dets.append(FrameDetection(
                frame_idx=frame_idx,
                category=category,
                specific_type=specific_type,
                color=color,
                color_hex=color_hex,
                material=material,
                pattern=pattern,
                confidence=confidence,
                bbox=list(bbox),
                cutout_image=cutout
            ))
        
        self.frame_detections[frame_idx] = frame_dets
    
    def _detect_scene_change(self, frame: np.ndarray, frame_idx: int) -> bool:
        """Detect outfit/scene change using color histogram."""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
            cv2.normalize(hist, hist)
            
            if self._prev_frame_hist is None:
                self._prev_frame_hist = hist
                return False
            
            correlation = cv2.compareHist(self._prev_frame_hist, hist, cv2.HISTCMP_CORREL)
            change_score = max(0.0, 1.0 - correlation)
            
            self._prev_frame_hist = hist
            
            # LOWERED THRESHOLD for more aggressive outfit detection
            # 0.15 is more sensitive than 0.35 - will detect more scene changes
            effective_threshold = 0.15  # Was: self.scene_change_threshold (0.35)
            
            # Only trigger if gap from last boundary (at least 1 frame)
            if change_score > effective_threshold:
                if frame_idx - self.outfit_boundaries[-1] >= 1:  # Was: 2
                    logger.info(f"🎬 Scene change at frame {frame_idx}: score={change_score:.3f} > threshold={effective_threshold}")
                    return True
            
            return False
        except Exception:
            return False
    
    def _detect_material(self, frame: np.ndarray, bbox: List[int]) -> str:
        """Detect material for a clothing region."""
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return ""
            
            crop = frame[y1:y2, x1:x2]
            
            if self.material_analyzer:
                result = self.material_analyzer.analyze(crop)
                return result.get("material", "")
            
            return ""
        except Exception:
            return ""
    
    def finalize(self) -> Dict[str, Any]:
        """
        Finalize analysis and return complete timeline.
        
        Returns:
            Dictionary with outfits, formatted timeline, and metadata
        """
        if self.frame_count == 0:
            return {
                "success": False,
                "outfits": [],
                "formattedTimeline": [],
                "totalDurationSec": 0,
                "processingTimeMs": 0
            }
        
        start_time = time.time()
        
        # Build outfit segments
        outfit_timelines = []
        outfit_boundaries = self.outfit_boundaries + [self.frame_count]
        
        for i in range(len(outfit_boundaries) - 1):
            start_frame = outfit_boundaries[i]
            end_frame = outfit_boundaries[i + 1] - 1
            
            # Get all items in this outfit segment
            items = self._aggregate_items_for_segment(start_frame, end_frame)
            
            if items:
                outfit = OutfitTimeline(
                    outfit_id=i + 1,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time_sec=start_frame / self.fps,
                    end_time_sec=(end_frame + 1) / self.fps,
                    items=items
                )
                outfit_timelines.append(outfit)
        
        # Generate formatted strings
        formatted_timeline = [outfit.formatted_string for outfit in outfit_timelines]
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.info(f"📊 Timeline: {len(outfit_timelines)} outfits, "
                   f"{sum(len(o.items) for o in outfit_timelines)} items")
        
        return {
            "success": True,
            "outfits": [o.to_dict() for o in outfit_timelines],
            "formattedTimeline": formatted_timeline,
            "totalDurationSec": round(self.frame_count / self.fps, 2),
            "processingTimeMs": round(processing_time, 2),
            "frameCount": self.frame_count,
            "fps": self.fps
        }
    
    def _aggregate_items_for_segment(
        self,
        start_frame: int,
        end_frame: int
    ) -> List[TimelinedItem]:
        """
        Aggregate all items detected in a frame range.
        
        Uses voting/consensus to determine:
        - Best specific type
        - Best color
        - Best material
        - Mean confidence
        """
        # 🎯 CATEGORY NORMALIZATION - Merge similar categories into main groups
        # This ensures only 1 item per main category per outfit
        CATEGORY_NORMALIZATION = {
            # All pants types → "pants"
            "pants": "pants",
            "jeans": "pants",
            "trousers": "pants",
            "slacks": "pants",
            "chinos": "pants",
            "cargo pants": "pants",
            "joggers": "pants",
            # All shoes types → "shoes"
            "shoes": "shoes",
            "left_shoe": "shoes",
            "right_shoe": "shoes",
            "boots": "shoes",
            "sneakers": "shoes",
            "loafers": "shoes",
            # All upper body → "upper"
            "upper_clothes": "upper",
            "jacket": "upper",
            "coat": "upper",
            "blazer": "upper",
            "sweater": "upper",
            "shirt": "upper",
            "top": "upper",
        }
        
        # Group detections by NORMALIZED category
        category_detections: Dict[str, List[FrameDetection]] = defaultdict(list)
        
        for frame_idx in range(start_frame, end_frame + 1):
            if frame_idx in self.frame_detections:
                for det in self.frame_detections[frame_idx]:
                    # Normalize the category
                    normalized_cat = CATEGORY_NORMALIZATION.get(det.category.lower(), det.category)
                    category_detections[normalized_cat].append(det)
        
        # Build items using weighted voting
        items = []
        
        for category, dets in category_detections.items():
            if len(dets) < self.min_item_frames:
                continue  # Skip items seen in too few frames
            
            # Weighted voting for specific type
            type_votes = defaultdict(float)
            for det in dets:
                type_votes[det.specific_type] += det.confidence
            best_type = max(type_votes, key=type_votes.get) if type_votes else category
            
            # Weighted voting for color
            color_votes = defaultdict(float)
            hex_for_color = {}
            for det in dets:
                if det.color:
                    color_votes[det.color] += det.confidence
                    hex_for_color[det.color] = det.color_hex
            best_color = max(color_votes, key=color_votes.get) if color_votes else ""
            best_hex = hex_for_color.get(best_color, "")
            
            # Weighted voting for material
            material_votes = defaultdict(float)
            for det in dets:
                if det.material:
                    material_votes[det.material] += det.confidence
            best_material = max(material_votes, key=material_votes.get) if material_votes else ""
            
            # Weighted voting for pattern
            pattern_votes = defaultdict(float)
            for det in dets:
                if det.pattern:
                    pattern_votes[det.pattern] += det.confidence
            best_pattern = max(pattern_votes, key=pattern_votes.get) if pattern_votes else "solid"
            
            # Mean confidence
            mean_conf = sum(d.confidence for d in dets) / len(dets)
            
            # Frame range for this item
            item_frames = [d.frame_idx for d in dets]
            item_start = min(item_frames)
            item_end = max(item_frames)
            
            # Best cutout (highest confidence)
            best_det = max(dets, key=lambda d: d.confidence)
            
            items.append(TimelinedItem(
                category=category,
                specific_type=best_type,
                color=best_color,
                color_hex=best_hex,
                material=best_material,
                pattern=best_pattern,
                start_frame=item_start,
                end_frame=item_end,
                start_time_sec=item_start / self.fps,
                end_time_sec=(item_end + 1) / self.fps,
                confidence=mean_conf,
                cutout_image=best_det.cutout_image,
                bbox=best_det.bbox
            ))
        
        # Sort items by category importance
        category_order = ["jacket", "coat", "sweater", "shirt", "top", 
                         "pants", "trousers", "skirt", "shorts",
                         "shoes", "boots", "sneakers",
                         "bag", "hat", "scarf", "accessories"]
        
        def sort_key(item):
            try:
                return category_order.index(item.category.lower())
            except ValueError:
                return len(category_order)
        
        items.sort(key=sort_key)
        
        return items
    
    def reset(self):
        """Reset analyzer state for new video."""
        self.frame_detections.clear()
        self.outfit_boundaries = [0]
        self.frame_count = 0
        self._prev_frame_hist = None
        logger.info("OutfitTimelineAnalyzer reset")


# === SINGLETON INSTANCE ===
_timeline_analyzer_instance: Optional[OutfitTimelineAnalyzer] = None


def get_outfit_timeline_analyzer(
    reset: bool = True,
    **kwargs
) -> OutfitTimelineAnalyzer:
    """
    Get singleton outfit timeline analyzer.
    
    Args:
        reset: Reset analyzer state (recommended for new video)
        **kwargs: Constructor arguments
    
    Returns:
        OutfitTimelineAnalyzer instance
    """
    global _timeline_analyzer_instance
    
    if _timeline_analyzer_instance is None:
        _timeline_analyzer_instance = OutfitTimelineAnalyzer(**kwargs)
    elif reset:
        _timeline_analyzer_instance.reset()
        # Update FPS if provided
        if "fps" in kwargs:
            _timeline_analyzer_instance.fps = kwargs["fps"]
    
    return _timeline_analyzer_instance


def analyze_video_timeline(
    frames: List[np.ndarray],
    frame_detections: List[List[Dict]],
    fps: float = 30.0,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function for batch video analysis.
    
    Args:
        frames: List of BGR images
        frame_detections: List of detection lists per frame
        fps: Video FPS
        **kwargs: Analyzer configuration
    
    Returns:
        Timeline analysis result
    """
    analyzer = OutfitTimelineAnalyzer(fps=fps, **kwargs)
    
    for frame_idx, (frame, detections) in enumerate(zip(frames, frame_detections)):
        analyzer.add_frame(frame, detections, frame_idx)
    
    return analyzer.finalize()
