"""
🎬 Full Video Clothing Analyzer
Aggregates detections across ALL frames for maximum item discovery

Key Features:
1. UNION-based detection (never miss items)
2. Cross-frame type refinement (partial → full visibility)
3. Confidence boosting for consistent items
4. Smart deduplication by category overlap
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import logging
import base64
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class VideoItem:
    """Aggregated item from full video analysis"""
    category: str
    specific_type: str
    primary_color: str
    color_hex: str
    confidence: float
    best_cutout_base64: str
    best_bbox: Tuple[int, int, int, int]
    
    # Multi-frame data
    frames_detected: int
    total_frames: int
    type_candidates: Dict[str, float] = field(default_factory=dict)
    color_candidates: Dict[str, float] = field(default_factory=dict)
    all_confidences: List[float] = field(default_factory=list)
    
    # Detailed attributes (aggregated)
    pattern: str = ""
    material: str = ""
    features: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "category": self.category,
            "specificType": self.specific_type,
            "primaryColor": self.primary_color,
            "colorHex": self.color_hex,
            "confidence": round(self.confidence, 3),
            "cutoutImage": self.best_cutout_base64,
            "bbox": list(self.best_bbox),
            "pattern": self.pattern,
            "material": self.material,
            "features": self.features,
            "multiFrameData": {
                "framesDetected": self.frames_detected,
                "totalFrames": self.total_frames,
                "agreement": round(self.frames_detected / max(1, self.total_frames), 2),
                "typeCandidates": {k: round(v, 2) for k, v in self.type_candidates.items()},
                "colorCandidates": {k: round(v, 2) for k, v in self.color_candidates.items()}
            }
        }


class FullVideoAnalyzer:
    """
    🎬 FULL VIDEO ANALYZER - Never miss an item!
    
    Unlike the voting-based multi-frame analyzer, this uses a UNION approach:
    - Detects items across ALL frames
    - Even partial visibility counts
    - Refines type/color as more frames are analyzed
    - Smart category merging (e.g., "upper_clothes" + "jacket" → "jacket")
    """
    
    # Category hierarchy for merging
    CATEGORY_HIERARCHY = {
        # Upper body
        "upper_clothes": ["jacket", "coat", "blazer", "shirt", "t-shirt", "sweater", "hoodie", "cardigan", "vest", "blouse"],
        "top": ["jacket", "coat", "blazer", "shirt", "t-shirt", "sweater", "hoodie", "cardigan", "vest", "blouse"],
        
        # Lower body
        "pants": ["jeans", "trousers", "chinos", "dress_pants", "joggers", "shorts"],
        "lower_clothes": ["pants", "jeans", "trousers", "shorts", "skirt"],
        
        # Footwear
        "shoes": ["sneakers", "boots", "loafers", "oxford", "sandals", "heels"],
        "left_shoe": ["sneakers", "boots", "loafers"],
        "right_shoe": ["sneakers", "boots", "loafers"],
        
        # Accessories
        "bag": ["handbag", "backpack", "tote", "clutch"],
    }
    
    # Categories that should be merged
    MERGE_CATEGORIES = {
        "left_shoe": "shoes",
        "right_shoe": "shoes",
        "upper_clothes": None,  # Will use specific type
        "lower_clothes": None,
        "top": None,
    }
    
    def __init__(
        self,
        min_confidence: float = 0.25,
        single_frame_penalty: float = 0.15
    ):
        """
        Args:
            min_confidence: Minimum confidence to consider a detection
            single_frame_penalty: Confidence penalty for items only in 1 frame
        """
        self.min_confidence = min_confidence
        self.single_frame_penalty = single_frame_penalty
    
    def analyze_full_video(
        self,
        frame_results: List[Dict],
        frame_images: List[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Analyze ALL frames and aggregate detections.
        
        Args:
            frame_results: List of segment-all results from each frame
            frame_images: Optional original frames for re-cropping
            
        Returns:
            Complete video analysis with all detected items
        """
        if not frame_results:
            return {"success": False, "items": [], "error": "No frames provided"}
        
        valid_results = [r for r in frame_results if r.get("success", False)]
        n_frames = len(valid_results)
        
        if not valid_results:
            return {"success": False, "items": [], "error": "All frames failed"}
        
        logger.info(f"🎬 Full Video Analysis: {n_frames} frames")
        
        # === COLLECT ALL DETECTIONS ===
        all_detections = defaultdict(list)  # category -> list of detections
        
        for frame_idx, result in enumerate(valid_results):
            for item in result.get("items", []):
                category = self._normalize_category(item)
                conf = item.get("confidence", 0.5)
                
                if conf >= self.min_confidence:
                    all_detections[category].append({
                        "frame": frame_idx,
                        "item": item,
                        "confidence": conf,
                        "specificType": item.get("specificType", ""),
                        "primaryColor": item.get("primaryColor", ""),
                        "colorHex": item.get("colorHex", "#000000"),
                        "cutoutImage": item.get("cutoutImage", ""),
                        "bbox": item.get("bbox", [0, 0, 0, 0]),
                        "attributes": item.get("attributes", {})
                    })
        
        logger.info(f"📊 Found {len(all_detections)} unique categories across {n_frames} frames")
        
        # === MERGE SIMILAR CATEGORIES ===
        merged_detections = self._merge_similar_categories(all_detections)
        
        # === CREATE FINAL ITEMS ===
        final_items = []
        
        for category, detections in merged_detections.items():
            video_item = self._aggregate_detections(category, detections, n_frames)
            
            if video_item:
                final_items.append(video_item)
                logger.info(f"  ✅ {category}: {video_item.frames_detected}/{n_frames} frames → {video_item.specific_type} ({video_item.primary_color})")
        
        # Sort by confidence
        final_items.sort(key=lambda x: x.confidence, reverse=True)
        
        return {
            "success": True,
            "items": [item.to_dict() for item in final_items],
            "itemCount": len(final_items),
            "framesAnalyzed": n_frames,
            "strategy": "full_video_union"
        }
    
    def _normalize_category(self, item: Dict) -> str:
        """Normalize category to handle variations."""
        category = item.get("category", "unknown").lower().strip()
        specific = item.get("specificType", "").lower().strip()
        
        # Map merge categories
        if category in self.MERGE_CATEGORIES:
            merged = self.MERGE_CATEGORIES[category]
            if merged:
                return merged
            # Use specific type if available
            if specific:
                return specific
        
        # Normalize shoe categories
        if "shoe" in category:
            return "shoes"
        
        return category
    
    def _merge_similar_categories(
        self,
        all_detections: Dict[str, List]
    ) -> Dict[str, List]:
        """Merge categories that represent the same item."""
        merged = defaultdict(list)
        
        # Track which categories we've already handled
        handled = set()
        
        for category, detections in all_detections.items():
            if category in handled:
                continue
            
            # Check for related categories
            related = [category]
            
            # Find parent category if this is a specific type
            for parent, children in self.CATEGORY_HIERARCHY.items():
                if category in children or category == parent:
                    related.append(parent)
                    related.extend(children)
            
            # Collect all related detections
            all_related_detections = []
            for rel_cat in set(related):
                if rel_cat in all_detections:
                    all_related_detections.extend(all_detections[rel_cat])
                    handled.add(rel_cat)
            
            # Use the most specific category name
            best_cat = self._get_best_category_name(all_related_detections, category)
            
            merged[best_cat].extend(all_related_detections)
        
        return merged
    
    def _get_best_category_name(
        self,
        detections: List[Dict],
        default: str
    ) -> str:
        """Get the most specific category name from detections."""
        # Count specific types
        type_counts = Counter()
        
        for det in detections:
            specific = det.get("specificType", "")
            if specific and specific != "unknown":
                type_counts[specific] += det["confidence"]
        
        if type_counts:
            # Return most confident specific type
            return type_counts.most_common(1)[0][0]
        
        return default
    
    def _aggregate_detections(
        self,
        category: str,
        detections: List[Dict],
        total_frames: int
    ) -> Optional[VideoItem]:
        """Aggregate all detections for a category into one VideoItem."""
        if not detections:
            return None
        
        # Count unique frames
        frames_detected = len(set(d["frame"] for d in detections))
        
        # Find best detection (highest confidence with cutout)
        best = None
        best_score = 0
        
        for det in detections:
            score = det["confidence"]
            # Bonus for having a cutout
            if det.get("cutoutImage"):
                score += 0.1
            if score > best_score:
                best_score = score
                best = det
        
        if not best:
            return None
        
        # Aggregate specific types (weighted by confidence)
        type_scores = defaultdict(float)
        for det in detections:
            t = det.get("specificType", "")
            if t:
                type_scores[t] += det["confidence"]
        
        best_type = max(type_scores.items(), key=lambda x: x[1])[0] if type_scores else category
        
        # Aggregate colors (weighted)
        color_scores = defaultdict(float)
        color_hexes = {}
        
        for det in detections:
            c = det.get("primaryColor", "")
            if c:
                color_scores[c] += det["confidence"]
                color_hexes[c] = det.get("colorHex", "#000000")
        
        best_color = max(color_scores.items(), key=lambda x: x[1])[0] if color_scores else "unknown"
        best_hex = color_hexes.get(best_color, "#000000")
        
        # Calculate final confidence
        avg_confidence = sum(d["confidence"] for d in detections) / len(detections)
        agreement_ratio = frames_detected / total_frames
        
        # Boost confidence for items seen in multiple frames
        if frames_detected > 1:
            confidence = min(0.98, avg_confidence + (agreement_ratio * 0.15))
        else:
            # Single-frame detection - apply penalty
            confidence = max(0.3, avg_confidence - self.single_frame_penalty)
        
        # Aggregate patterns and materials
        patterns = Counter()
        materials = Counter()
        
        for det in detections:
            attrs = det.get("attributes", {})
            if attrs.get("pattern"):
                p = attrs["pattern"]
                if isinstance(p, dict):
                    patterns[p.get("type", "solid")] += 1
                else:
                    patterns[p] += 1
            if attrs.get("material"):
                m = attrs["material"]
                if isinstance(m, dict):
                    materials[m.get("material", "unknown")] += 1
                else:
                    materials[m] += 1
        
        return VideoItem(
            category=category,
            specific_type=best_type,
            primary_color=best_color,
            color_hex=best_hex,
            confidence=confidence,
            best_cutout_base64=best.get("cutoutImage", ""),
            best_bbox=tuple(best.get("bbox", [0, 0, 0, 0])),
            frames_detected=frames_detected,
            total_frames=total_frames,
            type_candidates=dict(type_scores),
            color_candidates=dict(color_scores),
            all_confidences=[d["confidence"] for d in detections],
            pattern=patterns.most_common(1)[0][0] if patterns else "",
            material=materials.most_common(1)[0][0] if materials else "",
        )


def analyze_video_comprehensive(
    frames_base64: List[str],
    segment_func,
    max_frames: int = 10
) -> Dict[str, Any]:
    """
    🎬 COMPREHENSIVE VIDEO ANALYSIS
    
    Analyzes multiple frames and aggregates ALL detections.
    Never misses items that appear in ANY frame.
    
    Args:
        frames_base64: List of base64-encoded video frames
        segment_func: Function to segment each frame
        max_frames: Maximum frames to analyze (evenly sampled)
        
    Returns:
        Complete video analysis result
    """
    if not frames_base64:
        return {"success": False, "error": "No frames provided"}
    
    # Sample frames evenly if too many
    if len(frames_base64) > max_frames:
        step = len(frames_base64) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        selected_frames = [frames_base64[i] for i in indices]
    else:
        selected_frames = frames_base64
    
    logger.info(f"🎬 Comprehensive analysis: {len(selected_frames)} frames (of {len(frames_base64)} total)")
    
    # Process each frame
    frame_results = []
    
    for i, frame_b64 in enumerate(selected_frames):
        try:
            result = segment_func(frame_b64)
            if result.get("success"):
                frame_results.append(result)
                items = [f"{item.get('specificType') or item.get('category')}" for item in result.get("items", [])]
                logger.info(f"  Frame {i+1}: {len(result.get('items', []))} items → {items}")
            else:
                logger.warning(f"  Frame {i+1}: Failed")
        except Exception as e:
            logger.warning(f"  Frame {i+1}: Error - {e}")
    
    # Aggregate with UNION strategy
    analyzer = FullVideoAnalyzer()
    result = analyzer.analyze_full_video(frame_results)
    
    logger.info(f"📊 Final result: {result.get('itemCount', 0)} unique items detected")
    
    return result


# Singleton instance
_full_video_analyzer = None

def get_full_video_analyzer() -> FullVideoAnalyzer:
    """Get singleton FullVideoAnalyzer instance."""
    global _full_video_analyzer
    if _full_video_analyzer is None:
        _full_video_analyzer = FullVideoAnalyzer()
    return _full_video_analyzer
