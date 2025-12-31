"""
🎯 ByteTrack Identity-Consistent Aggregation (ICA)
Tracklet-based video analysis that eliminates duplicate noise

Key Features:
1. ByteTrack lightweight object tracking (IoU + Kalman)
2. Unique ID per item across frames (Item_01, Item_02...)
3. Attribute aggregation per tracklet
4. Best frame selection per tracklet for FashionCLIP
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


# ============================================
# 🎯 DATA STRUCTURES
# ============================================

@dataclass
class Detection:
    """Single frame detection"""
    frame_idx: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    category: str
    specific_type: str
    primary_color: str
    color_hex: str
    confidence: float
    cutout_base64: str = ""
    sharpness: float = 0.0
    attributes: Dict = field(default_factory=dict)


@dataclass
class Tracklet:
    """
    Object tracked across frames with unique ID.
    Aggregates all detections of the same physical item.
    """
    track_id: int
    category: str
    detections: List[Detection] = field(default_factory=list)
    state: str = "active"  # active, lost, removed
    
    # Kalman filter state
    mean: np.ndarray = None  # [cx, cy, aspect_ratio, height, vx, vy, va, vh]
    covariance: np.ndarray = None
    
    # Tracking history
    hits: int = 0
    age: int = 0
    time_since_update: int = 0
    
    def add_detection(self, det: Detection):
        """Add detection to tracklet."""
        self.detections.append(det)
        self.hits += 1
        self.time_since_update = 0
    
    def get_best_detection(self) -> Optional[Detection]:
        """Get highest quality detection (by sharpness + confidence)."""
        if not self.detections:
            return None
        
        return max(
            self.detections,
            key=lambda d: d.sharpness * 0.6 + d.confidence * 0.4
        )
    
    def aggregate_attributes(self) -> Dict[str, Any]:
        """Aggregate attributes across all detections."""
        if not self.detections:
            return {}
        
        # Weighted voting for specific type
        type_scores = defaultdict(float)
        color_scores = defaultdict(float)
        color_hexes = {}
        patterns = defaultdict(int)
        materials = defaultdict(int)
        
        for det in self.detections:
            weight = det.confidence
            
            if det.specific_type:
                type_scores[det.specific_type] += weight
            
            if det.primary_color:
                color_scores[det.primary_color] += weight
                color_hexes[det.primary_color] = det.color_hex
            
            attrs = det.attributes or {}
            if attrs.get("pattern"):
                p = attrs["pattern"]
                if isinstance(p, dict):
                    patterns[p.get("type", "solid")] += 1
                else:
                    patterns[str(p)] += 1
            
            if attrs.get("material"):
                m = attrs["material"]
                if isinstance(m, dict):
                    materials[m.get("material", "unknown")] += 1
                else:
                    materials[str(m)] += 1
        
        # Get best values
        best_type = max(type_scores.items(), key=lambda x: x[1])[0] if type_scores else self.category
        best_color = max(color_scores.items(), key=lambda x: x[1])[0] if color_scores else "unknown"
        best_hex = color_hexes.get(best_color, "#000000")
        best_pattern = max(patterns.items(), key=lambda x: x[1])[0] if patterns else ""
        best_material = max(materials.items(), key=lambda x: x[1])[0] if materials else ""
        
        # Confidence boosted by consistency
        avg_conf = sum(d.confidence for d in self.detections) / len(self.detections)
        frame_boost = min(0.15, len(self.detections) * 0.02)
        final_conf = min(0.98, avg_conf + frame_boost)
        
        return {
            "specificType": best_type,
            "primaryColor": best_color,
            "colorHex": best_hex,
            "pattern": best_pattern,
            "material": best_material,
            "confidence": round(final_conf, 3),
            "framesDetected": len(self.detections),
            "typeCandidates": dict(type_scores),
            "colorCandidates": dict(color_scores)
        }
    
    def to_dict(self) -> Dict:
        """Convert to API response format."""
        best_det = self.get_best_detection()
        aggregated = self.aggregate_attributes()
        
        return {
            "trackId": self.track_id,
            "category": self.category,
            "specificType": aggregated.get("specificType", self.category),
            "primaryColor": aggregated.get("primaryColor", "unknown"),
            "colorHex": aggregated.get("colorHex", "#000000"),
            "confidence": aggregated.get("confidence", 0.5),
            "pattern": aggregated.get("pattern", ""),
            "material": aggregated.get("material", ""),
            "cutoutImage": best_det.cutout_base64 if best_det else "",
            "bbox": list(best_det.bbox) if best_det else [0, 0, 0, 0],
            "multiFrameData": {
                "framesDetected": len(self.detections),
                "typeCandidates": aggregated.get("typeCandidates", {}),
                "colorCandidates": aggregated.get("colorCandidates", {}),
                "bestFrameIdx": best_det.frame_idx if best_det else 0,
                "bestSharpness": round(best_det.sharpness, 3) if best_det else 0
            }
        }


# ============================================
# 🚀 BYTETRACK IMPLEMENTATION
# ============================================

class KalmanFilter:
    """
    Simple Kalman filter for bbox tracking.
    State: [cx, cy, aspect_ratio, height, vx, vy, va, vh]
    """
    
    def __init__(self):
        # Motion model: constant velocity
        self.motion_mat = np.eye(8)
        for i in range(4):
            self.motion_mat[i, i + 4] = 1
        
        # Observation model: only position
        self.update_mat = np.eye(4, 8)
        
        # Noise covariances (tuned for fashion video)
        self._std_weight_position = 1 / 20
        self._std_weight_velocity = 1 / 160
    
    def initiate(self, measurement: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create track from unassociated measurement."""
        mean_pos = measurement  # cx, cy, a, h
        mean_vel = np.zeros(4)
        mean = np.concatenate([mean_pos, mean_vel])
        
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        
        return mean, covariance
    
    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter prediction step."""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        
        motion_cov = np.diag(np.square(np.concatenate([std_pos, std_vel])))
        
        mean = self.motion_mat @ mean
        covariance = self.motion_mat @ covariance @ self.motion_mat.T + motion_cov
        
        return mean, covariance
    
    def update(
        self, 
        mean: np.ndarray, 
        covariance: np.ndarray,
        measurement: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run Kalman filter correction step."""
        projected_mean = self.update_mat @ mean
        projected_cov = self.update_mat @ covariance @ self.update_mat.T
        
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = projected_cov + np.diag(np.square(std))
        
        kalman_gain = np.linalg.solve(
            innovation_cov.T,
            (covariance @ self.update_mat.T).T
        ).T
        
        innovation = measurement - projected_mean
        
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ innovation_cov @ kalman_gain.T
        
        return new_mean, new_covariance


class ByteTracker:
    """
    ByteTrack: Simple, High-Performance Multi-Object Tracking
    
    Key insight: Use IoU + Kalman prediction for association.
    Handles occluded objects by keeping "lost" tracks.
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 2
    ):
        """
        Args:
            iou_threshold: Minimum IoU for matching
            max_age: Frames to keep lost track before removing
            min_hits: Minimum detections before confirming track
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        
        self.kalman_filter = KalmanFilter()
        self.tracklets: Dict[int, Tracklet] = {}
        self.next_id = 1
        self.frame_count = 0
    
    def update(self, detections: List[Detection]) -> List[Tracklet]:
        """
        Update tracks with new frame detections.
        
        Returns list of active tracklets.
        """
        self.frame_count += 1
        
        # Predict new locations
        for tracklet in self.tracklets.values():
            if tracklet.mean is not None:
                tracklet.mean, tracklet.covariance = self.kalman_filter.predict(
                    tracklet.mean, tracklet.covariance
                )
            tracklet.age += 1
            tracklet.time_since_update += 1
        
        # Split detections by category for category-specific matching
        det_by_category = defaultdict(list)
        for det in detections:
            det_by_category[det.category].append(det)
        
        # Process each category
        matched_det_indices = set()
        
        for category, cat_detections in det_by_category.items():
            # Get active tracks of this category
            cat_tracks = [
                (tid, t) for tid, t in self.tracklets.items()
                if t.category == category and t.state != "removed"
            ]
            
            if not cat_tracks:
                # Create new tracks for all detections
                for det in cat_detections:
                    self._create_tracklet(det)
                continue
            
            # Calculate IoU matrix
            det_boxes = np.array([self._bbox_to_xyxy(d.bbox) for d in cat_detections])
            track_boxes = np.array([
                self._mean_to_xyxy(t.mean) if t.mean is not None 
                else self._bbox_to_xyxy(t.detections[-1].bbox)
                for _, t in cat_tracks
            ])
            
            iou_matrix = self._iou_batch(det_boxes, track_boxes)
            
            # Hungarian matching
            if iou_matrix.size > 0:
                det_indices, track_indices = linear_sum_assignment(-iou_matrix)
                
                for det_idx, track_idx in zip(det_indices, track_indices):
                    if iou_matrix[det_idx, track_idx] >= self.iou_threshold:
                        # Match found
                        tid, tracklet = cat_tracks[track_idx]
                        det = cat_detections[det_idx]
                        
                        # Update Kalman filter
                        measurement = self._bbox_to_z(det.bbox)
                        tracklet.mean, tracklet.covariance = self.kalman_filter.update(
                            tracklet.mean, tracklet.covariance, measurement
                        )
                        
                        tracklet.add_detection(det)
                        tracklet.state = "active"
                        matched_det_indices.add(id(det))
            
            # Create new tracks for unmatched detections
            for det in cat_detections:
                if id(det) not in matched_det_indices:
                    self._create_tracklet(det)
        
        # Mark lost tracks
        for tracklet in self.tracklets.values():
            if tracklet.time_since_update > 0:
                tracklet.state = "lost"
            if tracklet.time_since_update > self.max_age:
                tracklet.state = "removed"
        
        # Return confirmed tracklets
        return [
            t for t in self.tracklets.values()
            if t.state != "removed" and t.hits >= self.min_hits
        ]
    
    def _create_tracklet(self, det: Detection) -> Tracklet:
        """Create new tracklet from detection."""
        tracklet = Tracklet(
            track_id=self.next_id,
            category=det.category
        )
        
        # Initialize Kalman filter
        measurement = self._bbox_to_z(det.bbox)
        tracklet.mean, tracklet.covariance = self.kalman_filter.initiate(measurement)
        
        tracklet.add_detection(det)
        self.tracklets[self.next_id] = tracklet
        self.next_id += 1
        
        return tracklet
    
    def _bbox_to_z(self, bbox: Tuple) -> np.ndarray:
        """Convert bbox (x1, y1, x2, y2) to measurement [cx, cy, a, h]."""
        x1, y1, x2, y2 = bbox[:4]
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        a = w / h if h > 0 else 1
        return np.array([cx, cy, a, h])
    
    def _mean_to_xyxy(self, mean: np.ndarray) -> np.ndarray:
        """Convert state mean to bbox [x1, y1, x2, y2]."""
        cx, cy, a, h = mean[:4]
        w = a * h
        return np.array([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
    
    def _bbox_to_xyxy(self, bbox: Tuple) -> np.ndarray:
        """Ensure bbox is x1, y1, x2, y2 format."""
        if len(bbox) == 4:
            x1, y1, x2_or_w, y2_or_h = bbox
            # Check if it's x, y, w, h format
            if x2_or_w < x1 or y2_or_h < y1:
                # Convert from center or other format
                return np.array([x1, y1, x1 + x2_or_w, y1 + y2_or_h])
            return np.array([x1, y1, x2_or_w, y2_or_h])
        return np.array([0, 0, 100, 100])
    
    def _iou_batch(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Calculate IoU between two sets of boxes."""
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)))
        
        iou_matrix = np.zeros((len(boxes1), len(boxes2)))
        
        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                iou_matrix[i, j] = self._iou(box1, box2)
        
        return iou_matrix
    
    def _iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def get_final_tracklets(self) -> List[Tracklet]:
        """Get all confirmed tracklets."""
        return [
            t for t in self.tracklets.values()
            if t.hits >= self.min_hits
        ]


# ============================================
# 🎬 ICA VIDEO ANALYZER
# ============================================

class ICAVideoAnalyzer:
    """
    Identity-Consistent Aggregation (ICA) Video Analyzer.
    
    Uses ByteTrack to:
    1. Track items across frames with unique IDs
    2. Aggregate attributes per tracklet
    3. Select best frame per item for attribute extraction
    4. Eliminate duplicate detections
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 1,
        min_sharpness: float = 0.3
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.min_sharpness = min_sharpness
    
    def analyze_video(
        self,
        frame_detections: List[List[Dict]],
        frame_sharpness: List[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Analyze video with identity-consistent tracking.
        
        Args:
            frame_detections: List of detection lists per frame
            frame_sharpness: Optional per-item sharpness scores per frame
            
        Returns:
            ICA analysis result with deduplicated items
        """
        if not frame_detections:
            return {"success": False, "items": [], "error": "No frames"}
        
        logger.info(f"🎯 ICA Analysis: {len(frame_detections)} frames")
        
        # Create tracker
        tracker = ByteTracker(
            iou_threshold=self.iou_threshold,
            max_age=self.max_age,
            min_hits=self.min_hits
        )
        
        # Process each frame
        for frame_idx, frame_dets in enumerate(frame_detections):
            detections = []
            
            for det_data in frame_dets:
                # Convert bbox format
                bbox = det_data.get("bbox", [0, 0, 100, 100])
                if len(bbox) == 4:
                    # Normalize to x1, y1, x2, y2
                    x, y, w_or_x2, h_or_y2 = bbox
                    if w_or_x2 < x or h_or_y2 < y:
                        # It's already x1, y1, x2, y2
                        x1, y1, x2, y2 = bbox
                    else:
                        # Check if small values suggest x1y1x2y2
                        if w_or_x2 > 10 and h_or_y2 > 10:
                            x1, y1, x2, y2 = x, y, w_or_x2, h_or_y2
                        else:
                            x1, y1, x2, y2 = x, y, x + w_or_x2, y + h_or_y2
                else:
                    x1, y1, x2, y2 = 0, 0, 100, 100
                
                # Get sharpness
                sharpness = 0.5
                if frame_sharpness and frame_idx < len(frame_sharpness):
                    sharpness = frame_sharpness[frame_idx].get(
                        det_data.get("category", ""), 0.5
                    )
                
                detection = Detection(
                    frame_idx=frame_idx,
                    bbox=(x1, y1, x2, y2),
                    category=det_data.get("category", "unknown"),
                    specific_type=det_data.get("specificType", ""),
                    primary_color=det_data.get("primaryColor", ""),
                    color_hex=det_data.get("colorHex", "#000000"),
                    confidence=det_data.get("confidence", 0.5),
                    cutout_base64=det_data.get("cutoutImage", ""),
                    sharpness=sharpness,
                    attributes=det_data.get("attributes", {})
                )
                
                detections.append(detection)
            
            # Update tracker
            tracker.update(detections)
        
        # Get final tracklets
        tracklets = tracker.get_final_tracklets()
        
        logger.info(f"📊 Found {len(tracklets)} unique items (deduplicated)")
        
        # Convert to response format
        items = []
        for tracklet in tracklets:
            item_dict = tracklet.to_dict()
            items.append(item_dict)
            
            logger.info(
                f"  ✅ Track_{tracklet.track_id}: {item_dict['specificType']} "
                f"({item_dict['primaryColor']}) - {len(tracklet.detections)} frames"
            )
        
        # Sort by confidence
        items.sort(key=lambda x: x.get("confidence", 0), reverse=True)
        
        return {
            "success": True,
            "items": items,
            "itemCount": len(items),
            "framesAnalyzed": len(frame_detections),
            "totalDetections": sum(len(f) for f in frame_detections),
            "strategy": "identity_consistent_aggregation"
        }


# ============================================
# 🔧 UTILITY FUNCTIONS
# ============================================

def analyze_video_ica(
    frame_results: List[Dict],
    iou_threshold: float = 0.3
) -> Dict[str, Any]:
    """
    Utility function for ICA video analysis.
    
    Args:
        frame_results: List of segment-all results per frame
        iou_threshold: IoU threshold for matching
        
    Returns:
        Deduplicated analysis result
    """
    # Extract detection lists
    frame_detections = []
    
    for result in frame_results:
        if result.get("success"):
            frame_detections.append(result.get("items", []))
        else:
            frame_detections.append([])
    
    analyzer = ICAVideoAnalyzer(iou_threshold=iou_threshold)
    return analyzer.analyze_video(frame_detections)


# Singleton
_ica_analyzer = None

def get_ica_analyzer() -> ICAVideoAnalyzer:
    """Get singleton ICA analyzer."""
    global _ica_analyzer
    if _ica_analyzer is None:
        _ica_analyzer = ICAVideoAnalyzer()
    return _ica_analyzer
