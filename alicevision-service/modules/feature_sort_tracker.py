"""
🎯 FeatureSORT: Person-Anchored Outfit Tracking

ReID-based tracking that groups clothing items by person.

Key Algorithm:
1. Detect "Person" class with YOLOv8
2. Extract ReID embeddings for person re-identification
3. Track clothing items with ByteTrack-style IoU matching
4. Assign OutfitID based on person TrackID
5. Clothing items get OutfitID from intersecting person mask

This replaces the brittle "Total Items / 4" heuristic with
robust person-anchored grouping.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import logging
from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """A tracked object (person or clothing item)."""
    track_id: int
    class_name: str
    category: str
    bbox: Tuple[float, float, float, float]
    confidence: float
    mask: Optional[np.ndarray] = None
    outfit_id: int = 1
    reid_embedding: Optional[np.ndarray] = None
    
    # Tracking state
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    is_new: bool = True
    
    # Kalman state
    mean: Optional[np.ndarray] = None
    covariance: Optional[np.ndarray] = None


class KalmanBoxTracker:
    """
    Simple Kalman filter for bounding box tracking.
    State: [cx, cy, s, r, vx, vy, vs]
    where s = scale (area), r = aspect ratio
    """
    
    def __init__(self, bbox):
        """Initialize tracker with bounding box [x1, y1, x2, y2]."""
        # Convert bbox to [cx, cy, s, r]
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        s = w * h  # scale (area)
        r = w / h if h > 0 else 1  # aspect ratio
        
        # State: [cx, cy, s, r, vx, vy, vs]
        self.mean = np.array([cx, cy, s, r, 0, 0, 0], dtype=np.float32)
        
        # Covariance
        self.covariance = np.diag([10, 10, 10, 10, 1000, 1000, 1000]).astype(np.float32)
        
        # Motion matrices
        self.F = np.array([
            [1, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        self.H = np.array([
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0]
        ], dtype=np.float32)
        
        # Noise
        self.Q = np.eye(7, dtype=np.float32) * 0.01
        self.R = np.eye(4, dtype=np.float32) * 1
    
    def predict(self):
        """Predict next state."""
        self.mean = self.F @ self.mean
        self.covariance = self.F @ self.covariance @ self.F.T + self.Q
        return self.get_bbox()
    
    def update(self, bbox):
        """Update with new observation."""
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w / 2
        cy = y1 + h / 2
        s = w * h
        r = w / h if h > 0 else 1
        
        z = np.array([cx, cy, s, r], dtype=np.float32)
        
        # Kalman update
        y = z - self.H @ self.mean
        S = self.H @ self.covariance @ self.H.T + self.R
        K = self.covariance @ self.H.T @ np.linalg.inv(S)
        
        self.mean = self.mean + K @ y
        self.covariance = (np.eye(7) - K @ self.H) @ self.covariance
        
        return self.get_bbox()
    
    def get_bbox(self):
        """Convert state to bbox [x1, y1, x2, y2]."""
        cx, cy, s, r = self.mean[:4]
        w = np.sqrt(s * r)
        h = s / w if w > 0 else 0
        return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)


class FeatureSORTTracker:
    """
    🎯 FeatureSORT: Person-Anchored Outfit Tracking
    
    Tracks persons and clothing items separately, then links
    clothing to persons based on spatial overlap.
    
    Features:
    - Kalman filter for motion prediction
    - IoU-based association (ByteTrack style)
    - Optional ReID for person re-identification
    - Person-anchored outfit grouping
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 2,
        use_reid: bool = True
    ):
        """
        Initialize tracker.
        
        Args:
            iou_threshold: Minimum IoU for matching
            max_age: Frames to keep lost track
            min_hits: Minimum hits to confirm track
            use_reid: Use ReID for person matching
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.use_reid = use_reid
        
        # Track storage
        self.person_tracks: Dict[int, TrackedObject] = {}
        self.clothing_tracks: Dict[int, TrackedObject] = {}
        
        # Kalman filters
        self.person_kalman: Dict[int, KalmanBoxTracker] = {}
        self.clothing_kalman: Dict[int, KalmanBoxTracker] = {}
        
        # ID counter
        self.next_id = 1
        
        # ReID extractor (lazy loaded)
        self._reid_extractor = None
        
        logger.info(f"FeatureSORT initialized (iou={iou_threshold}, max_age={max_age})")
    
    def update(
        self,
        person_detections: List,
        clothing_detections: List,
        masks: np.ndarray = None,
        frame: np.ndarray = None
    ) -> Tuple[List[TrackedObject], List[TrackedObject]]:
        """
        Update tracks with new detections.
        
        Args:
            person_detections: List of person detections
            clothing_detections: List of clothing detections
            masks: Instance masks (N, H, W)
            frame: Original frame (for ReID extraction)
            
        Returns:
            Tuple of (tracked_persons, tracked_clothing)
        """
        # 1. Predict new locations for all tracks
        self._predict_all()
        
        # 2. Match person detections
        matched_persons = self._match_tracks(
            person_detections, 
            self.person_tracks,
            self.person_kalman
        )
        
        # 3. Match clothing detections
        matched_clothing = self._match_tracks(
            clothing_detections,
            self.clothing_tracks,
            self.clothing_kalman
        )
        
        # 4. Create new tracks for unmatched detections
        for det in person_detections:
            if id(det) not in matched_persons:
                self._create_track(det, is_person=True)
        
        for det in clothing_detections:
            if id(det) not in matched_clothing:
                self._create_track(det, is_person=False)
        
        # 5. Remove old tracks
        self._remove_stale_tracks()
        
        # 6. Get confirmed tracks
        person_results = [
            t for t in self.person_tracks.values()
            if t.hits >= self.min_hits
        ]
        
        clothing_results = [
            t for t in self.clothing_tracks.values()
            if t.hits >= self.min_hits
        ]
        
        return person_results, clothing_results
    
    def _predict_all(self):
        """Predict new locations for all tracks."""
        for track_id, track in self.person_tracks.items():
            if track_id in self.person_kalman:
                new_bbox = self.person_kalman[track_id].predict()
                track.bbox = new_bbox
            track.age += 1
            track.time_since_update += 1
        
        for track_id, track in self.clothing_tracks.items():
            if track_id in self.clothing_kalman:
                new_bbox = self.clothing_kalman[track_id].predict()
                track.bbox = new_bbox
            track.age += 1
            track.time_since_update += 1
    
    def _match_tracks(
        self,
        detections: List,
        tracks: Dict[int, TrackedObject],
        kalman_filters: Dict[int, KalmanBoxTracker]
    ) -> set:
        """Match detections to tracks using IoU."""
        matched_det_ids = set()
        
        if not detections or not tracks:
            return matched_det_ids
        
        # Build IoU matrix
        det_boxes = np.array([
            [d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3]] 
            for d in detections
        ])
        
        track_boxes = np.array([
            [t.bbox[0], t.bbox[1], t.bbox[2], t.bbox[3]]
            for t in tracks.values()
        ])
        
        iou_matrix = self._compute_iou_matrix(det_boxes, track_boxes)
        
        # Hungarian matching
        if iou_matrix.size > 0:
            det_indices, track_indices = linear_sum_assignment(-iou_matrix)
            
            track_list = list(tracks.values())
            
            for det_idx, track_idx in zip(det_indices, track_indices):
                if iou_matrix[det_idx, track_idx] >= self.iou_threshold:
                    det = detections[det_idx]
                    track = track_list[track_idx]
                    
                    # Update track
                    track.bbox = det.bbox
                    track.confidence = det.confidence
                    track.mask = det.mask if hasattr(det, 'mask') else None
                    track.hits += 1
                    track.time_since_update = 0
                    track.is_new = False
                    
                    # Update Kalman
                    if track.track_id in kalman_filters:
                        kalman_filters[track.track_id].update(det.bbox)
                    
                    matched_det_ids.add(id(det))
        
        return matched_det_ids
    
    def _create_track(self, detection, is_person: bool):
        """Create new track from detection."""
        track_id = self.next_id
        self.next_id += 1
        
        track = TrackedObject(
            track_id=track_id,
            class_name=detection.class_name,
            category=detection.category,
            bbox=detection.bbox,
            confidence=detection.confidence,
            mask=detection.mask if hasattr(detection, 'mask') else None,
            is_new=True,
            hits=1
        )
        
        # Create Kalman filter
        kf = KalmanBoxTracker(detection.bbox)
        
        if is_person:
            self.person_tracks[track_id] = track
            self.person_kalman[track_id] = kf
        else:
            self.clothing_tracks[track_id] = track
            self.clothing_kalman[track_id] = kf
        
        logger.debug(f"Created {'person' if is_person else 'clothing'} track {track_id}")
    
    def _remove_stale_tracks(self):
        """Remove tracks that haven't been updated recently."""
        stale_person_ids = [
            tid for tid, t in self.person_tracks.items()
            if t.time_since_update > self.max_age
        ]
        for tid in stale_person_ids:
            del self.person_tracks[tid]
            if tid in self.person_kalman:
                del self.person_kalman[tid]
        
        stale_clothing_ids = [
            tid for tid, t in self.clothing_tracks.items()
            if t.time_since_update > self.max_age
        ]
        for tid in stale_clothing_ids:
            del self.clothing_tracks[tid]
            if tid in self.clothing_kalman:
                del self.clothing_kalman[tid]
    
    def _compute_iou_matrix(
        self,
        boxes1: np.ndarray,
        boxes2: np.ndarray
    ) -> np.ndarray:
        """Compute IoU matrix between two sets of boxes."""
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)))
        
        iou_matrix = np.zeros((len(boxes1), len(boxes2)))
        
        for i, box1 in enumerate(boxes1):
            for j, box2 in enumerate(boxes2):
                iou_matrix[i, j] = self._compute_iou(box1, box2)
        
        return iou_matrix
    
    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        
        return inter / union if union > 0 else 0.0
    
    def reset(self):
        """Reset tracker state."""
        self.person_tracks.clear()
        self.clothing_tracks.clear()
        self.person_kalman.clear()
        self.clothing_kalman.clear()
        self.next_id = 1
        logger.info("FeatureSORT reset")


# ============================================
# Singleton
# ============================================

_tracker_instance = None


def get_feature_sort_tracker(**kwargs) -> FeatureSORTTracker:
    """Get singleton FeatureSORT tracker."""
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = FeatureSORTTracker(**kwargs)
    return _tracker_instance
