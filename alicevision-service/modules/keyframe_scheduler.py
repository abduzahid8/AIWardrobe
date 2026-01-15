"""
⏱️ Keyframe Scheduler

Determines when to run expensive slow-path analysis:
1. Time-based: Every N frames (default 30)
2. Event-based: New TrackID appears
3. Quality-based: Frame sharpness above threshold
4. Change-based: Significant appearance change

Smart scheduling ensures slow-path runs on highest quality frames
while maintaining smooth real-time performance.
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class KeyframeDecision:
    """Result of keyframe scheduling decision."""
    should_process: bool
    reason: str
    priority: int  # 1 = low, 2 = medium, 3 = high


class KeyframeScheduler:
    """
    ⏱️ Smart Keyframe Scheduling for Slow-Fast Architecture
    
    Decides when to trigger slow-path analysis based on:
    - Frame index (every N frames)
    - New track appearance
    - Frame quality (sharpness, visibility)
    - Appearance change detection
    
    Goals:
    - Minimize slow-path calls for efficiency
    - Maximize quality of analyzed frames
    - Ensure new items get processed quickly
    """
    
    def __init__(
        self,
        interval: int = 30,
        quality_threshold: float = 0.7,
        change_threshold: float = 0.3,
        min_interval: int = 10
    ):
        """
        Initialize scheduler.
        
        Args:
            interval: Base interval between keyframes
            quality_threshold: Minimum frame quality to consider
            change_threshold: Minimum change to force keyframe
            min_interval: Minimum frames between slow-path runs per track
        """
        self.interval = interval
        self.quality_threshold = quality_threshold
        self.change_threshold = change_threshold
        self.min_interval = min_interval
        
        # Per-track scheduling state
        self.last_keyframes: Dict[int, int] = {}  # track_id -> frame_idx
        self.last_appearances: Dict[int, np.ndarray] = {}  # track_id -> feature
        
        logger.info(f"KeyframeScheduler initialized (interval={interval})")
    
    def should_process_slow(
        self,
        track_id: int,
        frame_idx: int,
        is_new_track: bool = False,
        frame_quality: float = 1.0,
        crop: np.ndarray = None
    ) -> KeyframeDecision:
        """
        Determine if track needs slow-path processing.
        
        Args:
            track_id: Track identifier
            frame_idx: Current frame index
            is_new_track: Whether this is a newly created track
            frame_quality: Quality score (0-1)
            crop: Optional cropped image for change detection
            
        Returns:
            KeyframeDecision with processing recommendation
        """
        # 1. NEW TRACKS - Always process (high priority)
        if is_new_track:
            self.last_keyframes[track_id] = frame_idx
            return KeyframeDecision(
                should_process=True,
                reason="new_track",
                priority=3
            )
        
        # 2. Check minimum interval
        last_kf = self.last_keyframes.get(track_id, -self.interval)
        frames_since_last = frame_idx - last_kf
        
        if frames_since_last < self.min_interval:
            return KeyframeDecision(
                should_process=False,
                reason="too_soon",
                priority=0
            )
        
        # 3. SCHEDULED INTERVAL - Process if due (medium priority)
        if frames_since_last >= self.interval:
            # But prefer high quality frames
            if frame_quality >= self.quality_threshold:
                self.last_keyframes[track_id] = frame_idx
                return KeyframeDecision(
                    should_process=True,
                    reason="scheduled",
                    priority=2
                )
            # Allow processing even with lower quality if very overdue
            elif frames_since_last >= self.interval * 1.5:
                self.last_keyframes[track_id] = frame_idx
                return KeyframeDecision(
                    should_process=True,
                    reason="overdue",
                    priority=2
                )
        
        # 4. QUALITY-BASED - Process excellent frames between intervals
        if frame_quality >= 0.9 and frames_since_last >= self.min_interval * 2:
            self.last_keyframes[track_id] = frame_idx
            return KeyframeDecision(
                should_process=True,
                reason="high_quality",
                priority=1
            )
        
        # 5. CHANGE-BASED - Process if appearance changed significantly
        if crop is not None and track_id in self.last_appearances:
            change = self._compute_appearance_change(track_id, crop)
            if change >= self.change_threshold:
                self.last_keyframes[track_id] = frame_idx
                return KeyframeDecision(
                    should_process=True,
                    reason="appearance_change",
                    priority=2
                )
        
        # Store current appearance for future comparison
        if crop is not None:
            self._update_appearance(track_id, crop)
        
        return KeyframeDecision(
            should_process=False,
            reason="not_needed",
            priority=0
        )
    
    def compute_frame_quality(self, frame: np.ndarray) -> float:
        """
        Compute overall frame quality score.
        
        Considers:
        - Sharpness (Laplacian variance)
        - Brightness (exposure)
        - Contrast
        
        Returns:
            Quality score between 0 and 1
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Sharpness via Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            # Normalize (typical range 0-5000, max out at 2000)
            sharpness_score = min(1.0, sharpness / 2000)
            
            # Brightness
            mean_brightness = np.mean(gray) / 255.0
            # Penalize too dark or too bright
            brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2
            brightness_score = max(0, brightness_score)
            
            # Contrast
            contrast = np.std(gray) / 128.0
            contrast_score = min(1.0, contrast)
            
            # Weighted combination
            quality = (
                0.5 * sharpness_score +
                0.25 * brightness_score +
                0.25 * contrast_score
            )
            
            return quality
            
        except Exception as e:
            logger.warning(f"Frame quality computation failed: {e}")
            return 0.5
    
    def compute_crop_quality(self, crop: np.ndarray) -> float:
        """Compute quality score for a cropped region."""
        if crop.size == 0:
            return 0.0
        
        return self.compute_frame_quality(crop)
    
    def _compute_appearance_change(self, track_id: int, crop: np.ndarray) -> float:
        """
        Compute how much appearance has changed since last keyframe.
        
        Uses color histogram comparison.
        """
        try:
            current = self._extract_appearance_feature(crop)
            if current is None:
                return 0.0
            
            previous = self.last_appearances.get(track_id)
            if previous is None:
                return 0.0
            
            # Histogram correlation (1 = identical, 0 = completely different)
            correlation = cv2.compareHist(
                current.astype(np.float32),
                previous.astype(np.float32),
                cv2.HISTCMP_CORREL
            )
            
            # Convert to change score (0 = no change, 1 = complete change)
            change = 1.0 - max(0, correlation)
            
            return change
            
        except Exception as e:
            logger.debug(f"Appearance change computation failed: {e}")
            return 0.0
    
    def _extract_appearance_feature(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Extract appearance feature (color histogram) from crop."""
        try:
            if crop.size == 0:
                return None
            
            # Resize for consistency
            crop_resized = cv2.resize(crop, (64, 64))
            
            # HSV histogram
            hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist(
                [hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]
            )
            hist = cv2.normalize(hist, hist).flatten()
            
            return hist
            
        except Exception:
            return None
    
    def _update_appearance(self, track_id: int, crop: np.ndarray):
        """Update stored appearance for track."""
        feature = self._extract_appearance_feature(crop)
        if feature is not None:
            self.last_appearances[track_id] = feature
    
    def get_track_status(self, track_id: int) -> Dict[str, Any]:
        """Get scheduling status for a track."""
        return {
            "last_keyframe": self.last_keyframes.get(track_id),
            "has_appearance": track_id in self.last_appearances
        }
    
    def reset(self):
        """Reset scheduler state."""
        self.last_keyframes.clear()
        self.last_appearances.clear()
        logger.info("KeyframeScheduler reset")


# ============================================
# Singleton
# ============================================

_scheduler_instance = None


def get_keyframe_scheduler(**kwargs) -> KeyframeScheduler:
    """Get singleton KeyframeScheduler."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = KeyframeScheduler(**kwargs)
    return _scheduler_instance
