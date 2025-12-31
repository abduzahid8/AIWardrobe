"""
MediaPipe Pose Detection for Frame Quality Scoring
Selects best frames with frontal body pose for wardrobe photos
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import logging
import base64

logger = logging.getLogger(__name__)


@dataclass
class PoseResult:
    """Result of pose analysis on a single frame"""
    has_person: bool
    is_frontal: bool
    pose_score: float
    body_visibility: float
    landmarks: Optional[List[Dict]] = None
    analysis: Optional[Dict] = None


class PoseDetector:
    """
    MediaPipe-based pose detection for frame quality scoring.
    
    Evaluates:
    - Person presence
    - Frontal vs side view
    - Full body visibility
    - Pose stability
    """
    
    def __init__(self):
        self._pose = None
        self._loaded = False
    
    def _load_mediapipe(self):
        """Lazy load MediaPipe"""
        if self._loaded:
            return True
        
        try:
            import mediapipe as mp
            
            self._mp_pose = mp.solutions.pose
            self._pose = self._mp_pose.Pose(
                static_image_mode=True,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.5
            )
            self._loaded = True
            logger.info("✅ MediaPipe Pose loaded")
            return True
            
        except ImportError as e:
            logger.warning(f"MediaPipe not available: {e}")
            return False
    
    def analyze_pose(self, image: np.ndarray) -> PoseResult:
        """
        Analyze pose in an image.
        
        Args:
            image: BGR image
            
        Returns:
            PoseResult with pose analysis
        """
        if not self._load_mediapipe():
            # Return default if MediaPipe not available
            return PoseResult(
                has_person=True,
                is_frontal=True,
                pose_score=0.7,
                body_visibility=0.8
            )
        
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self._pose.process(rgb_image)
        
        if not results.pose_landmarks:
            return PoseResult(
                has_person=False,
                is_frontal=False,
                pose_score=0.0,
                body_visibility=0.0
            )
        
        landmarks = results.pose_landmarks.landmark
        
        # Calculate metrics
        visibility_score = self._calculate_visibility(landmarks)
        frontal_score = self._calculate_frontal_score(landmarks)
        clothing_area_score = self._calculate_clothing_area_score(landmarks, image.shape)
        
        # Combined score
        pose_score = (
            visibility_score * 0.3 +
            frontal_score * 0.5 +
            clothing_area_score * 0.2
        )
        
        # Extract key landmarks for reference
        key_landmarks = self._extract_key_landmarks(landmarks, image.shape)
        
        return PoseResult(
            has_person=True,
            is_frontal=frontal_score > 0.6,
            pose_score=pose_score,
            body_visibility=visibility_score,
            landmarks=key_landmarks,
            analysis={
                "visibilityScore": round(visibility_score, 3),
                "frontalScore": round(frontal_score, 3),
                "clothingAreaScore": round(clothing_area_score, 3)
            }
        )
    
    def _calculate_visibility(self, landmarks) -> float:
        """Calculate how many body landmarks are visible"""
        # Key landmarks for clothing visibility
        key_indices = [
            11, 12,  # Shoulders
            13, 14,  # Elbows
            15, 16,  # Wrists
            23, 24,  # Hips
            25, 26,  # Knees
        ]
        
        visible_count = 0
        for idx in key_indices:
            if landmarks[idx].visibility > 0.5:
                visible_count += 1
        
        return visible_count / len(key_indices)
    
    def _calculate_frontal_score(self, landmarks) -> float:
        """Calculate how frontal the pose is"""
        # Check shoulder alignment
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        # Shoulders should have similar z-depth for frontal pose
        shoulder_depth_diff = abs(left_shoulder.z - right_shoulder.z)
        
        # Shoulders should be at similar y-level
        shoulder_y_diff = abs(left_shoulder.y - right_shoulder.y)
        
        # Check hip alignment
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        hip_depth_diff = abs(left_hip.z - right_hip.z)
        
        # Score calculation
        depth_score = max(0, 1 - (shoulder_depth_diff + hip_depth_diff) * 5)
        alignment_score = max(0, 1 - shoulder_y_diff * 10)
        
        # Check if person is facing camera (nose between shoulders)
        nose = landmarks[0]
        if left_shoulder.x < nose.x < right_shoulder.x or right_shoulder.x < nose.x < left_shoulder.x:
            facing_score = 1.0
        else:
            facing_score = 0.5
        
        return (depth_score * 0.4 + alignment_score * 0.3 + facing_score * 0.3)
    
    def _calculate_clothing_area_score(self, landmarks, image_shape) -> float:
        """Calculate how much of the frame contains the person's torso"""
        h, w = image_shape[:2]
        
        # Get torso bounds
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        
        # Calculate torso area
        torso_width = abs(left_shoulder.x - right_shoulder.x)
        torso_height = abs((left_shoulder.y + right_shoulder.y) / 2 - (left_hip.y + right_hip.y) / 2)
        
        torso_area = torso_width * torso_height
        
        # Ideal: torso takes up 20-40% of frame
        if 0.15 < torso_area < 0.5:
            return 1.0
        elif torso_area < 0.1:
            return torso_area * 10  # Too small
        else:
            return max(0.5, 1.0 - (torso_area - 0.5) * 2)  # Too large
    
    def _extract_key_landmarks(self, landmarks, image_shape) -> List[Dict]:
        """Extract key landmarks with pixel coordinates"""
        h, w = image_shape[:2]
        
        key_points = [
            (0, "nose"),
            (11, "left_shoulder"),
            (12, "right_shoulder"),
            (23, "left_hip"),
            (24, "right_hip"),
        ]
        
        result = []
        for idx, name in key_points:
            lm = landmarks[idx]
            result.append({
                "name": name,
                "x": int(lm.x * w),
                "y": int(lm.y * h),
                "visibility": round(lm.visibility, 3)
            })
        
        return result


def score_frames_for_pose(frames_base64: List[str]) -> Dict:
    """
    Score multiple frames for pose quality.
    
    Args:
        frames_base64: List of base64-encoded frames
        
    Returns:
        Dictionary with best frame index and scores
    """
    detector = PoseDetector()
    scores = []
    
    for i, frame_b64 in enumerate(frames_base64):
        # Decode
        if ',' in frame_b64:
            frame_b64 = frame_b64.split(',')[1]
        
        img_bytes = base64.b64decode(frame_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if image is None:
            scores.append(0.0)
            continue
        
        result = detector.analyze_pose(image)
        scores.append(result.pose_score)
    
    if not scores:
        return {"error": "No valid frames"}
    
    best_index = int(np.argmax(scores))
    
    # Get detailed analysis of best frame
    if ',' in frames_base64[best_index]:
        frame_b64 = frames_base64[best_index].split(',')[1]
    else:
        frame_b64 = frames_base64[best_index]
    
    img_bytes = base64.b64decode(frame_b64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    best_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    best_result = detector.analyze_pose(best_image)
    
    return {
        "bestFrameIndex": best_index,
        "bestScore": round(scores[best_index], 4),
        "allScores": [round(s, 4) for s in scores],
        "bestFrameAnalysis": best_result.analysis,
        "isFrontal": best_result.is_frontal,
        "hasPerson": best_result.has_person
    }
