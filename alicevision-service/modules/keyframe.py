"""
AliceVision Keyframe Selection Module
Intelligent frame selection for optimal clothing capture
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class FrameScore:
    """Score components for a video frame"""
    index: int
    sharpness: float
    motion_blur: float
    centering: float
    brightness: float
    total_score: float
    

class KeyframeSelector:
    """
    Intelligent keyframe selection using multiple quality metrics.
    
    Replaces naive middle-frame selection with quality-based selection
    that considers sharpness, blur, composition, and lighting.
    """
    
    def __init__(
        self,
        sharpness_weight: float = 0.4,
        blur_penalty: float = 0.3,
        centering_weight: float = 0.2,
        brightness_weight: float = 0.1
    ):
        self.sharpness_weight = sharpness_weight
        self.blur_penalty = blur_penalty
        self.centering_weight = centering_weight
        self.brightness_weight = brightness_weight
    
    def calculate_sharpness(self, image: np.ndarray) -> float:
        """
        Calculate image sharpness using Laplacian variance.
        Higher values indicate sharper images.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Normalize to 0-1 range (typical range is 0-1000)
        return min(variance / 500.0, 1.0)
    
    def detect_motion_blur(self, image: np.ndarray) -> float:
        """
        Detect motion blur using FFT analysis.
        Returns blur score (higher = more blur, which is bad).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)
        
        # Calculate ratio of high to low frequencies
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # High frequency region (outer ring)
        outer_radius = min(rows, cols) // 4
        inner_radius = outer_radius // 2
        
        high_freq = magnitude[crow-outer_radius:crow+outer_radius, 
                              ccol-outer_radius:ccol+outer_radius].mean()
        low_freq = magnitude[crow-inner_radius:crow+inner_radius,
                             ccol-inner_radius:ccol+inner_radius].mean()
        
        if low_freq == 0:
            return 0.5
        
        # Lower ratio = more blur
        blur_ratio = high_freq / low_freq
        blur_score = 1.0 - min(blur_ratio / 0.1, 1.0)
        
        return blur_score
    
    def calculate_centering(self, image: np.ndarray) -> float:
        """
        Calculate how well-centered the subject is.
        Uses edge detection to find the subject and measure centering.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.5
        
        # Find largest contour (assumed to be the subject)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate center of subject
        subject_center_x = x + w / 2
        subject_center_y = y + h / 2
        
        # Calculate image center
        image_center_x = gray.shape[1] / 2
        image_center_y = gray.shape[0] / 2
        
        # Calculate distance from center (normalized)
        max_distance = np.sqrt(image_center_x**2 + image_center_y**2)
        distance = np.sqrt(
            (subject_center_x - image_center_x)**2 + 
            (subject_center_y - image_center_y)**2
        )
        
        centering_score = 1.0 - (distance / max_distance)
        
        return centering_score
    
    def calculate_brightness(self, image: np.ndarray) -> float:
        """
        Calculate brightness score (penalize too dark or too bright).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        mean_brightness = gray.mean() / 255.0
        
        # Optimal brightness is around 0.4-0.6
        if 0.35 <= mean_brightness <= 0.65:
            return 1.0
        elif mean_brightness < 0.35:
            return mean_brightness / 0.35
        else:
            return (1.0 - mean_brightness) / 0.35
    
    def score_frame(self, image: np.ndarray, index: int) -> FrameScore:
        """
        Calculate overall quality score for a frame.
        """
        sharpness = self.calculate_sharpness(image)
        motion_blur = self.detect_motion_blur(image)
        centering = self.calculate_centering(image)
        brightness = self.calculate_brightness(image)
        
        # Calculate weighted total score
        total = (
            sharpness * self.sharpness_weight +
            (1.0 - motion_blur) * self.blur_penalty +
            centering * self.centering_weight +
            brightness * self.brightness_weight
        )
        
        return FrameScore(
            index=index,
            sharpness=sharpness,
            motion_blur=motion_blur,
            centering=centering,
            brightness=brightness,
            total_score=total
        )
    
    def select_best_frames(
        self, 
        frames: List[np.ndarray], 
        top_n: int = 3
    ) -> List[FrameScore]:
        """
        Select the best frames from a list.
        
        Args:
            frames: List of frames as numpy arrays
            top_n: Number of best frames to return
            
        Returns:
            List of FrameScore objects for the best frames
        """
        if not frames:
            return []
        
        logger.info(f"Scoring {len(frames)} frames...")
        
        scores = []
        for i, frame in enumerate(frames):
            score = self.score_frame(frame, i)
            scores.append(score)
            logger.debug(f"Frame {i}: sharpness={score.sharpness:.3f}, "
                        f"blur={score.motion_blur:.3f}, "
                        f"centering={score.centering:.3f}, "
                        f"total={score.total_score:.3f}")
        
        # Sort by total score (descending)
        scores.sort(key=lambda x: x.total_score, reverse=True)
        
        return scores[:top_n]
    
    def select_from_video(
        self, 
        video_path: str, 
        sample_rate: int = 5,
        top_n: int = 3
    ) -> List[FrameScore]:
        """
        Select best frames directly from a video file.
        
        Args:
            video_path: Path to video file
            sample_rate: Extract every Nth frame
            top_n: Number of best frames to return
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        frames = []
        frame_indices = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                frames.append(frame)
                frame_indices.append(frame_count)
            
            frame_count += 1
        
        cap.release()
        
        logger.info(f"Extracted {len(frames)} frames from video "
                   f"(sample rate: every {sample_rate} frames)")
        
        # Score and select
        best_scores = self.select_best_frames(frames, top_n)
        
        # Update indices to reflect original video frame numbers
        for score in best_scores:
            score.index = frame_indices[score.index]
        
        return best_scores


def select_best_frame_from_base64(
    frames_base64: List[str],
    sharpness_weight: float = 0.4,
    blur_penalty: float = 0.3,
    centering_weight: float = 0.2
) -> Dict:
    """
    Utility function to select best frame from base64-encoded images.
    
    Args:
        frames_base64: List of base64-encoded image strings
        
    Returns:
        Dictionary with best frame info
    """
    import base64
    
    selector = KeyframeSelector(
        sharpness_weight=sharpness_weight,
        blur_penalty=blur_penalty,
        centering_weight=centering_weight
    )
    
    frames = []
    for b64 in frames_base64:
        # Remove data URL prefix if present
        if ',' in b64:
            b64 = b64.split(',')[1]
        
        # Decode base64 to image
        img_bytes = base64.b64decode(b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is not None:
            frames.append(img)
    
    if not frames:
        return {"error": "No valid frames found"}
    
    best_scores = selector.select_best_frames(frames, top_n=1)
    
    if not best_scores:
        return {"error": "Could not score frames"}
    
    best = best_scores[0]
    
    return {
        "bestFrameIndex": best.index,
        "scores": {
            "sharpness": round(best.sharpness, 4),
            "motionBlur": round(best.motion_blur, 4),
            "centering": round(best.centering, 4),
            "brightness": round(best.brightness, 4),
            "totalScore": round(best.total_score, 4)
        },
        "totalFramesAnalyzed": len(frames)
    }
