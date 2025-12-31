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
    # 🚀 NEW: Advanced metrics
    stillness: float = 0.0           # Optical flow stillness (0-1, higher = more still)
    semantic_relevance: float = 0.0   # CLIP similarity to fashion prompts (0-1)
    spectral_sharpness: float = 0.0   # FFT-based texture-invariant sharpness
    

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
        🎯 ENHANCED: Spectral blur detection using FFT magnitude analysis.
        
        Uses adaptive thresholding based on texture complexity to avoid
        false positives on soft fabrics (silk, cashmere) and false negatives
        on high-contrast patterns (houndstooth, plaid).
        
        Returns blur score (higher = more blur, which is bad).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Compute 2D FFT and shift zero frequency to center
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shift))
        
        h, w = magnitude.shape
        
        # Create radial frequency bins
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r_max = np.sqrt(center_x**2 + center_y**2)
        
        # Top 15% frequencies (high-frequency band) - edge definition
        high_freq_mask = r > (0.85 * r_max)
        high_freq_energy = np.mean(magnitude[high_freq_mask]) if np.any(high_freq_mask) else 0
        
        # Mid frequencies (texture band) - fabric patterns
        mid_freq_mask = (r > 0.3 * r_max) & (r < 0.7 * r_max)
        mid_freq_energy = np.mean(magnitude[mid_freq_mask]) if np.any(mid_freq_mask) else 1
        
        # Low frequencies (overall structure)
        low_freq_mask = r < 0.2 * r_max
        low_freq_energy = np.mean(magnitude[low_freq_mask]) if np.any(low_freq_mask) else 1
        
        # Adaptive threshold based on texture complexity
        # High mid-freq energy suggests complex texture (good), not blur
        texture_complexity = mid_freq_energy / (low_freq_energy + 1e-10)
        
        # Sharp images have high-frequency content
        # Blurry images act as low-pass filters, attenuating high frequencies
        blur_ratio = high_freq_energy / (mid_freq_energy + 1e-10)
        
        # Adaptive threshold: complex textures get more lenient blur threshold
        adaptive_threshold = max(0.1, 0.05 * texture_complexity)
        
        # Blur score: low high-freq energy = blurry
        blur_score = max(0, 1.0 - (blur_ratio / adaptive_threshold))
        
        return min(1.0, blur_score)
    
    def calculate_spectral_sharpness(self, image: np.ndarray) -> float:
        """
        🆕 Calculate texture-invariant sharpness using FFT spectral analysis.
        
        Unlike Laplacian variance (which confuses contrast with sharpness),
        this method analyzes the magnitude spectrum to detect true focus.
        
        Returns sharpness score (0-1, higher = sharper).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Compute 2D FFT
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shift))
        
        h, w = magnitude.shape
        center_y, center_x = h // 2, w // 2
        
        # Calculate radial distance from center
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r_max = np.sqrt(center_x**2 + center_y**2)
        
        # Weighted average emphasizing high frequencies
        # Sharp images have more energy in high-frequency bands
        weights = r / r_max  # Higher weight for outer (high-freq) regions
        weighted_magnitude = magnitude * weights
        
        # Normalize by total magnitude to get relative high-freq content
        total_mag = np.sum(magnitude) + 1e-10
        weighted_sum = np.sum(weighted_magnitude)
        
        # Spectral sharpness score
        sharpness = weighted_sum / total_mag
        
        # Normalize to 0-1 range (empirically tuned)
        normalized_sharpness = min(1.0, sharpness / 0.5)
        
        return normalized_sharpness
    
    def calculate_optical_flow_stillness(
        self, 
        prev_frame: np.ndarray, 
        curr_frame: np.ndarray
    ) -> float:
        """
        🆕 Calculate motion magnitude using Farneback optical flow.
        
        Detects the "pose" moments in fashion videos where the subject
        is still and the garment is displayed optimally.
        
        Returns stillness score (0-1, higher = more still = better for fashion).
        """
        # Convert to grayscale
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame
            
        if len(curr_frame.shape) == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame
        
        # Calculate dense optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5,    # Image pyramid scale
            levels=3,         # Number of pyramid levels
            winsize=15,       # Window size
            iterations=3,     # Iterations per level
            poly_n=5,         # Polynomial expansion neighborhood
            poly_sigma=1.2,   # Gaussian std for polynomial expansion
            flags=0
        )
        
        # Calculate motion magnitude
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        avg_motion = np.mean(magnitude)
        
        # Stillness = inverse of motion (normalize to 0-1)
        # max_expected_motion is typical movement in pixels per frame
        max_expected_motion = 20.0
        stillness = max(0, 1.0 - (avg_motion / max_expected_motion))
        
        return min(1.0, stillness)
    
    def calculate_brenner_gradient(self, image: np.ndarray) -> float:
        """
        🆕 Brenner Gradient: Fast alternative to FFT for real-time sharpness.
        
        Computes squared difference between pixels 2 apart.
        More stable than Laplacian, especially for fashion textiles.
        
        Returns sharpness score (0-1, higher = sharper).
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        gray = gray.astype(np.float64)
        
        # Horizontal Brenner gradient
        h_diff = gray[:, 2:] - gray[:, :-2]
        h_focus = np.sum(h_diff ** 2)
        
        # Vertical Brenner gradient  
        v_diff = gray[2:, :] - gray[:-2, :]
        v_focus = np.sum(v_diff ** 2)
        
        # Combined focus measure
        total_focus = h_focus + v_focus
        
        # Normalize by image size
        pixels = gray.shape[0] * gray.shape[1]
        normalized_focus = total_focus / pixels
        
        # Scale to 0-1 (empirically tuned)
        sharpness = min(1.0, normalized_focus / 5000.0)
        
        return sharpness
    
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
