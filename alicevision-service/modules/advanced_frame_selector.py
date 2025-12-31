"""
🚀 Advanced Frame Selection Module - Next-Generation Video Analytics
Implements the strategic roadmap for intelligent video ingestion

Key Features:
1. FFT Spectral Sharpness Analysis (texture-invariant)
2. Optical Flow Stillness Detection (pose moments)
3. CLIP Semantic Relevance Scoring
4. Adaptive Keyframe Sampling (AKS) with clustering
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
import logging
import base64
from PIL import Image
from io import BytesIO
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================
# 🎯 DATA STRUCTURES
# ============================================

@dataclass
class AdvancedFrameScore:
    """Comprehensive frame quality assessment"""
    index: int
    original_index: int  # Index in original video
    
    # Visual quality metrics
    spectral_sharpness: float = 0.0     # FFT-based sharpness (0-1)
    laplacian_sharpness: float = 0.0    # Traditional Laplacian (0-1)
    brenner_sharpness: float = 0.0      # Brenner gradient (0-1)
    motion_blur: float = 0.0            # Blur score (0-1, higher = more blur)
    
    # Temporal metrics
    stillness: float = 0.0              # Optical flow stillness (0-1)
    
    # Composition metrics
    centering: float = 0.0              # Subject centering (0-1)
    brightness: float = 0.0             # Lighting quality (0-1)
    
    # Semantic metrics
    semantic_relevance: float = 0.0     # CLIP similarity to fashion (0-1)
    
    # Combined score
    total_score: float = 0.0
    
    # Metadata
    cluster_id: int = -1                # For AKS clustering
    is_centroid: bool = False           # Whether this is a cluster centroid
    
    def to_dict(self) -> Dict:
        return {
            "index": self.index,
            "originalIndex": self.original_index,
            "scores": {
                "spectralSharpness": round(self.spectral_sharpness, 4),
                "laplacianSharpness": round(self.laplacian_sharpness, 4),
                "brennerSharpness": round(self.brenner_sharpness, 4),
                "motionBlur": round(self.motion_blur, 4),
                "stillness": round(self.stillness, 4),
                "centering": round(self.centering, 4),
                "brightness": round(self.brightness, 4),
                "semanticRelevance": round(self.semantic_relevance, 4),
                "totalScore": round(self.total_score, 4)
            },
            "clusterId": self.cluster_id,
            "isCentroid": self.is_centroid
        }


@dataclass
class FrameSelectionResult:
    """Result of advanced frame selection"""
    best_frames: List[AdvancedFrameScore]
    total_frames_analyzed: int
    clusters_formed: int
    processing_time_ms: float
    method_used: str
    
    def to_dict(self) -> Dict:
        return {
            "bestFrames": [f.to_dict() for f in self.best_frames],
            "totalFramesAnalyzed": self.total_frames_analyzed,
            "clustersFormed": self.clusters_formed,
            "processingTimeMs": round(self.processing_time_ms, 2),
            "methodUsed": self.method_used
        }


# ============================================
# 🚀 ADVANCED FRAME SELECTOR
# ============================================

class AdvancedFrameSelector:
    """
    Next-generation frame selection combining:
    1. FFT spectral sharpness (texture-invariant)
    2. Optical flow stillness detection
    3. CLIP semantic relevance scoring
    4. Adaptive Keyframe Sampling (AKS)
    
    This implements the strategic roadmap for optimal video ingestion.
    """
    
    # Default fashion prompts for semantic filtering
    DEFAULT_FASHION_PROMPTS = [
        "full body outfit",
        "fashion photography",
        "person wearing clothes",
        "clothing on display",
        "model showing outfit",
        "fashion pose",
        "clear view of clothing"
    ]
    
    # Weights for final score calculation
    SCORE_WEIGHTS = {
        "spectral_sharpness": 0.20,
        "stillness": 0.25,
        "semantic_relevance": 0.25,
        "centering": 0.15,
        "brightness": 0.10,
        "blur_penalty": 0.05  # Penalty for blur
    }
    
    def __init__(
        self,
        use_clip: bool = True,
        use_optical_flow: bool = True,
        clip_threshold: float = 0.25,
        stillness_weight: float = 0.25,
        device: str = "auto"
    ):
        """
        Initialize advanced frame selector.
        
        Args:
            use_clip: Whether to use CLIP for semantic scoring
            use_optical_flow: Whether to use optical flow for stillness
            clip_threshold: Minimum CLIP similarity to accept frame
            stillness_weight: Weight for stillness in scoring
            device: Compute device ("cuda", "cpu", "auto")
        """
        self.use_clip = use_clip
        self.use_optical_flow = use_optical_flow
        self.clip_threshold = clip_threshold
        self.stillness_weight = stillness_weight
        
        # Setup device
        self.device = self._setup_device(device)
        
        # Lazy-loaded models
        self._clip_model = None
        self._clip_preprocess = None
        self._tokenizer = None
        
        # Cache for CLIP text embeddings
        self._prompt_embeddings = None
        
        logger.info(f"AdvancedFrameSelector initialized (CLIP: {use_clip}, OpticalFlow: {use_optical_flow})")
    
    def _setup_device(self, device: str) -> str:
        """Setup compute device."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device
    
    def _load_clip(self):
        """Lazy load CLIP model."""
        if self._clip_model is not None:
            return
        
        try:
            import torch
            import open_clip
            
            logger.info("Loading CLIP model for semantic frame selection...")
            
            self._clip_model, _, self._clip_preprocess = open_clip.create_model_and_transforms(
                "ViT-B-32", pretrained="openai"
            )
            self._tokenizer = open_clip.get_tokenizer("ViT-B-32")
            self._clip_model = self._clip_model.to(self.device)
            self._clip_model.eval()
            
            # Pre-compute prompt embeddings
            self._precompute_prompt_embeddings()
            
            logger.info("CLIP model loaded successfully")
            
        except Exception as e:
            logger.warning(f"Could not load CLIP: {e}. Semantic scoring disabled.")
            self.use_clip = False
    
    def _precompute_prompt_embeddings(self):
        """Pre-compute embeddings for default fashion prompts."""
        if self._clip_model is None:
            return
        
        import torch
        
        with torch.no_grad():
            tokens = self._tokenizer(self.DEFAULT_FASHION_PROMPTS)
            tokens = tokens.to(self.device)
            self._prompt_embeddings = self._clip_model.encode_text(tokens)
            self._prompt_embeddings = self._prompt_embeddings / self._prompt_embeddings.norm(dim=-1, keepdim=True)
    
    # ============================================
    # 📊 QUALITY METRICS
    # ============================================
    
    def calculate_spectral_sharpness(self, image: np.ndarray) -> float:
        """
        Calculate texture-invariant sharpness using FFT spectral analysis.
        
        This method avoids the pitfall of Laplacian variance, which confuses
        high-contrast patterns with sharpness.
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
        
        # Create radial weights (higher for outer = high-freq regions)
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r_max = np.sqrt(center_x**2 + center_y**2)
        weights = r / r_max
        
        # Weighted average emphasizing high frequencies
        weighted_magnitude = magnitude * weights
        sharpness = np.sum(weighted_magnitude) / (np.sum(magnitude) + 1e-10)
        
        return min(1.0, sharpness / 0.5)
    
    def calculate_laplacian_sharpness(self, image: np.ndarray) -> float:
        """Traditional Laplacian variance sharpness (for comparison)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        return min(variance / 500.0, 1.0)
    
    def calculate_brenner_sharpness(self, image: np.ndarray) -> float:
        """Brenner gradient - fast, stable sharpness metric."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        gray = gray.astype(np.float64)
        
        h_diff = gray[:, 2:] - gray[:, :-2]
        h_focus = np.sum(h_diff ** 2)
        
        v_diff = gray[2:, :] - gray[:-2, :]
        v_focus = np.sum(v_diff ** 2)
        
        total_focus = h_focus + v_focus
        pixels = gray.shape[0] * gray.shape[1]
        normalized_focus = total_focus / pixels
        
        return min(1.0, normalized_focus / 5000.0)
    
    def calculate_motion_blur(self, image: np.ndarray) -> float:
        """Enhanced FFT-based motion blur detection with adaptive thresholding."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude = np.log1p(np.abs(fft_shift))
        
        h, w = magnitude.shape
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        r_max = np.sqrt(center_x**2 + center_y**2)
        
        # Frequency bands
        high_freq_mask = r > (0.85 * r_max)
        mid_freq_mask = (r > 0.3 * r_max) & (r < 0.7 * r_max)
        low_freq_mask = r < 0.2 * r_max
        
        high_freq_energy = np.mean(magnitude[high_freq_mask]) if np.any(high_freq_mask) else 0
        mid_freq_energy = np.mean(magnitude[mid_freq_mask]) if np.any(mid_freq_mask) else 1
        low_freq_energy = np.mean(magnitude[low_freq_mask]) if np.any(low_freq_mask) else 1
        
        # Adaptive threshold
        texture_complexity = mid_freq_energy / (low_freq_energy + 1e-10)
        blur_ratio = high_freq_energy / (mid_freq_energy + 1e-10)
        adaptive_threshold = max(0.1, 0.05 * texture_complexity)
        
        blur_score = max(0, 1.0 - (blur_ratio / adaptive_threshold))
        
        return min(1.0, blur_score)
    
    def calculate_optical_flow_stillness(
        self,
        prev_frame: np.ndarray,
        curr_frame: np.ndarray
    ) -> float:
        """
        Calculate stillness using Farneback optical flow.
        Detects "pose" moments where subject is still.
        """
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame
        
        if len(curr_frame.shape) == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame
        
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        avg_motion = np.mean(magnitude)
        
        max_expected_motion = 20.0
        stillness = max(0, 1.0 - (avg_motion / max_expected_motion))
        
        return min(1.0, stillness)
    
    def calculate_centering(self, image: np.ndarray) -> float:
        """Calculate subject centering using edge detection."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.5
        
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        subject_center_x = x + w / 2
        subject_center_y = y + h / 2
        image_center_x = gray.shape[1] / 2
        image_center_y = gray.shape[0] / 2
        
        max_distance = np.sqrt(image_center_x**2 + image_center_y**2)
        distance = np.sqrt(
            (subject_center_x - image_center_x)**2 +
            (subject_center_y - image_center_y)**2
        )
        
        return 1.0 - (distance / max_distance)
    
    def calculate_brightness(self, image: np.ndarray) -> float:
        """Calculate brightness score (optimal around 0.4-0.6)."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        mean_brightness = gray.mean() / 255.0
        
        if 0.35 <= mean_brightness <= 0.65:
            return 1.0
        elif mean_brightness < 0.35:
            return mean_brightness / 0.35
        else:
            return (1.0 - mean_brightness) / 0.35
    
    def calculate_semantic_relevance(
        self,
        image: np.ndarray,
        custom_prompts: List[str] = None
    ) -> float:
        """
        Calculate CLIP semantic relevance to fashion content.
        
        Solves the "Unboxing Problem" - rejects frames without clear fashion content.
        """
        if not self.use_clip:
            return 0.5  # Neutral if CLIP disabled
        
        self._load_clip()
        
        if self._clip_model is None:
            return 0.5
        
        try:
            import torch
            
            # Convert to PIL Image
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image)
            
            # Preprocess and encode image
            image_input = self._clip_preprocess(pil_image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                image_features = self._clip_model.encode_image(image_input)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                
                # Use custom prompts or default
                if custom_prompts:
                    tokens = self._tokenizer(custom_prompts)
                    tokens = tokens.to(self.device)
                    text_features = self._clip_model.encode_text(tokens)
                    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                else:
                    text_features = self._prompt_embeddings
                
                # Calculate similarity
                similarity = (image_features @ text_features.T).max().item()
            
            # Normalize to 0-1 (CLIP similarities typically range 0.15-0.35)
            normalized = (similarity - 0.15) / 0.25
            return max(0.0, min(1.0, normalized))
            
        except Exception as e:
            logger.warning(f"CLIP scoring failed: {e}")
            return 0.5
    
    # ============================================
    # 🎯 FRAME SELECTION
    # ============================================
    
    def score_frame(
        self,
        frame: np.ndarray,
        index: int,
        original_index: int,
        prev_frame: np.ndarray = None,
        custom_prompts: List[str] = None
    ) -> AdvancedFrameScore:
        """Calculate comprehensive quality score for a frame."""
        
        score = AdvancedFrameScore(
            index=index,
            original_index=original_index,
            spectral_sharpness=self.calculate_spectral_sharpness(frame),
            laplacian_sharpness=self.calculate_laplacian_sharpness(frame),
            brenner_sharpness=self.calculate_brenner_sharpness(frame),
            motion_blur=self.calculate_motion_blur(frame),
            centering=self.calculate_centering(frame),
            brightness=self.calculate_brightness(frame)
        )
        
        # Optical flow stillness (requires previous frame)
        if self.use_optical_flow and prev_frame is not None:
            score.stillness = self.calculate_optical_flow_stillness(prev_frame, frame)
        else:
            score.stillness = 0.5
        
        # CLIP semantic relevance
        if self.use_clip:
            score.semantic_relevance = self.calculate_semantic_relevance(frame, custom_prompts)
        else:
            score.semantic_relevance = 0.5
        
        # Calculate weighted total score
        w = self.SCORE_WEIGHTS
        score.total_score = (
            score.spectral_sharpness * w["spectral_sharpness"] +
            score.stillness * w["stillness"] +
            score.semantic_relevance * w["semantic_relevance"] +
            score.centering * w["centering"] +
            score.brightness * w["brightness"] -
            score.motion_blur * w["blur_penalty"]
        )
        
        return score
    
    def _extract_color_histogram(self, image: np.ndarray) -> np.ndarray:
        """Extract color histogram for clustering."""
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        else:
            hsv = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2HSV)
        
        # Calculate histograms for H, S, V channels
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])
        
        # Concatenate and normalize
        hist = np.concatenate([hist_h, hist_s, hist_v]).flatten()
        hist = hist / (hist.sum() + 1e-10)
        
        return hist
    
    def select_optimal_frames(
        self,
        frames: List[np.ndarray],
        frame_indices: List[int] = None,
        top_n: int = 5,
        semantic_query: str = None,
        use_aks: bool = True,
        min_semantic_score: float = None
    ) -> FrameSelectionResult:
        """
        Select optimal frames using advanced analysis.
        
        Args:
            frames: List of video frames
            frame_indices: Original indices of frames in video
            top_n: Number of best frames to return
            semantic_query: Optional custom semantic query (e.g., "red dress")
            use_aks: Use Adaptive Keyframe Sampling with clustering
            min_semantic_score: Minimum semantic score to accept frame
            
        Returns:
            FrameSelectionResult with scored frames
        """
        import time
        start_time = time.time()
        
        if not frames:
            return FrameSelectionResult(
                best_frames=[],
                total_frames_analyzed=0,
                clusters_formed=0,
                processing_time_ms=0,
                method_used="none"
            )
        
        frame_indices = frame_indices or list(range(len(frames)))
        min_semantic_score = min_semantic_score or self.clip_threshold
        custom_prompts = [semantic_query] if semantic_query else None
        
        clusters_formed = 0
        method_used = "full_analysis"
        
        # Adaptive Keyframe Sampling for efficiency
        if use_aks and len(frames) > 20:
            frames, frame_indices, clusters_formed = self._adaptive_keyframe_sampling(
                frames, frame_indices, n_clusters=min(10, len(frames) // 3)
            )
            method_used = "aks_clustering"
        
        # Score all frames
        scores = []
        prev_frame = None
        
        for i, (frame, orig_idx) in enumerate(zip(frames, frame_indices)):
            score = self.score_frame(
                frame=frame,
                index=i,
                original_index=orig_idx,
                prev_frame=prev_frame,
                custom_prompts=custom_prompts
            )
            
            # Apply semantic threshold
            if self.use_clip and score.semantic_relevance < min_semantic_score:
                continue
            
            scores.append(score)
            prev_frame = frame
        
        # Sort by total score
        scores.sort(key=lambda x: x.total_score, reverse=True)
        
        # Ensure diversity in selected frames
        selected = self._select_diverse_frames(scores, top_n)
        
        processing_time = (time.time() - start_time) * 1000
        
        return FrameSelectionResult(
            best_frames=selected,
            total_frames_analyzed=len(frames),
            clusters_formed=clusters_formed,
            processing_time_ms=processing_time,
            method_used=method_used
        )
    
    def _adaptive_keyframe_sampling(
        self,
        frames: List[np.ndarray],
        frame_indices: List[int],
        n_clusters: int = 10
    ) -> Tuple[List[np.ndarray], List[int], int]:
        """
        Adaptive Keyframe Sampling (AKS) using color histogram clustering.
        
        Reduces computational load by only analyzing cluster centroids.
        """
        # Extract features for clustering
        features = np.array([self._extract_color_histogram(f) for f in frames])
        
        try:
            from sklearn.cluster import KMeans
            
            # Cluster frames
            kmeans = KMeans(n_clusters=min(n_clusters, len(frames)), random_state=42)
            labels = kmeans.fit_predict(features)
            
            # Select centroid frames (closest to cluster center)
            selected_frames = []
            selected_indices = []
            
            for cluster_id in range(kmeans.n_clusters):
                cluster_mask = labels == cluster_id
                cluster_features = features[cluster_mask]
                cluster_center = kmeans.cluster_centers_[cluster_id]
                
                # Find closest frame to centroid
                distances = np.linalg.norm(cluster_features - cluster_center, axis=1)
                closest_idx = np.argmin(distances)
                
                # Get original index
                cluster_frame_indices = np.where(cluster_mask)[0]
                original_idx = cluster_frame_indices[closest_idx]
                
                selected_frames.append(frames[original_idx])
                selected_indices.append(frame_indices[original_idx])
            
            return selected_frames, selected_indices, kmeans.n_clusters
            
        except ImportError:
            logger.warning("sklearn not available, skipping AKS clustering")
            return frames, frame_indices, 0
    
    def _select_diverse_frames(
        self,
        scores: List[AdvancedFrameScore],
        top_n: int
    ) -> List[AdvancedFrameScore]:
        """Select diverse frames using temporal spacing."""
        if len(scores) <= top_n:
            return scores
        
        selected = [scores[0]]  # Always include best
        
        for score in scores[1:]:
            if len(selected) >= top_n:
                break
            
            # Check temporal distance from selected frames
            min_distance = min(
                abs(score.original_index - s.original_index) 
                for s in selected
            )
            
            # Accept if sufficiently far from existing selections
            if min_distance >= 5:  # At least 5 frames apart
                selected.append(score)
        
        # Fill remaining slots with best remaining
        if len(selected) < top_n:
            for score in scores:
                if score not in selected:
                    selected.append(score)
                    if len(selected) >= top_n:
                        break
        
        return selected
    
    def select_from_video(
        self,
        video_path: str,
        sample_rate: int = 3,
        top_n: int = 5,
        semantic_query: str = None
    ) -> FrameSelectionResult:
        """
        Select optimal frames from a video file.
        
        Args:
            video_path: Path to video file
            sample_rate: Extract every Nth frame
            top_n: Number of frames to select
            semantic_query: Optional semantic filter
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
        
        logger.info(f"Extracted {len(frames)} frames from video (sample rate: {sample_rate})")
        
        return self.select_optimal_frames(
            frames=frames,
            frame_indices=frame_indices,
            top_n=top_n,
            semantic_query=semantic_query
        )


# ============================================
# 🔧 UTILITY FUNCTIONS
# ============================================

def select_best_frames_advanced(
    frames_base64: List[str],
    top_n: int = 5,
    use_clip: bool = True,
    semantic_query: str = None
) -> Dict:
    """
    Utility function to select best frames from base64-encoded images.
    
    Args:
        frames_base64: List of base64-encoded images
        top_n: Number of best frames to return
        use_clip: Whether to use CLIP for semantic filtering
        semantic_query: Optional semantic query
        
    Returns:
        Dictionary with selection results
    """
    # Decode frames
    frames = []
    for b64 in frames_base64:
        if ',' in b64:
            b64 = b64.split(',')[1]
        
        img_bytes = base64.b64decode(b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is not None:
            frames.append(img)
    
    if not frames:
        return {"error": "No valid frames found"}
    
    # Run selection
    selector = AdvancedFrameSelector(use_clip=use_clip)
    result = selector.select_optimal_frames(
        frames=frames,
        top_n=top_n,
        semantic_query=semantic_query
    )
    
    return result.to_dict()


# Singleton instance
_advanced_selector_instance = None

def get_advanced_frame_selector(use_clip: bool = True) -> AdvancedFrameSelector:
    """Get singleton instance of AdvancedFrameSelector."""
    global _advanced_selector_instance
    if _advanced_selector_instance is None:
        _advanced_selector_instance = AdvancedFrameSelector(use_clip=use_clip)
    return _advanced_selector_instance
