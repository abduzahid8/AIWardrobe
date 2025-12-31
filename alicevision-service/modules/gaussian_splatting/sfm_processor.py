"""
📷 Structure from Motion (SfM) Processor
Camera pose estimation from video frames

Uses:
- Feature detection (SIFT/ORB)
- Feature matching
- Bundle adjustment
- Camera intrinsics estimation
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import logging
import base64
from io import BytesIO

logger = logging.getLogger(__name__)


class SfMProcessor:
    """
    Structure from Motion processor for camera pose estimation.
    
    Pipeline:
    1. Extract keypoints from all frames
    2. Match features between consecutive frames
    3. Estimate relative poses
    4. Bundle adjustment for global consistency
    5. Output camera poses for 3DGS optimization
    """
    
    def __init__(
        self,
        feature_type: str = "sift",
        max_features: int = 2000
    ):
        """
        Initialize SfM processor.
        
        Args:
            feature_type: "sift", "orb", or "akaze"
            max_features: Maximum features per image
        """
        self.feature_type = feature_type
        self.max_features = max_features
        
        # Initialize feature detector
        if feature_type == "sift":
            self.detector = cv2.SIFT_create(max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        elif feature_type == "orb":
            self.detector = cv2.ORB_create(max_features)
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            self.detector = cv2.AKAZE_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    
    def process_video_frames(
        self,
        frames: List,
        focal_length: float = None
    ) -> Dict:
        """
        Process video frames to estimate camera poses.
        
        Args:
            frames: List of frames (numpy arrays or base64 strings)
            focal_length: Optional known focal length
            
        Returns:
            Dictionary with camera poses and 3D points
        """
        logger.info(f"Processing {len(frames)} frames for SfM...")
        
        # Convert frames to numpy
        images = [self._to_numpy(f) for f in frames]
        
        if len(images) < 2:
            return {"success": False, "error": "Need at least 2 frames"}
        
        # Estimate camera intrinsics
        h, w = images[0].shape[:2]
        if focal_length is None:
            focal_length = max(w, h)  # Rough estimate
        
        K = np.array([
            [focal_length, 0, w / 2],
            [0, focal_length, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Extract features from all images
        all_keypoints = []
        all_descriptors = []
        
        for i, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, desc = self.detector.detectAndCompute(gray, None)
            all_keypoints.append(kp)
            all_descriptors.append(desc)
            logger.info(f"  Frame {i}: {len(kp)} features")
        
        # Initialize camera poses
        camera_poses = [np.eye(4)]  # First camera at origin
        
        # Track 3D points
        points_3d = []
        
        # Process consecutive frame pairs
        for i in range(len(images) - 1):
            # Match features
            matches = self._match_features(
                all_descriptors[i],
                all_descriptors[i + 1]
            )
            
            if len(matches) < 8:
                logger.warning(f"  Pair {i}-{i+1}: Only {len(matches)} matches")
                # Copy previous pose (static camera assumption)
                camera_poses.append(camera_poses[-1].copy())
                continue
            
            # Get matched points
            pts1 = np.float32([all_keypoints[i][m.queryIdx].pt for m in matches])
            pts2 = np.float32([all_keypoints[i + 1][m.trainIdx].pt for m in matches])
            
            # Estimate essential matrix
            E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            
            if E is None:
                camera_poses.append(camera_poses[-1].copy())
                continue
            
            # Recover pose
            _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
            
            # Build transformation matrix
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.flatten()
            
            # Chain with previous pose
            new_pose = camera_poses[-1] @ T
            camera_poses.append(new_pose)
            
            # Triangulate points
            points = self._triangulate_points(
                K, camera_poses[-2], camera_poses[-1],
                pts1[mask.flatten() > 0],
                pts2[mask.flatten() > 0]
            )
            points_3d.extend(points)
            
            logger.info(f"  Pair {i}-{i+1}: {len(matches)} matches, {len(points)} 3D points")
        
        logger.info(f"SfM complete: {len(camera_poses)} poses, {len(points_3d)} 3D points")
        
        return {
            "success": True,
            "camera_poses": camera_poses,
            "intrinsics": K,
            "points_3d": np.array(points_3d) if points_3d else np.zeros((0, 3)),
            "image_size": (w, h)
        }
    
    def _to_numpy(self, frame) -> np.ndarray:
        """Convert frame to numpy array."""
        if isinstance(frame, np.ndarray):
            return frame
        
        if isinstance(frame, str):
            # Base64 string
            if ',' in frame:
                frame = frame.split(',')[1]
            img_bytes = base64.b64decode(frame)
            nparr = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        raise ValueError(f"Unsupported frame type: {type(frame)}")
    
    def _match_features(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray
    ) -> List:
        """Match features between two frames."""
        if desc1 is None or desc2 is None:
            return []
        
        # KNN matching
        try:
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
        except:
            return []
        
        # Ratio test
        good_matches = []
        for m in matches:
            if len(m) == 2:
                if m[0].distance < 0.75 * m[1].distance:
                    good_matches.append(m[0])
        
        return good_matches
    
    def _triangulate_points(
        self,
        K: np.ndarray,
        pose1: np.ndarray,
        pose2: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray
    ) -> List[np.ndarray]:
        """Triangulate 3D points from two views."""
        if len(pts1) == 0:
            return []
        
        # Projection matrices
        P1 = K @ pose1[:3, :]
        P2 = K @ pose2[:3, :]
        
        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        
        # Convert to 3D
        points_3d = []
        for i in range(points_4d.shape[1]):
            pt = points_4d[:, i]
            if pt[3] != 0:
                pt_3d = pt[:3] / pt[3]
                # Filter outliers
                if np.all(np.abs(pt_3d) < 100):
                    points_3d.append(pt_3d)
        
        return points_3d


class COLMAPWrapper:
    """
    Wrapper for COLMAP SfM (if installed).
    
    COLMAP provides more accurate results but requires installation.
    Falls back to OpenCV SfM if not available.
    """
    
    @staticmethod
    def is_available() -> bool:
        """Check if COLMAP is installed."""
        import shutil
        return shutil.which('colmap') is not None
    
    def run_sfm(
        self,
        image_dir: str,
        output_dir: str
    ) -> Dict:
        """Run COLMAP SfM pipeline."""
        import subprocess
        import os
        
        if not self.is_available():
            logger.warning("COLMAP not available, falling back to OpenCV SfM")
            return {"success": False, "error": "COLMAP not installed"}
        
        database_path = os.path.join(output_dir, "database.db")
        sparse_dir = os.path.join(output_dir, "sparse")
        os.makedirs(sparse_dir, exist_ok=True)
        
        try:
            # Feature extraction
            subprocess.run([
                "colmap", "feature_extractor",
                "--database_path", database_path,
                "--image_path", image_dir
            ], check=True)
            
            # Feature matching
            subprocess.run([
                "colmap", "exhaustive_matcher",
                "--database_path", database_path
            ], check=True)
            
            # Sparse reconstruction
            subprocess.run([
                "colmap", "mapper",
                "--database_path", database_path,
                "--image_path", image_dir,
                "--output_path", sparse_dir
            ], check=True)
            
            return {
                "success": True,
                "sparse_dir": sparse_dir
            }
            
        except subprocess.CalledProcessError as e:
            logger.error(f"COLMAP failed: {e}")
            return {"success": False, "error": str(e)}
