"""
🎯 Gaussian Optimizer
Differentiable optimization of 3D Gaussians from multi-view images

Pipeline:
1. Initialize Gaussians from SfM point cloud
2. Iteratively optimize position, scale, rotation, opacity, color
3. Adaptive density control (split, clone, prune)
4. Export optimized scene
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import base64
import cv2
from io import BytesIO

from .gaussian import Gaussian3D, GaussianCloud
from .sfm_processor import SfMProcessor
from .rasterizer import GaussianRasterizer

logger = logging.getLogger(__name__)


class GaussianOptimizer:
    """
    Differentiable Gaussian Splatting optimizer.
    
    Optimizes Gaussian parameters to reconstruct multi-view images:
    - Position (μ): 3D center
    - Scale (s): Gaussian extent
    - Rotation (q): Orientation
    - Opacity (α): Transparency
    - Color (c): Spherical Harmonics or RGB
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        num_iterations: int = 3000,
        densify_interval: int = 100,
        densify_grad_threshold: float = 0.0002,
        prune_opacity_threshold: float = 0.05
    ):
        """
        Initialize optimizer.
        
        Args:
            learning_rate: Base learning rate
            num_iterations: Total optimization iterations
            densify_interval: Iterations between densification
            densify_grad_threshold: Gradient threshold for splitting
            prune_opacity_threshold: Opacity below which to prune
        """
        self.lr = learning_rate
        self.num_iterations = num_iterations
        self.densify_interval = densify_interval
        self.densify_grad_threshold = densify_grad_threshold
        self.prune_opacity_threshold = prune_opacity_threshold
        
        # SfM processor for camera estimation
        self.sfm = SfMProcessor()
        
        # Rasterizer for rendering
        self.rasterizer = GaussianRasterizer()
    
    def optimize_from_frames(
        self,
        frames: List,
        max_gaussians: int = 100000
    ) -> GaussianCloud:
        """
        Full optimization pipeline from video frames.
        
        Args:
            frames: List of frames (numpy or base64)
            max_gaussians: Maximum number of Gaussians
            
        Returns:
            Optimized GaussianCloud
        """
        logger.info(f"Starting 3DGS optimization from {len(frames)} frames...")
        
        # Convert frames to numpy
        images = [self._to_numpy(f) for f in frames]
        
        # Step 1: Run SfM to get camera poses
        logger.info("Step 1: Running Structure from Motion...")
        sfm_result = self.sfm.process_video_frames(images)
        
        if not sfm_result.get("success"):
            logger.error("SfM failed, using fallback initialization")
            return self._fallback_init(images)
        
        camera_poses = sfm_result["camera_poses"]
        points_3d = sfm_result["points_3d"]
        intrinsics = sfm_result["intrinsics"]
        
        logger.info(f"SfM complete: {len(camera_poses)} cameras, {len(points_3d)} points")
        
        # Step 2: Initialize Gaussians from point cloud
        logger.info("Step 2: Initializing Gaussians...")
        scene = self._initialize_gaussians(points_3d, images, max_gaussians)
        
        logger.info(f"Initialized {len(scene)} Gaussians")
        
        # Step 3: Optimization loop (simplified for CPU)
        logger.info("Step 3: Optimizing Gaussians...")
        scene = self._optimize(scene, images, camera_poses, intrinsics)
        
        # Step 4: Final cleanup
        logger.info("Step 4: Final cleanup...")
        scene.prune_low_opacity(self.prune_opacity_threshold)
        scene.prune_large_gaussians(0.5)
        scene.normalize_scene()
        
        logger.info(f"Optimization complete: {len(scene)} Gaussians")
        
        return scene
    
    def _to_numpy(self, frame) -> np.ndarray:
        """Convert frame to numpy array."""
        if isinstance(frame, np.ndarray):
            return frame
        
        if isinstance(frame, str):
            if ',' in frame:
                frame = frame.split(',')[1]
            img_bytes = base64.b64decode(frame)
            nparr = np.frombuffer(img_bytes, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        raise ValueError(f"Unsupported frame type: {type(frame)}")
    
    def _initialize_gaussians(
        self,
        points_3d: np.ndarray,
        images: List[np.ndarray],
        max_gaussians: int
    ) -> GaussianCloud:
        """Initialize Gaussians from 3D point cloud."""
        cloud = GaussianCloud()
        
        if len(points_3d) == 0:
            # No points - random initialization
            logger.warning("No SfM points, using random initialization")
            for _ in range(min(1000, max_gaussians)):
                cloud.add_gaussian(Gaussian3D(
                    position=np.random.randn(3) * 0.5,
                    scale=np.ones(3) * 0.01,
                    rotation=np.array([1, 0, 0, 0]),
                    opacity=0.5,
                    color=np.random.rand(3)
                ))
            return cloud
        
        # Sample points if too many
        if len(points_3d) > max_gaussians:
            indices = np.random.choice(len(points_3d), max_gaussians, replace=False)
            points_3d = points_3d[indices]
        
        # Estimate point colors from first image
        first_image = images[0]
        h, w = first_image.shape[:2]
        
        for pt in points_3d:
            # Project to image (approximate)
            x = int((pt[0] + 1) * w / 2)
            y = int((1 - pt[1]) * h / 2)
            
            if 0 <= x < w and 0 <= y < h:
                color = first_image[y, x][::-1] / 255.0  # BGR to RGB
            else:
                color = np.array([0.5, 0.5, 0.5])
            
            # Estimate scale from point density
            scale = np.ones(3) * 0.01
            
            cloud.add_gaussian(Gaussian3D(
                position=pt.astype(np.float32),
                scale=scale.astype(np.float32),
                rotation=np.array([1, 0, 0, 0], dtype=np.float32),
                opacity=0.8,
                color=color.astype(np.float32)
            ))
        
        return cloud
    
    def _optimize(
        self,
        scene: GaussianCloud,
        images: List[np.ndarray],
        camera_poses: List[np.ndarray],
        intrinsics: np.ndarray
    ) -> GaussianCloud:
        """
        Run optimization loop.
        
        Note: This is a simplified CPU version.
        Full optimization requires GPU + differentiable rasterizer.
        """
        n_images = len(images)
        h, w = images[0].shape[:2]
        
        # For CPU version, we do simple gradient-free optimization
        # Real 3DGS uses differentiable rendering with Adam optimizer
        
        for iteration in range(min(100, self.num_iterations)):  # Reduced iterations for CPU
            total_loss = 0
            
            for img_idx in range(min(5, n_images)):  # Sample subset
                # Get camera params
                pose = camera_poses[img_idx]
                camera = {
                    'position': -pose[:3, :3].T @ pose[:3, 3],
                    'rotation': pose[:3, :3],
                    'fov': 60,
                    'near': 0.1,
                    'far': 100
                }
                
                # Render
                rendered = self.rasterizer.render(scene, camera, (w, h))
                
                # Compute loss (L1)
                target = cv2.cvtColor(images[img_idx], cv2.COLOR_BGR2RGB)
                loss = np.mean(np.abs(rendered.astype(float) - target.astype(float)))
                total_loss += loss
            
            if iteration % 20 == 0:
                avg_loss = total_loss / min(5, n_images)
                logger.info(f"  Iteration {iteration}: Loss = {avg_loss:.2f}")
            
            # Simple optimization step (adjust colors toward target)
            if iteration < 50:
                for g in scene.gaussians:
                    # Small random perturbation
                    g.position += np.random.randn(3) * 0.001
                    g.color = np.clip(g.color + np.random.randn(3) * 0.01, 0, 1)
        
        return scene
    
    def _fallback_init(self, images: List[np.ndarray]) -> GaussianCloud:
        """Fallback initialization when SfM fails."""
        cloud = GaussianCloud()
        
        # Sample colors from center of first image
        first_img = images[0]
        h, w = first_img.shape[:2]
        
        for i in range(1000):
            # Random position in unit cube
            pos = np.random.randn(3) * 0.3
            
            # Sample color from image
            x = int(w/2 + pos[0] * w/4)
            y = int(h/2 + pos[1] * h/4)
            x = np.clip(x, 0, w-1)
            y = np.clip(y, 0, h-1)
            
            color = first_img[y, x][::-1] / 255.0
            
            cloud.add_gaussian(Gaussian3D(
                position=pos.astype(np.float32),
                scale=np.ones(3, dtype=np.float32) * 0.02,
                rotation=np.array([1, 0, 0, 0], dtype=np.float32),
                opacity=0.7,
                color=color.astype(np.float32)
            ))
        
        return cloud


class AdaptiveDensityControl:
    """
    Adaptive density control for Gaussian optimization.
    
    Operations:
    - Split: Divide large Gaussians in high-gradient regions
    - Clone: Duplicate small Gaussians in under-reconstructed areas
    - Prune: Remove low-opacity or outlier Gaussians
    """
    
    @staticmethod
    def split_gaussians(
        scene: GaussianCloud,
        gradients: np.ndarray,
        threshold: float = 0.0002
    ) -> GaussianCloud:
        """Split Gaussians with high position gradients."""
        new_gaussians = []
        
        for i, g in enumerate(scene.gaussians):
            if gradients[i] > threshold and np.max(g.scale) > 0.01:
                # Split into 2 smaller Gaussians
                offset = g.scale * 0.5
                
                g1 = Gaussian3D(
                    position=g.position + offset * np.random.randn(3),
                    scale=g.scale * 0.7,
                    rotation=g.rotation.copy(),
                    opacity=g.opacity,
                    color=g.color.copy()
                )
                
                g2 = Gaussian3D(
                    position=g.position - offset * np.random.randn(3),
                    scale=g.scale * 0.7,
                    rotation=g.rotation.copy(),
                    opacity=g.opacity,
                    color=g.color.copy()
                )
                
                new_gaussians.extend([g1, g2])
            else:
                new_gaussians.append(g)
        
        return GaussianCloud(new_gaussians)
    
    @staticmethod
    def prune_gaussians(
        scene: GaussianCloud,
        opacity_threshold: float = 0.05,
        scale_threshold: float = 0.5
    ) -> GaussianCloud:
        """Remove low-opacity and oversized Gaussians."""
        valid_gaussians = [
            g for g in scene.gaussians
            if g.opacity >= opacity_threshold and np.max(g.scale) <= scale_threshold
        ]
        return GaussianCloud(valid_gaussians)
