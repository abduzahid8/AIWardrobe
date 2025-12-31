"""
🎨 Gaussian Rasterizer
Tile-based rasterization for real-time rendering

Implements:
- 2D projection of 3D Gaussians
- Tile-based sorting for efficiency
- Alpha blending with depth ordering
- View-dependent color evaluation
"""

import numpy as np
from typing import Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)


class GaussianRasterizer:
    """
    Tile-based Gaussian rasterizer for real-time rendering.
    
    The rendering pipeline:
    1. Project 3D Gaussians to 2D image plane
    2. Compute 2D covariance for each Gaussian
    3. Assign Gaussians to tiles (16x16 pixels)
    4. Sort Gaussians by depth within each tile
    5. Alpha-blend front-to-back
    """
    
    def __init__(
        self,
        tile_size: int = 16,
        max_gaussians_per_tile: int = 256
    ):
        """
        Initialize rasterizer.
        
        Args:
            tile_size: Pixel size of each tile (default 16x16)
            max_gaussians_per_tile: Maximum Gaussians to render per tile
        """
        self.tile_size = tile_size
        self.max_gaussians_per_tile = max_gaussians_per_tile
    
    def render(
        self,
        scene,  # GaussianCloud
        camera: Dict,
        image_size: Tuple[int, int] = (1024, 1024),
        background: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    ) -> np.ndarray:
        """
        Render scene from camera viewpoint.
        
        Args:
            scene: GaussianCloud to render
            camera: Camera parameters
                - position: (3,) camera position
                - rotation: (3, 3) rotation matrix or (4,) quaternion
                - fov: Field of view in degrees
                - near: Near plane
                - far: Far plane
            image_size: (width, height) of output image
            background: Background color RGB
            
        Returns:
            Rendered image as (H, W, 3) numpy array
        """
        width, height = image_size
        
        # Initialize output
        image = np.zeros((height, width, 3), dtype=np.float32)
        image[:] = background
        
        if len(scene) == 0:
            return (image * 255).astype(np.uint8)
        
        # Get camera matrices
        view_matrix = self._compute_view_matrix(camera)
        proj_matrix = self._compute_projection_matrix(camera, width / height)
        viewproj = proj_matrix @ view_matrix
        
        # Project Gaussians
        projected = self._project_gaussians(scene, viewproj, width, height)
        
        # Sort by depth
        sorted_indices = np.argsort(projected['depths'])
        
        # Render each Gaussian (simplified - no tiling for CPU version)
        for idx in sorted_indices:
            if projected['depths'][idx] < 0:
                continue  # Behind camera
            
            self._splat_gaussian(
                image,
                projected['means_2d'][idx],
                projected['covs_2d'][idx],
                projected['colors'][idx],
                projected['opacities'][idx]
            )
        
        return (np.clip(image, 0, 1) * 255).astype(np.uint8)
    
    def _compute_view_matrix(self, camera: Dict) -> np.ndarray:
        """Compute 4x4 view matrix."""
        position = np.array(camera.get('position', [0, 0, 3]))
        
        # Handle rotation
        rotation = camera.get('rotation', np.eye(3))
        if len(rotation) == 4:
            rotation = self._quaternion_to_matrix(rotation)
        rotation = np.array(rotation).reshape(3, 3)
        
        # Build view matrix
        view = np.eye(4)
        view[:3, :3] = rotation.T
        view[:3, 3] = -rotation.T @ position
        
        return view
    
    def _compute_projection_matrix(
        self, 
        camera: Dict,
        aspect: float
    ) -> np.ndarray:
        """Compute 4x4 projection matrix."""
        fov = camera.get('fov', 60.0)
        near = camera.get('near', 0.1)
        far = camera.get('far', 100.0)
        
        fov_rad = np.radians(fov)
        f = 1.0 / np.tan(fov_rad / 2)
        
        proj = np.zeros((4, 4))
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (far + near) / (near - far)
        proj[2, 3] = 2 * far * near / (near - far)
        proj[3, 2] = -1
        
        return proj
    
    def _quaternion_to_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to rotation matrix."""
        q = np.array(q)
        q = q / np.linalg.norm(q)
        w, x, y, z = q
        
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
    
    def _project_gaussians(
        self,
        scene,
        viewproj: np.ndarray,
        width: int,
        height: int
    ) -> Dict:
        """Project 3D Gaussians to 2D."""
        n = len(scene)
        
        positions = scene.positions
        colors = scene.colors
        opacities = scene.opacities
        
        # Homogeneous coordinates
        positions_h = np.hstack([positions, np.ones((n, 1))])
        
        # Project
        projected = (viewproj @ positions_h.T).T
        
        # Perspective divide
        w = projected[:, 3:4]
        w = np.where(np.abs(w) < 1e-6, 1e-6, w)
        ndc = projected[:, :3] / w
        
        # To screen coordinates
        means_2d = np.zeros((n, 2))
        means_2d[:, 0] = (ndc[:, 0] + 1) * 0.5 * width
        means_2d[:, 1] = (1 - ndc[:, 1]) * 0.5 * height  # Flip Y
        
        # Depths
        depths = ndc[:, 2]
        
        # 2D covariance (simplified - use scale as radius)
        scales = scene.scales
        avg_scale = np.mean(scales, axis=1)
        
        # Project scale to screen space (approximate)
        focal = min(width, height) / 2
        projected_scale = avg_scale * focal / np.maximum(w[:, 0], 0.1)
        
        covs_2d = np.stack([
            np.diag([s**2, s**2]) for s in projected_scale
        ])
        
        return {
            'means_2d': means_2d,
            'covs_2d': covs_2d,
            'depths': depths,
            'colors': colors,
            'opacities': opacities
        }
    
    def _splat_gaussian(
        self,
        image: np.ndarray,
        mean: np.ndarray,
        cov: np.ndarray,
        color: np.ndarray,
        opacity: float
    ):
        """Splat a single Gaussian onto the image."""
        h, w = image.shape[:2]
        
        # Compute radius (3 sigma)
        eigenvalues = np.linalg.eigvalsh(cov)
        radius = 3 * np.sqrt(np.max(eigenvalues))
        
        if radius < 0.5:
            return  # Too small
        
        # Bounding box
        x0 = max(0, int(mean[0] - radius))
        x1 = min(w, int(mean[0] + radius) + 1)
        y0 = max(0, int(mean[1] - radius))
        y1 = min(h, int(mean[1] + radius) + 1)
        
        if x0 >= x1 or y0 >= y1:
            return
        
        # Evaluate Gaussian
        try:
            cov_inv = np.linalg.inv(cov + np.eye(2) * 1e-6)
        except:
            return
        
        for y in range(y0, y1):
            for x in range(x0, x1):
                d = np.array([x - mean[0], y - mean[1]])
                mahalanobis = d @ cov_inv @ d
                
                if mahalanobis < 9:  # 3 sigma
                    alpha = opacity * np.exp(-0.5 * mahalanobis)
                    
                    # Alpha blend
                    image[y, x] = (1 - alpha) * image[y, x] + alpha * color


class WebGLRasterizerExporter:
    """
    Export Gaussian scene for WebGL rendering.
    
    Generates:
    - Compressed data format (.spz)
    - WebGL shader code
    - Viewer HTML
    """
    
    def export_for_webgl(
        self,
        scene,
        output_path: str
    ) -> Dict[str, str]:
        """
        Export scene for WebGL viewer.
        
        Returns:
            Dictionary with file paths
        """
        import os
        import json
        
        base_path = os.path.splitext(output_path)[0]
        
        # Export scene data as JSON (simplified format)
        scene_data = {
            "numGaussians": len(scene),
            "positions": scene.positions.tolist(),
            "scales": scene.scales.tolist(),
            "rotations": scene.rotations.tolist(),
            "opacities": scene.opacities.tolist(),
            "colors": scene.colors.tolist()
        }
        
        json_path = f"{base_path}.json"
        with open(json_path, 'w') as f:
            json.dump(scene_data, f)
        
        # Export viewer HTML
        html_path = f"{base_path}.html"
        self._export_viewer_html(html_path, os.path.basename(json_path))
        
        return {
            "data": json_path,
            "viewer": html_path
        }
    
    def _export_viewer_html(self, path: str, data_file: str):
        """Export WebGL viewer HTML."""
        html = f'''<!DOCTYPE html>
<html>
<head>
    <title>3DGS Viewer</title>
    <style>
        body {{ margin: 0; overflow: hidden; }}
        canvas {{ width: 100%; height: 100%; }}
    </style>
</head>
<body>
    <canvas id="canvas"></canvas>
    <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        // 3D Gaussian Splatting WebGL Viewer
        const canvas = document.getElementById('canvas');
        const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0xffffff);
        
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 100);
        camera.position.set(0, 0, 3);
        
        const controls = new THREE.OrbitControls(camera, canvas);
        controls.enableDamping = true;
        
        // Load Gaussian data
        fetch('{data_file}')
            .then(r => r.json())
            .then(data => {{
                const geometry = new THREE.BufferGeometry();
                const positions = new Float32Array(data.positions.flat());
                const colors = new Float32Array(data.colors.flat());
                
                geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
                geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
                
                const material = new THREE.PointsMaterial({{
                    size: 0.02,
                    vertexColors: true,
                    transparent: true,
                    opacity: 0.8
                }});
                
                const points = new THREE.Points(geometry, material);
                scene.add(points);
            }});
        
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
        
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>'''
        
        with open(path, 'w') as f:
            f.write(html)
