"""
🌟 3D Gaussian Primitive
Core data structure for 3D Gaussian Splatting

A 3D Gaussian is defined by:
- Position (μ): 3D center point
- Covariance (Σ): 3x3 matrix defining shape/orientation
- Opacity (α): Transparency value
- Color (c): RGB or Spherical Harmonics coefficients
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class Gaussian3D:
    """
    Single 3D Gaussian primitive.
    
    The covariance matrix Σ is decomposed into:
    Σ = R * S * S^T * R^T
    Where R is rotation matrix and S is scale matrix.
    """
    
    # Position (x, y, z)
    position: np.ndarray
    
    # Scale (sx, sy, sz) - diagonal of S matrix
    scale: np.ndarray
    
    # Rotation quaternion (w, x, y, z)
    rotation: np.ndarray
    
    # Opacity (0-1)
    opacity: float = 1.0
    
    # Color as Spherical Harmonics coefficients
    # SH degree 0: 3 coefficients (RGB)
    # SH degree 1: 9 coefficients
    # SH degree 2: 15 coefficients
    # SH degree 3: 27 coefficients
    sh_coeffs: np.ndarray = None
    
    # Simple RGB color (fallback)
    color: np.ndarray = None
    
    def __post_init__(self):
        """Validate and initialize."""
        self.position = np.array(self.position, dtype=np.float32)
        self.scale = np.array(self.scale, dtype=np.float32)
        self.rotation = np.array(self.rotation, dtype=np.float32)
        
        if self.sh_coeffs is None and self.color is None:
            self.color = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        
        if self.color is not None:
            self.color = np.array(self.color, dtype=np.float32)
    
    @property
    def covariance(self) -> np.ndarray:
        """Compute covariance matrix from scale and rotation."""
        R = self._quaternion_to_rotation_matrix(self.rotation)
        S = np.diag(self.scale)
        return R @ S @ S.T @ R.T
    
    def _quaternion_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix."""
        w, x, y, z = q / np.linalg.norm(q)
        
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ], dtype=np.float32)
    
    def get_color_for_direction(self, direction: np.ndarray) -> np.ndarray:
        """
        Get view-dependent color using Spherical Harmonics.
        
        Args:
            direction: Viewing direction (normalized)
            
        Returns:
            RGB color
        """
        if self.sh_coeffs is None:
            return self.color
        
        # Evaluate SH for given direction
        return self._evaluate_sh(direction)
    
    def _evaluate_sh(self, d: np.ndarray) -> np.ndarray:
        """Evaluate Spherical Harmonics."""
        # SH basis functions (degree 0-3)
        C0 = 0.28209479177387814
        C1 = 0.4886025119029199
        C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        
        x, y, z = d
        
        # Degree 0
        if len(self.sh_coeffs) >= 3:
            result = C0 * self.sh_coeffs[:3]
        else:
            return self.color
        
        # Degree 1
        if len(self.sh_coeffs) >= 12:
            result += -C1 * y * self.sh_coeffs[3:6]
            result += C1 * z * self.sh_coeffs[6:9]
            result += -C1 * x * self.sh_coeffs[9:12]
        
        # Clamp to valid range
        return np.clip(result, 0, 1)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "position": self.position.tolist(),
            "scale": self.scale.tolist(),
            "rotation": self.rotation.tolist(),
            "opacity": float(self.opacity),
            "color": self.color.tolist() if self.color is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Gaussian3D':
        """Create from dictionary."""
        return cls(
            position=np.array(data["position"]),
            scale=np.array(data["scale"]),
            rotation=np.array(data["rotation"]),
            opacity=data.get("opacity", 1.0),
            color=np.array(data["color"]) if data.get("color") else None
        )


class GaussianCloud:
    """
    Collection of 3D Gaussians representing a scene.
    
    Optimized for:
    - Fast rendering via tile-based rasterization
    - Memory-efficient storage
    - GPU acceleration (when available)
    """
    
    def __init__(self, gaussians: List[Gaussian3D] = None):
        """Initialize with optional list of Gaussians."""
        self.gaussians = gaussians or []
        
        # Cached numpy arrays for fast access
        self._positions = None
        self._scales = None
        self._rotations = None
        self._opacities = None
        self._colors = None
        
        self._cache_valid = False
    
    def add_gaussian(self, gaussian: Gaussian3D):
        """Add a Gaussian to the cloud."""
        self.gaussians.append(gaussian)
        self._cache_valid = False
    
    def remove_gaussian(self, index: int):
        """Remove Gaussian by index."""
        del self.gaussians[index]
        self._cache_valid = False
    
    def __len__(self) -> int:
        return len(self.gaussians)
    
    def __getitem__(self, index: int) -> Gaussian3D:
        return self.gaussians[index]
    
    # ============================================
    # BATCH OPERATIONS
    # ============================================
    
    def _build_cache(self):
        """Build numpy array cache for fast access."""
        if self._cache_valid:
            return
        
        n = len(self.gaussians)
        if n == 0:
            self._positions = np.zeros((0, 3), dtype=np.float32)
            self._scales = np.zeros((0, 3), dtype=np.float32)
            self._rotations = np.zeros((0, 4), dtype=np.float32)
            self._opacities = np.zeros((0,), dtype=np.float32)
            self._colors = np.zeros((0, 3), dtype=np.float32)
            self._cache_valid = True
            return
        
        self._positions = np.stack([g.position for g in self.gaussians])
        self._scales = np.stack([g.scale for g in self.gaussians])
        self._rotations = np.stack([g.rotation for g in self.gaussians])
        self._opacities = np.array([g.opacity for g in self.gaussians])
        self._colors = np.stack([
            g.color if g.color is not None else np.array([1, 1, 1])
            for g in self.gaussians
        ])
        
        self._cache_valid = True
    
    @property
    def positions(self) -> np.ndarray:
        """Get all positions as (N, 3) array."""
        self._build_cache()
        return self._positions
    
    @property
    def scales(self) -> np.ndarray:
        """Get all scales as (N, 3) array."""
        self._build_cache()
        return self._scales
    
    @property
    def rotations(self) -> np.ndarray:
        """Get all rotations as (N, 4) array."""
        self._build_cache()
        return self._rotations
    
    @property
    def opacities(self) -> np.ndarray:
        """Get all opacities as (N,) array."""
        self._build_cache()
        return self._opacities
    
    @property
    def colors(self) -> np.ndarray:
        """Get all colors as (N, 3) array."""
        self._build_cache()
        return self._colors
    
    # ============================================
    # SCENE OPERATIONS
    # ============================================
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box."""
        if len(self.gaussians) == 0:
            return np.zeros(3), np.zeros(3)
        
        positions = self.positions
        return positions.min(axis=0), positions.max(axis=0)
    
    def get_center(self) -> np.ndarray:
        """Get scene center."""
        min_pt, max_pt = self.get_bounding_box()
        return (min_pt + max_pt) / 2
    
    def normalize_scene(self):
        """Normalize scene to unit cube centered at origin."""
        min_pt, max_pt = self.get_bounding_box()
        center = (min_pt + max_pt) / 2
        scale = np.max(max_pt - min_pt)
        
        if scale > 0:
            for g in self.gaussians:
                g.position = (g.position - center) / scale
                g.scale = g.scale / scale
        
        self._cache_valid = False
    
    def prune_low_opacity(self, threshold: float = 0.01):
        """Remove Gaussians with opacity below threshold."""
        self.gaussians = [g for g in self.gaussians if g.opacity >= threshold]
        self._cache_valid = False
        logger.info(f"Pruned to {len(self.gaussians)} Gaussians")
    
    def prune_large_gaussians(self, max_scale: float = 0.5):
        """Remove Gaussians that are too large (usually floaters)."""
        self.gaussians = [
            g for g in self.gaussians 
            if np.max(g.scale) <= max_scale
        ]
        self._cache_valid = False
    
    # ============================================
    # SERIALIZATION
    # ============================================
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "num_gaussians": len(self.gaussians),
            "gaussians": [g.to_dict() for g in self.gaussians]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'GaussianCloud':
        """Create from dictionary."""
        gaussians = [
            Gaussian3D.from_dict(g) 
            for g in data.get("gaussians", [])
        ]
        return cls(gaussians)
    
    def to_ply_data(self) -> bytes:
        """
        Export to PLY format (compatible with standard viewers).
        
        Returns:
            PLY file as bytes
        """
        from io import BytesIO
        
        n = len(self.gaussians)
        if n == 0:
            return b""
        
        # Build header
        header = f"""ply
format binary_little_endian 1.0
element vertex {n}
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float opacity
property float red
property float green
property float blue
end_header
"""
        
        # Build data
        self._build_cache()
        
        buffer = BytesIO()
        buffer.write(header.encode('ascii'))
        
        for i in range(n):
            # Position
            buffer.write(self._positions[i].astype(np.float32).tobytes())
            # Scale
            buffer.write(self._scales[i].astype(np.float32).tobytes())
            # Rotation
            buffer.write(self._rotations[i].astype(np.float32).tobytes())
            # Opacity
            buffer.write(np.array([self._opacities[i]], dtype=np.float32).tobytes())
            # Color
            buffer.write(self._colors[i].astype(np.float32).tobytes())
        
        return buffer.getvalue()
    
    @classmethod
    def from_ply_file(cls, filepath: str) -> 'GaussianCloud':
        """Load from PLY file."""
        try:
            from plyfile import PlyData
            
            plydata = PlyData.read(filepath)
            vertex = plydata['vertex']
            
            gaussians = []
            for i in range(len(vertex)):
                g = Gaussian3D(
                    position=np.array([vertex['x'][i], vertex['y'][i], vertex['z'][i]]),
                    scale=np.array([
                        vertex['scale_0'][i], 
                        vertex['scale_1'][i], 
                        vertex['scale_2'][i]
                    ]),
                    rotation=np.array([
                        vertex['rot_0'][i],
                        vertex['rot_1'][i],
                        vertex['rot_2'][i],
                        vertex['rot_3'][i]
                    ]),
                    opacity=vertex['opacity'][i],
                    color=np.array([
                        vertex['red'][i],
                        vertex['green'][i],
                        vertex['blue'][i]
                    ])
                )
                gaussians.append(g)
            
            return cls(gaussians)
            
        except ImportError:
            logger.warning("plyfile not installed, using fallback")
            return cls()
