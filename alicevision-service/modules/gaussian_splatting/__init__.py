"""
🌟 3D Gaussian Splatting Engine
Photorealistic volumetric rendering for fashion items

Features:
- Camera pose estimation via SfM
- Gaussian primitive optimization
- Real-time WebGL rendering
- Export to .ply / .spz formats
- Fabric texture preservation (velvet, lace, fur)
"""

from .gaussian import Gaussian3D, GaussianCloud
from .optimizer import GaussianOptimizer
from .rasterizer import GaussianRasterizer
from .sfm_processor import SfMProcessor
from .scene_exporter import SceneExporter

__all__ = [
    'Gaussian3D',
    'GaussianCloud',
    'GaussianOptimizer',
    'GaussianRasterizer',
    'SfMProcessor',
    'SceneExporter',
    'create_3dgs_from_video',
    'render_gaussian_scene'
]


def create_3dgs_from_video(video_frames: list, output_path: str = None):
    """
    Create 3D Gaussian Splatting scene from video frames.
    
    Args:
        video_frames: List of base64-encoded frames or file paths
        output_path: Optional output path for .ply file
        
    Returns:
        GaussianCloud object
    """
    from .optimizer import GaussianOptimizer
    
    optimizer = GaussianOptimizer()
    scene = optimizer.optimize_from_frames(video_frames)
    
    if output_path:
        from .scene_exporter import SceneExporter
        exporter = SceneExporter()
        exporter.export_ply(scene, output_path)
    
    return scene


def render_gaussian_scene(scene, camera_params: dict):
    """
    Render Gaussian scene from specific camera viewpoint.
    
    Args:
        scene: GaussianCloud object
        camera_params: Camera parameters (position, rotation, fov)
        
    Returns:
        Rendered image as numpy array
    """
    from .rasterizer import GaussianRasterizer
    
    rasterizer = GaussianRasterizer()
    return rasterizer.render(scene, camera_params)
