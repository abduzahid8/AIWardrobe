"""
💾 Scene Exporter
Export Gaussian scenes to various formats

Supported formats:
- .ply (Point Cloud Library format)
- .spz (Compressed Gaussian format)
- .json (Web-friendly format)
- .html (Self-contained WebGL viewer)
"""

import numpy as np
import json
import os
from typing import Dict, Optional
import logging
from io import BytesIO

from .gaussian import GaussianCloud

logger = logging.getLogger(__name__)


class SceneExporter:
    """
    Export Gaussian scenes to various formats for viewing and sharing.
    """
    
    def export_ply(
        self,
        scene: GaussianCloud,
        output_path: str,
        include_sh: bool = False
    ):
        """
        Export to PLY format (compatible with most 3D viewers).
        
        Args:
            scene: GaussianCloud to export
            output_path: Output file path
            include_sh: Include Spherical Harmonics coefficients
        """
        ply_data = scene.to_ply_data()
        
        with open(output_path, 'wb') as f:
            f.write(ply_data)
        
        logger.info(f"Exported {len(scene)} Gaussians to {output_path}")
    
    def export_json(
        self,
        scene: GaussianCloud,
        output_path: str
    ):
        """
        Export to JSON format (for web viewers).
        """
        data = {
            "format": "gaussian_splatting",
            "version": "1.0",
            "numGaussians": len(scene),
            "positions": scene.positions.tolist(),
            "scales": scene.scales.tolist(),
            "rotations": scene.rotations.tolist(),
            "opacities": scene.opacities.tolist(),
            "colors": scene.colors.tolist()
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Exported to JSON: {output_path}")
    
    def export_spz(
        self,
        scene: GaussianCloud,
        output_path: str
    ):
        """
        Export to SPZ format (compressed Gaussian format).
        
        SPZ uses:
        - Quantization for positions/colors
        - Huffman coding for compression
        """
        import struct
        import zlib
        
        n = len(scene)
        
        # Pack data
        buffer = BytesIO()
        
        # Header
        buffer.write(b'SPZ1')  # Magic
        buffer.write(struct.pack('<I', n))  # Num Gaussians
        
        # Positions (float16 for compression)
        positions = scene.positions.astype(np.float16)
        buffer.write(positions.tobytes())
        
        # Scales (float16)
        scales = scene.scales.astype(np.float16)
        buffer.write(scales.tobytes())
        
        # Rotations (float16)
        rotations = scene.rotations.astype(np.float16)
        buffer.write(rotations.tobytes())
        
        # Opacities (uint8, quantized)
        opacities = (scene.opacities * 255).astype(np.uint8)
        buffer.write(opacities.tobytes())
        
        # Colors (uint8)
        colors = (scene.colors * 255).astype(np.uint8)
        buffer.write(colors.tobytes())
        
        # Compress
        raw_data = buffer.getvalue()
        compressed = zlib.compress(raw_data, level=9)
        
        with open(output_path, 'wb') as f:
            f.write(compressed)
        
        ratio = len(compressed) / len(raw_data) * 100
        logger.info(f"Exported to SPZ: {output_path} ({ratio:.1f}% of raw size)")
    
    def export_webgl_viewer(
        self,
        scene: GaussianCloud,
        output_path: str,
        title: str = "3D Gaussian Splatting Viewer"
    ):
        """
        Export self-contained WebGL viewer HTML.
        
        The HTML file contains:
        - Embedded Gaussian data
        - Three.js renderer
        - Orbit controls
        """
        # Serialize scene data
        scene_json = json.dumps({
            "positions": scene.positions.tolist(),
            "scales": scene.scales.tolist(),
            "rotations": scene.rotations.tolist(),
            "opacities": scene.opacities.tolist(),
            "colors": scene.colors.tolist()
        })
        
        html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        * {{ margin: 0; padding: 0; }}
        body {{ overflow: hidden; background: #1a1a1a; }}
        canvas {{ display: block; }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            color: white;
            font-family: Arial, sans-serif;
            font-size: 14px;
            background: rgba(0,0,0,0.5);
            padding: 10px;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div id="info">
        <strong>3D Gaussian Splatting Viewer</strong><br>
        {len(scene)} Gaussians<br>
        <small>Drag to rotate, scroll to zoom</small>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/examples/js/controls/OrbitControls.js"></script>
    <script>
        // Gaussian scene data
        const sceneData = {scene_json};
        
        // Setup Three.js
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(0x1a1a1a);
        
        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.01, 100);
        camera.position.set(0, 0, 2);
        
        const renderer = new THREE.WebGLRenderer({{ antialias: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        document.body.appendChild(renderer.domElement);
        
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;
        
        // Create point cloud from Gaussians
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(sceneData.positions.flat());
        const colors = new Float32Array(sceneData.colors.flat());
        const sizes = new Float32Array(sceneData.scales.map(s => Math.max(...s) * 50));
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
        
        // Gaussian splat shader
        const material = new THREE.ShaderMaterial({{
            uniforms: {{
                pointSize: {{ value: 3.0 }}
            }},
            vertexShader: `
                attribute float size;
                varying vec3 vColor;
                void main() {{
                    vColor = color;
                    vec4 mvPosition = modelViewMatrix * vec4(position, 1.0);
                    gl_Position = projectionMatrix * mvPosition;
                    gl_PointSize = size * (300.0 / -mvPosition.z);
                }}
            `,
            fragmentShader: `
                varying vec3 vColor;
                void main() {{
                    vec2 cxy = 2.0 * gl_PointCoord - 1.0;
                    float r = dot(cxy, cxy);
                    if (r > 1.0) discard;
                    float alpha = exp(-r * 2.0);
                    gl_FragColor = vec4(vColor, alpha);
                }}
            `,
            transparent: true,
            vertexColors: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending
        }});
        
        const points = new THREE.Points(geometry, material);
        scene.add(points);
        
        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
        
        // Handle resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>'''
        
        with open(output_path, 'w') as f:
            f.write(html)
        
        logger.info(f"Exported WebGL viewer: {output_path}")
    
    def export_all(
        self,
        scene: GaussianCloud,
        output_dir: str,
        name: str = "scene"
    ) -> Dict[str, str]:
        """
        Export to all formats.
        
        Returns:
            Dictionary of format -> file path
        """
        os.makedirs(output_dir, exist_ok=True)
        
        paths = {}
        
        # PLY
        ply_path = os.path.join(output_dir, f"{name}.ply")
        self.export_ply(scene, ply_path)
        paths["ply"] = ply_path
        
        # JSON
        json_path = os.path.join(output_dir, f"{name}.json")
        self.export_json(scene, json_path)
        paths["json"] = json_path
        
        # SPZ
        spz_path = os.path.join(output_dir, f"{name}.spz")
        self.export_spz(scene, spz_path)
        paths["spz"] = spz_path
        
        # WebGL viewer
        html_path = os.path.join(output_dir, f"{name}_viewer.html")
        self.export_webgl_viewer(scene, html_path)
        paths["viewer"] = html_path
        
        return paths
