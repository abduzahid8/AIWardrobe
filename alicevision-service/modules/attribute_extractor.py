"""
Comprehensive Clothing Attribute Extraction
Color, pattern, material, and texture analysis
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image
from sklearn.cluster import KMeans
import logging
from dataclasses import dataclass
import base64

logger = logging.getLogger(__name__)


# Extended FASHION-SPECIFIC color names mapping (100+ colors)
COLOR_NAMES = {
    # === BLACKS & WHITES ===
    (0, 0, 0): "black",
    (25, 25, 25): "jet black",
    (50, 50, 50): "charcoal",
    (255, 255, 255): "white",
    (250, 249, 246): "off-white",
    (255, 253, 245): "cream",
    (255, 255, 240): "ivory",
    (245, 245, 240): "eggshell",
    
    # === GRAYS ===
    (70, 70, 70): "dark gray",
    (105, 105, 105): "dim gray",
    (128, 128, 128): "gray",
    (169, 169, 169): "medium gray",
    (192, 192, 192): "silver",
    (211, 211, 211): "light gray",
    (220, 220, 220): "gainsboro",
    (112, 128, 144): "slate gray",
    
    # === BROWNS & TANS ===
    (139, 69, 19): "saddle brown",
    (160, 82, 45): "sienna",
    (165, 42, 42): "brown",
    (101, 67, 33): "dark brown",
    (150, 75, 0): "chocolate",
    (193, 154, 107): "camel",
    (154, 70, 20): "cognac",
    (210, 180, 140): "tan",
    (245, 245, 220): "beige",
    (188, 143, 143): "rosy brown",
    (210, 105, 30): "rust",
    (72, 60, 50): "taupe",
    (194, 178, 128): "sand",
    
    # === DENIM COLORS ===
    (21, 96, 189): "denim blue",
    (40, 75, 120): "dark wash denim",
    (90, 130, 180): "medium wash denim",
    (140, 170, 210): "light wash denim",
    (30, 50, 80): "raw denim",
    (80, 80, 95): "black denim",
    
    # === REDS ===
    (255, 0, 0): "red",
    (220, 20, 60): "crimson",
    (178, 34, 34): "firebrick",
    (139, 0, 0): "dark red",
    (128, 0, 32): "burgundy",
    (128, 0, 0): "maroon",
    (114, 47, 55): "wine",
    (222, 49, 99): "cherry",
    (255, 99, 71): "tomato red",
    (255, 127, 80): "coral",
    (183, 65, 14): "rust red",
    (226, 114, 91): "terracotta",
    
    # === PINKS ===
    (255, 192, 203): "pink",
    (255, 182, 193): "light pink",
    (255, 105, 180): "hot pink",
    (255, 20, 147): "deep pink",
    (219, 112, 147): "dusty rose",
    (194, 129, 140): "mauve",
    (199, 21, 133): "magenta",
    (230, 190, 180): "blush",
    (255, 111, 255): "bright pink",
    
    # === ORANGES ===
    (255, 165, 0): "orange",
    (255, 140, 0): "dark orange",
    (255, 69, 0): "red orange",
    (255, 200, 150): "peach",
    (251, 206, 177): "apricot",
    (255, 127, 80): "coral orange",
    (204, 85, 0): "burnt orange",
    (255, 204, 0): "tangerine",
    
    # === YELLOWS ===
    (255, 255, 0): "yellow",
    (255, 215, 0): "gold",
    (255, 219, 88): "mustard",
    (238, 232, 170): "pale gold",
    (240, 230, 140): "khaki",
    (255, 255, 224): "pale yellow",
    (255, 255, 191): "butter",
    (255, 247, 0): "lemon",
    
    # === GREENS ===
    (0, 128, 0): "green",
    (0, 100, 0): "dark green",
    (34, 139, 34): "forest green",
    (53, 94, 59): "hunter green",
    (128, 128, 0): "olive",
    (75, 83, 32): "army green",
    (188, 184, 138): "sage",
    (189, 252, 201): "mint",
    (80, 200, 120): "emerald",
    (0, 128, 128): "teal",
    (50, 205, 50): "lime",
    (144, 238, 144): "light green",
    
    # === BLUES ===
    (0, 0, 255): "blue",
    (0, 0, 128): "navy blue",
    (0, 0, 139): "dark blue",
    (25, 25, 112): "midnight blue",
    (0, 71, 171): "cobalt",
    (65, 105, 225): "royal blue",
    (100, 149, 237): "cornflower blue",
    (70, 130, 180): "steel blue",
    (135, 206, 235): "sky blue",
    (173, 216, 230): "light blue",
    (137, 207, 240): "baby blue",
    (176, 224, 230): "powder blue",
    (64, 224, 208): "turquoise",
    (0, 255, 255): "cyan",
    
    # === PURPLES ===
    (128, 0, 128): "purple",
    (75, 0, 130): "indigo",
    (97, 64, 81): "eggplant",
    (142, 69, 133): "plum",
    (138, 43, 226): "violet",
    (230, 230, 250): "lavender",
    (200, 162, 200): "lilac",
    (186, 85, 211): "orchid",
    (148, 0, 211): "dark violet",
}


@dataclass
class ColorInfo:
    """Color information"""
    name: str
    rgb: Tuple[int, int, int]
    hex: str
    percentage: float


@dataclass
class PatternInfo:
    """Pattern detection result"""
    type: str
    confidence: float
    complexity: float
    description: str


@dataclass
class MaterialInfo:
    """Material/texture prediction"""
    material: str
    confidence: float
    texture: str
    features: Dict[str, float]


class AttributeExtractor:
    """
    Advanced clothing attribute extraction including:
    - Color analysis (dominant colors, palettes)
    - Pattern recognition (stripes, dots, floral, etc.)
    - Material/texture prediction
    - Style classification
    - Clothing details (collar, sleeves, closure, fit)
    """
    
    def __init__(self):
        self.pattern_types = [
            "solid", "striped", "plaid", "checkered", "polka dot",
            "floral", "geometric", "animal print", "camouflage",
            "tie-dye", "gradient", "abstract", "printed text"
        ]
        
        self.materials = [
            "cotton", "silk", "denim", "leather", "wool",
            "polyester", "linen", "velvet", "satin", "knit"
        ]
        
        # === NEW: Detailed clothing attributes ===
        
        # Collar/Neckline types
        self.collar_types = [
            "crew neck", "v-neck", "scoop neck", "turtleneck", "mock neck",
            "polo collar", "button-down collar", "spread collar", "mandarin collar",
            "hooded", "cowl neck", "boat neck", "square neck", "halter",
            "off-shoulder", "one-shoulder", "strapless", "peter pan collar"
        ]
        
        # Sleeve types
        self.sleeve_types = [
            "short sleeve", "long sleeve", "3/4 sleeve", "sleeveless",
            "cap sleeve", "raglan sleeve", "puff sleeve", "bell sleeve",
            "bishop sleeve", "dolman sleeve", "roll-up sleeve", "cuffed"
        ]
        
        # Closure/Opening types
        self.closure_types = [
            "pullover", "full zip", "half zip", "quarter zip",
            "button-up", "button-down", "snap buttons", "toggle buttons",
            "drawstring", "hook and eye", "velcro", "wrap", "tie front",
            "hidden placket", "double-breasted", "single-breasted"
        ]
        
        # Fit types
        self.fit_types = [
            "slim fit", "regular fit", "relaxed fit", "oversized",
            "tailored", "loose", "fitted", "cropped", "boxy"
        ]
        
        # Length types (for various clothing)
        self.length_types = [
            "cropped", "waist-length", "hip-length", "thigh-length",
            "knee-length", "midi", "maxi", "full-length", "ankle-length"
        ]
        
        # Pocket types
        self.pocket_types = [
            "no pockets", "side pockets", "front pockets", "back pockets",
            "kangaroo pocket", "chest pocket", "flap pockets", "zippered pockets",
            "patch pockets", "welt pockets", "cargo pockets"
        ]
    
    def extract_dominant_colors(
        self, 
        image: np.ndarray, 
        n_colors: int = 5,
        sample_size: int = 10000
    ) -> List[ColorInfo]:
        """
        Extract dominant colors using K-means clustering
        
        Args:
            image: BGR image
            n_colors: Number of dominant colors to extract
            sample_size: Number of pixels to sample
            
        Returns:
            List of ColorInfo sorted by percentage
        """
        # Convert to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Reshape to pixels
        pixels = rgb_image.reshape(-1, 3)
        
        # Sample if too large
        if len(pixels) > sample_size:
            indices = np.random.choice(len(pixels), sample_size, replace=False)
            pixels = pixels[indices]
        
        # Remove very dark and very bright pixels (likely background)
        mask = ~((pixels.max(axis=1) < 30) | (pixels.min(axis=1) > 240))
        pixels = pixels[mask]
        
        if len(pixels) < n_colors:
            return []
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Get colors and counts
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        counts = np.bincount(labels)
        
        # Sort by frequency
        sorted_indices = np.argsort(-counts)
        
        color_infos = []
        total = counts.sum()
        
        for idx in sorted_indices:
            rgb = tuple(colors[idx])
            percentage = float(counts[idx]) / total * 100
            
            color_name = self._get_color_name(rgb)
            hex_code = "#{:02x}{:02x}{:02x}".format(*rgb)
            
            color_infos.append(ColorInfo(
                name=color_name,
                rgb=rgb,
                hex=hex_code,
                percentage=percentage
            ))
        
        return color_infos
    
    def _get_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """Find closest color name"""
        min_distance = float('inf')
        closest_name = "unknown"
        
        for color_rgb, name in COLOR_NAMES.items():
            # Euclidean distance in RGB space
            distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(rgb, color_rgb)))
            
            if distance < min_distance:
                min_distance = distance
                closest_name = name
        
        return closest_name
    
    def detect_pattern(self, image: np.ndarray) -> PatternInfo:
        """
        Detect clothing pattern type using computer vision
        
        Args:
            image: BGR image
            
        Returns:
            PatternInfo with pattern type and confidence
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate features
        features = {}
        
        # 1. Edge density (high for patterns)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # 2. Frequency analysis
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = np.log(np.abs(fshift) + 1)
        features['frequency_variance'] = np.var(magnitude_spectrum)
        
        # 3. Texture uniformity
        features['std_dev'] = np.std(gray)
        features['mean'] = np.mean(gray)
        
        # 4. Color variation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features['hue_variance'] = np.var(hsv[:, :, 0])
        features['saturation_mean'] = np.mean(hsv[:, :, 1])
        
        # Pattern classification based on features
        pattern_type, confidence = self._classify_pattern(features)
        
        # Calculate complexity
        complexity = self._calculate_pattern_complexity(features)
        
        # Generate description
        description = self._generate_pattern_description(pattern_type, complexity, features)
        
        return PatternInfo(
            type=pattern_type,
            confidence=confidence,
            complexity=complexity,
            description=description
        )
    
    def _classify_pattern(self, features: Dict[str, float]) -> Tuple[str, float]:
        """Classify pattern based on features"""
        edge_density = features['edge_density']
        freq_var = features['frequency_variance']
        std_dev = features['std_dev']
        hue_var = features['hue_variance']
        
        # Solid/minimal pattern
        if edge_density < 0.1 and std_dev < 30:
            return "solid", 0.95
        
        # Striped (high frequency in one direction)
        if edge_density > 0.15 and freq_var > 5000:
            return "striped", 0.75
        
        # Polka dot (circular patterns)
        if 0.1 < edge_density < 0.2 and freq_var < 3000:
            return "polka dot", 0.65
        
        # Floral/complex (high color variation)
        if hue_var > 500 and edge_density > 0.15:
            return "floral", 0.70
        
        # Checkered/plaid
        if edge_density > 0.20 and 3000 < freq_var < 6000:
            return "checkered", 0.70
        
        # Geometric
        if edge_density > 0.15 and std_dev > 40:
            return "geometric", 0.65
        
        # Default
        if edge_density < 0.12:
            return "solid", 0.80
        else:
            return "printed", 0.60
    
    def _calculate_pattern_complexity(self, features: Dict[str, float]) -> float:
        """Calculate pattern complexity score 0-1"""
        # Normalize and combine features
        edge_norm = min(features['edge_density'] / 0.3, 1.0)
        freq_norm = min(features['frequency_variance'] / 10000, 1.0)
        std_norm = min(features['std_dev'] / 100, 1.0)
        
        complexity = (edge_norm * 0.4 + freq_norm * 0.3 + std_norm * 0.3)
        return float(complexity)
    
    def _generate_pattern_description(
        self, 
        pattern_type: str, 
        complexity: float,
        features: Dict[str, float]
    ) -> str:
        """Generate human-readable pattern description"""
        complexity_desc = "simple" if complexity < 0.3 else "moderate" if complexity < 0.7 else "complex"
        
        if pattern_type == "solid":
            return f"Solid color with minimal pattern"
        elif pattern_type == "striped":
            return f"{complexity_desc.capitalize()} striped pattern"
        elif pattern_type == "polka dot":
            return f"{complexity_desc.capitalize()} polka dot pattern"
        elif pattern_type == "floral":
            return f"{complexity_desc.capitalize()} floral print"
        elif pattern_type == "checkered":
            return f"{complexity_desc.capitalize()} checkered pattern"
        else:
            return f"{complexity_desc.capitalize()} {pattern_type} pattern"
    
    def predict_material(self, image: np.ndarray) -> MaterialInfo:
        """
        Predict material/texture from image features
        
        Note: This is a heuristic-based approach. For production,
        consider fine-tuning a CNN on material datasets.
        
        Args:
            image: BGR image
            
        Returns:
            MaterialInfo with material prediction
        """
        # Extract texture features
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        features = {}
        
        # 1. Smoothness (low variance = smooth materials)
        features['smoothness'] = 1.0 / (1.0 + np.std(gray) / 50)
        
        # 2. Shininess (detect highlights)
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        features['shininess'] = np.sum(bright > 0) / bright.size
        
        # 3. Texture energy (Gabor filters would be better)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['texture_energy'] = np.var(laplacian) / 1000
        
        # 4. Color saturation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features['saturation'] = np.mean(hsv[:, :, 1]) / 255
        
        # Simple rule-based classification
        material, confidence, texture = self._classify_material(features)
        
        return MaterialInfo(
            material=material,
            confidence=confidence,
            texture=texture,
            features=features
        )
    
    def _classify_material(self, features: Dict[str, float]) -> Tuple[str, float, str]:
        """Classify material based on features"""
        smoothness = features['smoothness']
        shininess = features['shininess']
        texture_energy = features['texture_energy']
        saturation = features['saturation']
        
        # Silk/satin (smooth + shiny)
        if smoothness > 0.7 and shininess > 0.05:
            return "silk/satin", 0.70, "smooth"
        
        # Leather (smooth, medium shine, high saturation)
        if smoothness > 0.6 and shininess > 0.02 and saturation > 0.4:
            return "leather", 0.65, "smooth"
        
        # Denim (textured, low shine)
        if texture_energy > 1.5 and shininess < 0.02 and smoothness < 0.5:
            return "denim", 0.70, "rough"
        
        # Wool/knit (textured, low shine)
        if texture_energy > 2.0 and smoothness < 0.4:
            return "wool/knit", 0.65, "knitted"
        
        # Cotton (medium texture, low shine)
        if smoothness > 0.4 and shininess < 0.03:
            return "cotton", 0.75, "woven"
        
        # Polyester (smooth, slight shine)
        if smoothness > 0.5 and 0.01 < shininess < 0.04:
            return "polyester", 0.60, "smooth"
        
        # Default
        return "cotton blend", 0.50, "woven"
    
    def extract_all_attributes(self, image: np.ndarray) -> Dict:
        """
        Extract all attributes from clothing image
        
        Args:
            image: BGR image
            
        Returns:
            Dictionary with all attributes
        """
        import time
        start_time = time.time()
        
        # Extract colors
        colors = self.extract_dominant_colors(image, n_colors=5)
        
        # Detect pattern
        pattern = self.detect_pattern(image)
        
        # Predict material
        material = self.predict_material(image)
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "colors": [
                {
                    "name": c.name,
                    "rgb": c.rgb,
                    "hex": c.hex,
                    "percentage": round(c.percentage, 2)
                }
                for c in colors
            ],
            "primaryColor": colors[0].name if colors else "unknown",
            "colorPalette": [c.hex for c in colors[:3]],
            "pattern": {
                "type": pattern.type,
                "confidence": round(pattern.confidence, 3),
                "complexity": round(pattern.complexity, 3),
                "description": pattern.description
            },
            "material": {
                "type": material.material,
                "confidence": round(material.confidence, 3),
                "texture": material.texture
            },
            "processingTimeMs": round(processing_time, 1)
        }


def extract_attributes_from_base64(image_base64: str) -> Dict:
    """
    Utility function for base64 image attribute extraction
    
    Args:
        image_base64: Base64-encoded image
        
    Returns:
        Attributes dictionary
    """
    # Remove data URL prefix
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]
    
    # Decode
    img_bytes = base64.b64decode(image_base64)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "Could not decode image"}
    
    # Extract attributes
    extractor = AttributeExtractor()
    return extractor.extract_all_attributes(image)
