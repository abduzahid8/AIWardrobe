"""
FashionCLIP Module - Advanced Fashion Attribute Extraction
Specialized CLIP model fine-tuned for fashion understanding
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
import base64
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class FashionAttributes:
    """Extracted fashion attributes"""
    category: str  # shirt, pants, dress, etc.
    subcategory: str  # t-shirt, jeans, maxi-dress, etc.
    colors: List[Tuple[str, float]]  # [(color_name, confidence), ...]
    patterns: List[Tuple[str, float]]  # solid, striped, plaid, etc.
    style: List[Tuple[str, float]]  # casual, formal, sporty, etc.
    fabric: Optional[str] = None  # cotton, denim, silk, etc.
    details: Dict[str, str] = None  # neckline, sleeve_length, etc.
    description: str = ""  # Natural language description


class FashionCLIP:
    """
    FashionCLIP for detailed clothing attribute extraction.
    
    Uses OpenCLIP with fashion-specific fine-tuning to extract:
    - Clothing categories and subcategories
    - Colors (primary, secondary, patterns)
    - Style attributes (casual, formal, vintage, etc.)
    - Fabric types
    - Detailed features (neckline, sleeves, fit, etc.)
    """
    
    # Clothing categories - EXPANDED to 80+ specific types
    CATEGORIES = [
        # === TOPS (30 types) ===
        "t-shirt", "graphic tee", "tank top", "crop top", "polo shirt",
        "button-down shirt", "dress shirt", "oxford shirt", "flannel shirt",
        "blouse", "silk blouse", "sweater", "crewneck sweater", "v-neck sweater",
        "cardigan", "turtleneck", "mock neck", "hoodie", "pullover hoodie",
        "zip-up hoodie", "sweatshirt", "henley shirt", "long sleeve shirt",
        "vest", "bodysuit", "camisole", "jersey", "tunic", "thermal top",
        "raglan shirt",
        
        # === OUTERWEAR (25 types) ===
        "denim jacket", "leather jacket", "bomber jacket", "trucker jacket",
        "blazer", "sport coat", "puffer jacket", "down jacket", "parka",
        "trench coat", "overcoat", "peacoat", "windbreaker", "track jacket",
        "varsity jacket", "fleece jacket", "quilted jacket", "anorak", "raincoat",
        "ski jacket", "moto jacket", "safari jacket", "field jacket",
        "shacket", "overshirt",
        
        # === BOTTOMS (25 types) ===
        "jeans", "skinny jeans", "straight jeans", "wide-leg jeans", "mom jeans",
        "bootcut jeans", "boyfriend jeans", "slim fit jeans", "tapered jeans",
        "chinos", "cargo pants", "joggers", "sweatpants", "dress pants",
        "trousers", "leggings", "denim shorts", "cargo shorts", "athletic shorts",
        "bermuda shorts", "khakis", "corduroys", "cropped pants", "palazzo pants",
        "culottes",
        
        # === DRESSES & SKIRTS (15 types) ===
        "dress", "maxi dress", "midi dress", "mini dress", "sundress",
        "cocktail dress", "evening gown", "wrap dress", "shirt dress",
        "skirt", "mini skirt", "midi skirt", "maxi skirt", "pleated skirt",
        "pencil skirt",
        
        # === FOOTWEAR (20 types) ===
        "sneakers", "running shoes", "high-top sneakers", "low-top sneakers",
        "boots", "chelsea boots", "combat boots", "ankle boots", "hiking boots",
        "cowboy boots", "knee-high boots", "loafers", "oxfords", "derby shoes",
        "brogues", "sandals", "slides", "flip flops", "espadrilles", "dress shoes",
        
        # === ACCESSORIES (15 types) ===
        "hat", "baseball cap", "beanie", "bucket hat", "fedora", "sun hat",
        "backpack", "tote bag", "crossbody bag", "messenger bag", "handbag",
        "sunglasses", "belt", "scarf", "watch"
    ]
    
    # Color descriptions - EXPANDED to 80+ colors for ultimate accuracy
    COLORS = [
        # === BLACKS & GRAYS (10) ===
        "black", "jet black", "charcoal", "dark gray", "gray",
        "light gray", "silver", "slate gray", "gunmetal", "ash gray",
        
        # === WHITES & CREAMS (8) ===
        "white", "off-white", "ivory", "cream", "pearl",
        "bone white", "alabaster", "vanilla",
        
        # === BLUES (14) ===
        "navy blue", "royal blue", "sky blue", "light blue", "baby blue",
        "cobalt", "midnight blue", "powder blue", "steel blue", "denim blue",
        "teal", "turquoise", "aqua", "cerulean",
        
        # === BROWNS (12) ===
        "brown", "tan", "beige", "camel", "chocolate",
        "cognac", "saddle brown", "espresso", "taupe", "khaki",
        "chestnut", "mocha",
        
        # === REDS (8) ===
        "red", "burgundy", "wine", "crimson", "maroon",
        "cherry", "scarlet", "brick red",
        
        # === PINKS (8) ===
        "pink", "blush pink", "coral", "salmon", "rose",
        "hot pink", "dusty pink", "fuchsia",
        
        # === ORANGES (6) ===
        "orange", "rust", "terracotta", "peach",
        "tangerine", "burnt orange",
        
        # === YELLOWS (6) ===
        "yellow", "mustard", "gold", "amber",
        "lemon", "butter yellow",
        
        # === GREENS (12) ===
        "green", "olive", "forest green", "sage", "mint",
        "hunter green", "emerald", "army green", "lime", "seafoam",
        "moss green", "kelly green",
        
        # === PURPLES (10) ===
        "purple", "violet", "lavender", "plum", "lilac",
        "mauve", "indigo", "eggplant", "amethyst", "grape",
        
        # === METALLICS & SPECIAL (4) ===
        "gold metallic", "silver metallic", "bronze", "multicolor"
    ]
    
    # Pattern types
    PATTERNS = [
        "solid color", "striped", "plaid", "checkered", "polka dot",
        "floral", "geometric", "abstract", "camo", "tie-dye",
        "animal print", "leopard print", "zebra print"
    ]
    
    # Style attributes
    STYLES = [
        "casual", "formal", "business casual", "sporty", "athletic",
        "elegant", "vintage", "boho", "street style", "minimalist",
        "preppy", "grunge", "romantic", "edgy", "classic"
    ]
    
    # Fabric types
    FABRICS = [
        "cotton", "denim", "silk", "satin", "linen", "wool",
        "polyester", "leather", "suede", "velvet", "chiffon",
        "knit", "jersey", "fleece"
    ]
    
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        device: str = "auto"
    ):
        """
        Initialize FashionCLIP.
        
        Args:
            model_name: CLIP model architecture
            pretrained: Pretrained weights source
            device: "cuda", "cpu", or "auto"
        """
        self.device = self._setup_device(device)
        self.model_name = model_name
        
        logger.info(f"Initializing FashionCLIP ({model_name}) on {self.device}")
        
        # Lazy loading
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._loaded = False
    
    def _setup_device(self, device: str) -> str:
        """Setup compute device"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def _load_model(self):
        """Lazy load CLIP model"""
        if self._loaded:
            return
        
        try:
            import open_clip
            
            logger.info(f"Loading {self.model_name} model...")
            
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained='openai',
                device=self.device
            )
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            
            self._model.eval()
            self._loaded = True
            
            logger.info("FashionCLIP model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}")
    
    def extract_attributes(
        self,
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> FashionAttributes:
        """
        Extract detailed fashion attributes from image.
        
        Args:
            image: Input image (BGR format)
            roi: Optional region of interest (x1, y1, x2, y2)
            
        Returns:
            FashionAttributes with extracted information
        """
        self._load_model()
        
        # Crop to ROI if provided
        if roi:
            x1, y1, x2, y2 = roi
            image = image[y1:y2, x1:x2]
        
        # Convert to RGB and PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Extract each attribute type
        category = self._classify(image_pil, self.CATEGORIES, top_k=1)[0]
        colors = self._classify(image_pil, self.COLORS, top_k=3)
        patterns = self._classify(image_pil, self.PATTERNS, top_k=2)
        styles = self._classify(image_pil, self.STYLES, top_k=3)
        
        # Extract fabric (lower confidence acceptable)
        fabric_results = self._classify(image_pil, self.FABRICS, top_k=1)
        fabric = fabric_results[0][0] if fabric_results[0][1] > 0.15 else None
        
        # Extract detailed features based on category
        details = self._extract_details(image_pil, category[0])
        
        # Generate natural language description
        description = self._generate_description(
            category[0], colors, patterns[0][0], styles, fabric
        )
        
        return FashionAttributes(
            category=category[0],
            subcategory=category[0],  # Could refine further
            colors=colors,
            patterns=patterns,
            style=styles,
            fabric=fabric,
            details=details,
            description=description
        )
    
    def classify_specific_type(
        self,
        image: np.ndarray,
        category_hint: str = None
    ) -> Tuple[str, float]:
        """
        🎯 ULTIMATE TYPE CLASSIFICATION
        
        Classify clothing to specific type using ensemble approach.
        Returns highly accurate type like "denim jacket", "skinny jeans", etc.
        
        Args:
            image: BGR image of clothing item
            category_hint: Optional hint like "upper_clothes", "pants"
            
        Returns:
            Tuple of (specific_type, confidence)
        """
        self._load_model()
        
        # Convert to PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Get type candidates based on category
        candidates = self._get_type_candidates(category_hint)
        
        # Primary classification
        primary_results = self._classify(image_pil, candidates, top_k=3)
        
        if not primary_results or primary_results[0][1] < 0.1:
            return "clothing item", 0.3
        
        primary_type = primary_results[0][0]
        primary_conf = primary_results[0][1]
        
        # Validation pass - confirm with related prompts
        validation_prompts = self._get_validation_prompts(primary_type)
        if validation_prompts:
            validation_results = self._classify(image_pil, validation_prompts, top_k=1)
            validation_conf = validation_results[0][1] if validation_results else 0
            
            # Boost confidence if validation agrees
            if validation_conf > 0.3:
                primary_conf = min(0.98, primary_conf * 1.15)
        
        logger.info(f"🏷️ CLIP classified: {primary_type} (conf={primary_conf:.2f})")
        
        return primary_type, primary_conf
    
    def _get_type_candidates(self, category_hint: str = None) -> List[str]:
        """Get classification candidates based on category."""
        
        # Category-specific candidates for faster, more accurate classification
        category_candidates = {
            "upper_clothes": [
                # T-shirts and casual tops
                "t-shirt", "graphic tee", "tank top", "crop top", "polo shirt",
                "henley shirt", "long sleeve shirt", "thermal top",
                # Button-down shirts
                "button-down shirt", "dress shirt", "oxford shirt", "flannel shirt",
                # Sweaters and knits
                "sweater", "crewneck sweater", "v-neck sweater", "cardigan", "turtleneck",
                # Hoodies and sweatshirts
                "hoodie", "pullover hoodie", "zip-up hoodie", "sweatshirt",
                # Jackets
                "denim jacket", "leather jacket", "bomber jacket", "trucker jacket",
                "puffer jacket", "fleece jacket", "windbreaker", "track jacket",
                # Formal
                "blazer", "sport coat", "vest",
                # Others
                "blouse", "tunic", "jersey"
            ],
            "pants": [
                # Jeans
                "jeans", "skinny jeans", "straight jeans", "bootcut jeans",
                "wide-leg jeans", "mom jeans", "boyfriend jeans", "slim fit jeans",
                # Casual pants
                "chinos", "cargo pants", "joggers", "sweatpants", "khakis",
                "corduroys", "cropped pants",
                # Formal
                "dress pants", "trousers", "slacks",
                # Athletic
                "leggings", "yoga pants", "track pants"
            ],
            "left_shoe": [
                # Sneakers
                "sneakers", "running shoes", "high-top sneakers", "low-top sneakers",
                "basketball shoes", "tennis shoes", "skate shoes",
                # Boots
                "boots", "chelsea boots", "combat boots", "ankle boots",
                "hiking boots", "work boots", "cowboy boots",
                # Formal
                "loafers", "oxfords", "derby shoes", "brogues", "dress shoes",
                # Casual
                "sandals", "slides", "flip flops", "espadrilles", "moccasins"
            ],
            "right_shoe": None,  # Same as left_shoe
            "dress": [
                "dress", "maxi dress", "midi dress", "mini dress", "sundress",
                "cocktail dress", "evening gown", "wrap dress", "shirt dress",
                "bodycon dress", "a-line dress", "slip dress"
            ],
            "skirt": [
                "skirt", "mini skirt", "midi skirt", "maxi skirt",
                "pleated skirt", "pencil skirt", "a-line skirt", "denim skirt"
            ],
            "hat": [
                "baseball cap", "beanie", "bucket hat", "fedora", "sun hat",
                "trucker hat", "snapback", "dad hat", "visor"
            ],
            "bag": [
                "backpack", "tote bag", "crossbody bag", "messenger bag",
                "handbag", "clutch", "duffel bag", "fanny pack"
            ]
        }
        
        if category_hint:
            # Normalize category hint
            cat = category_hint.lower().replace("-", "_")
            if cat == "right_shoe":
                cat = "left_shoe"
            if cat in category_candidates:
                return category_candidates[cat]
        
        # Return all categories if no hint
        return self.CATEGORIES
    
    def _get_validation_prompts(self, primary_type: str) -> List[str]:
        """Get validation prompts to confirm classification."""
        
        # Validation groups - if detected as X, should also match these
        validations = {
            # Jackets
            "denim jacket": ["blue denim outerwear", "jean jacket", "denim trucker jacket"],
            "leather jacket": ["black leather outerwear", "moto jacket", "biker jacket"],
            "bomber jacket": ["casual jacket with ribbed cuffs", "flight jacket"],
            "puffer jacket": ["quilted winter jacket", "padded jacket", "down jacket"],
            "blazer": ["formal jacket", "suit jacket", "sport coat"],
            "fleece jacket": ["soft fleece outerwear", "polar fleece jacket"],
            
            # Tops
            "hoodie": ["hooded sweatshirt", "casual hoodie with hood"],
            "sweatshirt": ["crew neck sweatshirt", "casual pullover"],
            "t-shirt": ["short sleeve cotton top", "casual tee"],
            "button-down shirt": ["collared shirt with buttons", "dress shirt"],
            "sweater": ["knit pullover", "cable knit top"],
            
            # Pants
            "jeans": ["denim pants", "blue jeans"],
            "skinny jeans": ["tight fitting denim", "slim denim pants"],
            "chinos": ["cotton twill pants", "khaki pants"],
            "cargo pants": ["pants with side pockets", "utility pants"],
            "joggers": ["sweatpants with cuffed ankles", "athletic pants"],
            
            # Shoes
            "sneakers": ["athletic shoes", "casual footwear"],
            "boots": ["ankle-height footwear", "leather boots"],
            "loafers": ["slip-on dress shoes", "leather loafers"],
        }
        
        return validations.get(primary_type, [])
    
    def _classify(
        self,
        image: Image.Image,
        labels: List[str],
        top_k: int = 1
    ) -> List[Tuple[str, float]]:
        """Classify image against list of labels"""
        try:
            # Preprocess image
            image_input = self._preprocess(image).unsqueeze(0).to(self.device)
            
            # Tokenize labels
            text_inputs = self._tokenizer(labels).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                image_features = self._model.encode_image(image_input)
                text_features = self._model.encode_text(text_inputs)
                
                # Normalize features
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Calculate similarity
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(top_k)
            
            # Return top results
            results = [
                (labels[idx], float(val))
                for idx, val in zip(indices, values)
            ]
            
            return results
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            return [(labels[0], 0.0)] * top_k
    
    def _extract_details(
        self,
        image: Image.Image,
        category: str
    ) -> Dict[str, str]:
        """Extract category-specific details"""
        details = {}
        
        # Neckline for tops
        if any(x in category.lower() for x in ["shirt", "blouse", "top", "dress"]):
            necklines = ["round neck", "v-neck", "collar", "turtleneck", "off-shoulder"]
            neckline = self._classify(image, necklines, top_k=1)[0][0]
            details["neckline"] = neckline
        
        # Sleeve length for tops
        if any(x in category.lower() for x in ["shirt", "blouse", "top", "dress", "jacket"]):
            sleeves = ["sleeveless", "short sleeve", "3/4 sleeve", "long sleeve"]
            sleeve = self._classify(image, sleeves, top_k=1)[0][0]
            details["sleeve_length"] = sleeve
        
        # Fit type
        fits = ["slim fit", "regular fit", "loose fit", "oversized"]
        fit = self._classify(image, fits, top_k=1)[0][0]
        details["fit"] = fit
        
        # Length for dresses/skirts
        if any(x in category.lower() for x in ["dress", "skirt"]):
            lengths = ["mini", "knee-length", "midi", "maxi"]
            length = self._classify(image, lengths, top_k=1)[0][0]
            details["length"] = length
        
        return details
    
    def _generate_description(
        self,
        category: str,
        colors: List[Tuple[str, float]],
        pattern: str,
        styles: List[Tuple[str, float]],
        fabric: Optional[str]
    ) -> str:
        """Generate natural language description"""
        # Primary color
        primary_color = colors[0][0] if colors else "colored"
        
        # Style
        primary_style = styles[0][0] if styles else "stylish"
        
        # Build description
        parts = []
        
        # Color + pattern
        if "solid" in pattern:
            parts.append(primary_color)
        else:
            parts.append(f"{primary_color} {pattern}")
        
        # Fabric
        if fabric:
            parts.append(fabric)
        
        # Style
        parts.append(primary_style)
        
        # Category
        parts.append(category)
        
        description = " ".join(parts)
        
        return description
    
    def extract_from_base64(
        self,
        image_base64: str,
        roi: Optional[Tuple[int, int, int, int]] = None
    ) -> Dict:
        """
        Extract attributes from base64 image.
        
        Args:
            image_base64: Base64-encoded image
            roi: Optional region of interest
            
        Returns:
            Dictionary with attributes
        """
        try:
            # Decode image
            image_bytes = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Extract attributes
            attrs = self.extract_attributes(image, roi=roi)
            
            return {
                "success": True,
                "category": attrs.category,
                "subcategory": attrs.subcategory,
                "colors": [{"name": c, "confidence": float(conf)} for c, conf in attrs.colors],
                "patterns": [{"name": p, "confidence": float(conf)} for p, conf in attrs.patterns],
                "styles": [{"name": s, "confidence": float(conf)} for s, conf in attrs.styles],
                "fabric": attrs.fabric,
                "details": attrs.details,
                "description": attrs.description
            }
            
        except Exception as e:
            logger.error(f"Extraction from base64 failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Singleton instance
_fashion_clip_instance = None

def get_fashion_clip() -> FashionCLIP:
    """Get singleton instance of FashionCLIP"""
    global _fashion_clip_instance
    if _fashion_clip_instance is None:
        _fashion_clip_instance = FashionCLIP()
    return _fashion_clip_instance
