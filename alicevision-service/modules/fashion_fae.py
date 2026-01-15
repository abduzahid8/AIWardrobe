"""
🧵 FashionFAE: Fine-grained Attribute Extraction

Extracts detailed fashion attributes that generic CLIP struggles with:
- Materials: "chiffon" vs "silk" vs "satin"
- Necklines: "crew neck" vs "scoop neck" vs "v-neck"
- Patterns: "herringbone" vs "tweed" vs "houndstooth"
- Sleeve Types: "bishop" vs "bell" vs "puff"
- Construction Details: "double-breasted" vs "single-breasted"

Implementation: Uses enhanced CLIP prompts with fashion-specific vocabularies.
Falls back gracefully if FashionFAE model not available.
"""

import cv2
import numpy as np
import torch
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class FashionFAEClassifier:
    """
    🧵 Fine-grained Attribute Extraction for Fashion
    
    Uses attribute-specific CLIP prompts to extract detailed
    fashion attributes like material, neckline, sleeve type, etc.
    
    This is a drop-in implementation that can be upgraded to
    the actual FashionFAE model when available.
    """
    
    # === FINE-GRAINED VOCABULARIES ===
    
    MATERIALS = [
        # Natural fabrics
        "cotton", "linen", "silk", "wool", "cashmere", "mohair", "alpaca",
        # Synthetic fabrics
        "polyester", "nylon", "rayon", "viscose", "spandex", "lycra",
        # Special fabrics
        "denim", "corduroy", "velvet", "velour", "satin", "chiffon",
        "organza", "tulle", "lace", "mesh", "fleece", "terry cloth",
        # Leather and alternatives
        "leather", "suede", "faux leather", "patent leather", "pvc",
        # Woven patterns
        "tweed", "herringbone", "houndstooth", "gabardine", "flannel",
        "chambray", "oxford cloth", "poplin", "broadcloth"
    ]
    
    NECKLINES = [
        "crew neck", "v-neck", "scoop neck", "boat neck", "bateau neck",
        "turtleneck", "mock neck", "cowl neck", "funnel neck",
        "square neck", "sweetheart neck", "off-shoulder", "one-shoulder",
        "halter neck", "keyhole neck", "split neck", "henley",
        "collar", "mandarin collar", "peter pan collar", "spread collar",
        "button-down collar", "hood", "shawl collar", "notch lapel"
    ]
    
    SLEEVE_TYPES = [
        "sleeveless", "cap sleeve", "short sleeve", "elbow sleeve",
        "3/4 sleeve", "bracelet sleeve", "long sleeve",
        "bishop sleeve", "bell sleeve", "flare sleeve", "flutter sleeve",
        "puff sleeve", "balloon sleeve", "leg-of-mutton sleeve",
        "dolman sleeve", "batwing sleeve", "kimono sleeve", "raglan sleeve",
        "set-in sleeve", "dropped shoulder", "cold shoulder"
    ]
    
    PATTERNS = [
        "solid", "striped", "horizontal stripes", "vertical stripes",
        "pinstripes", "candy stripes", "breton stripes",
        "plaid", "tartan", "gingham", "buffalo check", "windowpane",
        "checkered", "houndstooth check",
        "polka dot", "pin dot", "ditsy print",
        "floral", "tropical", "botanical", "paisley", "damask",
        "geometric", "abstract", "chevron", "zigzag",
        "animal print", "leopard", "zebra", "snake print", "cow print",
        "camouflage", "tie-dye", "ombre", "color block",
        "herringbone pattern", "argyle", "fair isle", "cable knit"
    ]
    
    FITS = [
        "slim fit", "skinny fit", "regular fit", "relaxed fit",
        "loose fit", "oversized", "boxy", "tailored",
        "fitted", "body-con", "a-line", "straight cut",
        "cropped", "longline", "tunic length"
    ]
    
    CLOSURES = [
        "button-front", "zip-front", "pullover", "wrap style",
        "snap buttons", "toggles", "hook and eye", "tie closure",
        "double-breasted", "single-breasted", "cardigan style",
        "open front", "belted", "drawstring"
    ]
    
    STYLE_TAGS = [
        "casual", "formal", "business casual", "smart casual",
        "sporty", "athletic", "activewear", "athleisure",
        "streetwear", "urban", "minimalist", "classic",
        "vintage", "retro", "bohemian", "preppy", "punk",
        "romantic", "feminine", "masculine", "androgynous",
        "luxury", "designer", "high fashion", "avant-garde"
    ]
    
    def __init__(self, model_name: str = "ViT-B-32", device: str = "auto"):
        """
        Initialize FashionFAE classifier.
        
        Args:
            model_name: CLIP model architecture
            device: "cuda", "cpu", or "auto"
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model_name = model_name
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._loaded = False
        
        logger.info(f"FashionFAE initialized (model={model_name}, device={self.device})")
    
    def _load_model(self):
        """Lazy load CLIP model."""
        if self._loaded:
            return
        
        try:
            import open_clip
            
            logger.info(f"Loading CLIP model {self.model_name}...")
            
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                self.model_name,
                pretrained='openai',
                device=self.device
            )
            self._tokenizer = open_clip.get_tokenizer(self.model_name)
            self._model.eval()
            self._loaded = True
            
            logger.info("✅ FashionFAE CLIP model loaded")
            
        except ImportError:
            logger.warning("open_clip not installed, using fallback")
            self._loaded = True  # Mark as loaded to prevent retry
        except Exception as e:
            logger.error(f"Failed to load CLIP: {e}")
            self._loaded = True
    
    def extract(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract comprehensive fashion attributes.
        
        Args:
            image: BGR image (numpy array)
            
        Returns:
            Dictionary with extracted attributes
        """
        self._load_model()
        
        if self._model is None:
            # Fallback when CLIP not available
            return self._fallback_extract(image)
        
        # Convert to PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        
        # Extract each attribute type
        material = self._classify_attribute(image_pil, self.MATERIALS, "made of")
        neckline = self._classify_attribute(image_pil, self.NECKLINES, "with")
        sleeve = self._classify_attribute(image_pil, self.SLEEVE_TYPES, "with")
        pattern = self._classify_attribute(image_pil, self.PATTERNS, "with")
        fit = self._classify_attribute(image_pil, self.FITS, "with")
        closure = self._classify_attribute(image_pil, self.CLOSURES, "with")
        style_tags = self._classify_multi(image_pil, self.STYLE_TAGS, "style", top_k=3)
        
        return {
            "material": material[0] if material else "",
            "material_confidence": material[1] if material else 0.0,
            "neckline": neckline[0] if neckline else "",
            "neckline_confidence": neckline[1] if neckline else 0.0,
            "sleeve_type": sleeve[0] if sleeve else "",
            "sleeve_confidence": sleeve[1] if sleeve else 0.0,
            "pattern": pattern[0] if pattern else "",
            "pattern_confidence": pattern[1] if pattern else 0.0,
            "fit": fit[0] if fit else "",
            "fit_confidence": fit[1] if fit else 0.0,
            "closure": closure[0] if closure else "",
            "closure_confidence": closure[1] if closure else 0.0,
            "style_tags": [s[0] for s in style_tags],
            "style_confidences": [s[1] for s in style_tags]
        }
    
    def _classify_attribute(
        self,
        image: Image.Image,
        labels: List[str],
        prefix: str = "with"
    ) -> Tuple[str, float]:
        """Classify image against attribute labels."""
        try:
            # Prepare prompts
            prompts = [f"a clothing item {prefix} {label}" for label in labels]
            
            # Preprocess image
            image_input = self._preprocess(image).unsqueeze(0).to(self.device)
            text_inputs = self._tokenizer(prompts).to(self.device)
            
            with torch.no_grad():
                image_features = self._model.encode_image(image_input)
                text_features = self._model.encode_text(text_inputs)
                
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                value, idx = similarity[0].topk(1)
            
            return (labels[idx.item()], float(value.item()))
            
        except Exception as e:
            logger.warning(f"Attribute classification failed: {e}")
            return ("", 0.0)
    
    def _classify_multi(
        self,
        image: Image.Image,
        labels: List[str],
        context: str = "",
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """Classify image against multiple labels."""
        try:
            prompts = [f"{label} {context}" for label in labels]
            
            image_input = self._preprocess(image).unsqueeze(0).to(self.device)
            text_inputs = self._tokenizer(prompts).to(self.device)
            
            with torch.no_grad():
                image_features = self._model.encode_image(image_input)
                text_features = self._model.encode_text(text_inputs)
                
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
                values, indices = similarity[0].topk(top_k)
            
            results = [
                (labels[idx.item()], float(val.item()))
                for idx, val in zip(indices, values)
            ]
            return results
            
        except Exception as e:
            logger.warning(f"Multi-classification failed: {e}")
            return []
    
    def _fallback_extract(self, image: np.ndarray) -> Dict[str, Any]:
        """Fallback extraction when CLIP not available."""
        # Use simple heuristics or return empty
        return {
            "material": "",
            "material_confidence": 0.0,
            "neckline": "",
            "neckline_confidence": 0.0,
            "sleeve_type": "",
            "sleeve_confidence": 0.0,
            "pattern": "solid",
            "pattern_confidence": 0.5,
            "fit": "regular fit",
            "fit_confidence": 0.5,
            "closure": "",
            "closure_confidence": 0.0,
            "style_tags": ["casual"],
            "style_confidences": [0.5]
        }


# ============================================
# Singleton
# ============================================

_fae_instance = None


def get_fashion_fae(**kwargs) -> FashionFAEClassifier:
    """Get singleton FashionFAE classifier."""
    global _fae_instance
    if _fae_instance is None:
        _fae_instance = FashionFAEClassifier(**kwargs)
    return _fae_instance
