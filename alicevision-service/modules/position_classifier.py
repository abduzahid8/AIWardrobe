"""
🎯 Position-Aware Classification Head
Adds spatial awareness to clothing classification.

Key insight: A hat CANNOT be at the bottom of an image.
This module encodes position information to improve classification accuracy.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class PositionEncoder(nn.Module):
    """
    Encodes bounding box position into feature space.
    
    Input: bbox (x, y, w, h) normalized to [0, 1]
    Output: 64-dim position embedding
    """
    
    def __init__(self, output_dim: int = 64):
        super().__init__()
        
        # 4 bbox values → 64 dim
        self.encoder = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.ReLU()
        )
        
    def forward(self, bbox_normalized: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bbox_normalized: (batch, 4) with [x_center, y_center, width, height] in [0,1]
        """
        return self.encoder(bbox_normalized)


class PositionAwareClassifier(nn.Module):
    """
    Combines visual features with position embeddings for better classification.
    
    Architecture:
        Visual Features (768) → 
        Position Encoding (64) → 
        Concat (832) →
        MLP → 
        25 clothing categories
    """
    
    def __init__(
        self, 
        visual_dim: int = 768, 
        position_dim: int = 64,
        num_classes: int = 25
    ):
        super().__init__()
        
        self.position_encoder = PositionEncoder(position_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(visual_dim + position_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.classes = [
            # Upper body (0-7)
            "t-shirt", "shirt", "blouse", "sweater", "hoodie", 
            "cardigan", "jacket", "blazer",
            # Lower body (8-12)
            "pants", "jeans", "shorts", "skirt", "leggings",
            # Full body (13-14)
            "dress", "jumpsuit",
            # Footwear (15-18)
            "sneakers", "boots", "sandals", "heels",
            # Accessories (19-24)
            "hat", "cap", "scarf", "bag", "belt", "sunglasses"
        ]
        
    def forward(
        self, 
        visual_features: torch.Tensor, 
        bbox: torch.Tensor,
        image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Args:
            visual_features: (batch, 768) from CLIP/Florence
            bbox: (batch, 4) bbox as [x, y, w, h] in pixels
            image_size: (height, width)
            
        Returns:
            logits: (batch, 25) class logits
        """
        # Normalize bbox to [0, 1]
        h, w = image_size
        bbox_normalized = bbox.clone().float()
        bbox_normalized[:, 0] /= w  # x
        bbox_normalized[:, 1] /= h  # y
        bbox_normalized[:, 2] /= w  # width
        bbox_normalized[:, 3] /= h  # height
        
        # Encode position
        position_features = self.position_encoder(bbox_normalized)
        
        # Concatenate and classify
        combined = torch.cat([visual_features, position_features], dim=-1)
        logits = self.classifier(combined)
        
        return logits
    
    def predict(
        self,
        visual_features: np.ndarray,
        bbox: Tuple[float, float, float, float],
        image_size: Tuple[int, int]
    ) -> Tuple[str, float]:
        """
        Predict clothing type from features and position.
        
        Returns:
            (predicted_class, confidence)
        """
        self.eval()
        
        with torch.no_grad():
            vis_tensor = torch.tensor(visual_features).unsqueeze(0)
            bbox_tensor = torch.tensor(list(bbox)).unsqueeze(0)
            
            logits = self.forward(vis_tensor, bbox_tensor, image_size)
            probs = torch.softmax(logits, dim=-1)
            
            confidence, idx = probs.max(dim=-1)
            predicted_class = self.classes[idx.item()]
            
            return predicted_class, confidence.item()


def apply_position_prior(
    category: str,
    bbox: Tuple[float, float, float, float],
    image_height: int,
    image_width: int
) -> Tuple[float, str]:
    """
    Apply position-based prior probability adjustment.
    
    Returns:
        (adjustment_factor, reason)
        
    Factor > 1.0 = boost confidence
    Factor < 1.0 = reduce confidence
    """
    if not bbox or len(bbox) < 4:
        return 1.0, "No bbox"
    
    # Calculate center Y position
    y_center = (bbox[1] + bbox[3] / 2) / image_height
    
    category_lower = category.lower()
    
    # HATS should be in top 40%
    if any(x in category_lower for x in ["hat", "cap", "beanie"]):
        if y_center < 0.40:
            return 1.3, "Hat at top - boosted"
        elif y_center > 0.60:
            return 0.5, "Hat too low - reduced"
    
    # SHOES should be in bottom 40%
    if any(x in category_lower for x in ["shoe", "sneaker", "boot", "sandal"]):
        if y_center > 0.60:
            return 1.3, "Shoes at bottom - boosted"
        elif y_center < 0.40:
            return 0.5, "Shoes too high - reduced"
    
    # PANTS should be in lower 60%
    if any(x in category_lower for x in ["pants", "jeans", "shorts", "skirt"]):
        if y_center > 0.40:
            return 1.2, "Bottoms in lower half - boosted"
        elif y_center < 0.25:
            return 0.6, "Bottoms too high - reduced"
    
    # TOPS should be in upper 60%
    if any(x in category_lower for x in ["shirt", "blouse", "sweater", "jacket"]):
        if y_center < 0.60:
            return 1.1, "Top in upper half - boosted"
    
    return 1.0, "No position adjustment"
