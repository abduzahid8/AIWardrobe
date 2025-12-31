"""
Wardrobe Memory Module
Vector embedding store for semantic wardrobe search

Uses CLIP embeddings to enable natural language search:
- "Find my warm jackets"
- "Something blue for a party"
- "Casual everyday shoes"
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging
from pathlib import Path
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class StoredItem:
    """Item stored in wardrobe memory."""
    id: str
    user_id: str
    category: str
    specific_type: str
    primary_color: str
    color_hex: str
    pattern: Optional[str] = None
    material: Optional[str] = None
    style_tags: List[str] = field(default_factory=list)
    occasion_tags: List[str] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    image_hash: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "userId": self.user_id,
            "category": self.category,
            "specificType": self.specific_type,
            "primaryColor": self.primary_color,
            "colorHex": self.color_hex,
            "pattern": self.pattern,
            "material": self.material,
            "styleTags": self.style_tags,
            "occasionTags": self.occasion_tags,
            "hasEmbedding": self.embedding is not None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "StoredItem":
        return cls(
            id=data.get("id", ""),
            user_id=data.get("userId", ""),
            category=data.get("category", ""),
            specific_type=data.get("specificType", ""),
            primary_color=data.get("primaryColor", ""),
            color_hex=data.get("colorHex", ""),
            pattern=data.get("pattern"),
            material=data.get("material"),
            style_tags=data.get("styleTags", []),
            occasion_tags=data.get("occasionTags", []),
            metadata=data.get("metadata", {})
        )


class WardrobeEmbeddingStore:
    """
    🧠 Vector Store for Wardrobe Items
    
    Stores CLIP embeddings for all wardrobe items, enabling:
    - Natural language semantic search
    - Similar item recommendations
    - Style profile analysis
    
    Uses in-memory storage with pickle serialization for persistence.
    """
    
    def __init__(
        self,
        user_id: str,
        storage_dir: str = None,
        auto_save: bool = True
    ):
        """
        Initialize wardrobe store for a user.
        
        Args:
            user_id: User identifier
            storage_dir: Directory to store embeddings (optional)
            auto_save: Automatically save after modifications
        """
        self.user_id = user_id
        self.auto_save = auto_save
        
        # Storage path
        if storage_dir:
            self.storage_dir = Path(storage_dir)
        else:
            self.storage_dir = Path.home() / ".aiwardrobe" / "embeddings"
        
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.storage_path = self.storage_dir / f"{user_id}_wardrobe.pkl"
        
        # In-memory storage
        self.items: Dict[str, StoredItem] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        
        # CLIP encoder (lazy loaded)
        self.clip_encoder = None
        
        # Load existing data
        self._load()
        
        logger.info(f"WardrobeEmbeddingStore initialized for user {user_id}")
    
    def _load_clip(self):
        """Lazy load CLIP encoder."""
        if self.clip_encoder is None:
            try:
                from .fashion_clip import get_fashion_clip
                self.clip_encoder = get_fashion_clip()
                logger.info("Fashion-CLIP loaded for embedding generation")
            except Exception as e:
                logger.warning(f"Fashion-CLIP not available: {e}")
    
    def _load(self):
        """Load stored data from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'rb') as f:
                    data = pickle.load(f)
                    self.items = data.get('items', {})
                    self.embeddings = data.get('embeddings', {})
                logger.info(f"Loaded {len(self.items)} items from storage")
            except Exception as e:
                logger.error(f"Failed to load storage: {e}")
    
    def _save(self):
        """Save data to disk."""
        try:
            data = {
                'items': self.items,
                'embeddings': self.embeddings
            }
            with open(self.storage_path, 'wb') as f:
                pickle.dump(data, f)
            logger.debug(f"Saved {len(self.items)} items to storage")
        except Exception as e:
            logger.error(f"Failed to save storage: {e}")
    
    def _generate_id(self, item_data: Dict) -> str:
        """Generate unique ID for item."""
        key_data = f"{self.user_id}_{item_data.get('category')}_{item_data.get('specificType')}_{item_data.get('primaryColor')}"
        return hashlib.md5(key_data.encode()).hexdigest()[:12]
    
    def _compute_image_hash(self, image_data: bytes) -> str:
        """Compute hash of image for deduplication."""
        return hashlib.md5(image_data).hexdigest()
    
    def add_item(
        self,
        item_data: Dict,
        image_base64: str = None,
        embedding: np.ndarray = None
    ) -> StoredItem:
        """
        Add item to wardrobe store.
        
        Args:
            item_data: Item information dict
            image_base64: Optional base64 image for embedding
            embedding: Optional pre-computed embedding
            
        Returns:
            StoredItem that was added
        """
        item_id = item_data.get('id') or self._generate_id(item_data)
        
        stored_item = StoredItem(
            id=item_id,
            user_id=self.user_id,
            category=item_data.get('category', ''),
            specific_type=item_data.get('specificType', ''),
            primary_color=item_data.get('primaryColor', ''),
            color_hex=item_data.get('colorHex', ''),
            pattern=item_data.get('pattern'),
            material=item_data.get('material'),
            style_tags=item_data.get('styleTags', []),
            occasion_tags=item_data.get('occasionTags', []),
            metadata=item_data.get('metadata', {})
        )
        
        # Generate embedding if image provided
        if image_base64 and embedding is None:
            embedding = self._generate_embedding(image_base64)
        
        if embedding is not None:
            stored_item.embedding = embedding
            self.embeddings[item_id] = embedding
        
        self.items[item_id] = stored_item
        
        if self.auto_save:
            self._save()
        
        logger.info(f"Added item {item_id} to wardrobe store")
        return stored_item
    
    def _generate_embedding(self, image_base64: str) -> Optional[np.ndarray]:
        """Generate CLIP embedding for image."""
        self._load_clip()
        
        if self.clip_encoder is None:
            return None
        
        try:
            import cv2
            import base64
            import numpy as np
            
            # Decode image
            if image_base64.startswith('data:'):
                image_base64 = image_base64.split(',')[1]
            
            image_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return None
            
            # Get CLIP embedding
            from PIL import Image
            import io
            
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Use CLIP to get embedding
            if hasattr(self.clip_encoder, 'model') and self.clip_encoder.model is not None:
                self.clip_encoder._load_model()
                
                preprocessed = self.clip_encoder.preprocess(pil_image).unsqueeze(0)
                if hasattr(self.clip_encoder, 'device'):
                    preprocessed = preprocessed.to(self.clip_encoder.device)
                
                import torch
                with torch.no_grad():
                    embedding = self.clip_encoder.model.encode_image(preprocessed)
                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                    return embedding.cpu().numpy().flatten()
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def remove_item(self, item_id: str) -> bool:
        """Remove item from store."""
        if item_id in self.items:
            del self.items[item_id]
            if item_id in self.embeddings:
                del self.embeddings[item_id]
            
            if self.auto_save:
                self._save()
            
            logger.info(f"Removed item {item_id}")
            return True
        return False
    
    def get_item(self, item_id: str) -> Optional[StoredItem]:
        """Get item by ID."""
        return self.items.get(item_id)
    
    def get_all_items(self) -> List[StoredItem]:
        """Get all items in wardrobe."""
        return list(self.items.values())
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[StoredItem, float]]:
        """
        Search wardrobe using natural language.
        
        Args:
            query: Natural language query (e.g., "warm winter jacket")
            top_k: Number of results
            
        Returns:
            List of (StoredItem, similarity_score) tuples
        """
        self._load_clip()
        
        if self.clip_encoder is None or not self.embeddings:
            # Fallback to keyword search
            return self._keyword_search(query, top_k)
        
        try:
            import torch
            
            # Encode query
            self.clip_encoder._load_model()
            query_tokens = self.clip_encoder.tokenizer([query])
            
            if hasattr(self.clip_encoder, 'device'):
                query_tokens = query_tokens.to(self.clip_encoder.device)
            
            with torch.no_grad():
                query_embedding = self.clip_encoder.model.encode_text(query_tokens)
                query_embedding = query_embedding / query_embedding.norm(dim=-1, keepdim=True)
                query_embedding = query_embedding.cpu().numpy().flatten()
            
            # Calculate similarities
            results = []
            for item_id, item_embedding in self.embeddings.items():
                if item_id in self.items:
                    similarity = np.dot(query_embedding, item_embedding)
                    results.append((self.items[item_id], float(similarity)))
            
            # Sort by similarity
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return self._keyword_search(query, top_k)
    
    def _keyword_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[StoredItem, float]]:
        """Fallback keyword-based search."""
        query_lower = query.lower()
        query_words = query_lower.split()
        
        results = []
        for item in self.items.values():
            score = 0.0
            
            # Check specific type
            if item.specific_type and query_lower in item.specific_type.lower():
                score += 0.5
            
            # Check category
            if item.category and query_lower in item.category.lower():
                score += 0.3
            
            # Check color
            if item.primary_color and query_lower in item.primary_color.lower():
                score += 0.3
            
            # Check individual words
            for word in query_words:
                if item.specific_type and word in item.specific_type.lower():
                    score += 0.1
                if item.primary_color and word in item.primary_color.lower():
                    score += 0.1
                if any(word in tag.lower() for tag in item.style_tags):
                    score += 0.1
                if item.material and word in item.material.lower():
                    score += 0.1
            
            if score > 0:
                results.append((item, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def find_similar(
        self,
        item_id: str,
        top_k: int = 5,
        exclude_same_category: bool = False
    ) -> List[Tuple[StoredItem, float]]:
        """
        Find items similar to a given item.
        
        Args:
            item_id: Reference item ID
            top_k: Number of results
            exclude_same_category: Exclude items in same category
            
        Returns:
            List of (StoredItem, similarity_score) tuples
        """
        if item_id not in self.embeddings:
            return []
        
        reference_embedding = self.embeddings[item_id]
        reference_item = self.items.get(item_id)
        
        results = []
        for other_id, other_embedding in self.embeddings.items():
            if other_id == item_id:
                continue
            
            other_item = self.items.get(other_id)
            if not other_item:
                continue
            
            if exclude_same_category and reference_item:
                if other_item.category == reference_item.category:
                    continue
            
            similarity = np.dot(reference_embedding, other_embedding)
            results.append((other_item, float(similarity)))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def find_compatible(
        self,
        item_id: str,
        top_k: int = 5
    ) -> List[Tuple[StoredItem, float]]:
        """
        Find items that would go well with a given item.
        
        Uses embedding similarity for visual compatibility,
        filtering by different categories.
        
        Args:
            item_id: Reference item ID
            top_k: Number of results
            
        Returns:
            List of compatible items with scores
        """
        return self.find_similar(item_id, top_k, exclude_same_category=True)
    
    def get_style_profile(self) -> Dict[str, Any]:
        """
        Analyze wardrobe style profile.
        
        Returns:
            Dict with style analysis
        """
        if not self.items:
            return {
                "totalItems": 0,
                "categories": {},
                "colors": {},
                "styles": {},
                "materials": {}
            }
        
        categories = {}
        colors = {}
        styles = {}
        materials = {}
        
        for item in self.items.values():
            # Count categories
            cat = item.category
            categories[cat] = categories.get(cat, 0) + 1
            
            # Count colors
            color = item.primary_color
            if color:
                colors[color] = colors.get(color, 0) + 1
            
            # Count styles
            for style in item.style_tags:
                styles[style] = styles.get(style, 0) + 1
            
            # Count materials
            mat = item.material
            if mat:
                materials[mat] = materials.get(mat, 0) + 1
        
        return {
            "totalItems": len(self.items),
            "categories": categories,
            "colors": colors,
            "styles": styles,
            "materials": materials
        }
    
    def clear(self):
        """Clear all items from store."""
        self.items.clear()
        self.embeddings.clear()
        if self.auto_save:
            self._save()
        logger.info("Cleared wardrobe store")


# User wardrobe stores cache
_user_stores: Dict[str, WardrobeEmbeddingStore] = {}


def get_user_wardrobe_store(
    user_id: str,
    storage_dir: str = None
) -> WardrobeEmbeddingStore:
    """
    Get or create wardrobe store for a user.
    
    Args:
        user_id: User identifier
        storage_dir: Optional custom storage directory
        
    Returns:
        WardrobeEmbeddingStore for the user
    """
    global _user_stores
    
    if user_id not in _user_stores:
        _user_stores[user_id] = WardrobeEmbeddingStore(
            user_id=user_id,
            storage_dir=storage_dir
        )
    
    return _user_stores[user_id]
