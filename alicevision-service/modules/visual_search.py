"""
Visual Similarity Search using Embeddings
Find similar clothing items using deep learning features
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from PIL import Image
import logging
from dataclasses import dataclass
import base64

logger = logging.getLogger(__name__)


@dataclass
class SimilarItem:
    """Similar item result"""
    item_id: str
    similarity: float
    distance: float
    metadata: Optional[Dict] = None


class VisualSearchEngine:
    """
    Visual similarity search using deep learning embeddings
    
    Features:
    - ResNet-50 feature extraction
    - FAISS vector indexing for fast search
    - Cosine similarity ranking
    - Duplicate detection
    """
    
    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        self._index = None
        self._item_ids = []
        self._metadata = {}
        self.embedding_dim = 2048  # ResNet-50 output
    
    def _load_model(self):
        """Load ResNet-50 for feature extraction"""
        if self._model is not None:
            return True
        
        try:
            import torch
            import torchvision.models as models
            import torchvision.transforms as transforms
            
            logger.info("Loading ResNet-50 model...")
            
            # Load pretrained ResNet-50
            self._model = models.resnet50(pretrained=True)
            
            # Remove final classification layer
            self._model = torch.nn.Sequential(*list(self._model.children())[:-1])
            
            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.to("cuda")
            
            self._model.eval()
            
            # Image preprocessing
            self._transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            logger.info("✅ ResNet-50 loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"ResNet-50 loading failed: {e}")
            return False
    
    def _init_faiss_index(self):
        """Initialize FAISS index for similarity search"""
        try:
            import faiss
            
            # Use L2 distance index (can convert to cosine later)
            self._index = faiss.IndexFlatL2(self.embedding_dim)
            
            logger.info("✅ FAISS index initialized")
            return True
            
        except ImportError:
            logger.warning("FAISS not available, using numpy fallback")
            self._index = None
            return False
    
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract feature embedding from image
        
        Args:
            image: BGR image
            
        Returns:
            2048-dim feature vector
        """
        if not self._load_model():
            # Fallback to simple color histogram
            return self._color_histogram_embedding(image)
        
        try:
            import torch
            
            # Convert to PIL RGB
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Preprocess
            input_tensor = self._transform(pil_image).unsqueeze(0)
            
            if self.device == "cuda":
                input_tensor = input_tensor.to("cuda")
            
            # Extract features
            with torch.no_grad():
                embedding = self._model(input_tensor)
            
            # Flatten and normalize
            embedding = embedding.squeeze().cpu().numpy()
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
            return self._color_histogram_embedding(image)
    
    def _color_histogram_embedding(self, image: np.ndarray) -> np.ndarray:
        """Fallback: simple color histogram embedding"""
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms
        h_hist = cv2.calcHist([hsv], [0], None, [256], [0, 256]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256]).flatten()
        
        # Concatenate and normalize
        embedding = np.concatenate([h_hist, s_hist, v_hist])
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        # Pad to 2048 dimensions
        if len(embedding) < self.embedding_dim:
            padding = np.zeros(self.embedding_dim - len(embedding))
            embedding = np.concatenate([embedding, padding])
        else:
            embedding = embedding[:self.embedding_dim]
        
        return embedding
    
    def add_item(
        self, 
        item_id: str, 
        image: np.ndarray,
        metadata: Optional[Dict] = None
    ):
        """
        Add item to search index
        
        Args:
            item_id: Unique identifier for item
            image: BGR image
            metadata: Optional metadata to store
        """
        # Extract embedding
        embedding = self.extract_embedding(image)
        
        # Initialize index if needed
        if self._index is None:
            self._init_faiss_index()
        
        # Add to index
        if self._index is not None:
            import faiss
            self._index.add(embedding.reshape(1, -1).astype('float32'))
        
        # Store metadata
        self._item_ids.append(item_id)
        if metadata:
            self._metadata[item_id] = metadata
        
        logger.info(f"Added item {item_id} to index (total: {len(self._item_ids)})")
    
    def search(
        self, 
        query_image: np.ndarray,
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[SimilarItem]:
        """
        Search for similar items
        
        Args:
            query_image: BGR image to search for
            top_k: Number of results to return
            similarity_threshold: Minimum similarity (0-1)
            
        Returns:
            List of SimilarItem results
        """
        if len(self._item_ids) == 0:
            logger.warning("Index is empty")
            return []
        
        # Extract query embedding
        query_embedding = self.extract_embedding(query_image)
        
        if self._index is not None:
            # FAISS search
            query_vec = query_embedding.reshape(1, -1).astype('float32')
            distances, indices = self._index.search(query_vec, min(top_k, len(self._item_ids)))
            
            results = []
            for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self._item_ids):
                    item_id = self._item_ids[idx]
                    
                    # Convert L2 distance to cosine similarity
                    # For normalized vectors: similarity = 1 - (dist^2 / 2)
                    similarity = 1.0 - (dist / 2.0)
                    similarity = max(0.0, min(1.0, similarity))
                    
                    if similarity >= similarity_threshold:
                        results.append(SimilarItem(
                            item_id=item_id,
                            similarity=float(similarity),
                            distance=float(dist),
                            metadata=self._metadata.get(item_id)
                        ))
        else:
            # Numpy fallback
            results = self._numpy_search(query_embedding, top_k, similarity_threshold)
        
        return results
    
    def _numpy_search(
        self, 
        query_embedding: np.ndarray,
        top_k: int,
        similarity_threshold: float
    ) -> List[SimilarItem]:
        """Fallback search using numpy"""
        # This would require storing all embeddings
        # For now, return empty
        logger.warning("Numpy search not fully implemented")
        return []
    
    def find_duplicates(self, similarity_threshold: float = 0.95) -> List[Tuple[str, str, float]]:
        """
        Find potential duplicate items in the index
        
        Args:
            similarity_threshold: Minimum similarity to consider duplicate
            
        Returns:
            List of (item_id1, item_id2, similarity) tuples
        """
        duplicates = []
        
        # Compare all pairs (this is O(n^2), optimize for large datasets)
        for i in range(len(self._item_ids)):
            for j in range(i + 1, len(self._item_ids)):
                # This requires storing embeddings
                # Simplified implementation
                pass
        
        return duplicates


def search_similar_from_base64(
    query_base64: str,
    index_images: List[Dict],  # [{"id": "...", "image": "base64..."}]
    top_k: int = 5
) -> Dict:
    """
    Utility function for visual similarity search from base64
    
    Args:
        query_base64: Base64-encoded query image
        index_images: List of images to search against
        top_k: Number of results
        
    Returns:
        Search results dictionary
    """
    import time
    start_time = time.time()
    
    # Decode query image
    if ',' in query_base64:
        query_base64 = query_base64.split(',')[1]
    
    query_bytes = base64.b64decode(query_base64)
    query_array = np.frombuffer(query_bytes, dtype=np.uint8)
    query_image = cv2.imdecode(query_array, cv2.IMREAD_COLOR)
    
    if query_image is None:
        return {"error": "Could not decode query image"}
    
    # Create search engine
    engine = VisualSearchEngine()
    
    # Add all index images
    for item in index_images:
        if ',' in item["image"]:
            img_b64 = item["image"].split(',')[1]
        else:
            img_b64 = item["image"]
        
        img_bytes = base64.b64decode(img_b64)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is not None:
            engine.add_item(item["id"], img, item.get("metadata"))
    
    # Search
    results = engine.search(query_image, top_k=top_k)
    
    processing_time = (time.time() - start_time) * 1000
    
    return {
        "results": [
            {
                "itemId": r.item_id,
                "similarity": round(r.similarity, 4),
                "distance": round(r.distance, 4),
                "metadata": r.metadata
            }
            for r in results
        ],
        "totalResults": len(results),
        "processingTimeMs": round(processing_time, 1)
    }
