"""
⚡ ASYNC INGESTION ENGINE - Decoupled Heavy Processing
=======================================================

Separates GPU-intensive ingestion from low-latency serving:

OFFLINE LAYER (Heavy):
- Video processing (Florence-2 + SAM 2)
- Embedding generation (Fashion-SigLIP)
- 3D asset creation

ONLINE LAYER (Fast):
- Vector search (HNSW)
- Metadata retrieval
- Real-time queries

This architecture enables million-user scale.
"""

import asyncio
import json
import logging
import base64
import time
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of an ingestion job"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class EmbeddingModel(Enum):
    """Available embedding models"""
    CLIP = "openai/clip-vit-large-patch14"
    SIGLIP = "google/siglip-so400m-patch14-384"
    FASHION_CLIP = "patrickjohncyh/fashion-clip"


@dataclass
class IngestionJob:
    """Single ingestion job"""
    job_id: str
    image_id: str
    status: JobStatus
    created_at: float
    
    # Input
    image_data: Optional[str] = None  # Base64
    video_frames: Optional[List[str]] = None
    
    # Results
    detections: List[Dict] = field(default_factory=list)
    embeddings: List[List[float]] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    asset_url: Optional[str] = None
    
    # Processing info
    processing_time_ms: float = 0
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result['status'] = self.status.value
        return result


@dataclass
class VectorRecord:
    """Record stored in vector database"""
    id: str
    embedding: List[float]
    metadata: Dict
    
    # Fashion-specific fields
    category: str
    specific_type: str
    colors: List[str]
    material: Optional[str]
    pattern: Optional[str]
    style_tags: List[str]
    
    # Asset reference
    asset_url: Optional[str] = None
    cutout_url: Optional[str] = None


class FashionEmbeddingModel:
    """
    Fashion-specialized embedding model.
    
    Uses Fashion-SigLIP for better multi-label fashion attributes
    compared to standard CLIP.
    """
    
    def __init__(
        self,
        model_type: EmbeddingModel = EmbeddingModel.FASHION_CLIP
    ):
        self.model_type = model_type
        self._model = None
        self._processor = None
        
        logger.info(f"📊 Embedding model: {model_type.value}")
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self._model is not None:
            return
        
        try:
            from transformers import AutoProcessor, AutoModel
            import torch
            
            logger.info(f"📥 Loading {self.model_type.value}...")
            
            self._processor = AutoProcessor.from_pretrained(
                self.model_type.value,
                trust_remote_code=True
            )
            self._model = AutoModel.from_pretrained(
                self.model_type.value,
                trust_remote_code=True
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self._model = self._model.cuda()
            
            self._model.eval()
            logger.info("✅ Embedding model loaded")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def embed_image(self, image) -> List[float]:
        """
        Generate embedding for a single image.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Embedding vector (768 or 1024 dims typically)
        """
        import torch
        from PIL import Image
        import numpy as np
        
        self._load_model()
        
        # Convert numpy to PIL if needed
        if isinstance(image, np.ndarray):
            import cv2
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Process
        inputs = self._processor(images=image, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model.get_image_features(**inputs)
        
        # Normalize
        embedding = outputs[0].cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for text query.
        
        Args:
            text: Query text (e.g., "red vintage jacket")
            
        Returns:
            Embedding vector
        """
        import torch
        import numpy as np
        
        self._load_model()
        
        inputs = self._processor(text=[text], return_tensors="pt", padding=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model.get_text_features(**inputs)
        
        embedding = outputs[0].cpu().numpy()
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()


class VectorDatabase:
    """
    Vector database abstraction layer.
    
    Supports multiple backends:
    - In-memory HNSW (for development)
    - OpenSearch (for production)
    - Weaviate (alternative)
    - Milvus (alternative)
    """
    
    def __init__(
        self,
        backend: str = "memory",
        index_params: Optional[Dict] = None
    ):
        self.backend = backend
        self.index_params = index_params or {
            "M": 16,  # HNSW connections per layer
            "ef_construction": 100,
            "ef_search": 50
        }
        
        # In-memory storage for development
        self._records: Dict[str, VectorRecord] = {}
        self._index = None
        
        logger.info(f"📦 Vector DB initialized (backend={backend})")
    
    def _build_index(self):
        """Build HNSW index from records."""
        if not self._records:
            return
        
        try:
            import hnswlib
            import numpy as np
            
            # Get dimension from first record
            first_record = next(iter(self._records.values()))
            dim = len(first_record.embedding)
            
            # Create index
            self._index = hnswlib.Index(space='cosine', dim=dim)
            self._index.init_index(
                max_elements=len(self._records) * 2,
                M=self.index_params["M"],
                ef_construction=self.index_params["ef_construction"]
            )
            
            # Add vectors
            ids = list(self._records.keys())
            vectors = np.array([
                self._records[id].embedding for id in ids
            ], dtype=np.float32)
            
            # Use integer labels for hnswlib
            self._id_to_label = {id: i for i, id in enumerate(ids)}
            self._label_to_id = {i: id for id, i in self._id_to_label.items()}
            
            self._index.add_items(vectors, list(range(len(ids))))
            self._index.set_ef(self.index_params["ef_search"])
            
            logger.info(f"Built HNSW index with {len(ids)} vectors")
            
        except ImportError:
            logger.warning("hnswlib not available, using brute force search")
    
    def insert(self, record: VectorRecord):
        """Insert a record into the database."""
        self._records[record.id] = record
        
        # Rebuild index periodically
        if len(self._records) % 100 == 0:
            self._build_index()
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Tuple[VectorRecord, float]]:
        """
        Search for similar items.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results
            filters: Optional metadata filters
            
        Returns:
            List of (record, similarity_score) tuples
        """
        import numpy as np
        
        if not self._records:
            return []
        
        query = np.array(query_embedding, dtype=np.float32)
        
        # Use HNSW if available
        if self._index is not None:
            try:
                labels, distances = self._index.knn_query(query, k=top_k)
                
                results = []
                for label, dist in zip(labels[0], distances[0]):
                    record_id = self._label_to_id.get(label)
                    if record_id and record_id in self._records:
                        record = self._records[record_id]
                        
                        # Apply filters
                        if filters:
                            if not self._matches_filters(record, filters):
                                continue
                        
                        # Convert distance to similarity
                        similarity = 1 - dist
                        results.append((record, similarity))
                
                return results
                
            except Exception as e:
                logger.warning(f"HNSW search failed, falling back to brute force: {e}")
        
        # Brute force fallback
        results = []
        for record in self._records.values():
            # Apply filters
            if filters and not self._matches_filters(record, filters):
                continue
            
            # Compute similarity
            record_vec = np.array(record.embedding, dtype=np.float32)
            similarity = np.dot(query, record_vec)
            results.append((record, similarity))
        
        # Sort and return top-k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def _matches_filters(self, record: VectorRecord, filters: Dict) -> bool:
        """Check if record matches filters."""
        for key, value in filters.items():
            record_value = getattr(record, key, None)
            
            if record_value is None:
                return False
            
            if isinstance(value, list):
                if record_value not in value:
                    return False
            else:
                if record_value != value:
                    return False
        
        return True
    
    def get_by_id(self, record_id: str) -> Optional[VectorRecord]:
        """Get a record by ID."""
        return self._records.get(record_id)
    
    def delete(self, record_id: str) -> bool:
        """Delete a record."""
        if record_id in self._records:
            del self._records[record_id]
            return True
        return False
    
    def count(self) -> int:
        """Get total record count."""
        return len(self._records)


class AsyncIngestionEngine:
    """
    ⚡ Asynchronous Ingestion Engine
    
    Decouples heavy GPU processing from user-facing interactions.
    
    Architecture:
    ┌────────────────────────────────────────┐
    │         OFFLINE INGESTION LAYER        │
    │  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
    │  │Florence-2│  │ SAM 2  │  │Embedding│ │
    │  │Detection │  │Tracking│  │   Gen   │ │
    │  └────┬────┘  └────┬────┘  └────┬────┘ │
    │       └───────────┬─────────────┘      │
    │              ┌────▼────┐               │
    │              │  Queue  │               │
    │              └────┬────┘               │
    └───────────────────┼────────────────────┘
                        │
                   ┌────▼────┐
                   │Vector DB│
                   └────┬────┘
                        │
    ┌───────────────────┼────────────────────┐
    │       ONLINE SERVING LAYER             │
    │  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
    │  │ Search  │  │Metadata │  │  Asset  │ │
    │  │  API    │  │Retrieval│  │  URLs   │ │
    │  └─────────┘  └─────────┘  └─────────┘ │
    └────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        embedding_model: EmbeddingModel = EmbeddingModel.FASHION_CLIP,
        vector_db_backend: str = "memory",
        max_workers: int = 4
    ):
        self.max_workers = max_workers
        
        # Components
        self.embedding_model = FashionEmbeddingModel(embedding_model)
        self.vector_db = VectorDatabase(backend=vector_db_backend)
        
        # Job queue
        self._job_queue: asyncio.Queue = None
        self._jobs: Dict[str, IngestionJob] = {}
        self._workers: List[asyncio.Task] = []
        
        # Pipeline reference
        self._unified_pipeline = None
        
        logger.info(f"⚡ Async Ingestion Engine initialized (workers={max_workers})")
    
    def _ensure_queue(self):
        """Ensure job queue exists."""
        if self._job_queue is None:
            self._job_queue = asyncio.Queue()
    
    async def submit_job(
        self,
        image_data: str,
        image_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Submit an image for asynchronous processing.
        
        Args:
            image_data: Base64-encoded image
            image_id: Optional unique ID (auto-generated if not provided)
            metadata: Optional metadata to attach
            
        Returns:
            Job ID for tracking
        """
        self._ensure_queue()
        
        # Generate IDs
        job_id = str(uuid.uuid4())
        if image_id is None:
            image_id = hashlib.md5(image_data.encode()[:1000]).hexdigest()
        
        # Create job
        job = IngestionJob(
            job_id=job_id,
            image_id=image_id,
            status=JobStatus.PENDING,
            created_at=time.time(),
            image_data=image_data,
            metadata=metadata or {}
        )
        
        self._jobs[job_id] = job
        
        # Add to queue
        await self._job_queue.put(job_id)
        
        logger.info(f"📥 Submitted job {job_id}")
        
        return job_id
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get status of a job."""
        job = self._jobs.get(job_id)
        if job:
            return job.to_dict()
        return None
    
    async def _process_job(self, job_id: str):
        """Process a single job."""
        job = self._jobs.get(job_id)
        if not job:
            return
        
        job.status = JobStatus.PROCESSING
        start_time = time.time()
        
        try:
            import cv2
            import numpy as np
            
            # Decode image
            img_bytes = base64.b64decode(job.image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                raise ValueError("Failed to decode image")
            
            # Run unified pipeline
            if self._unified_pipeline is None:
                from modules.unified_pipeline import get_unified_pipeline
                self._unified_pipeline = get_unified_pipeline()
            
            result = self._unified_pipeline.process(image)
            
            # Generate embeddings for each detection
            embeddings = []
            for det in result.detections:
                if det.cutout_base64:
                    # Embed the cutout
                    cutout_bytes = base64.b64decode(det.cutout_base64)
                    cutout_arr = np.frombuffer(cutout_bytes, np.uint8)
                    cutout_img = cv2.imdecode(cutout_arr, cv2.IMREAD_COLOR)
                    
                    if cutout_img is not None:
                        emb = self.embedding_model.embed_image(cutout_img)
                        embeddings.append(emb)
                        
                        # Store in vector DB
                        record = VectorRecord(
                            id=f"{job.image_id}_{len(embeddings)}",
                            embedding=emb,
                            metadata=det.attributes,
                            category=det.category,
                            specific_type=det.specific_type,
                            colors=[det.primary_color],
                            material=det.attributes.get("material"),
                            pattern=det.attributes.get("pattern"),
                            style_tags=[],
                            cutout_url=det.cutout_base64
                        )
                        self.vector_db.insert(record)
            
            # Update job
            job.detections = [d.to_dict() for d in result.detections]
            job.embeddings = embeddings
            job.status = JobStatus.COMPLETED
            job.processing_time_ms = (time.time() - start_time) * 1000
            
            logger.info(f"✅ Job {job_id} completed: {len(result.detections)} items in {job.processing_time_ms:.0f}ms")
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            logger.error(f"❌ Job {job_id} failed: {e}")
    
    async def _worker(self, worker_id: int):
        """Background worker that processes jobs from queue."""
        logger.info(f"👷 Worker {worker_id} started")
        
        while True:
            try:
                # Get job from queue
                job_id = await asyncio.wait_for(
                    self._job_queue.get(),
                    timeout=1.0
                )
                
                await self._process_job(job_id)
                self._job_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    async def start_workers(self):
        """Start background workers."""
        self._ensure_queue()
        
        for i in range(self.max_workers):
            task = asyncio.create_task(self._worker(i))
            self._workers.append(task)
        
        logger.info(f"🚀 Started {self.max_workers} ingestion workers")
    
    async def stop_workers(self):
        """Stop background workers."""
        for task in self._workers:
            task.cancel()
        
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers = []
        
        logger.info("⏹️ Stopped ingestion workers")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search for fashion items by text query.
        
        Args:
            query: Natural language query (e.g., "red vintage jacket")
            top_k: Number of results to return
            filters: Optional filters (category, color, etc.)
            
        Returns:
            List of matching items with metadata
        """
        # Embed query
        query_embedding = self.embedding_model.embed_text(query)
        
        # Search vector DB
        results = self.vector_db.search(
            query_embedding,
            top_k=top_k,
            filters=filters
        )
        
        # Format results
        formatted = []
        for record, score in results:
            formatted.append({
                "id": record.id,
                "category": record.category,
                "specificType": record.specific_type,
                "colors": record.colors,
                "material": record.material,
                "pattern": record.pattern,
                "score": round(score, 3),
                "cutoutUrl": record.cutout_url
            })
        
        return formatted
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        pending = sum(1 for j in self._jobs.values() if j.status == JobStatus.PENDING)
        processing = sum(1 for j in self._jobs.values() if j.status == JobStatus.PROCESSING)
        completed = sum(1 for j in self._jobs.values() if j.status == JobStatus.COMPLETED)
        failed = sum(1 for j in self._jobs.values() if j.status == JobStatus.FAILED)
        
        return {
            "totalJobs": len(self._jobs),
            "pending": pending,
            "processing": processing,
            "completed": completed,
            "failed": failed,
            "vectorDbCount": self.vector_db.count(),
            "activeWorkers": len([w for w in self._workers if not w.done()])
        }


# ============================================
# 🔧 SINGLETON INSTANCE
# ============================================

_ingestion_engine = None


def get_ingestion_engine() -> AsyncIngestionEngine:
    """Get singleton ingestion engine."""
    global _ingestion_engine
    if _ingestion_engine is None:
        _ingestion_engine = AsyncIngestionEngine()
    return _ingestion_engine
