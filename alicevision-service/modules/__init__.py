"""
AliceVision Modules
Advanced AI vision for AIWardrobe
"""

from .segmentation import (
    segment_clothing_from_base64,
    AdvancedClothingSegmentor,
    AdvancedSegmentationResult,
    ClothingItem,
    CLOTHING_CATEGORIES
)

from .keyframe import (
    select_best_frame_from_base64,
    KeyframeSelector
)

from .lighting import (
    normalize_lighting_from_base64,
    LightingNormalizer
)

from .card_styling import (
    create_product_card_from_base64,
    ProductCardStylist,
    TEMPLATES as CARD_TEMPLATES
)

from .pose_detection import (
    score_frames_for_pose,
    PoseDetector,
    PoseResult
)

# New AI modules
from .product_analyzer import (
    analyze_product_from_base64,
    ProductAnalyzer,
    DetectedProduct,
    ProductClassification
)

from .attribute_extractor import (
    extract_attributes_from_base64,
    AttributeExtractor,
    ColorInfo,
    PatternInfo,
    MaterialInfo
)

from .quality_assessor import (
    assess_photo_quality_from_base64,
    QualityAssessor,
    QualityScore
)

from .visual_search import (
    search_similar_from_base64,
    VisualSearchEngine,
    SimilarItem
)

__all__ = [
    # Segmentation
    "segment_clothing_from_base64",
    "AdvancedClothingSegmentor",
    "AdvancedSegmentationResult",
    "ClothingItem",
    "CLOTHING_CATEGORIES",
    
    # Keyframe
    "select_best_frame_from_base64",
    "KeyframeSelector",
    
    # Lighting
    "normalize_lighting_from_base64",
    "LightingNormalizer",
    
    # Card Styling
    "create_product_card_from_base64",
    "ProductCardStylist",
    "CARD_TEMPLATES",
    
    # Pose Detection
    "score_frames_for_pose",
    "PoseDetector",
    "PoseResult",
    
    # Product Analysis
    "analyze_product_from_base64",
    "ProductAnalyzer",
    "DetectedProduct",
    "ProductClassification",
    
    # Attribute Extraction
    "extract_attributes_from_base64",
    "AttributeExtractor",
    "ColorInfo",
    "PatternInfo",
    "MaterialInfo",
    
    # Quality Assessment
    "assess_photo_quality_from_base64",
    "QualityAssessor",
    "QualityScore",
    
    # Visual Search
    "search_similar_from_base64",
    "VisualSearchEngine",
    "SimilarItem",
]

# V2 Modules (Grounded SAM2 + FashionCLIP)
# These are imported lazily to save memory
try:
    from .grounded_sam import (
        get_grounded_sam,
        GroundedSAM2,
        SegmentationResult,
        DetectionResult
    )
    __all__.extend([
        "get_grounded_sam",
        "GroundedSAM2",
        "SegmentationResult",
        "DetectionResult"
    ])
except ImportError as e:
    # Grounded SAM2 dependencies not installed
    pass

try:
    from .fashion_clip import (
        get_fashion_clip,
        FashionCLIP,
        FashionAttributes
    )
    __all__.extend([
        "get_fashion_clip",
        "FashionCLIP",
        "FashionAttributes"
    ])
except ImportError as e:
    # FashionCLIP dependencies not installed
    pass

try:
    from .card_generator import (
        generate_card_prompt,
        CardPromptGenerator,
        CardPrompt
    )
    __all__.extend([
        "generate_card_prompt",
        "CardPromptGenerator",
        "CardPrompt"
    ])
except ImportError as e:
    # Card generator dependencies not installed
    pass


# === NEW WORLD-CLASS AI MODULES (2024) ===

import logging as _logging
_logger = _logging.getLogger(__name__)

try:
    from .yolo_detector import (
        get_yolo_detector,
        YOLOClothingDetector,
        YOLODetection,
        YOLOResult,
        detect_clothing_fast
    )
    __all__.extend([
        "get_yolo_detector",
        "YOLOClothingDetector",
        "YOLODetection",
        "YOLOResult",
        "detect_clothing_fast"
    ])
except ImportError as e:
    _logger.debug(f"YOLO detector not available: {e}")

try:
    from .hierarchical_classifier import (
        get_hierarchical_classifier,
        HierarchicalClothingClassifier,
        ClassificationResult,
        CLOTHING_TAXONOMY,
        classify_clothing
    )
    __all__.extend([
        "get_hierarchical_classifier",
        "HierarchicalClothingClassifier",
        "ClassificationResult",
        "CLOTHING_TAXONOMY",
        "classify_clothing"
    ])
except ImportError as e:
    _logger.debug(f"Hierarchical classifier not available: {e}")

try:
    from .material_analyzer import (
        get_material_analyzer,
        MaterialAnalyzer,
        MaterialAnalysis,
        analyze_material
    )
    __all__.extend([
        "get_material_analyzer",
        "MaterialAnalyzer",
        "MaterialAnalysis",
        "analyze_material"
    ])
except ImportError as e:
    _logger.debug(f"Material analyzer not available: {e}")

try:
    from .pattern_detector import (
        get_pattern_detector,
        PatternDetector,
        PatternAnalysis,
        detect_pattern
    )
    __all__.extend([
        "get_pattern_detector",
        "PatternDetector",
        "PatternAnalysis",
        "detect_pattern"
    ])
except ImportError as e:
    _logger.debug(f"Pattern detector not available: {e}")

try:
    from .ensemble_detector import (
        get_ensemble_detector,
        EnsembleDetector,
        EnsembleDetection,
        EnsembleResult,
        detect_ensemble
    )
    __all__.extend([
        "get_ensemble_detector",
        "EnsembleDetector",
        "EnsembleDetection",
        "EnsembleResult",
        "detect_ensemble"
    ])
except ImportError as e:
    _logger.debug(f"Ensemble detector not available: {e}")

try:
    from .style_intelligence import (
        get_style_intelligence,
        StyleIntelligence,
        StyleAnalysis,
        analyze_style,
        match_occasion
    )
    __all__.extend([
        "get_style_intelligence",
        "StyleIntelligence",
        "StyleAnalysis",
        "analyze_style",
        "match_occasion"
    ])
except ImportError as e:
    _logger.debug(f"Style intelligence not available: {e}")

try:
    from .color_harmony import (
        get_color_harmony_analyzer,
        ColorHarmonyAnalyzer,
        ColorHarmonyAnalysis,
        analyze_color_harmony
    )
    __all__.extend([
        "get_color_harmony_analyzer",
        "ColorHarmonyAnalyzer",
        "ColorHarmonyAnalysis",
        "analyze_color_harmony"
    ])
except ImportError as e:
    _logger.debug(f"Color harmony not available: {e}")

try:
    from .outfit_analyzer import (
        get_outfit_analyzer,
        OutfitCoherenceAnalyzer,
        OutfitAnalysis,
        OutfitItem,
        analyze_outfit
    )
    __all__.extend([
        "get_outfit_analyzer",
        "OutfitCoherenceAnalyzer",
        "OutfitAnalysis",
        "OutfitItem",
        "analyze_outfit"
    ])
except ImportError as e:
    _logger.debug(f"Outfit analyzer not available: {e}")

try:
    from .garment_3d import (
        get_garment_analyzer_3d,
        GarmentAnalyzer3D,
        GarmentAnalysis3D,
        analyze_garment_3d
    )
    __all__.extend([
        "get_garment_analyzer_3d",
        "GarmentAnalyzer3D",
        "GarmentAnalysis3D",
        "analyze_garment_3d"
    ])
except ImportError as e:
    _logger.debug(f"Garment 3D not available: {e}")

try:
    from .model_optimizer import (
        get_model_optimizer,
        ModelOptimizer,
        OptimizationResult,
        BenchmarkResult,
        benchmark_all_pipelines,
        get_optimization_recommendations
    )
    __all__.extend([
        "get_model_optimizer",
        "ModelOptimizer",
        "OptimizationResult",
        "BenchmarkResult",
        "benchmark_all_pipelines",
        "get_optimization_recommendations"
    ])
except ImportError as e:
    _logger.debug(f"Model optimizer not available: {e}")

try:
    from .ultimate_detector import (
        get_ultimate_detector,
        UltimateClothingDetector,
        UltimateDetection,
        detect_ultimate,
        CLOTHING_PROMPTS,
        ALL_CLOTHING_PROMPTS
    )
    __all__.extend([
        "get_ultimate_detector",
        "UltimateClothingDetector",
        "UltimateDetection",
        "detect_ultimate",
        "CLOTHING_PROMPTS",
        "ALL_CLOTHING_PROMPTS"
    ])
except ImportError as e:
    _logger.debug(f"Ultimate detector not available: {e}")

# === UNIFIED MULTIMODAL PIPELINE (SOTA 2025) ===
try:
    from .unified_pipeline import (
        get_unified_pipeline,
        UnifiedMultimodalPipeline,
        UnifiedDetection,
        PipelineResult,
        process_with_unified_pipeline
    )
    __all__.extend([
        "get_unified_pipeline",
        "UnifiedMultimodalPipeline",
        "UnifiedDetection",
        "PipelineResult",
        "process_with_unified_pipeline"
    ])
except ImportError as e:
    _logger.debug(f"Unified pipeline not available: {e}")

try:
    from .samurai_tracker import (
        get_samurai_tracker,
        SAMURAITracker,
        TrackedObject,
        KalmanFilter
    )
    __all__.extend([
        "get_samurai_tracker",
        "SAMURAITracker",
        "TrackedObject",
        "KalmanFilter"
    ])
except ImportError as e:
    _logger.debug(f"SAMURAI tracker not available: {e}")

try:
    from .distillation_engine import (
        get_distillation_engine,
        DistillationEngine,
        SocraticTrace,
        TeacherModel
    )
    __all__.extend([
        "get_distillation_engine",
        "DistillationEngine",
        "SocraticTrace",
        "TeacherModel"
    ])
except ImportError as e:
    _logger.debug(f"Distillation engine not available: {e}")

try:
    from .async_ingestion import (
        get_ingestion_engine,
        AsyncIngestionEngine,
        VectorDatabase,
        FashionEmbeddingModel
    )
    __all__.extend([
        "get_ingestion_engine",
        "AsyncIngestionEngine",
        "VectorDatabase",
        "FashionEmbeddingModel"
    ])
except ImportError as e:
    _logger.debug(f"Async ingestion not available: {e}")

try:
    from .gaussian_splatting import (
        get_splat_trainer,
        get_splat_renderer,
        get_hybrid_strategy,
        GaussianSplatTrainer,
        GaussianSplatRenderer,
        GaussianSplatScene
    )
    __all__.extend([
        "get_splat_trainer",
        "get_splat_renderer",
        "get_hybrid_strategy",
        "GaussianSplatTrainer",
        "GaussianSplatRenderer",
        "GaussianSplatScene"
    ])
except ImportError as e:
    _logger.debug(f"Gaussian splatting not available: {e}")

try:
    from .florence2_detector import (
        get_florence2_detector,
        Florence2Detector,
        detect_with_florence2
    )
    __all__.extend([
        "get_florence2_detector",
        "Florence2Detector",
        "detect_with_florence2"
    ])
except ImportError as e:
    _logger.debug(f"Florence-2 detector not available: {e}")

# === GOLDEN PATH MODULES (LATE 2025 SOTA) ===
try:
    from .qwen_vision import (
        get_qwen_analyzer,
        QwenVisionAnalyzer,
        QwenFashionAnalysis,
        analyze_with_qwen
    )
    __all__.extend([
        "get_qwen_analyzer",
        "QwenVisionAnalyzer",
        "QwenFashionAnalysis",
        "analyze_with_qwen"
    ])
except ImportError as e:
    _logger.debug(f"Qwen vision not available: {e}")

try:
    from .meshsplats import (
        get_meshsplats_pipeline,
        MeshSplatsPipeline,
        GSplatTrainer,
        MeshSplatsConverter,
        process_video_to_mesh
    )
    __all__.extend([
        "get_meshsplats_pipeline",
        "MeshSplatsPipeline",
        "GSplatTrainer",
        "MeshSplatsConverter",
        "process_video_to_mesh"
    ])
except ImportError as e:
    _logger.debug(f"MeshSplats not available: {e}")

try:
    from .golden_path_pipeline import (
        get_golden_path_pipeline,
        GoldenPathPipeline,
        GoldenPathDetection,
        GoldenPathResult,
        analyze_with_golden_path
    )
    __all__.extend([
        "get_golden_path_pipeline",
        "GoldenPathPipeline",
        "GoldenPathDetection",
        "GoldenPathResult",
        "analyze_with_golden_path"
    ])
except ImportError as e:
    _logger.debug(f"Golden Path not available: {e}")
