"""
Model Optimization Module - ONNX Export and Performance Tuning
Optimizes AI models for faster inference in production

This module provides:
- ONNX export for models
- Model quantization
- Batch inference optimization
- Memory management
- Performance benchmarking
"""

import cv2
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import logging
import time
import os

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Result from model optimization"""
    model_name: str
    original_size_mb: float
    optimized_size_mb: float
    size_reduction: float
    original_latency_ms: float
    optimized_latency_ms: float
    speedup: float
    optimization_type: str  # "onnx", "quantization", "pruning"
    status: str  # "success", "failed", "not_applicable"
    notes: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BenchmarkResult:
    """Benchmark results for a model/pipeline"""
    name: str
    input_size: Tuple[int, int]
    latency_ms: float
    throughput_fps: float
    memory_mb: float
    device: str
    batch_size: int
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ModelOptimizer:
    """
    ⚡ Model Optimization for Production
    
    Optimizes AI models for faster inference:
    - ONNX export for cross-platform deployment
    - Quantization (INT8/FP16) for smaller models
    - Batch processing optimization
    - Memory-efficient inference
    
    Supports:
    - PyTorch models
    - CLIP models
    - YOLO models
    - Transformers (SegFormer, etc.)
    
    Usage:
        optimizer = ModelOptimizer()
        result = optimizer.export_to_onnx(model, "segformer")
        result = optimizer.benchmark_model("segformer", image)
    """
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize model optimizer.
        
        Args:
            cache_dir: Directory for cached optimized models
        """
        self.cache_dir = cache_dir or os.path.expanduser("~/.cache/alicevision/optimized")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self._setup_device()
        logger.info(f"ModelOptimizer initialized (device={self.device})")
    
    def _setup_device(self):
        """Setup compute device."""
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
    
    def export_to_onnx(
        self,
        model: torch.nn.Module,
        model_name: str,
        input_shape: Tuple[int, ...] = (1, 3, 224, 224),
        dynamic_axes: Dict = None
    ) -> OptimizationResult:
        """
        Export PyTorch model to ONNX format.
        
        Args:
            model: PyTorch model
            model_name: Name for saved model
            input_shape: Input tensor shape
            dynamic_axes: Dynamic axes for variable input sizes
            
        Returns:
            OptimizationResult with export details
        """
        try:
            import onnx
            import onnxruntime as ort
            
            # Original size
            original_size = self._get_model_size(model)
            
            # Benchmark original
            dummy_input = torch.randn(*input_shape)
            original_latency = self._benchmark_pytorch(model, dummy_input)
            
            # Export path
            onnx_path = os.path.join(self.cache_dir, f"{model_name}.onnx")
            
            # Default dynamic axes for batch dimension
            if dynamic_axes is None:
                dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}
            
            # Export
            model.eval()
            with torch.no_grad():
                torch.onnx.export(
                    model,
                    dummy_input,
                    onnx_path,
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes=dynamic_axes,
                    opset_version=14
                )
            
            # Verify
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            
            # Get optimized size
            optimized_size = os.path.getsize(onnx_path) / (1024 * 1024)
            
            # Benchmark ONNX
            session = ort.InferenceSession(onnx_path)
            optimized_latency = self._benchmark_onnx(session, dummy_input.numpy())
            
            return OptimizationResult(
                model_name=model_name,
                original_size_mb=original_size,
                optimized_size_mb=optimized_size,
                size_reduction=(original_size - optimized_size) / original_size,
                original_latency_ms=original_latency,
                optimized_latency_ms=optimized_latency,
                speedup=original_latency / optimized_latency if optimized_latency > 0 else 1.0,
                optimization_type="onnx",
                status="success",
                notes=f"Exported to {onnx_path}"
            )
            
        except ImportError:
            return OptimizationResult(
                model_name=model_name,
                original_size_mb=0, optimized_size_mb=0, size_reduction=0,
                original_latency_ms=0, optimized_latency_ms=0, speedup=0,
                optimization_type="onnx",
                status="failed",
                notes="ONNX/ONNXRuntime not installed. Run: pip install onnx onnxruntime"
            )
        except Exception as e:
            return OptimizationResult(
                model_name=model_name,
                original_size_mb=0, optimized_size_mb=0, size_reduction=0,
                original_latency_ms=0, optimized_latency_ms=0, speedup=0,
                optimization_type="onnx",
                status="failed",
                notes=str(e)
            )
    
    def quantize_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        method: str = "dynamic"
    ) -> OptimizationResult:
        """
        Quantize model for smaller size and faster inference.
        
        Args:
            model: PyTorch model
            model_name: Name for saved model
            method: "dynamic" (post-training) or "static" (calibrated)
            
        Returns:
            OptimizationResult with quantization details
        """
        try:
            # Original metrics
            original_size = self._get_model_size(model)
            dummy_input = torch.randn(1, 3, 224, 224)
            original_latency = self._benchmark_pytorch(model, dummy_input)
            
            if method == "dynamic":
                # Dynamic quantization (no calibration needed)
                quantized = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear, torch.nn.Conv2d},
                    dtype=torch.qint8
                )
            else:
                # Static quantization would require calibration data
                logger.warning("Static quantization not fully implemented, using dynamic")
                quantized = torch.quantization.quantize_dynamic(
                    model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
            
            # Save quantized model
            save_path = os.path.join(self.cache_dir, f"{model_name}_quantized.pt")
            torch.save(quantized.state_dict(), save_path)
            
            optimized_size = os.path.getsize(save_path) / (1024 * 1024)
            optimized_latency = self._benchmark_pytorch(quantized, dummy_input)
            
            return OptimizationResult(
                model_name=model_name,
                original_size_mb=original_size,
                optimized_size_mb=optimized_size,
                size_reduction=(original_size - optimized_size) / original_size if original_size > 0 else 0,
                original_latency_ms=original_latency,
                optimized_latency_ms=optimized_latency,
                speedup=original_latency / optimized_latency if optimized_latency > 0 else 1.0,
                optimization_type=f"quantization_{method}",
                status="success",
                notes=f"Quantized with {method} method"
            )
            
        except Exception as e:
            return OptimizationResult(
                model_name=model_name,
                original_size_mb=0, optimized_size_mb=0, size_reduction=0,
                original_latency_ms=0, optimized_latency_ms=0, speedup=0,
                optimization_type="quantization",
                status="failed",
                notes=str(e)
            )
    
    def benchmark_pipeline(
        self,
        pipeline_name: str,
        test_images: List[np.ndarray] = None,
        num_runs: int = 10
    ) -> BenchmarkResult:
        """
        Benchmark an AliceVision pipeline.
        
        Args:
            pipeline_name: "segformer", "fashion_clip", "ensemble", etc.
            test_images: Optional list of test images
            num_runs: Number of runs for averaging
            
        Returns:
            BenchmarkResult with performance metrics
        """
        import tracemalloc
        
        # Create test image if not provided
        if test_images is None:
            test_images = [np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)]
        
        image = test_images[0]
        h, w = image.shape[:2]
        
        # Get pipeline
        pipeline_func = self._get_pipeline(pipeline_name)
        
        if pipeline_func is None:
            return BenchmarkResult(
                name=pipeline_name,
                input_size=(w, h),
                latency_ms=0,
                throughput_fps=0,
                memory_mb=0,
                device=self.device,
                batch_size=1
            )
        
        # Warmup
        for _ in range(2):
            try:
                pipeline_func(image)
            except:
                pass
        
        # Benchmark
        tracemalloc.start()
        
        times = []
        for _ in range(num_runs):
            start = time.time()
            try:
                pipeline_func(image)
            except Exception as e:
                logger.warning(f"Pipeline error: {e}")
            times.append((time.time() - start) * 1000)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        avg_latency = np.mean(times) if times else 0
        fps = 1000 / avg_latency if avg_latency > 0 else 0
        
        return BenchmarkResult(
            name=pipeline_name,
            input_size=(w, h),
            latency_ms=round(avg_latency, 2),
            throughput_fps=round(fps, 2),
            memory_mb=round(peak / (1024 * 1024), 2),
            device=self.device,
            batch_size=1
        )
    
    def optimize_for_device(self, device: str = None) -> Dict[str, Any]:
        """
        Get optimization recommendations for current device.
        
        Args:
            device: "mps", "cuda", "cpu" or None for auto-detect
            
        Returns:
            Dict with optimization recommendations
        """
        if device is None:
            device = self.device
        
        recommendations = {
            "device": device,
            "optimizations": [],
            "settings": {}
        }
        
        if device == "mps":
            recommendations["optimizations"] = [
                "Use Metal Performance Shaders (MPS) for tensor operations",
                "Enable half precision (FP16) for faster inference",
                "Use ONNX Runtime with CoreML backend for CPU fallback",
                "Batch processing improves throughput on M-series chips"
            ]
            recommendations["settings"] = {
                "use_fp16": True,
                "batch_size": 4,
                "num_threads": 8,
                "memory_fraction": 0.7
            }
        elif device == "cuda":
            recommendations["optimizations"] = [
                "Use TensorRT for maximum NVIDIA GPU performance",
                "Enable FP16/INT8 quantization for faster inference",
                "Use CUDA shared memory for batch processing",
                "Enable cuDNN autotuning"
            ]
            recommendations["settings"] = {
                "use_fp16": True,
                "use_tensorrt": True,
                "batch_size": 8,
                "cudnn_benchmark": True
            }
        else:
            recommendations["optimizations"] = [
                "Use ONNX Runtime for optimized CPU inference",
                "Enable multi-threading for parallel processing",
                "Consider INT8 quantization for 2-3x speedup",
                "Use smaller models (MobileNet, EfficientNet-B0)"
            ]
            recommendations["settings"] = {
                "use_onnx": True,
                "num_threads": os.cpu_count(),
                "batch_size": 1,
                "use_quantization": True
            }
        
        return recommendations
    
    def _get_model_size(self, model: torch.nn.Module) -> float:
        """Get model size in MB."""
        temp_path = "/tmp/temp_model.pt"
        torch.save(model.state_dict(), temp_path)
        size = os.path.getsize(temp_path) / (1024 * 1024)
        os.remove(temp_path)
        return size
    
    def _benchmark_pytorch(
        self,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
        num_runs: int = 10
    ) -> float:
        """Benchmark PyTorch model inference."""
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(2):
                _ = model(input_tensor)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.time()
                _ = model(input_tensor)
                times.append((time.time() - start) * 1000)
        
        return np.mean(times)
    
    def _benchmark_onnx(
        self,
        session,
        input_array: np.ndarray,
        num_runs: int = 10
    ) -> float:
        """Benchmark ONNX model inference."""
        input_name = session.get_inputs()[0].name
        
        # Warmup
        for _ in range(2):
            _ = session.run(None, {input_name: input_array})
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.time()
            _ = session.run(None, {input_name: input_array})
            times.append((time.time() - start) * 1000)
        
        return np.mean(times)
    
    def _get_pipeline(self, name: str):
        """Get pipeline function by name."""
        pipelines = {
            "segformer": self._get_segformer_pipeline,
            "fashion_clip": self._get_clip_pipeline,
            "material": self._get_material_pipeline,
            "pattern": self._get_pattern_pipeline,
            "style": self._get_style_pipeline
        }
        
        getter = pipelines.get(name)
        if getter:
            return getter()
        return None
    
    def _get_segformer_pipeline(self):
        """Get SegFormer pipeline."""
        try:
            from modules.segmentation import AdvancedClothingSegmentor
            segmentor = AdvancedClothingSegmentor()
            return lambda img: segmentor.segment(img)
        except:
            return None
    
    def _get_clip_pipeline(self):
        """Get Fashion-CLIP pipeline."""
        try:
            from modules.hierarchical_classifier import get_hierarchical_classifier
            classifier = get_hierarchical_classifier()
            return lambda img: classifier.classify(img)
        except:
            return None
    
    def _get_material_pipeline(self):
        """Get material analyzer pipeline."""
        try:
            from modules.material_analyzer import get_material_analyzer
            analyzer = get_material_analyzer()
            return lambda img: analyzer.analyze(img)
        except:
            return None
    
    def _get_pattern_pipeline(self):
        """Get pattern detector pipeline."""
        try:
            from modules.pattern_detector import get_pattern_detector
            detector = get_pattern_detector()
            return lambda img: detector.analyze(img)
        except:
            return None
    
    def _get_style_pipeline(self):
        """Get style intelligence pipeline."""
        try:
            from modules.style_intelligence import get_style_intelligence
            analyzer = get_style_intelligence()
            return lambda img: analyzer.analyze_style(img)
        except:
            return None


# === SINGLETON INSTANCE ===
_optimizer_instance = None


def get_model_optimizer() -> ModelOptimizer:
    """Get singleton instance."""
    global _optimizer_instance
    if _optimizer_instance is None:
        _optimizer_instance = ModelOptimizer()
    return _optimizer_instance


def benchmark_all_pipelines() -> Dict[str, Dict]:
    """Benchmark all available pipelines."""
    optimizer = get_model_optimizer()
    
    pipelines = ["segformer", "fashion_clip", "material", "pattern", "style"]
    results = {}
    
    for name in pipelines:
        result = optimizer.benchmark_pipeline(name)
        results[name] = result.to_dict()
    
    return results


def get_optimization_recommendations() -> Dict:
    """Get device-specific optimization recommendations."""
    optimizer = get_model_optimizer()
    return optimizer.optimize_for_device()
