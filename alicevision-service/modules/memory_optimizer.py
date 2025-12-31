"""
Memory-optimized configuration for free tier deployment
Lazy loading and model cleanup to fit within 512MB RAM
"""

import gc
import torch
from functools import wraps

# Global flag for free tier optimization
FREE_TIER_MODE = True  # Set to False for paid tier (faster but more RAM)

def cleanup_memory(func):
    """Decorator to clean up memory after model inference"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        # Aggressive garbage collection for free tier
        if FREE_TIER_MODE:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return result
    return wrapper


class ModelManager:
    """
    Singleton model manager for efficient memory usage.
    Loads models on-demand and unloads when not in use.
    """
    _instance = None
    _models = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_model(self, model_name: str):
        """Get model, loading if necessary"""
        if model_name not in self._models:
            self._load_model(model_name)
        return self._models[model_name]
    
    def _load_model(self, model_name: str):
        """Load a specific model"""
        if model_name == "yolo":
            from ultralytics import YOLO
            # Use nano model for free tier (smallest, fastest)
            self._models["yolo"] = YOLO('yolov8n.pt')
            
        elif model_name == "clip":
            import open_clip
            # Use smaller CLIP model
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-B-32',  # Smaller than ViT-L
                pretrained='openai'
            )
            model.eval()
            self._models["clip"] = {
                "model": model,
                "preprocess": preprocess
            }
            
        elif model_name == "segformer":
            from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
            processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
            model = SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
            model.eval()
            self._models["segformer"] = {
                "model": model,
                "processor": processor
            }
            
        elif model_name == "resnet":
            import torchvision.models as models
            import torchvision.transforms as transforms
            model = models.resnet50(pretrained=True)
            model = torch.nn.Sequential(*list(model.children())[:-1])
            model.eval()
            
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
            
            self._models["resnet"] = {
                "model": model,
                "transform": transform
            }
    
    def unload_model(self, model_name: str):
        """Unload a model to free memory"""
        if model_name in self._models:
            del self._models[model_name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def unload_all(self):
        """Unload all models"""
        self._models.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# Singleton instance
model_manager = ModelManager()
