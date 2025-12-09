"""
AliceVision Vercel Serverless API
FastAPI application adapted for Vercel deployment
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
import base64
from PIL import Image
import io

# Initialize FastAPI app
app = FastAPI(
    title="AliceVision Service",
    description="Computer vision microservice for AIWardrobe",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Request/Response Models
# ============================================

class KeyframeRequest(BaseModel):
    frames: List[str] = Field(..., description="List of base64-encoded video frames")
    sharpness_weight: float = Field(0.4, ge=0, le=1)
    blur_penalty: float = Field(0.3, ge=0, le=1)
    centering_weight: float = Field(0.2, ge=0, le=1)


class SegmentationRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image")
    add_white_background: bool = Field(True)


class LightingRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded image")
    target_brightness: float = Field(0.55, ge=0, le=1)
    target_temperature: float = Field(6000, ge=3000, le=10000)


# ============================================
# Helper Functions
# ============================================

def decode_base64_image(b64_string: str) -> np.ndarray:
    """Decode base64 to OpenCV image"""
    if ',' in b64_string:
        b64_string = b64_string.split(',')[1]
    img_bytes = base64.b64decode(b64_string)
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)


def encode_image_base64(image: np.ndarray, format: str = 'jpeg') -> str:
    """Encode OpenCV image to base64"""
    if format == 'png':
        _, buffer = cv2.imencode('.png', image)
    else:
        _, buffer = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buffer).decode('utf-8')


def calculate_sharpness(image: np.ndarray) -> float:
    """Calculate image sharpness using Laplacian variance"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return min(laplacian.var() / 500.0, 1.0)


def calculate_brightness(image: np.ndarray) -> float:
    """Calculate brightness score"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    mean_brightness = gray.mean() / 255.0
    if 0.35 <= mean_brightness <= 0.65:
        return 1.0
    elif mean_brightness < 0.35:
        return mean_brightness / 0.35
    else:
        return (1.0 - mean_brightness) / 0.35


def segment_with_grabcut(image: np.ndarray) -> np.ndarray:
    """GrabCut-based segmentation"""
    h, w = image.shape[:2]
    margin_x, margin_y = int(w * 0.1), int(h * 0.05)
    rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)
    
    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 3, cv2.GC_INIT_WITH_RECT)
    
    binary_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    
    # Create RGBA image
    b, g, r = cv2.split(image)
    result = cv2.merge([r, g, b, binary_mask])
    
    return result


def add_white_background(rgba_image: np.ndarray) -> np.ndarray:
    """Add white background to RGBA image"""
    if rgba_image.shape[2] != 4:
        return rgba_image
    
    h, w = rgba_image.shape[:2]
    white_bg = np.ones((h, w, 3), dtype=np.uint8) * 255
    
    alpha = rgba_image[:, :, 3:4] / 255.0
    rgb = rgba_image[:, :, :3]
    
    result = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
    return result


def normalize_lighting(image: np.ndarray, target_brightness: float = 0.55) -> np.ndarray:
    """Simple lighting normalization"""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    current_brightness = lab[:, :, 0].mean() / 255.0
    adjustment = (target_brightness - current_brightness) * 255
    lab[:, :, 0] = np.clip(lab[:, :, 0] + adjustment, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


# ============================================
# API Endpoints
# ============================================

@app.get("/")
async def root():
    return {
        "service": "AliceVision",
        "status": "healthy",
        "version": "1.0.0",
        "platform": "vercel"
    }


@app.get("/health")
async def health_check():
    return {"status": "ok", "modules": {"keyframe": "loaded", "segmentation": "loaded", "lighting": "loaded"}}


@app.post("/keyframe")
async def select_keyframe(request: KeyframeRequest):
    """Select the best frame from video frames"""
    try:
        if not request.frames:
            raise HTTPException(status_code=400, detail="No frames provided")
        
        scores = []
        for i, b64 in enumerate(request.frames[:10]):  # Limit to 10 frames for serverless
            img = decode_base64_image(b64)
            if img is None:
                continue
            
            sharpness = calculate_sharpness(img)
            brightness = calculate_brightness(img)
            
            total = (sharpness * request.sharpness_weight + 
                    brightness * (1 - request.sharpness_weight - request.blur_penalty))
            
            scores.append({"index": i, "sharpness": sharpness, "brightness": brightness, "total": total})
        
        if not scores:
            raise HTTPException(status_code=400, detail="No valid frames found")
        
        best = max(scores, key=lambda x: x["total"])
        
        return {
            "success": True,
            "bestFrameIndex": best["index"],
            "scores": {
                "sharpness": round(best["sharpness"], 4),
                "brightness": round(best["brightness"], 4),
                "totalScore": round(best["total"], 4)
            },
            "totalFramesAnalyzed": len(scores)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment")
async def segment_clothing(request: SegmentationRequest):
    """Segment clothing from image"""
    try:
        image = decode_base64_image(request.image)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        # Use GrabCut for segmentation
        segmented = segment_with_grabcut(image)
        
        if request.add_white_background:
            result = add_white_background(segmented)
            result_b64 = encode_image_base64(result, 'jpeg')
            mime = "image/jpeg"
        else:
            result_b64 = encode_image_base64(segmented, 'png')
            mime = "image/png"
        
        return {
            "success": True,
            "segmentedImage": f"data:{mime};base64,{result_b64}",
            "confidence": 0.7,
            "hasTransparency": not request.add_white_background
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lighting")
async def lighting_normalize(request: LightingRequest):
    """Normalize image lighting"""
    try:
        image = decode_base64_image(request.image)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        normalized = normalize_lighting(image, request.target_brightness)
        result_b64 = encode_image_base64(normalized, 'jpeg')
        
        return {
            "success": True,
            "normalizedImage": f"data:image/jpeg;base64,{result_b64}",
            "originalAnalysis": {
                "brightness": round(calculate_brightness(image), 4)
            },
            "appliedCorrections": {
                "brightness": round(request.target_brightness - calculate_brightness(image), 4)
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
