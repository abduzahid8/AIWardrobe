"""
AliceVision Microservice for AIWardrobe
FastAPI application providing computer vision endpoints
"""

import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
from modules import (
    select_best_frame_from_base64,
    segment_clothing_from_base64,
    normalize_lighting_from_base64
)

# Initialize FastAPI app
app = FastAPI(
    title="AliceVision Service",
    description="Computer vision microservice for AIWardrobe - keyframe selection, segmentation, and lighting normalization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================
# Request/Response Models
# ============================================

class KeyframeRequest(BaseModel):
    """Request model for keyframe selection"""
    frames: List[str] = Field(..., description="List of base64-encoded video frames")
    sharpness_weight: float = Field(0.4, ge=0, le=1, description="Weight for sharpness scoring")
    blur_penalty: float = Field(0.3, ge=0, le=1, description="Penalty for motion blur")
    centering_weight: float = Field(0.2, ge=0, le=1, description="Weight for subject centering")


class KeyframeResponse(BaseModel):
    """Response model for keyframe selection"""
    success: bool
    bestFrameIndex: int
    scores: Dict[str, float]
    totalFramesAnalyzed: int


class SegmentationRequest(BaseModel):
    """Request model for clothing segmentation"""
    image: str = Field(..., description="Base64-encoded image")
    add_white_background: bool = Field(True, description="Add white background to result")


class SegmentationResponse(BaseModel):
    """Response model for clothing segmentation"""
    success: bool
    segmentedImage: str
    confidence: float
    boundingBox: Optional[List[int]] = None
    hasTransparency: bool


class LightingRequest(BaseModel):
    """Request model for lighting normalization"""
    image: str = Field(..., description="Base64-encoded image")
    target_brightness: float = Field(0.55, ge=0, le=1, description="Target brightness level")
    target_temperature: float = Field(6000, ge=3000, le=10000, description="Target color temperature in Kelvin")
    add_vignette: bool = Field(False, description="Add subtle vignette effect")


class LightingResponse(BaseModel):
    """Response model for lighting normalization"""
    success: bool
    normalizedImage: str
    originalAnalysis: Dict[str, Any]
    appliedCorrections: Dict[str, float]


class FullPipelineRequest(BaseModel):
    """Request model for full processing pipeline"""
    frames: List[str] = Field(..., description="List of base64-encoded video frames")
    add_white_background: bool = Field(True, description="Add white background after segmentation")
    normalize_lighting: bool = Field(True, description="Apply lighting normalization")
    target_brightness: float = Field(0.55, ge=0, le=1)
    target_temperature: float = Field(6000, ge=3000, le=10000)


class FullPipelineResponse(BaseModel):
    """Response model for full pipeline"""
    success: bool
    finalImage: str
    bestFrameIndex: int
    keyframeScores: Dict[str, float]
    segmentationConfidence: float
    lightingCorrections: Optional[Dict[str, float]] = None
    processingSteps: List[str]


# ============================================
# API Endpoints
# ============================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "AliceVision",
        "status": "healthy",
        "version": "1.0.0",
        "endpoints": [
            "/keyframe - Smart frame selection",
            "/segment - Clothing segmentation", 
            "/lighting - Lighting normalization",
            "/process - Full pipeline"
        ]
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "ok",
        "modules": {
            "keyframe": "loaded",
            "segmentation": "loaded",
            "lighting": "loaded"
        }
    }


@app.post("/keyframe", response_model=KeyframeResponse)
async def select_keyframe(request: KeyframeRequest):
    """
    Select the best frame from a video based on quality metrics.
    
    Analyzes frames for:
    - Sharpness (Laplacian variance)
    - Motion blur (FFT analysis)
    - Subject centering (contour analysis)
    - Brightness (histogram analysis)
    """
    try:
        logger.info(f"Received {len(request.frames)} frames for keyframe selection")
        
        result = select_best_frame_from_base64(
            request.frames,
            sharpness_weight=request.sharpness_weight,
            blur_penalty=request.blur_penalty,
            centering_weight=request.centering_weight
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        logger.info(f"Selected frame {result['bestFrameIndex']} with score {result['scores']['totalScore']:.4f}")
        
        return KeyframeResponse(
            success=True,
            bestFrameIndex=result["bestFrameIndex"],
            scores=result["scores"],
            totalFramesAnalyzed=result["totalFramesAnalyzed"]
        )
        
    except Exception as e:
        logger.error(f"Keyframe selection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/segment", response_model=SegmentationResponse)
async def segment_clothing(request: SegmentationRequest):
    """
    Segment clothing from an image.
    
    Uses rembg with clothing-specific model, falling back to GrabCut.
    Applies edge refinement for smooth transitions.
    """
    try:
        logger.info("Starting clothing segmentation")
        
        result = segment_clothing_from_base64(
            request.image,
            add_white_bg=request.add_white_background
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        logger.info(f"Segmentation complete with confidence {result['confidence']:.4f}")
        
        bbox = result.get("boundingBox")
        if bbox:
            bbox = list(bbox)
        
        return SegmentationResponse(
            success=True,
            segmentedImage=result["segmentedImage"],
            confidence=result["confidence"],
            boundingBox=bbox,
            hasTransparency=result["hasTransparency"]
        )
        
    except Exception as e:
        logger.error(f"Segmentation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lighting", response_model=LightingResponse)
async def normalize_lighting(request: LightingRequest):
    """
    Normalize image lighting for catalog-quality photos.
    
    Adjusts:
    - White balance to target temperature
    - Exposure for optimal brightness
    - Contrast for clarity
    - Optional vignette for professional look
    """
    try:
        logger.info(f"Starting lighting normalization (target: {request.target_temperature}K, brightness: {request.target_brightness})")
        
        result = normalize_lighting_from_base64(
            request.image,
            target_brightness=request.target_brightness,
            target_temperature=request.target_temperature,
            add_vignette=request.add_vignette
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        logger.info(f"Lighting normalized with corrections: {result['appliedCorrections']}")
        
        return LightingResponse(
            success=True,
            normalizedImage=result["normalizedImage"],
            originalAnalysis=result["originalAnalysis"],
            appliedCorrections=result["appliedCorrections"]
        )
        
    except Exception as e:
        logger.error(f"Lighting normalization error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process", response_model=FullPipelineResponse)
async def full_pipeline(request: FullPipelineRequest):
    """
    Full processing pipeline: keyframe selection → segmentation → lighting normalization.
    
    This combines all three modules into a single optimized pipeline
    for processing video frames into catalog-quality product photos.
    """
    try:
        logger.info(f"Starting full pipeline with {len(request.frames)} frames")
        steps = []
        
        # Step 1: Keyframe selection
        logger.info("Step 1: Keyframe selection")
        keyframe_result = select_best_frame_from_base64(request.frames)
        
        if "error" in keyframe_result:
            raise HTTPException(status_code=400, detail=f"Keyframe: {keyframe_result['error']}")
        
        best_frame = request.frames[keyframe_result["bestFrameIndex"]]
        steps.append("keyframe_selection")
        logger.info(f"Selected frame {keyframe_result['bestFrameIndex']}")
        
        # Step 2: Segmentation
        logger.info("Step 2: Segmentation")
        segment_result = segment_clothing_from_base64(
            best_frame,
            add_white_bg=request.add_white_background
        )
        
        if "error" in segment_result:
            raise HTTPException(status_code=400, detail=f"Segmentation: {segment_result['error']}")
        
        current_image = segment_result["segmentedImage"]
        steps.append("segmentation")
        logger.info(f"Segmentation complete (confidence: {segment_result['confidence']:.4f})")
        
        # Step 3: Lighting normalization (optional)
        lighting_corrections = None
        if request.normalize_lighting:
            logger.info("Step 3: Lighting normalization")
            
            # Extract base64 from data URL
            if ',' in current_image:
                image_b64 = current_image.split(',')[1]
            else:
                image_b64 = current_image
            
            lighting_result = normalize_lighting_from_base64(
                image_b64,
                target_brightness=request.target_brightness,
                target_temperature=request.target_temperature
            )
            
            if "error" not in lighting_result:
                current_image = lighting_result["normalizedImage"]
                lighting_corrections = lighting_result["appliedCorrections"]
                steps.append("lighting_normalization")
                logger.info("Lighting normalization complete")
        
        logger.info(f"Full pipeline complete: {' → '.join(steps)}")
        
        return FullPipelineResponse(
            success=True,
            finalImage=current_image,
            bestFrameIndex=keyframe_result["bestFrameIndex"],
            keyframeScores=keyframe_result["scores"],
            segmentationConfidence=segment_result["confidence"],
            lightingCorrections=lighting_corrections,
            processingSteps=steps
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Full pipeline error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Main Entry Point
# ============================================

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5050))
    debug = os.getenv("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting AliceVision service on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug
    )
