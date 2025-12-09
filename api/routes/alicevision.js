/**
 * AliceVision Integration Routes
 * 
 * Routes for the AliceVision Python microservice
 * Provides enhanced keyframe selection, segmentation, and lighting normalization
 */

import express from "express";
import axios from "axios";

const router = express.Router();

// AliceVision service URL (environment variable or default)
const ALICEVISION_URL = process.env.ALICEVISION_URL || "http://localhost:5050";

/**
 * Helper function to call AliceVision service
 */
const callAliceVision = async (endpoint, data, timeout = 60000) => {
    try {
        const response = await axios.post(`${ALICEVISION_URL}${endpoint}`, data, {
            timeout,
            headers: { "Content-Type": "application/json" }
        });
        return response.data;
    } catch (error) {
        console.error(`AliceVision ${endpoint} error:`, error.message);
        if (error.response) {
            throw new Error(error.response.data?.detail || error.message);
        }
        throw error;
    }
};

// ============================================
// KEYFRAME SELECTION
// ============================================

/**
 * POST /alicevision/keyframe
 * Select the best frame from video frames
 */
router.post("/keyframe", async (req, res) => {
    try {
        const { frames, sharpness_weight, blur_penalty, centering_weight } = req.body;

        if (!frames || !Array.isArray(frames) || frames.length === 0) {
            return res.status(400).json({ error: "Frames array is required" });
        }

        console.log(`🎬 Selecting best frame from ${frames.length} frames...`);

        const result = await callAliceVision("/keyframe", {
            frames,
            sharpness_weight: sharpness_weight || 0.4,
            blur_penalty: blur_penalty || 0.3,
            centering_weight: centering_weight || 0.2
        });

        console.log(`✅ Best frame: ${result.bestFrameIndex} (score: ${result.scores.totalScore})`);

        res.json(result);
    } catch (error) {
        console.error("Keyframe selection error:", error.message);
        res.status(500).json({ error: error.message });
    }
});

// ============================================
// CLOTHING SEGMENTATION
// ============================================

/**
 * POST /alicevision/segment
 * Segment clothing from an image with edge refinement
 */
router.post("/segment", async (req, res) => {
    try {
        const { image, add_white_background } = req.body;

        if (!image) {
            return res.status(400).json({ error: "Image is required" });
        }

        console.log("✂️ Segmenting clothing from image...");

        const result = await callAliceVision("/segment", {
            image,
            add_white_background: add_white_background !== false
        });

        console.log(`✅ Segmentation complete (confidence: ${result.confidence})`);

        res.json(result);
    } catch (error) {
        console.error("Segmentation error:", error.message);
        res.status(500).json({ error: error.message });
    }
});

// ============================================
// LIGHTING NORMALIZATION
// ============================================

/**
 * POST /alicevision/lighting
 * Normalize image lighting for catalog-quality photos
 */
router.post("/lighting", async (req, res) => {
    try {
        const { image, target_brightness, target_temperature, add_vignette } = req.body;

        if (!image) {
            return res.status(400).json({ error: "Image is required" });
        }

        console.log("💡 Normalizing image lighting...");

        const result = await callAliceVision("/lighting", {
            image,
            target_brightness: target_brightness || 0.55,
            target_temperature: target_temperature || 6000,
            add_vignette: add_vignette || false
        });

        console.log("✅ Lighting normalization complete");

        res.json(result);
    } catch (error) {
        console.error("Lighting normalization error:", error.message);
        res.status(500).json({ error: error.message });
    }
});

// ============================================
// FULL PROCESSING PIPELINE
// ============================================

/**
 * POST /alicevision/process
 * Full pipeline: keyframe → segmentation → lighting
 */
router.post("/process", async (req, res) => {
    try {
        const {
            frames,
            add_white_background,
            normalize_lighting,
            target_brightness,
            target_temperature
        } = req.body;

        if (!frames || !Array.isArray(frames) || frames.length === 0) {
            return res.status(400).json({ error: "Frames array is required" });
        }

        console.log(`🚀 Running full AliceVision pipeline on ${frames.length} frames...`);

        const result = await callAliceVision("/process", {
            frames,
            add_white_background: add_white_background !== false,
            normalize_lighting: normalize_lighting !== false,
            target_brightness: target_brightness || 0.55,
            target_temperature: target_temperature || 6000
        }, 120000); // 2 minute timeout for full pipeline

        console.log(`✅ Full pipeline complete: ${result.processingSteps.join(" → ")}`);

        res.json(result);
    } catch (error) {
        console.error("Full pipeline error:", error.message);
        res.status(500).json({ error: error.message });
    }
});

// ============================================
// HEALTH CHECK
// ============================================

/**
 * GET /alicevision/health
 * Check if AliceVision service is available
 */
router.get("/health", async (req, res) => {
    try {
        const response = await axios.get(`${ALICEVISION_URL}/health`, { timeout: 5000 });
        res.json({
            status: "connected",
            alicevision: response.data
        });
    } catch (error) {
        res.status(503).json({
            status: "disconnected",
            error: error.message,
            help: "Start the AliceVision service with: docker-compose up -d"
        });
    }
});

export default router;
