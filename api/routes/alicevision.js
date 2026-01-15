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
// CLOTHING DETECTION (NO OpenAI/Gemini needed!)
// ============================================

/**
 * POST /alicevision/analyze
 * Local AI clothing detection - uses SegFormer + attribute extraction
 * No external API keys required!
 */
router.post("/analyze", async (req, res) => {
    try {
        const { frames } = req.body;

        if (!frames || !Array.isArray(frames) || frames.length === 0) {
            return res.status(400).json({ error: "Frames array is required" });
        }

        console.log(`🤖 Local AI analyzing ${frames.length} frames...`);

        // Use first frame for analysis
        const image = frames[0].replace(/^data:image\/\w+;base64,/, '');

        // Call AliceVision segment endpoint to detect clothing items
        const segResult = await callAliceVision("/segment", {
            image,
            add_white_background: false,
            use_advanced: true
        });

        // Map SegFormer items to our format
        const detectedItems = [];

        if (segResult.items && segResult.items.length > 0) {
            for (const item of segResult.items) {
                detectedItems.push({
                    itemType: item.category || "Clothing Item",
                    color: "Detected",
                    style: "Casual",
                    description: `${item.category} detected with ${Math.round(item.confidence * 100)}% confidence`,
                    position: getPosition(item.category),
                    confidence: Math.round(item.confidence * 100)
                });
            }
        } else {
            // Fallback: create basic item from segmentation confidence
            detectedItems.push({
                itemType: "Clothing Item",
                color: "Unknown",
                style: "Casual",
                description: "Clothing detected in frame",
                position: "upper",
                confidence: Math.round((segResult.confidence || 0.7) * 100)
            });
        }

        console.log(`✅ Local AI detected ${detectedItems.length} items`);

        res.json({
            detectedItems,
            source: "alicevision-local",
            segmentationConfidence: segResult.confidence
        });
    } catch (error) {
        console.error("Local AI analysis error:", error.message);
        res.status(500).json({ error: error.message });
    }
});

// Helper to map clothing category to body position
function getPosition(category) {
    const upper = ["shirt", "blouse", "sweater", "jacket", "coat", "top", "t-shirt", "hoodie", "upper-clothes"];
    const lower = ["pants", "jeans", "shorts", "skirt", "trousers"];
    const full = ["dress", "jumpsuit", "romper"];
    const feet = ["shoes", "boots", "sneakers", "sandals"];

    const cat = (category || "").toLowerCase();
    if (upper.some(u => cat.includes(u))) return "upper";
    if (lower.some(l => cat.includes(l))) return "lower";
    if (full.some(f => cat.includes(f))) return "full";
    if (feet.some(f => cat.includes(f))) return "feet";
    return "upper";
}

// ============================================
// VIDEO TIMELINE ANALYSIS
// ============================================

/**
 * POST /alicevision/analyze-video-timeline
 * Analyze video and detect outfit changes
 */
router.post("/analyze-video-timeline", async (req, res) => {
    try {
        const { frames, detect_outfit_changes, min_agreement } = req.body;

        if (!frames || !Array.isArray(frames) || frames.length < 2) {
            return res.status(400).json({ error: "At least 2 frames required" });
        }

        console.log(`📹 Timeline analysis: ${frames.length} frames...`);

        const result = await callAliceVision("/analyze-video-timeline", {
            frames,
            detect_outfit_changes: detect_outfit_changes !== false,
            min_agreement: min_agreement || 0.5
        }, 180000);

        console.log(`✅ Timeline analysis complete: ${result.items?.length || 0} items, ${result.outfits?.length || 0} outfits`);

        res.json(result);
    } catch (error) {
        console.error("Timeline analysis error:", error.message);
        res.status(500).json({ error: error.message });
    }
});

// ============================================
// ENSEMBLE DETECTION (Best Accuracy)
// ============================================

/**
 * POST /alicevision/detect-ensemble
 * Multi-model ensemble detection for maximum accuracy
 */
router.post("/detect-ensemble", async (req, res) => {
    try {
        const { image } = req.body;

        if (!image) {
            return res.status(400).json({ error: "Image is required" });
        }

        console.log("🎯 Running ensemble detection...");

        const result = await callAliceVision("/detect-ensemble", {
            image
        }, 90000);

        console.log(`✅ Ensemble: ${result.items?.length || 0} items detected`);

        res.json(result);
    } catch (error) {
        console.error("Ensemble detection error:", error.message);
        res.status(500).json({ error: error.message });
    }
});

// ============================================
// MULTI-FRAME SEGMENTATION
// ============================================

/**
 * POST /alicevision/segment-multi-frame
 * Analyze multiple frames with temporal voting
 */
router.post("/segment-multi-frame", async (req, res) => {
    try {
        const { frames, min_agreement } = req.body;

        if (!frames || !Array.isArray(frames) || frames.length < 2) {
            return res.status(400).json({ error: "At least 2 frames required" });
        }

        console.log(`📹 Multi-frame segmentation: ${frames.length} frames...`);

        const result = await callAliceVision("/segment-multi-frame", {
            frames,
            min_agreement: min_agreement || 0.5
        }, 120000);

        console.log(`✅ Multi-frame: ${result.items?.length || 0} items from ${result.framesAnalyzed} frames`);

        res.json(result);
    } catch (error) {
        console.error("Multi-frame error:", error.message);
        res.status(500).json({ error: error.message });
    }
});

// ============================================
// OUTFIT RECOMMENDATIONS
// ============================================

/**
 * POST /alicevision/outfit/recommend
 * AI-powered outfit recommendations
 */
router.post("/outfit/recommend", async (req, res) => {
    try {
        const { wardrobe_items, occasion, weather, preferences, max_outfits } = req.body;

        if (!wardrobe_items || !Array.isArray(wardrobe_items)) {
            return res.status(400).json({ error: "Wardrobe items required" });
        }

        console.log(`🧠 Generating outfit for: ${occasion}`);

        const result = await callAliceVision("/outfit/recommend", {
            wardrobe_items,
            occasion: occasion || "casual",
            weather,
            preferences,
            max_outfits: max_outfits || 3
        }, 60000);

        console.log(`✅ Generated ${result.outfits?.length || 0} outfit recommendations`);

        res.json(result);
    } catch (error) {
        console.error("Outfit recommendation error:", error.message);
        res.status(500).json({ error: error.message });
    }
});

// ============================================
// AI STYLIST CHAT
// ============================================

/**
 * POST /alicevision/outfit/chat
 * Conversational AI stylist
 */
router.post("/outfit/chat", async (req, res) => {
    try {
        const { message, wardrobe_items, conversation_history, context } = req.body;

        if (!message) {
            return res.status(400).json({ error: "Message is required" });
        }

        console.log(`💬 Stylist chat: ${message.slice(0, 50)}...`);

        const result = await callAliceVision("/outfit/chat", {
            message,
            wardrobe_items,
            conversation_history,
            context
        }, 30000);

        console.log(`✅ Chat response generated`);

        res.json(result);
    } catch (error) {
        console.error("Chat error:", error.message);
        res.status(500).json({ error: error.message });
    }
});

// ============================================
// WARDROBE SEMANTIC SEARCH
// ============================================

/**
 * POST /alicevision/wardrobe/search
 * Natural language wardrobe search
 */
router.post("/wardrobe/search", async (req, res) => {
    try {
        const { query, wardrobe_items, top_k } = req.body;

        if (!query) {
            return res.status(400).json({ error: "Query is required" });
        }

        console.log(`🔎 Wardrobe search: ${query}`);

        const result = await callAliceVision("/wardrobe/search", {
            query,
            wardrobe_items,
            top_k: top_k || 5
        }, 15000);

        console.log(`✅ Found ${result.results?.length || 0} matching items`);

        res.json(result);
    } catch (error) {
        console.error("Search error:", error.message);
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
            alicevision: response.data,
            endpoints: [
                "/alicevision/keyframe",
                "/alicevision/segment",
                "/alicevision/lighting",
                "/alicevision/process",
                "/alicevision/analyze",
                "/alicevision/analyze-video-timeline",
                "/alicevision/detect-ensemble",
                "/alicevision/segment-multi-frame",
                "/alicevision/outfit/recommend",
                "/alicevision/outfit/chat",
                "/alicevision/wardrobe/search"
            ]
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
