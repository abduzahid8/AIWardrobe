/**
 * AI Clothing Analysis & Image Generation Routes
 * POST /openai/analyze-clothing  — Gemini Vision + HuggingFace CLIP fallback
 * POST /openai/generate-image    — OpenAI DALL-E 3 image generation
 * POST /generate-product-image   — Alias
 * POST /remove-background        — AliceVision → Replicate fallback
 */
import express from "express";
import axios from "axios";
import { ALICEVISION_URL } from "../../config.js";
import { authenticateToken } from "../../middleware/auth.js";
import { aiLimiter } from "../../middleware/rateLimit.js";
import logger from "../../utils/logger.js";
import { validateImageData } from "../../middleware/validators.js";
import hfService from "../../services/huggingface.js";
import geminiService from "../../services/gemini.js";
import openaiService from "../../services/openai.js";
import replicateService from "../../services/replicate.js";

const router = express.Router();

// ── POST /openai/analyze-clothing ──
router.post("/openai/analyze-clothing", authenticateToken, aiLimiter, validateImageData, async (req, res) => {
    try {
        const { imageBase64 } = req.body;

        if (!imageBase64) {
            return res.status(400).json({ error: "Image base64 is required" });
        }

        logger.info("👔 Analyzing clothing...");

        // Strategy 1: Gemini Vision (fast, accurate, free)
        if (geminiService.isAvailable()) {
            try {
                const analysis = await geminiService.analyzeClothingImage(imageBase64);
                logger.info(`✅ Gemini detected: ${analysis.category}`);

                return res.json({
                    detectedItems: [{
                        itemType: analysis.category,
                        color: analysis.primaryColor || "Detected from image",
                        style: analysis.style,
                        material: analysis.material || "Detected",
                        description: analysis.description,
                        confidence: 95,
                        position: "detected",
                    }],
                    provider: "gemini",
                });
            } catch (geminiErr) {
                logger.warn("Gemini clothing analysis failed, falling back to HF:", geminiErr.message);
            }
        }

        // Strategy 2: HuggingFace BLIP-2 + CLIP (free tier)
        const analysis = await hfService.analyzeClothingImage(imageBase64);

        const detectedItems = [{
            itemType: analysis.itemType.charAt(0).toUpperCase() + analysis.itemType.slice(1),
            color: "Detected from image",
            style: analysis.style.charAt(0).toUpperCase() + analysis.style.slice(1),
            material: "Detected",
            description: analysis.description,
            confidence: analysis.confidence,
            position: analysis.position,
        }];

        if (analysis.categories.length > 1 && analysis.categories[1].score > 0.15) {
            detectedItems.push({
                itemType: analysis.categories[1].label.charAt(0).toUpperCase() + analysis.categories[1].label.slice(1),
                color: "Detected from image",
                style: analysis.style.charAt(0).toUpperCase() + analysis.style.slice(1),
                material: "Detected",
                description: analysis.description,
                confidence: Math.round(analysis.categories[1].score * 100),
                position: analysis.position,
            });
        }

        logger.info(`✅ HF detected ${detectedItems.length} items`);
        res.json({ detectedItems, provider: "huggingface" });
    } catch (error) {
        logger.error("Clothing analysis error:", error.message);
        res.status(500).json({ error: "Analysis failed" });
    }
});

// ── POST /openai/generate-image ──
// Now actually generates images using DALL-E 3
router.post("/openai/generate-image", authenticateToken, aiLimiter, async (req, res) => {
    try {
        const { prompt, size, quality } = req.body;

        if (!prompt) {
            return res.status(400).json({ error: "Prompt is required" });
        }

        logger.info("🎨 Generating image with DALL-E 3...");

        if (!openaiService.isAvailable()) {
            return res.status(503).json({
                error: "Image generation not configured",
                message: "Set OPENAI_API_KEY in api/.env to enable DALL-E 3 image generation.",
            });
        }

        const result = await openaiService.generateImage(prompt, size, quality);

        logger.info("✅ Image generated successfully");
        res.json({
            imageUrl: result.imageUrl,
            revisedPrompt: result.revisedPrompt,
            provider: "openai_dalle3",
        });
    } catch (error) {
        logger.error("Image generation error:", error.message);
        res.status(500).json({ error: "Image generation failed", detail: error.message });
    }
});

// ── POST /generate-product-image ──
router.post("/generate-product-image", authenticateToken, aiLimiter, async (req, res) => {
    if (openaiService.isAvailable()) {
        const { prompt } = req.body;
        try {
            const result = await openaiService.generateImage(
                prompt || "Professional product photo of a clothing item on white background, studio lighting, Massimo Dutti style"
            );
            return res.json({ imageUrl: result.imageUrl, provider: "openai_dalle3" });
        } catch (err) {
            logger.warn("DALL-E product image failed:", err.message);
        }
    }

    res.json({
        imageUrl: null,
        message: "Use /api/product-photo/process for AliceVision-enhanced images",
        useAliceVision: true,
    });
});

// ── POST /remove-background ──
// Strategy: AliceVision (free) → Replicate rembg (paid fallback)
router.post("/remove-background", authenticateToken, aiLimiter, async (req, res) => {
    try {
        const { imageBase64 } = req.body;

        if (!imageBase64) {
            return res.status(400).json({ error: "Image base64 is required" });
        }

        logger.info("✂️ Removing background from clothing image...");

        // Strategy 1: AliceVision (free, local)
        try {
            const avResponse = await axios.post(
                `${ALICEVISION_URL}/remove-bg`,
                { image: imageBase64 },
                { timeout: 60000 }
            );

            if (avResponse.data?.success) {
                logger.info(`✅ Background removed via AliceVision (free, ${avResponse.data.processingTimeMs}ms)`);
                return res.json({
                    imageUrl: `data:image/png;base64,${avResponse.data.image}`,
                    provider: "alicevision",
                    cost: 0,
                });
            }
        } catch (avError) {
            logger.warn(`AliceVision unavailable: ${avError.message}`);
        }

        // Strategy 2: Replicate rembg (paid fallback)
        if (replicateService.isAvailable()) {
            try {
                const result = await replicateService.removeBackground(imageBase64);
                logger.info(`✅ Background removed via Replicate (${result.processingTimeMs}ms)`);
                return res.json({
                    imageUrl: `data:image/png;base64,${result.image}`,
                    provider: "replicate",
                    cost: 0.01,
                });
            } catch (repError) {
                logger.warn(`Replicate bg removal failed: ${repError.message}`);
            }
        }

        // No provider available
        res.status(503).json({
            error: "Background removal service unavailable",
            message: "AliceVision is not running and Replicate is not configured.",
            provider: "none",
        });
    } catch (error) {
        logger.error("Background removal error:", error.message);
        res.status(500).json({ error: "Background removal failed" });
    }
});

export default router;
