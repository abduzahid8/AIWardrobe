/**
 * AI Clothing Analysis & Image Generation Routes
 * POST /openai/analyze-clothing  — OpenAI Vision clothing detection
 * POST /openai/generate-image    — DALL-E product image generation
 * POST /generate-product-image   — Replicate SDXL (disabled)
 * POST /remove-background        — AliceVision local rembg (free) → Replicate fallback
 */
import express from "express";
import axios from "axios";
import Replicate from "replicate";
import { ALICEVISION_URL } from "../../config.js";
import { authenticateToken } from "../../middleware/auth.js";
import { requireTier } from "../../middleware/subscriptionGuard.js";
import { aiLimiter } from "../../middleware/rateLimit.js";
import logger from "../../utils/logger.js";
import { validateImageData } from "../../middleware/validators.js";

const router = express.Router();
const replicate = new Replicate({ auth: process.env.REPLICATE_API_TOKEN });

// ── POST /openai/analyze-clothing ──
router.post("/openai/analyze-clothing", authenticateToken, aiLimiter, validateImageData, async (req, res) => {
    try {
        const { imageBase64 } = req.body;

        if (!imageBase64) {
            return res.status(400).json({ error: "Image base64 is required" });
        }

        const response = await axios.post(
            "https://api.openai.com/v1/chat/completions",
            {
                model: "gpt-4o-mini",
                messages: [
                    {
                        role: "user",
                        content: [
                            {
                                type: "text",
                                text: `You are a precise fashion AI. Analyze this image and identify ONLY DISTINCT, CLEARLY VISIBLE clothing items.

RULES:
- Count each item ONLY ONCE (don't count the same jacket as both "jacket" and "outerwear")
- Only include items you can clearly see (minimum 50% visible)
- DO NOT count partially hidden items behind other clothes
- DO NOT count accessories like watches, jewelry, or belts unless specifically asked
- If you see a layered outfit (shirt under jacket), count each as separate ONLY if both are clearly visible

For each DISTINCT item, provide:
1. itemType: specific type (e.g., "Denim Jacket", "Crew-neck T-shirt", "Slim-fit Jeans")
2. color: primary color(s)
3. style: Casual/Formal/Sport/Streetwear
4. material: fabric type if visible
5. description: 1-sentence product description
6. confidence: your confidence 0-100 (only include items with 70%+ confidence)
7. position: upper/lower/full/feet

Return ONLY items with 70%+ confidence as JSON array:
[{"itemType": "...", "color": "...", "style": "...", "material": "...", "description": "...", "confidence": 85, "position": "upper"}]

Be conservative - it's better to miss an item than to add a false one.`,
                            },
                            {
                                type: "image_url",
                                image_url: { url: `data:image/jpeg;base64,${imageBase64}` },
                            },
                        ],
                    },
                ],
                max_tokens: 1000,
            },
            {
                headers: {
                    Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
                    "Content-Type": "application/json",
                },
                timeout: 45000,
            }
        );

        const text = response.data.choices?.[0]?.message?.content || "[]";
        logger.debug("OpenAI Clothing Analysis (raw):", text);

        const jsonMatch = text.match(/\[[\s\S]*\]/);
        let detectedItems = jsonMatch ? JSON.parse(jsonMatch[0]) : [];

        // Filter by confidence (only keep 70%+ confidence items)
        const originalCount = detectedItems.length;
        detectedItems = detectedItems.filter(item => {
            const confidence = item.confidence || 100;
            return confidence >= 70;
        });

        logger.info(` OpenAI detected ${originalCount} items, ${detectedItems.length} passed 70% confidence threshold`);

        res.json({ detectedItems });
    } catch (error) {
        logger.error("OpenAI analysis error:", error.response?.data || error.message);
        res.status(500).json({ error: "Analysis failed" });
    }
});

// ── POST /openai/generate-image ──
router.post("/openai/generate-image", authenticateToken, requireTier('premium'), aiLimiter, async (req, res) => {
    try {
        const { prompt } = req.body;

        if (!prompt) {
            return res.status(400).json({ error: "Prompt is required" });
        }

        const response = await axios.post(
            "https://api.openai.com/v1/images/generations",
            {
                model: "dall-e-3",
                prompt: `${prompt}, professional product photography, clean white background, studio lighting, high quality, e-commerce style, centered, full garment visible`,
                n: 1,
                size: "1024x1024",
                quality: "standard",
            },
            {
                headers: {
                    Authorization: `Bearer ${process.env.OPENAI_API_KEY}`,
                    "Content-Type": "application/json",
                },
                timeout: 60000,
            }
        );

        const imageUrl = response.data.data[0].url;
        res.json({ imageUrl });
    } catch (error) {
        logger.error("DALL-E generation error:", error.response?.data || error.message);
        res.status(500).json({ error: "Image generation failed" });
    }
});

// ── POST /generate-product-image (DISABLED) ──
router.post("/generate-product-image", authenticateToken, aiLimiter, async (req, res) => {
    try {
        const { description, itemType } = req.body;

        if (!description && !itemType) {
            return res.status(400).json({ error: "Description or item type required" });
        }

        logger.warn(" generate-product-image called but Replicate is disabled");
        res.json({
            imageUrl: null,
            message: "Replicate disabled. Use /api/product-photo/process for AliceVision-enhanced images",
            useAliceVision: true
        });
    } catch (error) {
        logger.error("Image generation error:", error.message);
        res.status(500).json({ error: "Image generation failed" });
    }
});

// ── POST /remove-background ──
// Strategy: AliceVision local rembg (free) → Replicate rembg (paid fallback)
router.post("/remove-background", authenticateToken, aiLimiter, async (req, res) => {
    try {
        const { imageBase64 } = req.body;

        if (!imageBase64) {
            return res.status(400).json({ error: "Image base64 is required" });
        }

        logger.info(" Removing background from clothing image...");

        // ── Try AliceVision (free, local) ──
        try {
            const avResponse = await axios.post(
                `${ALICEVISION_URL}/remove-bg`,
                { image: imageBase64 },
                { timeout: 60000 }
            );

            if (avResponse.data && avResponse.data.success) {
                logger.info(` ✅ Background removed via AliceVision (free, ${avResponse.data.processingTimeMs}ms)`);
                return res.json({
                    imageUrl: `data:image/png;base64,${avResponse.data.image}`,
                    provider: "alicevision",
                    cost: 0,
                });
            }
        } catch (avError) {
            logger.warn(` AliceVision unavailable: ${avError.message}. Falling back to Replicate...`);
        }

        // ── Fallback: Replicate rembg (paid) ──
        const output = await replicate.run(
            "cjwbw/rembg:fb8af171cfa1616ddcf1242c093f9c46bcada5ad4cf6f2fbe8b81b330ec5c003",
            {
                input: {
                    image: `data:image/jpeg;base64,${imageBase64}`,
                },
            }
        );

        logger.info(" Background removed via Replicate (paid fallback)");
        res.json({ imageUrl: output, provider: "replicate", cost: 0.005 });
    } catch (error) {
        logger.error("Background removal error:", error.message);
        res.status(500).json({ error: "Background removal failed" });
    }
});

export default router;
