/**
 * Unified Image Processor Routes
 * POST /api/process-upload  — Full pipeline: remove bg → classify → enhance → describe
 *
 * Calls the local AliceVision Python service (free, zero API cost).
 * Falls back gracefully if AliceVision is unavailable.
 */
import express from "express";
import axios from "axios";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { authenticateToken } from "../../middleware/auth.js";
import { aiLimiter } from "../../middleware/rateLimit.js";
import { ALICEVISION_URL } from "../../config.js";
import logger from "../../utils/logger.js";
import { validateImageData } from "../../middleware/validators.js";

const router = express.Router();

// Gemini for fallback text descriptions (free tier)
const genAI = process.env.GEMINI_API_KEY
    ? new GoogleGenerativeAI(process.env.GEMINI_API_KEY)
    : null;

// Massimo Dutti-style prompts for fallback when AliceVision is down
const STYLE_DESCRIPTIONS = {
    tops: "Premium cotton piece with impeccable tailoring and a contemporary silhouette",
    bottoms: "Tailored trouser with precise fit and refined craftsmanship",
    shoes: "Sleek footwear combining artisan quality with understated luxury",
    outerwear: "Sophisticated outerwear with clean lines and a luxurious drape",
    dresses: "Fluid dress with effortless elegance and premium fabric selection",
    accessories: "Refined accessory with minimalist design and premium materials",
    other: "Fashion piece with considered design and quality construction",
};

/**
 * POST /api/process-upload
 *
 * Unified image processing endpoint.
 * Input:  { imageBase64: string, generateDescription?: boolean }
 * Output: { imageUrl, cutoutUrl, classification, description, style, steps }
 */
router.post(
    "/process-upload",
    authenticateToken,
    aiLimiter,
    validateImageData,
    async (req, res) => {
        try {
            const { imageBase64, generateDescription = true } = req.body;

            if (!imageBase64) {
                return res.status(400).json({ error: "imageBase64 is required" });
            }

            logger.info("🎨 Process-upload: Starting unified pipeline...");

            // ── Strategy 1: AliceVision full pipeline (preferred, free) ──
            try {
                const response = await axios.post(
                    `${ALICEVISION_URL}/process`,
                    {
                        image: imageBase64,
                        generate_description: generateDescription,
                    },
                    { timeout: 120000 }
                );

                if (response.data && response.data.success) {
                    logger.info(
                        `✅ AliceVision pipeline complete: ${response.data.classification?.category || "unknown"} (${response.data.totalProcessingMs}ms)`
                    );

                    return res.json({
                        success: true,
                        imageUrl: `data:image/jpeg;base64,${response.data.image}`,
                        cutoutUrl: `data:image/png;base64,${response.data.cutout}`,
                        classification: response.data.classification,
                        description: response.data.description,
                        style: response.data.style,
                        dimensions: response.data.dimensions,
                        steps: response.data.steps,
                        totalProcessingMs: response.data.totalProcessingMs,
                        provider: "alicevision",
                        costs: response.data.costs,
                    });
                }
            } catch (avError) {
                logger.warn(
                    `⚠️ AliceVision unavailable: ${avError.message}. Falling back to API...`
                );
            }

            // ── Strategy 2: Fallback — AliceVision individual endpoints ──
            let cutoutB64 = null;
            let classification = null;
            let enhancedB64 = null;
            const steps = [];

            // 2a. Try background removal endpoint
            try {
                const bgRes = await axios.post(
                    `${ALICEVISION_URL}/remove-bg`,
                    { image: imageBase64 },
                    { timeout: 60000 }
                );
                if (bgRes.data?.success) {
                    cutoutB64 = bgRes.data.image;
                    steps.push("alicevision_remove_bg");
                }
            } catch (e) {
                logger.warn("Background removal fallback failed");
            }

            // 2b. Try classification endpoint
            try {
                const clsRes = await axios.post(
                    `${ALICEVISION_URL}/classify`,
                    { image: imageBase64 },
                    { timeout: 30000 }
                );
                if (clsRes.data?.success) {
                    classification = {
                        category: clsRes.data.category,
                        section: clsRes.data.section,
                        confidence: clsRes.data.confidence,
                        top5: clsRes.data.top5,
                    };
                    steps.push("alicevision_classify");
                }
            } catch (e) {
                logger.warn("Classification fallback failed");
            }

            // 2c. Try enhancement endpoint
            if (cutoutB64) {
                try {
                    const enhRes = await axios.post(
                        `${ALICEVISION_URL}/enhance`,
                        { image: cutoutB64 },
                        { timeout: 30000 }
                    );
                    if (enhRes.data?.success) {
                        enhancedB64 = enhRes.data.image;
                        steps.push("alicevision_enhance");
                    }
                } catch (e) {
                    logger.warn("Enhancement fallback failed");
                }
            }

            // ── Strategy 3: Gemini-only fallback ──
            if (!classification && genAI) {
                try {
                    const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash" });

                    const result = await model.generateContent([
                        {
                            inlineData: {
                                data: imageBase64.replace(/^data:image\/\w+;base64,/, ""),
                                mimeType: "image/jpeg",
                            },
                        },
                        `Analyze this clothing image. Return ONLY a JSON object with these exact fields:
{
  "category": "one of: T-shirt/top, Shirt, Pullover, Coat, Dress, Trouser, Sandal, Sneaker, Ankle boot, Bag",
  "section": "one of: tops, bottoms, shoes, outerwear, dresses, accessories",
  "confidence": 0.0-1.0,
  "color": "primary color",
  "material": "fabric type",
  "pattern": "solid/striped/plaid/printed/etc",
  "style": "casual/formal/sport/streetwear"
}
Return ONLY the JSON, no markdown or explanation.`,
                    ]);

                    const text = result.response.text();
                    const jsonMatch = text.match(/\{[\s\S]*\}/);
                    if (jsonMatch) {
                        const parsed = JSON.parse(jsonMatch[0]);
                        classification = {
                            category: parsed.category || "unknown",
                            section: parsed.section || "other",
                            confidence: parsed.confidence || 0.7,
                            attributes: {
                                color: parsed.color,
                                material: parsed.material,
                                pattern: parsed.pattern,
                                style: parsed.style,
                            },
                        };
                        steps.push("gemini_classify");
                    }
                } catch (geminiErr) {
                    logger.warn(`Gemini fallback failed: ${geminiErr.message}`);
                }
            }

            // ── Build description ──
            let description = null;
            if (classification) {
                const section = classification.section || "other";
                description =
                    STYLE_DESCRIPTIONS[section] || STYLE_DESCRIPTIONS.other;
            }

            // ── Final response ──
            const finalImage = enhancedB64 || cutoutB64 || imageBase64;
            const finalFormat = enhancedB64 ? "jpeg" : cutoutB64 ? "png" : "jpeg";

            res.json({
                success: true,
                imageUrl: `data:image/${finalFormat};base64,${finalImage}`,
                cutoutUrl: cutoutB64
                    ? `data:image/png;base64,${cutoutB64}`
                    : null,
                classification,
                description,
                style: "massimo_dutti",
                steps,
                provider: steps.length > 0 ? "hybrid" : "passthrough",
                costs: { api_calls: steps.includes("gemini_classify") ? 1 : 0, paid: false },
            });
        } catch (error) {
            logger.error("Process-upload error:", error.message);
            res.status(500).json({ error: "Image processing failed" });
        }
    }
);

export default router;
