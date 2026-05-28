/**
 * Virtual Try-On Route
 * POST /try-on — Virtual try-on with multi-provider fallback
 *
 * Strategy:
 *   1. HF Space URL (if configured via HF_TRYON_SPACE_URL)
 *   2. Replicate IDM-VTON (using REPLICATE_API_TOKEN)
 *   3. Return 503 if neither is available
 */
import express from "express";
import axios from "axios";
import { authenticateToken } from "../../middleware/auth.js";
import { aiLimiter } from "../../middleware/rateLimit.js";
import logger from "../../utils/logger.js";
import replicateService from "../../services/replicate.js";

const router = express.Router();

// User sets this to their HF Space, ngrok URL, or any try-on API
const TRYON_API_URL = process.env.HF_TRYON_SPACE_URL || null;

// ── POST /try-on ──
router.post("/try-on", authenticateToken, aiLimiter, async (req, res) => {
    const { human_image, garment_image, description, garment_type } = req.body;

    // ── Validation ──
    if (!human_image) {
        return res.status(400).json({ error: "human_image (base64) is required" });
    }
    if (!garment_image) {
        return res.status(400).json({ error: "garment_image (base64) is required" });
    }

    // Clean base64 prefixes
    const cleanPerson = human_image.replace(/^data:image\/\w+;base64,/, "");
    const cleanGarment = garment_image.replace(/^data:image\/\w+;base64,/, "");

    logger.info("👗 Starting virtual try-on...");

    // ── Strategy 1: HuggingFace Space (if configured) ──
    if (TRYON_API_URL) {
        try {
            const response = await axios.post(
                `${TRYON_API_URL}/tryon`,
                {
                    person_image: cleanPerson,
                    garment_image: cleanGarment,
                    garment_description: description || "A stylish " + (garment_type || "clothing").replace("_", " "),
                    auto_crop: true,
                    denoise_steps: 30,
                    seed: 42,
                },
                {
                    headers: {
                        "Content-Type": "application/json",
                        "Authorization": process.env.HF_TOKEN ? `Bearer ${process.env.HF_TOKEN}` : undefined,
                    },
                    timeout: 120000,
                }
            );

            if (response.data?.result_image) {
                logger.info("✅ Try-on complete via HF Space");
                return res.json({
                    image: response.data.result_image,
                    elapsed_seconds: response.data.elapsed_seconds || null,
                    provider: "hf_space",
                });
            } else if (response.data?.image) {
                logger.info("✅ Try-on complete via HF Space");
                return res.json({ image: response.data.image, provider: "hf_space" });
            }
        } catch (hfError) {
            logger.warn(`HF Space try-on failed: ${hfError.message}. Falling back to Replicate...`);
        }
    }

    // ── Strategy 2: Replicate IDM-VTON ──
    if (replicateService.isAvailable()) {
        try {
            const result = await replicateService.virtualTryOn(cleanPerson, cleanGarment, {
                garmentType: garment_type || "upper_body",
                description: description || "A stylish " + (garment_type || "clothing").replace("_", " "),
                steps: 30,
                seed: 42,
            });

            logger.info("✅ Try-on complete via Replicate IDM-VTON");
            return res.json({
                image: result.image,
                elapsed_seconds: result.elapsed_seconds,
                provider: "replicate",
            });
        } catch (repError) {
            logger.error("Replicate try-on error:", repError.message);
        }
    }

    // ── No provider available ──
    return res.status(503).json({
        error: "Virtual try-on API not available",
        message: "Neither HF Space nor Replicate is configured/working.",
        instructions: [
            "Option A: Set HF_TRYON_SPACE_URL in api/.env to a HuggingFace Space URL",
            "Option B: Set REPLICATE_API_TOKEN in api/.env for Replicate IDM-VTON",
            "Then restart the server",
        ],
    });
});

export default router;
