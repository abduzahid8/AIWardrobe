/**
 * Frame Analysis Route
 * POST /api/analyze-frames — Analyze video frames using HuggingFace BLIP-2 + CLIP
 * (Replaced Gemini Pro Vision)
 */
import express from "express";
import { authenticateToken } from "../../middleware/auth.js";
import { aiLimiter } from "../../middleware/rateLimit.js";
import logger from "../../utils/logger.js";
import hfService from "../../services/huggingface.js";

const router = express.Router();

/**
 * POST /api/analyze-frames
 * Analyze video frames for clothing items using HuggingFace models
 */
router.post("/analyze-frames", authenticateToken, aiLimiter, async (req, res) => {
    try {
        const { frames } = req.body;

        if (!frames || !Array.isArray(frames) || frames.length === 0) {
            return res.status(400).json({ error: "No frames provided" });
        }

        logger.info(`📸 Received ${frames.length} frames for analysis (HuggingFace)`);

        // Analyze up to 3 frames (balance between accuracy and speed)
        const framesToAnalyze = frames.slice(0, 3);
        const allItems = [];

        for (let i = 0; i < framesToAnalyze.length; i++) {
            try {
                const frame = framesToAnalyze[i].replace(/^data:image\/\w+;base64,/, "");

                // Get clothing analysis for this frame
                const analysis = await hfService.analyzeClothingImage(frame);

                // Check if this item type was already detected in a previous frame
                const isDuplicate = allItems.some(
                    (item) => item.itemType.toLowerCase() === analysis.itemType.toLowerCase()
                );

                if (!isDuplicate) {
                    allItems.push({
                        itemType: analysis.itemType.charAt(0).toUpperCase() + analysis.itemType.slice(1),
                        color: "Detected from image",
                        style: analysis.style.charAt(0).toUpperCase() + analysis.style.slice(1),
                        description: analysis.description,
                        position: analysis.position,
                        confidence: analysis.confidence > 50 ? "high" : analysis.confidence > 30 ? "medium" : "low",
                    });
                }

                // Add strong secondary categories too
                for (const cat of analysis.categories.slice(1, 3)) {
                    if (cat.score > 0.15) {
                        const catDup = allItems.some(
                            (item) => item.itemType.toLowerCase() === cat.label.toLowerCase()
                        );
                        if (!catDup) {
                            allItems.push({
                                itemType: cat.label.charAt(0).toUpperCase() + cat.label.slice(1),
                                color: "Detected from image",
                                style: analysis.style.charAt(0).toUpperCase() + analysis.style.slice(1),
                                description: analysis.description,
                                position: analysis.position,
                                confidence: cat.score > 0.3 ? "high" : "medium",
                            });
                        }
                    }
                }
            } catch (frameErr) {
                logger.warn(`Frame ${i + 1} analysis failed:`, frameErr.message);
            }
        }

        if (allItems.length === 0) {
            // Return a fallback item if no analysis worked
            allItems.push({
                itemType: "Clothing Item",
                color: "Unknown",
                style: "Casual",
                description: "Could not identify specific items",
                position: "upper",
                confidence: "low",
            });
        }

        logger.info(`✅ Detected ${allItems.length} unique clothing items across ${framesToAnalyze.length} frames`);
        res.json({ detectedItems: allItems });
    } catch (error) {
        logger.error("Frame analysis error:", error.message);
        res.status(500).json({ error: error.message });
    }
});

export default router;
