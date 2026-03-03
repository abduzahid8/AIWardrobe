import express from "express";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { authenticateToken } from "../../middleware/auth.js";
import { aiLimiter } from "../../middleware/rateLimit.js";
import logger from "../../utils/logger.js";

const router = express.Router();
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

/**
 * POST /api/analyze-frames
 * Analyze video frames using Gemini Vision
 */
router.post("/analyze-frames", authenticateToken, aiLimiter, async (req, res) => {
    try {
        const { frames } = req.body;

        if (!frames || !Array.isArray(frames) || frames.length === 0) {
            return res.status(400).json({ error: "No frames provided" });
        }

        logger.info(` Received ${frames.length} frames for analysis`);

        // Check if API key is set
        if (!process.env.GEMINI_API_KEY) {
            logger.error(' GEMINI_API_KEY not set in .env file!');
            return res.status(500).json({ error: "Gemini API key not configured" });
        }
        logger.debug(` Gemini API key: ${process.env.GEMINI_API_KEY.substring(0, 6)}...`);

        // Use gemini-pro-vision for image analysis (v1beta API)
        const model = genAI.getGenerativeModel({ model: "gemini-pro-vision" });
        logger.info(` Using model: gemini-pro-vision`);

        const imageParts = frames.slice(0, 5).map((base64Data) => ({
            inlineData: {
                data: base64Data.replace(/^data:image\/\w+;base64,/, ""),
                mimeType: "image/jpeg",
            },
        }));

        const prompt = `IMPORTANT: Identify EVERY SINGLE clothing item visible in these video frames.
    There are likely MULTIPLE items (2-5 or more). Check ALL body areas carefully:
    
    1. 👕 UPPER BODY: shirts, t-shirts, blouses, jackets, coats, hoodies, sweaters
    2. 👖 LOWER BODY: pants, jeans, shorts, skirts, trousers
    3. 👗 FULL BODY: dresses, jumpsuits, overalls
    4. 👟 FEET: shoes, sneakers, boots, sandals, heels
    5. 👜 ACCESSORIES: bags, hats, scarves, belts, watches, jewelry
    
    For EACH item found, provide:
    - itemType: specific type (e.g., "Denim Jacket", "V-neck T-shirt", "Slim-fit Jeans")
    - color: exact color(s)
    - style: Casual, Formal, Sport, or Streetwear
    - description: brief product description
    - position: where on body (upper, lower, feet, accessory, full)
    - confidence: your confidence level (high, medium, low)
    
    CRITICAL: Do NOT return just 1 item if multiple are visible!
    Return EVERY item as a JSON array:
    [{"itemType": "...", "color": "...", "style": "...", "description": "...", "position": "...", "confidence": "..."}]`;

        const result = await model.generateContent([prompt, ...imageParts]);
        const responseText = result.response.text();

        logger.info(" Gemini response:", responseText);

        let detectedItems = [];
        try {
            const jsonMatch = responseText.match(/\[[\s\S]*\]/);
            if (jsonMatch) {
                detectedItems = JSON.parse(jsonMatch[0]);
            }
        } catch (parseError) {
            logger.error("Parse error:", parseError);
            detectedItems = [
                {
                    itemType: "Unknown Item",
                    color: "Unknown",
                    style: "Casual",
                    description: "Could not parse response",
                },
            ];
        }

        res.json({ detectedItems });
    } catch (error) {
        logger.error("Frame analysis error:", error);
        res.status(500).json({ error: error.message });
    }
});

export default router;
