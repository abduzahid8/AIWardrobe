import express from "express";
import logger from '../utils/logger.js';
import { authenticateToken } from "../middleware/auth.js";
import { aiLimiter } from "../middleware/rateLimit.js";
import "dotenv/config";

const router = express.Router();

const GEMINI_API_KEY = process.env.GEMINI_API_KEY;

if (!GEMINI_API_KEY) {
    console.warn("⚠️  GEMINI_API_KEY not set — Gemini proxy routes will return 503");
}

/**
 * POST /api/gemini/analyze-image
 * Proxy for Gemini Vision API — analyzes clothing in images server-side.
 * Keeps the API key out of the client bundle.
 */
router.post("/analyze-image", authenticateToken, aiLimiter, async (req, res) => {
    try {
        if (!GEMINI_API_KEY) {
            return res.status(503).json({ error: "AI service not configured" });
        }

        const { imageBase64, prompt } = req.body;

        if (!imageBase64) {
            return res.status(400).json({ error: "imageBase64 is required" });
        }

        const defaultPrompt = `Analyze this clothing item image. Return a JSON object with:
{
  "category": "top|bottom|dress|outerwear|shoes|accessories|other",
  "specificType": "e.g. t-shirt, jeans, sneakers",
  "primaryColor": "main color name",
  "colorHex": "#hex code",
  "pattern": "solid|striped|checkered|floral|printed|other",
  "material": "e.g. cotton, denim, leather",
  "style": "casual|formal|sport|streetwear|beach|elegant|business",
  "confidence": 0.0 to 1.0
}
Only return valid JSON, no markdown.`;

        const response = await fetch(
            `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GEMINI_API_KEY}`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    contents: [{
                        parts: [
                            { text: prompt || defaultPrompt },
                            {
                                inline_data: {
                                    mime_type: "image/jpeg",
                                    data: imageBase64.replace(/^data:image\/\w+;base64,/, ""),
                                },
                            },
                        ],
                    }],
                    generationConfig: {
                        temperature: 0.2,
                        maxOutputTokens: 1024,
                    },
                }),
            }
        );

        if (!response.ok) {
            const errorBody = await response.text();
            logger.error("Gemini API error:", response.status, errorBody);
            return res.status(502).json({ error: "AI service error", status: response.status });
        }

        const data = await response.json();
        const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";

        // Try to parse as JSON for structured responses
        try {
            const parsed = JSON.parse(text.replace(/```json\n?|\n?```/g, "").trim());
            return res.json({ success: true, result: parsed, raw: text });
        } catch {
            return res.json({ success: true, result: null, raw: text });
        }
    } catch (error) {
        logger.error("Gemini proxy error:", error.message);
        res.status(500).json({ error: "AI analysis failed" });
    }
});

/**
 * POST /api/gemini/chat
 * Proxy for Gemini text generation — style advice, outfit explanations.
 */
router.post("/chat", authenticateToken, aiLimiter, async (req, res) => {
    try {
        if (!GEMINI_API_KEY) {
            return res.status(503).json({ error: "AI service not configured" });
        }

        const { prompt, conversationHistory } = req.body;

        if (!prompt) {
            return res.status(400).json({ error: "prompt is required" });
        }

        // Build conversation context
        const contents = [];
        if (conversationHistory?.length) {
            for (const msg of conversationHistory) {
                contents.push({
                    role: msg.role === "assistant" ? "model" : "user",
                    parts: [{ text: msg.content }],
                });
            }
        }
        contents.push({ role: "user", parts: [{ text: prompt }] });

        const response = await fetch(
            `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${GEMINI_API_KEY}`,
            {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    contents,
                    generationConfig: {
                        temperature: 0.7,
                        maxOutputTokens: 2048,
                    },
                }),
            }
        );

        if (!response.ok) {
            const errorBody = await response.text();
            logger.error("Gemini chat API error:", response.status, errorBody);
            return res.status(502).json({ error: "AI service error" });
        }

        const data = await response.json();
        const text = data.candidates?.[0]?.content?.parts?.[0]?.text || "";

        res.json({ success: true, text });
    } catch (error) {
        logger.error("Gemini chat proxy error:", error.message);
        res.status(500).json({ error: "AI chat failed" });
    }
});

export default router;
