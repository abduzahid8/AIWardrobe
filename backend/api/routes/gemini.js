/**
 * AI Proxy Routes (Gemini path — now fully multi-provider)
 * POST /api/gemini/analyze-image  — Vision failover chain → HF CLIP fallback
 * POST /api/gemini/chat           — Text failover chain (all providers)
 *
 * Path kept as /api/gemini/* for backward compatibility with client code.
 */
import express from "express";
import logger from '../utils/logger.js';
import { authenticateToken } from "../middleware/auth.js";
import { aiLimiter } from "../middleware/rateLimit.js";
import aiRouter from "../services/ai/index.js";
import hfService from "../services/huggingface.js";

const router = express.Router();

/**
 * POST /api/gemini/analyze-image
 * LLM vision via the failover router (cycles all vision-capable providers),
 * then HuggingFace BLIP-2 + CLIP as the deterministic last resort.
 */
router.post("/analyze-image", authenticateToken, aiLimiter, async (req, res) => {
    try {
        const { imageBase64, prompt } = req.body;

        if (!imageBase64) {
            return res.status(400).json({ error: "imageBase64 is required" });
        }

        const cleanBase64 = imageBase64.replace(/^data:image\/\w+;base64,/, "");

        // Strategy 1: failover vision chain (Gemini → OpenAI → OpenRouter…)
        try {
            const analysisResult = await aiRouter.analyzeClothingImage(cleanBase64);
            const analysis = analysisResult.result;
            logger.info(`✅ ${analysisResult.provider} Vision analysis: ${analysis.category}`);
            return res.json({
                success: true,
                result: {
                    category: analysis.category || "clothing",
                    specificType: analysis.specificType || analysis.category,
                    primaryColor: analysis.primaryColor || "detected from image",
                    pattern: analysis.pattern || "solid",
                    style: analysis.style || "casual",
                    confidence: 0.95,
                    description: analysis.description || "",
                    material: analysis.material || "unknown",
                    season: analysis.season || "all-season",
                },
                raw: analysis.description,
                provider: analysisResult.provider,
            });
        } catch (visionErr) {
            const attempts = visionErr.attempts?.map((a) => `${a.provider}:${a.error}`).join(", ");
            logger.warn("Vision failover chain failed, using HF pipeline:", attempts || visionErr.message);
        }

        // Strategy 3: HuggingFace BLIP-2 + CLIP
        const description = await hfService.imageToText(cleanBase64);

        const categoryLabels = ["top", "bottom", "dress", "outerwear", "shoes", "accessories", "other"];
        const categories = await hfService.zeroShotImageClassify(cleanBase64, categoryLabels);

        const styleLabels = ["casual", "formal", "sport", "semi_classic", "beach", "elegant", "business"];
        const styles = await hfService.zeroShotImageClassify(cleanBase64, styleLabels);

        const patternLabels = ["solid", "striped", "checkered", "floral", "printed", "other"];
        const patterns = await hfService.zeroShotImageClassify(cleanBase64, patternLabels);

        const topCat = categories[0] || { label: "other", score: 0.5 };
        const topStyle = styles[0] || { label: "casual", score: 0.5 };
        const topPattern = patterns[0] || { label: "solid", score: 0.5 };

        const result = {
            category: topCat.label,
            specificType: description.split(" ").slice(0, 3).join(" "),
            primaryColor: "detected from image",
            pattern: topPattern.label,
            style: topStyle.label,
            confidence: topCat.score,
            description: description,
        };

        return res.json({ success: true, result, raw: description, provider: "huggingface" });
    } catch (error) {
        logger.error("Image analysis error:", error.message);
        res.status(500).json({ error: "AI analysis failed" });
    }
});

/**
 * POST /api/gemini/chat
 * Chat via the failover router — cycles all configured text providers.
 */
router.post("/chat", authenticateToken, aiLimiter, async (req, res) => {
    try {
        const { prompt, conversationHistory } = req.body;

        if (!prompt) {
            return res.status(400).json({ error: "prompt is required" });
        }

        // Build conversation messages
        const messages = [];
        messages.push({
            role: "system",
            content: "You are an expert fashion stylist AI assistant. Provide helpful, concise fashion advice with specific recommendations. Use emojis for a friendly tone. Be knowledgeable about brands, trends, and seasonal styles."
        });

        if (conversationHistory?.length) {
            for (const msg of conversationHistory) {
                messages.push({
                    role: msg.role === "assistant" ? "assistant" : "user",
                    content: msg.content,
                });
            }
        }
        messages.push({ role: "user", content: prompt });

        const chatResult = await aiRouter.chatWithHistory(messages, 800);
        logger.info(`✅ Chat response via ${chatResult.provider}`);
        res.json({ success: true, text: chatResult.text, provider: chatResult.provider });
    } catch (error) {
        const attempts = error.attempts?.map((a) => `${a.provider}:${a.error}`).join(", ");
        logger.error("Chat failed on all providers:", attempts || error.message);
        res.status(503).json({ error: "AI chat failed" });
    }
});

export default router;
