/**
 * AI Outfit Generation Routes
 * POST /generate-outfits — AI-powered outfit recommendations
 */
import express from "express";
import { GoogleGenerativeAI } from "@google/generative-ai";
import { authenticateToken } from "../../middleware/auth.js";
import { aiLimiter } from "../../middleware/rateLimit.js";
import logger from "../../utils/logger.js";

const router = express.Router();
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

// ── POST /generate-outfits ──
router.post("/generate-outfits", authenticateToken, aiLimiter, async (req, res) => {
    try {
        const { occasion, stylePreferences, limit = 5 } = req.body;

        if (!occasion && !stylePreferences) {
            return res.status(400).json({
                error: "Please provide an occasion or style preferences"
            });
        }

        logger.info(" Generating outfits with AI for:", { occasion, stylePreferences });

        // Import outfit database
        const { curatedOutfits } = await import("../../data/curatedOutfits.js");

        // Try AI-powered matching first, fallback to keyword matching
        let scoredOutfits;

        try {
            const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

            const prompt = `You are a fashion AI stylist. Analyze these outfit requests and score each outfit from 0-100.

USER REQUEST:
Occasion: ${occasion || 'any'}
Style Preferences: ${stylePreferences || 'none specified'}

AVAILABLE OUTFITS:
${curatedOutfits.map((outfit, idx) => `
${idx + 1}. ${outfit.description}
   - Occasions: ${outfit.occasion.join(', ')}
   - Styles: ${outfit.style.join(', ')}
`).join('\n')}

Return ONLY a JSON array of scores, one number (0-100) for each outfit in order. Example: [95, 70, 85, 60, ...]`;

            const result = await model.generateContent(prompt);
            const responseText = result.response.text();

            const scores = JSON.parse(responseText.match(/\[[\d,\s]+\]/)?.[0] || '[]');

            if (scores.length === curatedOutfits.length) {
                logger.info(" Using AI-powered matching");
                scoredOutfits = curatedOutfits.map((outfit, idx) => ({
                    ...outfit,
                    matchScore: scores[idx] / 100
                }));
            } else {
                throw new Error("AI returned invalid scores");
            }
        } catch (aiError) {
            logger.warn(" AI matching failed, using keyword fallback:", aiError.message);

            // Fallback: Enhanced keyword matching
            const styleKeywords = stylePreferences
                ? stylePreferences.toLowerCase().split(/[\s,]+/).filter(w => w.length > 2)
                : [];

            scoredOutfits = curatedOutfits.map(outfit => {
                let score = 0;

                if (occasion && outfit.occasion.includes(occasion.toLowerCase())) {
                    score += 10;
                }

                if (occasion && outfit.occasion.some(occ =>
                    occ.includes(occasion.toLowerCase()) || occasion.toLowerCase().includes(occ)
                )) {
                    score += 5;
                }

                styleKeywords.forEach(keyword => {
                    if (outfit.style.some(s => s === keyword)) {
                        score += 4;
                    } else if (outfit.style.some(s => s.includes(keyword) || keyword.includes(s))) {
                        score += 2;
                    }
                    if (outfit.description.toLowerCase().includes(keyword)) {
                        score += 1;
                    }
                });

                return { ...outfit, matchScore: Math.min(score / 15, 1) };
            });
        }

        const topOutfits = scoredOutfits
            .filter(o => o.matchScore > 0)
            .sort((a, b) => b.matchScore - a.matchScore)
            .slice(0, limit);

        logger.info(` Found ${topOutfits.length} matching outfits`);

        res.json({
            success: true,
            outfits: topOutfits,
            query: { occasion, stylePreferences },
            aiPowered: topOutfits[0]?.matchScore > 0.5
        });

    } catch (error) {
        logger.error("Outfit generation error:", error.message);
        res.status(500).json({ error: "Outfit generation failed" });
    }
});

export default router;
