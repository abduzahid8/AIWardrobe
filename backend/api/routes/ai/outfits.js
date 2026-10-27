/**
 * AI Outfit Generation Routes
 * POST /generate-outfits — AI-powered outfit recommendations
 *
 * Scoring cycles through all configured providers via the failover router
 * (invalid JSON output counts as a failure and cycles to the next provider),
 * with keyword matching as the deterministic last resort.
 */
import express from "express";
import { authenticateToken } from "../../middleware/auth.js";
import { aiLimiter } from "../../middleware/rateLimit.js";
import logger from "../../utils/logger.js";
import aiRouter from "../../services/ai/index.js";

import { getAllStyleRules } from "../../data/styleRules.js";

const router = express.Router();

// ── POST /generate-outfits ──
router.post("/generate-outfits", authenticateToken, aiLimiter, async (req, res) => {
    try {
        const { occasion, stylePreferences, limit = 5 } = req.body;

        if (!occasion && !stylePreferences) {
            return res.status(400).json({
                error: "Please provide an occasion or style preferences"
            });
        }

        logger.info("👗 Generating outfits:", { occasion, stylePreferences });

        // Import outfit database
        const { curatedOutfits } = await import("../../data/curatedOutfits.js");

        let scoredOutfits;

        const scoringPrompt = `You are a fashion AI stylist. Score each outfit from 0-100 based on how well it matches the request and adheres to core style principles.

FOLLOW THESE STRICT STYLE RULES:
${getAllStyleRules()}

USER REQUEST:
Occasion: ${occasion || 'any'}
Style Preferences: ${stylePreferences || 'none specified'}

AVAILABLE OUTFITS:
${curatedOutfits.map((outfit, idx) => `
${idx + 1}. ${outfit.description}
   - Occasions: ${outfit.occasion.join(', ')}
   - Styles: ${outfit.style.join(', ')}
`).join('\n')}

Return ONLY a JSON array of scores, one number (0-100) for each outfit in order. Example: [95, 70, 85, 60, ...]
No explanation, no markdown, ONLY the JSON array.`;

        const parseScores = (text, outfitCount) => {
            const scores = JSON.parse(text.match(/\[[\d,\s]+\]/)?.[0] || '[]');
            if (scores.length === outfitCount) return scores;
            throw new Error(`Invalid scores count: got ${scores.length}, expected ${outfitCount}`);
        };

        // ── AI scoring: failover router cycles every configured provider ──
        try {
            const scoredResult = await aiRouter.generateJSON(
                scoringPrompt,
                "You MUST respond with ONLY a JSON array of numbers.",
                200,
                {
                    validateJSON: (parsed) => parseScores(JSON.stringify(parsed), curatedOutfits.length),
                }
            );
            const scores = scoredResult.result;
            logger.info(`✅ Using ${scoredResult.provider}-powered matching`);
            scoredOutfits = curatedOutfits.map((outfit, idx) => ({
                ...outfit, matchScore: scores[idx] / 100
            }));
        } catch (aiErr) {
            const attempts = aiErr.attempts?.map((a) => `${a.provider}:${a.error}`).join(", ");
            logger.warn("AI outfit scoring failed on all providers:", attempts || aiErr.message);
        }

        // ── Strategy 3: Keyword matching (always works, no API) ──
        if (!scoredOutfits) {
            logger.info("📝 Using keyword fallback matching");

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

        logger.info(`✅ Found ${topOutfits.length} matching outfits`);

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
