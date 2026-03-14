/**
 * AI Outfit Generation Routes
 * POST /generate-outfits — AI-powered outfit recommendations
 *
 * Primary: OpenAI GPT-4o-mini → Gemini → HuggingFace → keyword matching
 */
import express from "express";
import { authenticateToken } from "../../middleware/auth.js";
import { aiLimiter } from "../../middleware/rateLimit.js";
import logger from "../../utils/logger.js";
import openaiService from "../../services/openai.js";
import geminiService from "../../services/gemini.js";
import hfService from "../../services/huggingface.js";

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

        const scoringPrompt = `You are a fashion AI stylist. Score each outfit from 0-100 based on how well it matches the request.

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

        // ── Strategy 1: OpenAI GPT-4o-mini ──
        if (!scoredOutfits && openaiService.isAvailable()) {
            try {
                const responseText = await openaiService.generateText(scoringPrompt, "You MUST respond with ONLY a JSON array of numbers.", 200);
                const scores = parseScores(responseText, curatedOutfits.length);
                logger.info("✅ Using OpenAI-powered matching");
                scoredOutfits = curatedOutfits.map((outfit, idx) => ({
                    ...outfit, matchScore: scores[idx] / 100
                }));
            } catch (oaiErr) {
                logger.warn("OpenAI outfit scoring failed:", oaiErr.message);
            }
        }

        // ── Strategy 2: Gemini 2.0 Flash ──
        if (!scoredOutfits && geminiService.isAvailable()) {
            try {
                const responseText = await geminiService.generateText(scoringPrompt, "You MUST respond with ONLY a JSON array of numbers.", 200);
                const scores = parseScores(responseText, curatedOutfits.length);
                logger.info("✅ Using Gemini-powered matching");
                scoredOutfits = curatedOutfits.map((outfit, idx) => ({
                    ...outfit, matchScore: scores[idx] / 100
                }));
            } catch (geminiErr) {
                logger.warn("Gemini outfit scoring failed:", geminiErr.message);
            }
        }

        // ── Strategy 2: HuggingFace (fallback) ──
        if (!scoredOutfits) {
            try {
                const prompt = `You are a fashion AI stylist. Score each outfit from 0-100 based on how well it matches the request.

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

                const responseText = await hfService.generateText(
                    prompt,
                    "You are a fashion stylist AI. You MUST respond with ONLY a JSON array of numbers. No explanation.",
                    200
                );

                const scores = JSON.parse(responseText.match(/\[[\d,\s]+\]/)?.[0] || '[]');

                if (scores.length === curatedOutfits.length) {
                    logger.info("✅ Using HuggingFace-powered matching");
                    scoredOutfits = curatedOutfits.map((outfit, idx) => ({
                        ...outfit,
                        matchScore: scores[idx] / 100
                    }));
                } else {
                    throw new Error("HF returned invalid scores count");
                }
            } catch (hfError) {
                logger.warn("HF outfit scoring failed:", hfError.message);
            }
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
