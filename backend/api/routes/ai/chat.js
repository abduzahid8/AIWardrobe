/**
 * AI Chat & Smart Search Routes
 * GET  /smart-search — Semantic outfit search with embeddings
 * POST /ai-chat      — OpenAI GPT-4o-mini fashion stylist chat (Gemini → HF fallback)
 */
import express from "express";
import cosineSimilarity from "compute-cosine-similarity";
import { supabase } from "../../lib/supabase.js";
import { authenticateToken } from "../../middleware/auth.js";
import logger from "../../utils/logger.js";
import { validateAIChat } from "../../middleware/validators.js";
import hfService from "../../services/huggingface.js";
import geminiService from "../../services/gemini.js";
import openaiService from "../../services/openai.js";

import { getAllStyleRules } from "../../data/styleRules.js";

const router = express.Router();

/** Normalize search query */
const normalizeQuery = (query) => {
    const synonyms = {
        "coffee date": "coffee date",
        "dinner date": "date",
        "job interview": "interview",
        work: "interview",
        casual: "casual",
        formal: "formal",
        outfit: "",
        "give me": "",
        a: "",
        an: "",
        for: "",
    };

    let normalized = query.toLowerCase();
    Object.keys(synonyms).forEach((key) => {
        normalized = normalized.replace(new RegExp(`\\b\${key}\\b`, "gi"), synonyms[key]);
    });
    return [...new Set(normalized.trim().split(/\s+/).filter(Boolean))].join(" ");
};

// ── GET /smart-search ──
// NOTE: there is no `outfits` table — user outfits live in `saved_outfits`
// (see 001_full_rebuild.sql). That table has no embedding/style columns, so
// embedding search is best-effort with a keyword fallback that always works.
router.get("/smart-search", authenticateToken, async (req, res) => {
    const { query } = req.query;
    if (!query) return res.status(400).json({ error: "Query required" });

    try {
        const normalizedQuery = normalizeQuery(query);
        const { data: rawOutfits, error: outfitsError } = await supabase
            .from('saved_outfits')
            .select('id, occasion, season, name, caption, items, visibility')
            .eq('user_id', req.user.id)
            .limit(100);
        if (outfitsError) throw outfitsError;
        const outfits = (rawOutfits || []).map(o => ({
            ...o,
            _id: o.id,
            toObject: function () { return this; }
        }));

        const queryTerms = normalizedQuery.split(" ").filter(Boolean);
        const keywordScored = outfits
            .filter((o) =>
                queryTerms.some(
                    (term) =>
                        (o.occasion || "").toLowerCase().includes(term) ||
                        (o.caption || "").toLowerCase().includes(term) ||
                        (o.name || "").toLowerCase().includes(term) ||
                        JSON.stringify(o.items || []).toLowerCase().includes(term)
                )
            )
            .map((o) => ({ ...o.toObject(), score: 0.1 }));

        // Embedding re-rank is optional — HF may be down or outfits may lack
        // embeddings. Never let it fail the request.
        try {
            const queryEmbedding = await hfService.generateEmbedding(normalizedQuery);
            const MIN_SIMILARITY = query.length > 20 ? 0.3 : 0.4;
            const embedded = outfits
                .filter((o) => Array.isArray(o.embedding))
                .map((o) => {
                    const score = cosineSimilarity(queryEmbedding, o.embedding);
                    return { ...o.toObject(), score };
                })
                .filter((o) => o.score >= MIN_SIMILARITY)
                .sort((a, b) => b.score - a.score);
            if (embedded.length > 0) return res.json(embedded.slice(0, 5));
        } catch (embErr) {
            logger.warn("smart-search embedding fallback:", embErr.message);
        }

        res.json(keywordScored.slice(0, 5));
    } catch (err) {
        logger.error("AI ERROR:", err);
        res.status(500).json({ error: "Search failed" });
    }
});

// ── POST /ai-chat ──
// Primary: OpenAI GPT-4o-mini → Gemini → HuggingFace
router.post("/ai-chat", authenticateToken, validateAIChat, async (req, res) => {
    const { query } = req.body;
    logger.info("💬 Chat request:", query);

    const systemPrompt = `You are Ava, a helpful, friendly fashion stylist AI assistant. Keep answers concise, actionable, and fun.

CLOTHING PRICES AND BUDGET:
- The user message may include owned wardrobe (cost 0) and a shop catalog with listed prices.
- ALWAYS name prices when recommending shop/catalog pieces.
- If the user gives a budget (e.g. "under $150"), shop items in a recommended look MUST sum to that amount or less. Owned items are free.
- Prefer cheaper shop substitutes over exceeding the cap. If a required slot has nothing affordable, say so.

FOLLOW THESE STRICT STYLE RULES:
${getAllStyleRules()}`;

    // Strategy 1: OpenAI GPT-4o-mini
    if (openaiService.isAvailable()) {
        try {
            const text = await openaiService.generateText(query, systemPrompt, 500);
            logger.info("✅ Chat response via OpenAI");
            return res.json({ text, provider: "openai" });
        } catch (oaiErr) {
            logger.warn("OpenAI chat failed:", oaiErr.message);
        }
    }

    // Strategy 2: Gemini 2.0 Flash
    if (geminiService.isAvailable()) {
        try {
            const text = await geminiService.generateText(query, systemPrompt, 500);
            logger.info("✅ Chat response via Gemini");
            return res.json({ text, provider: "gemini" });
        } catch (geminiErr) {
            logger.warn("Gemini chat failed:", geminiErr.message);
        }
    }

    // Strategy 3: HuggingFace Llama 3
    try {
        const text = await hfService.generateText(
            query,
            "You are a helpful fashion stylist. Keep answers short and fun with emojis.",
            500
        );
        res.json({ text, provider: "huggingface" });
    } catch (err) {
        logger.error("All AI providers failed:", err.message);
        res.status(500).json({ error: "AI model is busy, try again later." });
    }
});

export default router;
