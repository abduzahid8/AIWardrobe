/**
 * AI Chat & Smart Search Routes
 * GET  /smart-search — Semantic outfit search with embeddings
 * POST /ai-chat      — HuggingFace fashion stylist chat
 */
import express from "express";
import { HfInference } from "@huggingface/inference";
import cosineSimilarity from "compute-cosine-similarity";
import Outfit from "../../models/outfit.js";
import { authenticateToken } from "../../middleware/auth.js";
import logger from "../../utils/logger.js";
import { validateAIChat } from "../../middleware/validators.js";

const router = express.Router();
const hf = new HfInference(process.env.HF_TOKEN);

// ── Helpers ──

/** Generate text embedding using HuggingFace */
const generateEmbedding = async (text) => {
    const response = await hf.featureExtraction({
        model: "sentence-transformers/all-MiniLM-L6-v2",
        inputs: text,
    });
    return response;
};

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
        normalized = normalized.replace(new RegExp(`\\b${key}\\b`, "gi"), synonyms[key]);
    });
    return [...new Set(normalized.trim().split(/\s+/).filter(Boolean))].join(" ");
};

// ── GET /smart-search ──
router.get("/smart-search", authenticateToken, async (req, res) => {
    const { query } = req.query;
    if (!query) return res.status(400).json({ error: "Query required" });

    try {
        const normalizedQuery = normalizeQuery(query);
        const queryEmbedding = await generateEmbedding(normalizedQuery);
        const outfits = await Outfit.find();

        const MIN_SIMILARITY = query.length > 20 ? 0.3 : 0.4;

        let scored = outfits
            .map((o) => {
                const score = cosineSimilarity(queryEmbedding, o.embedding);
                return { ...o.toObject(), score };
            })
            .filter((o) => o.score >= MIN_SIMILARITY)
            .sort((a, b) => b.score - a.score);

        if (scored.length === 0) {
            const queryTerms = normalizedQuery.split(" ");
            scored = outfits
                .filter((o) =>
                    queryTerms.some(
                        (term) =>
                            (o.occasion || "").toLowerCase().includes(term) ||
                            (o.style || "").toLowerCase().includes(term) ||
                            (o.items || []).some((item) => (item || "").toLowerCase().includes(term))
                    )
                )
                .map((o) => ({ ...o.toObject(), score: 0.1 }));
        }

        res.json(scored.slice(0, 5));
    } catch (err) {
        logger.error(" AI ERROR:", err);
        res.status(500).json({ error: "Search failed" });
    }
});

// ── POST /ai-chat ──
router.post("/ai-chat", authenticateToken, validateAIChat, async (req, res) => {
    const { query } = req.body;
    logger.info(" Chat request:", query);

    try {
        const result = await hf.chatCompletion({
            model: "meta-llama/Meta-Llama-3-8B-Instruct",
            messages: [
                {
                    role: "system",
                    content: "You are a helpful fashion stylist. Keep answers short and fun with emojis.",
                },
                { role: "user", content: query },
            ],
            max_tokens: 500,
            temperature: 0.7,
        });

        if (result && result.choices && result.choices.length > 0) {
            res.json({ text: result.choices[0].message.content });
        } else {
            throw new Error("AI returned empty response");
        }
    } catch (err) {
        logger.error(" HF Error:", err.message);
        res.status(500).json({ error: "AI model is busy, try again later." });
    }
});

export default router;
