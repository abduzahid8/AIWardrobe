/**
 * Gemini AI Service — Centralized Google Gemini 2.0 Flash client
 *
 * Uses the existing GEMINI_API_KEY from .env.
 * Gemini 2.0 Flash is free-tier, fast, and high-quality.
 *
 * Exports:
 *   - generateText(prompt, systemInstruction, maxTokens)
 *   - analyzeImage(imageBase64, prompt)
 *   - chatWithHistory(messages, maxTokens)
 */
import { GoogleGenerativeAI } from "@google/generative-ai";
import logger from "../utils/logger.js";

// ── Singleton ──
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
if (!GEMINI_API_KEY) {
    logger.warn("⚠️  GEMINI_API_KEY not set — Gemini AI features will fall back to HuggingFace");
}

const genAI = GEMINI_API_KEY ? new GoogleGenerativeAI(GEMINI_API_KEY) : null;

// Model IDs
const TEXT_MODEL = "gemini-1.5-flash";
const VISION_MODEL = "gemini-1.5-flash";

// ═══════════════════════════════════════════
// TEXT GENERATION
// ═══════════════════════════════════════════

/**
 * Generate text with Gemini 2.0 Flash.
 * @param {string} userPrompt - User message
 * @param {string} systemInstruction - System instruction
 * @param {number} maxTokens - Max output tokens (default 800)
 * @returns {Promise<string>} Generated text
 */
export async function generateText(userPrompt, systemInstruction = "", maxTokens = 800) {
    if (!genAI) throw new Error("Gemini API key not configured");

    const model = genAI.getGenerativeModel({
        model: TEXT_MODEL,
        generationConfig: {
            maxOutputTokens: maxTokens,
            temperature: 0.7,
        },
        ...(systemInstruction ? { systemInstruction } : {}),
    });

    const result = await model.generateContent(userPrompt);
    const response = result.response;
    const text = response.text();

    if (!text) throw new Error("Gemini returned empty response");
    return text;
}

// ═══════════════════════════════════════════
// CHAT WITH HISTORY
// ═══════════════════════════════════════════

/**
 * Multi-turn chat with conversation history.
 * @param {Array<{role: string, content: string}>} messages - Chat messages
 * @param {number} maxTokens
 * @returns {Promise<string>}
 */
export async function chatWithHistory(messages, maxTokens = 800) {
    if (!genAI) throw new Error("Gemini API key not configured");

    // Extract system instruction from messages
    const systemMsg = messages.find(m => m.role === "system");
    const chatMessages = messages.filter(m => m.role !== "system");

    const model = genAI.getGenerativeModel({
        model: TEXT_MODEL,
        generationConfig: {
            maxOutputTokens: maxTokens,
            temperature: 0.7,
        },
        ...(systemMsg ? { systemInstruction: systemMsg.content } : {}),
    });

    // Convert to Gemini format: { role: "user"|"model", parts: [{text}] }
    const history = chatMessages.slice(0, -1).map(m => ({
        role: m.role === "assistant" ? "model" : "user",
        parts: [{ text: m.content }],
    }));

    const lastMessage = chatMessages[chatMessages.length - 1];

    const chat = model.startChat({ history });
    const result = await chat.sendMessage(lastMessage.content);
    const text = result.response.text();

    if (!text) throw new Error("Gemini returned empty response");
    return text;
}

// ═══════════════════════════════════════════
// IMAGE / VISION ANALYSIS
// ═══════════════════════════════════════════

/**
 * Analyze an image with Gemini Vision.
 * @param {string} imageBase64 - Base64 image (with or without data URI prefix)
 * @param {string} prompt - What to analyze
 * @returns {Promise<string>} Analysis result text
 */
export async function analyzeImage(imageBase64, prompt = "Describe this clothing item in detail.") {
    if (!genAI) throw new Error("Gemini API key not configured");

    const model = genAI.getGenerativeModel({
        model: VISION_MODEL,
        generationConfig: {
            maxOutputTokens: 1000,
            temperature: 0.3,
        },
    });

    const cleanBase64 = imageBase64.replace(/^data:image\/\w+;base64,/, "");

    const result = await model.generateContent([
        prompt,
        {
            inlineData: {
                mimeType: "image/jpeg",
                data: cleanBase64,
            },
        },
    ]);

    const text = result.response.text();
    if (!text) throw new Error("Gemini Vision returned empty response");
    return text;
}

/**
 * Analyze a clothing image and return structured JSON.
 * @param {string} imageBase64
 * @returns {Promise<object>} Structured clothing data
 */
export async function analyzeClothingImage(imageBase64) {
    const prompt = `Analyze this clothing image and return ONLY a JSON object with these fields:
{
  "category": "the clothing category (e.g. T-shirt/top, Shirt, Dress, Coat, Trouser, Sneaker, etc.)",
  "specificType": "specific type (e.g. polo shirt, maxi dress, chino pants)",
  "primaryColor": "the dominant color",
  "secondaryColors": ["other colors present"],
  "pattern": "solid, striped, checkered, floral, printed, or other",
  "style": "casual, formal, sport, semi_classic, elegant, or business",
  "material": "best guess at fabric (cotton, polyester, wool, denim, leather, etc.)",
  "season": "spring, summer, fall, winter, or all-season",
  "description": "A brief 1-sentence description of the item"
}
Return ONLY the JSON, no markdown, no explanation.`;

    const text = await analyzeImage(imageBase64, prompt);

    try {
        // Extract JSON from response (handle markdown code blocks)
        const jsonMatch = text.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
            return JSON.parse(jsonMatch[0]);
        }
    } catch {
        logger.warn("Gemini clothing analysis returned non-JSON, returning raw text");
    }

    // Fallback structure
    return {
        category: "clothing",
        specificType: "unknown",
        primaryColor: "unknown",
        pattern: "solid",
        style: "casual",
        material: "unknown",
        season: "all-season",
        description: text.slice(0, 200),
    };
}

// ═══════════════════════════════════════════
// UTILITY
// ═══════════════════════════════════════════

/**
 * Check if Gemini is available (API key configured).
 */
export function isAvailable() {
    return !!genAI;
}

// ═══════════════════════════════════════════
// EXPORTS
// ═══════════════════════════════════════════

export default {
    generateText,
    chatWithHistory,
    analyzeImage,
    analyzeClothingImage,
    isAvailable,
};
