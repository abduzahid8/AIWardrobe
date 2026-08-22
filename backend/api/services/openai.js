/**
 * OpenAI Service — GPT-4o-mini Text/Chat/Vision + DALL-E 3 Image Generation
 *
 * Uses the existing OPENAI_API_KEY from .env.
 * GPT-4o-mini: fast, cheap, high-quality text/chat/vision
 * DALL-E 3: image generation
 *
 * Exports:
 *   - generateText(prompt, systemInstruction, maxTokens)
 *   - chatWithHistory(messages, maxTokens)
 *   - analyzeImage(imageBase64, prompt)
 *   - analyzeClothingImage(imageBase64)
 *   - generateImage(prompt, size, quality)
 */
import axios from "axios";
import logger from "../utils/logger.js";

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
if (!OPENAI_API_KEY) {
    logger.warn("⚠️  OPENAI_API_KEY not set — OpenAI features will be unavailable");
}

const OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions";
const OPENAI_IMAGE_URL = "https://api.openai.com/v1/images/generations";
const TEXT_MODEL = "gpt-4o-mini";

const headers = () => ({
    "Content-Type": "application/json",
    "Authorization": `Bearer ${OPENAI_API_KEY}`,
});

// ═══════════════════════════════════════════
// TEXT GENERATION
// ═══════════════════════════════════════════

/**
 * Generate text with GPT-4o-mini.
 * @param {string} userPrompt - User message
 * @param {string} systemInstruction - System instruction
 * @param {number} maxTokens - Max output tokens (default 800)
 * @returns {Promise<string>} Generated text
 */
export async function generateText(userPrompt, systemInstruction = "", maxTokens = 800) {
    if (!OPENAI_API_KEY) throw new Error("OpenAI API key not configured");

    const messages = [];
    if (systemInstruction) messages.push({ role: "system", content: systemInstruction });
    messages.push({ role: "user", content: userPrompt });

    const response = await axios.post(OPENAI_CHAT_URL, {
        model: TEXT_MODEL,
        messages,
        max_tokens: maxTokens,
        temperature: 0.7,
    }, { headers: headers(), timeout: 30000 });

    const text = response.data?.choices?.[0]?.message?.content;
    if (!text) throw new Error("OpenAI returned empty response");
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
    if (!OPENAI_API_KEY) throw new Error("OpenAI API key not configured");

    // OpenAI uses "assistant" not "model"
    const formattedMessages = messages.map(m => ({
        role: m.role === "model" ? "assistant" : m.role,
        content: m.content,
    }));

    const response = await axios.post(OPENAI_CHAT_URL, {
        model: TEXT_MODEL,
        messages: formattedMessages,
        max_tokens: maxTokens,
        temperature: 0.7,
    }, { headers: headers(), timeout: 30000 });

    const text = response.data?.choices?.[0]?.message?.content;
    if (!text) throw new Error("OpenAI returned empty response");
    return text;
}

// ═══════════════════════════════════════════
// VISION / IMAGE ANALYSIS
// ═══════════════════════════════════════════

/**
 * Analyze an image with GPT-4o-mini vision.
 * @param {string} imageBase64 - Base64 image
 * @param {string} prompt - What to analyze
 * @returns {Promise<string>} Analysis result text
 */
export async function analyzeImage(imageBase64, prompt = "Describe this clothing item in detail.") {
    if (!OPENAI_API_KEY) throw new Error("OpenAI API key not configured");

    const cleanBase64 = imageBase64.replace(/^data:image\/\w+;base64,/, "");

    const response = await axios.post(OPENAI_CHAT_URL, {
        model: TEXT_MODEL,
        messages: [
            {
                role: "user",
                content: [
                    { type: "text", text: prompt },
                    {
                        type: "image_url",
                        image_url: {
                            url: `data:image/jpeg;base64,${cleanBase64}`,
                            detail: "low",
                        },
                    },
                ],
            },
        ],
        max_tokens: 1000,
        temperature: 0.3,
    }, { headers: headers(), timeout: 30000 });

    const text = response.data?.choices?.[0]?.message?.content;
    if (!text) throw new Error("OpenAI Vision returned empty response");
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
        const jsonMatch = text.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
            return JSON.parse(jsonMatch[0]);
        }
    } catch {
        logger.warn("OpenAI clothing analysis returned non-JSON, returning raw text");
    }

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
// IMAGE GENERATION (DALL-E 3)
// ═══════════════════════════════════════════

/**
 * Generate an image using DALL-E 3.
 * @param {string} prompt - Image description
 * @param {string} size - Image size: "1024x1024", "1792x1024", or "1024x1792"
 * @param {string} quality - "standard" or "hd"
 * @returns {Promise<{imageUrl: string, revisedPrompt: string}>}
 */
export async function generateImage(prompt, size = "1024x1024", quality = "standard") {
    if (!OPENAI_API_KEY) throw new Error("OpenAI API key not configured");

    logger.info("🎨 OpenAI: Generating image with DALL-E 3...");

    const response = await axios.post(OPENAI_IMAGE_URL, {
        model: "dall-e-3",
        prompt,
        n: 1,
        size,
        quality,
        response_format: "b64_json",
    }, {
        headers: headers(),
        timeout: 60000,
    });

    if (response.data?.data?.[0]) {
        const result = response.data.data[0];
        logger.info("✅ DALL-E 3 image generated");
        return {
            imageBase64: result.b64_json,
            imageUrl: `data:image/png;base64,${result.b64_json}`,
            revisedPrompt: result.revised_prompt || prompt,
        };
    }

    throw new Error("OpenAI returned empty response");
}

// ═══════════════════════════════════════════
// UTILITY
// ═══════════════════════════════════════════

export function isAvailable() {
    return !!OPENAI_API_KEY;
}

import sharp from "sharp";
import FormData from "form-data";

// ═══════════════════════════════════════════
// IMAGE EDITING (DALL-E 2)
// ═══════════════════════════════════════════

/**
 * Edit an image to create a studio photo.
 * Uses the alpha channel as a mask to preserve the opaque garment and generate a background.
 * @param {string} imageBase64 - Base64 image (must be transparent PNG and square)
 * @param {string} prompt - Description of desired output background
 * @returns {Promise<string>} Edited image as base64
 */
export async function editImage(imageBase64, prompt) {
    if (!OPENAI_API_KEY) throw new Error("OpenAI API key not configured");

    logger.info("🎨 OpenAI: Generating DALL-E studio background...");

    try {
        const cleanBase64 = imageBase64.replace(/^data:image\/\w+;base64,/, "");
        const imageBuffer = Buffer.from(cleanBase64, "base64");

        // DALL-E 2 edit requires a square transparent PNG under 4MB
        // We use sharp to pad the cutout into a 1024x1024 square with a transparent background
        const squaredBuffer = await sharp(imageBuffer)
            .resize(1024, 1024, {
                fit: "contain",
                background: { r: 0, g: 0, b: 0, alpha: 0 }
            })
            .png()
            .toBuffer();

        const form = new FormData();
        form.append("image", squaredBuffer, {
            filename: "garment.png",
            contentType: "image/png",
        });

        form.append("prompt", prompt);
        form.append("n", 1);
        form.append("size", "1024x1024");
        form.append("response_format", "b64_json");
        // We use dall-e-2 as it supports alpha-channel based editing
        form.append("model", "dall-e-2");

        const response = await axios.post("https://api.openai.com/v1/images/edits", form, {
            headers: {
                ...form.getHeaders(),
                "Authorization": `Bearer ${OPENAI_API_KEY}`
            },
            timeout: 60000,
            maxBodyLength: Infinity,
            maxContentLength: Infinity
        });

        if (response.data?.data?.[0]) {
            logger.info("✅ DALL-E studio background generated");
            return response.data.data[0].b64_json;
        }

        throw new Error("OpenAI returned empty response");
    } catch (error) {
        logger.error("DALL-E edit error:", error.response?.data?.error?.message || error.message);
        throw error;
    }
}

export default {
    generateText,
    chatWithHistory,
    analyzeImage,
    analyzeClothingImage,
    generateImage,
    editImage,
    isAvailable,
};
