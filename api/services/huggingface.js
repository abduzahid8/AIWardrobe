/**
 * HuggingFace Inference API — Centralized Service
 *
 * Single source of truth for all HF model calls.
 * Every AI route imports from here instead of initializing its own clients.
 *
 * Models used (all free-tier serverless inference):
 *   - Text generation:    meta-llama/Meta-Llama-3-8B-Instruct
 *   - Image-to-text:      Salesforce/blip2-opt-2.7b
 *   - Zero-shot image:    openai/clip-vit-large-patch14
 *   - Embeddings:         sentence-transformers/all-MiniLM-L6-v2
 *   - Image classify:     google/vit-base-patch16-224
 */
import { HfInference } from "@huggingface/inference";
import logger from "../utils/logger.js";

// ── Singleton ──
const HF_TOKEN = process.env.HF_TOKEN;
if (!HF_TOKEN) {
    logger.warn("⚠️  HF_TOKEN not set — HuggingFace AI features will fail");
}
const hf = new HfInference(HF_TOKEN);

// ── Models (cheapest free-tier options that work well) ──
const MODELS = {
    TEXT_GEN: "meta-llama/Meta-Llama-3-8B-Instruct",
    IMAGE_TO_TEXT: "Salesforce/blip2-opt-2.7b",
    ZERO_SHOT_IMAGE: "openai/clip-vit-large-patch14",
    EMBEDDINGS: "sentence-transformers/all-MiniLM-L6-v2",
    IMAGE_CLASSIFY: "google/vit-base-patch16-224",
};

// ═══════════════════════════════════════════
// TEXT GENERATION (replaces Gemini / OpenAI chat)
// ═══════════════════════════════════════════

/**
 * Chat-style text generation.
 * @param {string} userPrompt - User message
 * @param {string} systemPrompt - System instruction
 * @param {number} maxTokens - Max output tokens (default 800)
 * @returns {Promise<string>} Generated text
 */
export async function generateText(userPrompt, systemPrompt = "", maxTokens = 800) {
    const messages = [];
    if (systemPrompt) messages.push({ role: "system", content: systemPrompt });
    messages.push({ role: "user", content: userPrompt });

    const result = await hf.chatCompletion({
        model: MODELS.TEXT_GEN,
        messages,
        max_tokens: maxTokens,
        temperature: 0.7,
    });

    if (result?.choices?.length > 0) {
        return result.choices[0].message.content;
    }
    throw new Error("HF returned empty response");
}

/**
 * Chat with conversation history.
 * @param {Array<{role: string, content: string}>} messages
 * @param {number} maxTokens
 * @returns {Promise<string>}
 */
export async function chatWithHistory(messages, maxTokens = 800) {
    const result = await hf.chatCompletion({
        model: MODELS.TEXT_GEN,
        messages,
        max_tokens: maxTokens,
        temperature: 0.7,
    });

    if (result?.choices?.length > 0) {
        return result.choices[0].message.content;
    }
    throw new Error("HF returned empty response");
}

// ═══════════════════════════════════════════
// IMAGE ANALYSIS (replaces OpenAI Vision / Gemini Vision)
// ═══════════════════════════════════════════

/**
 * Convert base64 image to a Blob for HF API.
 */
function base64ToBlob(b64, mimeType = "image/jpeg") {
    const raw = b64.replace(/^data:image\/\w+;base64,/, "");
    const buffer = Buffer.from(raw, "base64");
    return new Blob([buffer], { type: mimeType });
}

/**
 * Image → text description using BLIP-2.
 * @param {string} imageBase64 - Base64 image (with or without data URI prefix)
 * @returns {Promise<string>} Text description
 */
export async function imageToText(imageBase64) {
    const blob = base64ToBlob(imageBase64);
    const result = await hf.imageToText({
        model: MODELS.IMAGE_TO_TEXT,
        data: blob,
    });
    return result?.generated_text || "";
}

/**
 * Image classification (general categories).
 * Uses ViT — very fast, free.
 * @param {string} imageBase64
 * @returns {Promise<Array<{label: string, score: number}>>}
 */
export async function classifyImage(imageBase64) {
    const blob = base64ToBlob(imageBase64);
    const result = await hf.imageClassification({
        model: MODELS.IMAGE_CLASSIFY,
        data: blob,
    });
    return result; // [{label, score}, ...]
}

/**
 * Zero-shot image classification with custom labels (CLIP).
 * Perfect for fashion categories.
 * @param {string} imageBase64
 * @param {string[]} candidateLabels - e.g. ["t-shirt", "jeans", "dress"]
 * @returns {Promise<Array<{label: string, score: number}>>}
 */
export async function zeroShotImageClassify(imageBase64, candidateLabels) {
    const blob = base64ToBlob(imageBase64);
    const result = await hf.zeroShotImageClassification({
        model: MODELS.ZERO_SHOT_IMAGE,
        inputs: { image: blob },
        parameters: { candidate_labels: candidateLabels },
    });
    return result; // [{label, score}, ...]
}

/**
 * Comprehensive clothing analysis combining BLIP-2 + CLIP + LLM.
 * This replaces OpenAI Vision / Gemini Vision for clothing detection.
 * @param {string} imageBase64
 * @returns {Promise<object>} Structured clothing data
 */
export async function analyzeClothingImage(imageBase64) {
    // Step 1: Get text description from BLIP-2
    const description = await imageToText(imageBase64);

    // Step 2: Zero-shot classify into fashion categories
    const fashionLabels = [
        "t-shirt", "shirt", "blouse", "sweater", "hoodie", "jacket", "coat",
        "dress", "skirt", "pants", "jeans", "shorts", "sneakers", "boots",
        "sandals", "bag", "hat", "scarf", "belt", "watch",
    ];
    const categories = await zeroShotImageClassify(imageBase64, fashionLabels);

    // Step 3: Zero-shot classify style
    const styleLabels = ["casual", "formal", "sport", "semi_classic", "elegant", "business"];
    const styles = await zeroShotImageClassify(imageBase64, styleLabels);

    // Step 4: Zero-shot classify position
    const positionLabels = ["upper body clothing", "lower body clothing", "full body clothing", "footwear", "accessory"];
    const positions = await zeroShotImageClassify(imageBase64, positionLabels);

    // Build structured result
    const topCategory = categories[0] || { label: "unknown", score: 0 };
    const topStyle = styles[0] || { label: "casual", score: 0 };
    const topPosition = positions[0] || { label: "upper body clothing", score: 0 };

    // Map position to simplified key
    const posMap = {
        "upper body clothing": "upper",
        "lower body clothing": "lower",
        "full body clothing": "full",
        "footwear": "feet",
        "accessory": "accessory",
    };

    return {
        itemType: topCategory.label,
        color: "detected",  // CLIP doesn't do color well, description has it
        style: topStyle.label,
        description: description,
        position: posMap[topPosition.label] || "upper",
        confidence: Math.round(topCategory.score * 100),
        categories: categories.slice(0, 5),
        styles: styles.slice(0, 3),
    };
}

// ═══════════════════════════════════════════
// EMBEDDINGS (already used in chat.js — centralized here)
// ═══════════════════════════════════════════

/**
 * Generate text embedding for semantic search.
 * @param {string} text
 * @returns {Promise<number[]>} Embedding vector
 */
export async function generateEmbedding(text) {
    const result = await hf.featureExtraction({
        model: MODELS.EMBEDDINGS,
        inputs: text,
    });
    return result;
}

// ═══════════════════════════════════════════
// EXPORTS
// ═══════════════════════════════════════════

export { hf, MODELS };

export default {
    hf,
    MODELS,
    generateText,
    chatWithHistory,
    imageToText,
    classifyImage,
    zeroShotImageClassify,
    analyzeClothingImage,
    generateEmbedding,
};
