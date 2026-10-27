/**
 * AI Failover Facade — the single entry point for LLM work in the API.
 *
 * Route handlers call these functions exactly like the old per-vendor
 * services; under the hood every call cycles through all configured
 * providers (Gemini → OpenAI → Aion Labs → Cohere → Ollama Cloud →
 * OpenRouter → Cloudflare → HuggingFace) with per-provider circuit breakers.
 *
 * Every function returns `{ ...result, provider, attempts }` so responses
 * keep exposing which provider actually served the request.
 *
 * JSON-mode helpers treat malformed/empty model output as a provider failure,
 * which automatically cycles to the next provider instead of erroring out.
 */
import { execute, AllProvidersFailedError, NoProvidersConfiguredError } from "./failover.js";
import { getProviders } from "./registry.js";
import { snapshotAll } from "./circuitBreaker.js";
import { extractJSON } from "./jsonUtils.js";
import logger from "../../utils/logger.js";

const CLOTHING_PROMPT = `Analyze this clothing image and return ONLY a JSON object with these fields:
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

/** Extract the first JSON object/array from arbitrary model text. */
export { extractJSON } from "./jsonUtils.js";

// ═══════════════════════════════════════════
// TEXT
// ═══════════════════════════════════════════

/**
 * Single-turn text generation with a system instruction.
 * @returns {Promise<{text: string, provider: string, attempts: Array}>}
 */
export function generateText(prompt, systemInstruction = "", maxTokens = 800, opts = {}) {
    return run("text", { prompt, systemInstruction, maxTokens }, opts);
}

/**
 * Multi-turn chat with conversation history.
 * @param {Array<{role: string, content: string}>} messages
 * @returns {Promise<{text: string, provider: string, attempts: Array}>}
 */
export function chatWithHistory(messages, maxTokens = 800, opts = {}) {
    return run("text", { messages, maxTokens }, opts);
}

/**
 * Text generation that must parse as JSON — invalid output cycles providers.
 * @returns {Promise<{result: any, provider: string, attempts: Array}>}
 */
export function generateJSON(prompt, systemInstruction = "", maxTokens = 800, opts = {}) {
    return run("text", { prompt, systemInstruction, maxTokens }, {
        ...opts,
        validate: (r) => {
            const parsed = extractJSON(r.text);
            if (opts.validateJSON) return opts.validateJSON(parsed);
            return parsed;
        },
    });
}

// ═══════════════════════════════════════════
// VISION
// ═══════════════════════════════════════════

/**
 * Free-form image analysis.
 * @returns {Promise<{text: string, provider: string, attempts: Array}>}
 */
export function analyzeImage(imageBase64, prompt = "Describe this clothing item in detail.", opts = {}) {
    return run("vision", { imageBase64, prompt, maxTokens: opts.maxTokens || 1000 }, opts);
}

/**
 * Structured clothing analysis (same JSON contract as the legacy services).
 * Providers returning garbage JSON are skipped and the next one is tried.
 * @returns {Promise<{result: object, provider: string, attempts: Array}>}
 */
export function analyzeClothingImage(imageBase64, opts = {}) {
    return run("vision", { imageBase64, prompt: CLOTHING_PROMPT, maxTokens: 1000 }, {
        ...opts,
        validate: (r) => {
            const parsed = extractJSON(r.text);
            // Minimum viable schema — anything else means the provider
            // didn't actually understand the image.
            if (!parsed.category && !parsed.primaryColor) {
                throw new Error("missing category/primaryColor");
            }
            return parsed;
        },
    });
}

// ═══════════════════════════════════════════
// EMBEDDINGS
// ═══════════════════════════════════════════

/**
 * 384-dim embedding (HF MiniLM, Cohere light fallback — same dimensionality).
 * @returns {Promise<{result: number[], provider: string, attempts: Array}>}
 */
export function generateEmbedding(text, opts = {}) {
    return run("embeddings", { text }, opts);
}

// ═══════════════════════════════════════════
// INTERNAL + STATUS
// ═══════════════════════════════════════════

async function run(capability, args, opts = {}) {
    return execute(capability, args, {
        providers: await getProviders(),
        logger,
        ...opts,
    });
}

/**
 * Health snapshot for the /ai-status endpoint — provider config, breaker
 * states, retry timers. Contains no secrets.
 */
export async function status() {
    return {
        providers: (await getProviders()).map((p) => ({
            name: p.name,
            capabilities: p.capabilities,
            configured: p.isConfigured(),
            breaker: snapshotAll()[p.name] || null,
        })),
    };
}

export { AllProvidersFailedError, NoProvidersConfiguredError };

export default {
    generateText,
    chatWithHistory,
    generateJSON,
    analyzeImage,
    analyzeClothingImage,
    generateEmbedding,
    extractJSON,
    status,
    AllProvidersFailedError,
    NoProvidersConfiguredError,
};
