/**
 * OpenRouter provider — gateway to hundreds of models behind one key.
 *
 * Used as a late-stage text fallback and, notably, as a *vision* fallback:
 * `google/gemini-2.0-flash-001` accepts OpenAI-style image_url content, which
 * gives the vision chain a third LLM leg after Gemini and OpenAI.
 */
import { makeOpenAICompatProvider } from "./openaiCompat.js";

const API_KEY = process.env.OPENROUTER_API_KEY;
const BASE_URL = process.env.OPENROUTER_BASE_URL || "https://openrouter.ai/api/v1";
const MODEL = process.env.OPENROUTER_MODEL || "google/gemini-2.5-flash";
const VISION_MODEL = process.env.OPENROUTER_VISION_MODEL || "google/gemini-2.5-flash";

export default makeOpenAICompatProvider({
    name: "openrouter",
    baseUrl: BASE_URL,
    apiKey: API_KEY,
    model: MODEL,
    visionModel: VISION_MODEL,
    vision: true,
    headers: (h) => ({
        ...h,
        "HTTP-Referer": process.env.OPENROUTER_SITE_URL || "https://aiwardrobe.app",
        "X-Title": "AIWardrobe",
    }),
});

export const isConfigured = () => Boolean(API_KEY);
