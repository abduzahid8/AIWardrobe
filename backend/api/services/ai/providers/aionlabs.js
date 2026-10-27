/**
 * Aion Labs provider — OpenAI-compatible API at https://api.aionlabs.ai/v1.
 *
 * Models are narrative/reasoning-tuned; `aion-3.5-mini` is the fast tier.
 * Reasoning models may burn tokens thinking before answering, so requests
 * carry extra headroom (tokenBuffer) and empty content cycles the router.
 */
import { makeOpenAICompatProvider } from "./openaiCompat.js";

const API_KEY = process.env.AIONLABS_API_KEY;
const BASE_URL = process.env.AIONLABS_BASE_URL || "https://api.aionlabs.ai/v1";
const MODEL = process.env.AIONLABS_MODEL || "aion-labs/aion-3.5-mini";

export default makeOpenAICompatProvider({
    name: "aionlabs",
    baseUrl: BASE_URL,
    apiKey: API_KEY,
    model: MODEL,
    // Aion models are text→text (no vision modality).
    tokenBuffer: 384,
});

export const isConfigured = () => Boolean(API_KEY);
