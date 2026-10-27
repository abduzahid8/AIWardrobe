/**
 * Ollama Cloud provider — hosted Ollama at https://ollama.com.
 *
 * Exposes the OpenAI-compatible /v1 contract with cloud-hosted models.
 * Default `gpt-oss:20b` is a reasoning model — like Aion, it gets extra
 * token headroom and empty-content responses cycle the router.
 */
import { makeOpenAICompatProvider } from "./openaiCompat.js";

const API_KEY = process.env.OLLAMA_API_KEY;
const BASE_URL = process.env.OLLAMA_BASE_URL || "https://ollama.com/v1";
const MODEL = process.env.OLLAMA_MODEL || "gpt-oss:20b";

export default makeOpenAICompatProvider({
    name: "ollamacloud",
    baseUrl: BASE_URL,
    apiKey: API_KEY,
    model: MODEL,
    tokenBuffer: 384,
});

export const isConfigured = () => Boolean(API_KEY);
