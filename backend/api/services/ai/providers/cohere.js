/**
 * Cohere provider — chat (command-r7b) + embeddings.
 *
 * Embeddings default to `embed-english-light-v3.0` which is 384-dimensional —
 * dimensionally identical to HuggingFace sentence-transformers/all-MiniLM-L6-v2
 * used elsewhere in the app, so vectors stay interchangeable in smart-search.
 */
import axios from "axios";

const API_KEY = process.env.COHERE_API_KEY;
const CHAT_MODEL = process.env.COHERE_MODEL || "command-r7b-12-2024";
const EMBED_MODEL = process.env.COHERE_EMBED_MODEL || "embed-english-light-v3.0";
const EMBED_DIM = Number(process.env.COHERE_EMBED_DIM || 384);
const BASE_URL = "https://api.cohere.com/v2";

function extractChatText(data) {
    const parts = data?.message?.content;
    const text = Array.isArray(parts)
        ? parts.map((p) => p?.text || "").join("").trim()
        : "";
    if (!text) throw new Error("empty_response");
    return text;
}

/** Accept either a ready `messages` array or {prompt, systemInstruction}. */
function buildMessages(args) {
    if (Array.isArray(args.messages)) return args.messages;
    const out = [];
    if (args.systemInstruction) out.push({ role: "system", content: args.systemInstruction });
    out.push({ role: "user", content: args.prompt });
    return out;
}

async function chat(args, timeoutMs) {
    const { data } = await axios.post(
        `${BASE_URL}/chat`,
        {
            model: CHAT_MODEL,
            messages: buildMessages(args),
            max_tokens: args.maxTokens || 800,
            temperature: args.temperature ?? 0.7,
        },
        { headers: { "Content-Type": "application/json", Authorization: `Bearer ${API_KEY}` }, timeout: timeoutMs }
    );
    return { text: extractChatText(data) };
}

async function embed({ text }, timeoutMs) {
    const { data } = await axios.post(
        `${BASE_URL}/embed`,
        {
            model: EMBED_MODEL,
            texts: [text],
            embedding_types: ["float"],
            input_type: "search_query",
        },
        { headers: { "Content-Type": "application/json", Authorization: `Bearer ${API_KEY}` }, timeout: timeoutMs }
    );
    const vector = data?.embeddings?.float?.[0];
    if (!Array.isArray(vector) || vector.length === 0) throw new Error("empty_embedding");
    if (vector.length !== EMBED_DIM) {
        throw new Error(`embedding_dim_mismatch: got ${vector.length}, expected ${EMBED_DIM}`);
    }
    return vector;
}

export default {
    name: "cohere",
    capabilities: ["text", "embeddings"],
    isConfigured: () => Boolean(API_KEY),
    run: async (capability, args, timeoutMs) => {
        if (capability === "text") return chat(args, timeoutMs);
        if (capability === "embeddings") return embed(args, timeoutMs);
        throw new Error(`cohere does not support capability ${capability}`);
    },
};

export const isConfigured = () => Boolean(API_KEY);
