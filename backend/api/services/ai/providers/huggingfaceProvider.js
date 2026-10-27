/**
 * HuggingFace provider adapter — wraps the existing services/huggingface.js
 * (Llama-3-8B text + MiniLM embeddings) into the uniform failover interface.
 * HF remains the always-free last leg of the text chain and the primary
 * embedding source (Cohere light matches its 384-dim vectors).
 */
import * as hfService from "../../huggingface.js";

async function toMessages({ prompt, systemInstruction, messages }) {
    if (Array.isArray(messages)) return messages;
    const out = [];
    if (systemInstruction) out.push({ role: "system", content: systemInstruction });
    out.push({ role: "user", content: prompt });
    return out;
}

export default {
    name: "huggingface",
    capabilities: ["text", "embeddings"],
    isConfigured: () => Boolean(process.env.HF_TOKEN),
    run: async (capability, args) => {
        if (capability === "text") {
            const messages = await toMessages(args);
            const text = await hfService.chatWithHistory(messages, args.maxTokens || 800);
            return { text };
        }
        if (capability === "embeddings") {
            const vector = await hfService.generateEmbedding(args.text);
            if (!Array.isArray(vector) || vector.length === 0) throw new Error("empty_embedding");
            return vector;
        }
        throw new Error(`huggingface does not support capability ${capability}`);
    },
};

export const isConfigured = () => Boolean(process.env.HF_TOKEN);
