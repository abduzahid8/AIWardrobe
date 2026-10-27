/**
 * OpenAI provider adapter — wraps the existing services/openai.js
 * (GPT-4o-mini text/vision) into the uniform failover interface instead of
 * reimplementing the vendor calls.
 */
import * as openaiService from "../../openai.js";

async function toMessages({ prompt, systemInstruction, messages }) {
    if (Array.isArray(messages)) {
        return messages.map((m) => ({
            role: m.role === "model" ? "assistant" : m.role,
            content: m.content,
        }));
    }
    const out = [];
    if (systemInstruction) out.push({ role: "system", content: systemInstruction });
    out.push({ role: "user", content: prompt });
    return out;
}

export default {
    name: "openai",
    capabilities: ["text", "vision"],
    isConfigured: () => openaiService.isAvailable(),
    run: async (capability, args) => {
        if (capability === "text") {
            const messages = await toMessages(args);
            const text = await openaiService.chatWithHistory(messages, args.maxTokens || 800);
            return { text };
        }
        if (capability === "vision") {
            const text = await openaiService.analyzeImage(
                args.imageBase64,
                args.prompt || "Describe this clothing item in detail."
            );
            return { text };
        }
        throw new Error(`openai does not support capability ${capability}`);
    },
};

export const isConfigured = () => openaiService.isAvailable();
