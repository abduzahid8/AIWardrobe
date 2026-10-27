/**
 * Gemini provider adapter — wraps the existing services/gemini.js
 * (Gemini 2.0 Flash text/vision) into the uniform failover interface.
 */
import * as geminiService from "../../gemini.js";

async function toMessages({ prompt, systemInstruction, messages }) {
    if (Array.isArray(messages)) return messages;
    return [
        ...(systemInstruction ? [{ role: "system", content: systemInstruction }] : []),
        { role: "user", content: prompt },
    ];
}

export default {
    name: "gemini",
    capabilities: ["text", "vision"],
    isConfigured: () => geminiService.isAvailable(),
    run: async (capability, args) => {
        if (capability === "text") {
            const messages = await toMessages(args);
            const text = await geminiService.chatWithHistory(messages, args.maxTokens || 800);
            return { text };
        }
        if (capability === "vision") {
            const text = await geminiService.analyzeImage(
                args.imageBase64,
                args.prompt || "Describe this clothing item in detail."
            );
            return { text };
        }
        throw new Error(`gemini does not support capability ${capability}`);
    },
};

export const isConfigured = () => geminiService.isAvailable();
