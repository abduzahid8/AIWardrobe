/**
 * OpenAI-compatible provider factory.
 *
 * Aion Labs, Ollama Cloud, OpenRouter, NVIDIA and dozens of other vendors
 * expose the same /chat/completions contract. One factory, many providers —
 * each with its own base URL, key, model and capabilities.
 *
 * Reasoning models (Aion, gpt-oss…) may spend tokens in a `reasoning` field
 * and return empty `content` when the token budget is exhausted — we detect
 * that and treat it as a failure so the router cycles onward.
 */
import axios from "axios";

/**
 * @param {object} cfg
 * @param {string} cfg.name
 * @param {string} cfg.baseUrl - e.g. "https://api.aionlabs.ai/v1"
 * @param {string|undefined} cfg.apiKey
 * @param {string} cfg.model - Default chat model
 * @param {string} [cfg.visionModel] - Model used for image prompts
 * @param {boolean} [cfg.vision] - Advertise the vision capability
 * @param {number} [cfg.tokenBuffer] - Extra max_tokens headroom for reasoning models
 * @param {(headers:object)=>object} [cfg.headers] - Extra headers
 */
export function makeOpenAICompatProvider(cfg) {
    const {
        name,
        baseUrl,
        apiKey,
        model,
        visionModel,
        vision: visionEnabled = false,
        tokenBuffer = 0,
        headers: extraHeaders,
    } = cfg;

    const capabilities = ["text"];
    if (visionEnabled) capabilities.push("vision");

    function authHeaders() {
        const h = { "Content-Type": "application/json", Authorization: `Bearer ${apiKey}` };
        return extraHeaders ? extraHeaders(h) : h;
    }

    function extractContent(data) {
        const msg = data?.choices?.[0]?.message;
        const text = msg?.content;
        if (typeof text === "string" && text.trim().length > 0) return text.trim();
        if (msg?.reasoning) {
            throw new Error("empty_content_reasoning_exhausted_tokens");
        }
        throw new Error("empty_response");
    }

    /** Accept either a ready `messages` array or {prompt, systemInstruction}. */
    function buildMessages(args) {
        if (Array.isArray(args.messages)) {
            return args.messages.map((m) => ({
                role: m.role === "model" ? "assistant" : m.role,
                content: m.content,
            }));
        }
        const out = [];
        if (args.systemInstruction) out.push({ role: "system", content: args.systemInstruction });
        out.push({ role: "user", content: args.prompt });
        return out;
    }

    async function chat(args, timeoutMs) {
        const { data } = await axios.post(
            `${baseUrl}/chat/completions`,
            {
                model,
                messages: buildMessages(args),
                max_tokens: (args.maxTokens || 800) + tokenBuffer,
                temperature: args.temperature ?? 0.7,
            },
            { headers: authHeaders(), timeout: timeoutMs }
        );
        return { text: extractContent(data) };
    }

    async function vision({ imageBase64, prompt, maxTokens = 1000, temperature = 0.3 }, timeoutMs) {
        const cleanBase64 = imageBase64.replace(/^data:image\/\w+;base64,/, "");
        const { data } = await axios.post(
            `${baseUrl}/chat/completions`,
            {
                model: visionModel || model,
                messages: [
                    {
                        role: "user",
                        content: [
                            { type: "text", text: prompt },
                            {
                                type: "image_url",
                                image_url: { url: `data:image/jpeg;base64,${cleanBase64}`, detail: "low" },
                            },
                        ],
                    },
                ],
                max_tokens: maxTokens + tokenBuffer,
                temperature,
            },
            { headers: authHeaders(), timeout: timeoutMs }
        );
        return { text: extractContent(data) };
    }

    return {
        name,
        capabilities,
        isConfigured: () => Boolean(apiKey),
        run: async (capability, args, timeoutMs) => {
            if (capability === "text") return chat(args, timeoutMs);
            if (capability === "vision") return vision(args, timeoutMs);
            throw new Error(`${name} does not support capability ${capability}`);
        },
    };
}

export default makeOpenAICompatProvider;
