/**
 * Cloudflare Workers AI provider.
 *
 * Runs serverless models (default @cf/meta/llama-3.1-8b-instruct) via
 * POST /accounts/{account_id}/ai/run/{model}.
 *
 * Requires BOTH env vars:
 *   CLOUDFLARE_API_TOKEN  — token with Workers AI permissions
 *   CLOUDFLARE_ACCOUNT_ID — the 32-hex account ID (Tokens → verify only
 *                           validates the token; the account ID must be
 *                           supplied separately)
 * If either is missing the provider is simply unconfigured and skipped.
 */
import axios from "axios";

const API_TOKEN = process.env.CLOUDFLARE_API_TOKEN;
const ACCOUNT_ID = process.env.CLOUDFLARE_ACCOUNT_ID;
const MODEL = process.env.CLOUDFLARE_MODEL || "@cf/meta/llama-3.1-8b-instruct";

/** Accept either a ready `messages` array or {prompt, systemInstruction}. */
function buildMessages(args) {
    if (Array.isArray(args.messages)) return args.messages;
    const out = [];
    if (args.systemInstruction) out.push({ role: "system", content: args.systemInstruction });
    out.push({ role: "user", content: args.prompt });
    return out;
}

async function run(capability, args, timeoutMs) {
    const { data } = await axios.post(
        `https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/ai/run/${MODEL}`,
        { messages: buildMessages(args), max_tokens: args.maxTokens || 800 },
        {
            headers: { "Content-Type": "application/json", Authorization: `Bearer ${API_TOKEN}` },
            timeout: timeoutMs,
        }
    );

    if (data?.success === false) {
        throw new Error(`workers_ai_error: ${data.errors?.[0]?.message || "unknown"}`);
    }
    const text = data?.result?.response;
    if (!text) throw new Error("empty_response");
    return { text };
}

export default {
    name: "cloudflare",
    capabilities: ["text"],
    isConfigured: () => Boolean(API_TOKEN && ACCOUNT_ID),
    run,
};

export const isConfigured = () => Boolean(API_TOKEN && ACCOUNT_ID);
