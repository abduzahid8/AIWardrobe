/**
 * Cloudflare Workers AI provider.
 *
 * Two auth modes (checked in order):
 *
 *   1. PROXY MODE (preferred) — calls the aiwardrobe-ai-proxy Worker, which
 *      holds the Workers AI binding. No expiring API tokens; auth via a
 *      shared secret header. Requires:
 *        CLOUDFLARE_PROXY_URL    e.g. https://aiwardrobe-ai-proxy.<subdomain>.workers.dev
 *        CLOUDFLARE_PROXY_SECRET shared secret (Worker env PROXY_SECRET)
 *
 *   2. LEGACY MODE — direct REST call with a long-lived API token:
 *        CLOUDFLARE_API_TOKEN    token with Workers AI permissions
 *        CLOUDFLARE_ACCOUNT_ID   32-hex account ID
 *      (user tokens with limited scope, e.g. verify-only, will 401 at
 *      request time and simply cycle via the circuit breaker)
 *
 * Default model: @cf/meta/llama-3.1-8b-instruct (override: CLOUDFLARE_MODEL).
 * If neither mode is fully configured the provider is skipped.
 */
import axios from "axios";

const API_TOKEN = process.env.CLOUDFLARE_API_TOKEN;
const ACCOUNT_ID = process.env.CLOUDFLARE_ACCOUNT_ID;
const PROXY_URL = (process.env.CLOUDFLARE_PROXY_URL || "").replace(/\/+$/, "");
const PROXY_SECRET = process.env.CLOUDFLARE_PROXY_SECRET;
const MODEL = process.env.CLOUDFLARE_MODEL || "@cf/meta/llama-3.1-8b-instruct";

const useProxy = () => Boolean(PROXY_URL && PROXY_SECRET);
const useLegacy = () => Boolean(API_TOKEN && ACCOUNT_ID);

/** Accept either a ready `messages` array or {prompt, systemInstruction}. */
function buildMessages(args) {
    if (Array.isArray(args.messages)) return args.messages;
    const out = [];
    if (args.systemInstruction) out.push({ role: "system", content: args.systemInstruction });
    out.push({ role: "user", content: args.prompt });
    return out;
}

async function run(_capability, args, timeoutMs) {
    const payload = { messages: buildMessages(args), max_tokens: args.maxTokens || 800 };
    let data;

    if (useProxy()) {
        const res = await axios.post(
            `${PROXY_URL}/`,
            { model: MODEL, ...payload },
            {
                headers: { "Content-Type": "application/json", "x-proxy-secret": PROXY_SECRET },
                timeout: timeoutMs,
            }
        );
        data = res.data;
        if (data?.success === false) {
            throw new Error(`workers_ai_error: ${data.errors?.[0]?.message || "unknown"}`);
        }
        const text = data?.result?.response;
        if (!text) throw new Error("empty_response");
        return { text };
    }

    // Legacy direct mode
    const { data: legacy } = await axios.post(
        `https://api.cloudflare.com/client/v4/accounts/${ACCOUNT_ID}/ai/run/${MODEL}`,
        payload,
        {
            headers: { "Content-Type": "application/json", Authorization: `Bearer ${API_TOKEN}` },
            timeout: timeoutMs,
        }
    );
    if (legacy?.success === false) {
        throw new Error(`workers_ai_error: ${legacy.errors?.[0]?.message || "unknown"}`);
    }
    const text = legacy?.result?.response;
    if (!text) throw new Error("empty_response");
    return { text };
}

export default {
    name: "cloudflare",
    capabilities: ["text"],
    isConfigured: () => useProxy() || useLegacy(),
    run,
};

export const isConfigured = () => useProxy() || useLegacy();
