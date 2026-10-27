/**
 * Provider registry — builds the failover chain from environment config.
 *
 * Providers are loaded lazily and ISOLATED: a missing vendor SDK (or any
 * module-level error) only disables that one provider instead of taking
 * down every AI route. The chain is built once per process and cached.
 *
 * Order resolution:
 *   1. AI_PROVIDER_ORDER env (comma-separated) filters and re-orders,
 *      e.g. "gemini,aionlabs,cohere,openai,huggingface".
 *   2. Unknown names are ignored; configured providers missing from the
 *      env order are appended (never silently dropped).
 *   3. Unconfigured providers are skipped at request time by the engine.
 */
import logger from "../../utils/logger.js";

/** Full chain in default priority order. */
const LOADERS = [
    { name: "gemini", load: () => import("./providers/geminiProvider.js") },
    { name: "openai", load: () => import("./providers/openaiProvider.js") },
    { name: "aionlabs", load: () => import("./providers/aionlabs.js") },
    { name: "cohere", load: () => import("./providers/cohere.js") },
    { name: "ollamacloud", load: () => import("./providers/ollamacloud.js") },
    { name: "openrouter", load: () => import("./providers/openrouter.js") },
    { name: "cloudflare", load: () => import("./providers/cloudflare.js") },
    { name: "huggingface", load: () => import("./providers/huggingfaceProvider.js") },
];

let chainPromise = null;

async function buildChain() {
    const loaded = [];
    for (const { name, load } of LOADERS) {
        try {
            const mod = await load();
            loaded.push(mod.default);
        } catch (err) {
            logger.warn(`[ai-failover] provider "${name}" unavailable: ${err.message}`);
        }
    }

    // Apply AI_PROVIDER_ORDER, appending providers not mentioned in it.
    const byName = new Map(loaded.map((p) => [p.name, p]));
    const orderEnv = (process.env.AI_PROVIDER_ORDER || "")
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);

    const ordered = [];
    const seen = new Set();
    for (const name of orderEnv) {
        const p = byName.get(name);
        if (p && !seen.has(name)) {
            ordered.push(p);
            seen.add(name);
        }
    }
    for (const p of loaded) {
        if (!seen.has(p.name)) ordered.push(p);
    }
    return ordered;
}

/**
 * @returns {Promise<Array<object>>} Ordered provider adapters
 */
export function getProviders() {
    if (!chainPromise) chainPromise = buildChain();
    return chainPromise;
}

/** Names + configuration state (for /ai-status, never exposes keys). */
export async function describeProviders() {
    const providers = await getProviders();
    return providers.map((p) => ({
        name: p.name,
        capabilities: p.capabilities,
        configured: p.isConfigured(),
    }));
}

/** Test helper. */
export function resetRegistry() {
    chainPromise = null;
}

export default { getProviders, describeProviders, resetRegistry };
