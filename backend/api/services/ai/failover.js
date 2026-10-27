/**
 * Failover Engine — the core of the multi-provider AI architecture.
 *
 * Given a capability ("text" | "vision" | "embeddings") and a list of provider
 * adapters, it:
 *
 *   1. CYCLES: keeps a rotating start index (round-robin), so requests are
 *      spread across providers and, when one dies, each subsequent request
 *      naturally starts from the next healthy provider.
 *   2. FAILOVER: on any failure (timeout, 4xx/5xx, empty response, invalid
 *      JSON, validation error) it instantly moves to the next provider in
 *      the cycle until one succeeds or all fail.
 *   3. SHORT-CIRCUITS: every provider has a circuit breaker — once a provider
 *      fails repeatedly, it is skipped without spending the attempt timeout
 *      on it (fast recovery after cooldown, probed half-open).
 *
 * The engine is dependency-injected (providers + logger come from the caller)
 * so it stays trivially unit-testable and free of vendor imports.
 */
import { getBreaker } from "./circuitBreaker.js";

const TIMEOUTS = {
    text: Number(process.env.AI_TIMEOUT_TEXT_MS || 30_000),
    vision: Number(process.env.AI_TIMEOUT_VISION_MS || 45_000),
    embeddings: Number(process.env.AI_TIMEOUT_EMBEDDINGS_MS || 20_000),
};

/** Round-robin rotation counter — shared across all capabilities. */
let rotation = 0;

export class AllProvidersFailedError extends Error {
    constructor(capability, attempts) {
        const summary = attempts
            .map((a) => `${a.provider}:${a.error}`)
            .join(" | ");
        super(`All AI providers failed for capability "${capability}" — ${summary}`);
        this.name = "AllProvidersFailedError";
        this.capability = capability;
        this.attempts = attempts;
    }
}

export class NoProvidersConfiguredError extends Error {
    constructor(capability) {
        super(`No AI providers configured for capability "${capability}"`);
        this.name = "NoProvidersConfiguredError";
        this.capability = capability;
    }
}

/**
 * Normalize a provider error into a short reason string.
 */
function errorReason(err) {
    if (!err) return "unknown";
    if (err.code === "ECONNABORTED" || err.name === "AbortError" || err.name === "TimeoutError") {
        return "timeout";
    }
    const status = err.response?.status;
    if (status) return `http_${status}`;
    return (err.message || String(err)).slice(0, 120);
}

/**
 * Run one capability through the provider cycle.
 *
 * @param {'text'|'vision'|'embeddings'} capability
 * @param {object} args - Capability args forwarded to the provider
 *   (e.g. { messages, maxTokens } for text, { imageBase64, prompt } for vision).
 * @param {object} opts
 * @param {Array<object>} opts.providers - Registered provider adapters
 * @param {(result:any)=>any} [opts.validate] - Post-process/validate the raw
 *   provider result; throwing marks the attempt as failed and cycles onward.
 *   Whatever it returns becomes `result` on the response object.
 * @param {object} [opts.logger] - Optional logger ({ info, warn, error })
 * @param {number} [opts.timeoutMs] - Per-attempt timeout override
 * @returns {Promise<{result:any, provider:string, attempts:Array, fromCacheRotation:number}>}
 * @throws {AllProvidersFailedError|NoProvidersConfiguredError}
 */
export async function execute(capability, args, opts) {
    const { providers, validate, logger = console } = opts;
    const timeoutMs = opts.timeoutMs || TIMEOUTS[capability] || TIMEOUTS.text;

    // Only providers that are configured AND support this capability.
    const eligible = providers.filter(
        (p) => p.capabilities.includes(capability) && p.isConfigured()
    );

    if (eligible.length === 0) throw new NoProvidersConfiguredError(capability);

    // ── Rotation: start each request from the next provider in the cycle ──
    const start = rotation++ % eligible.length;
    const ordered = [
        ...eligible.slice(start),
        ...eligible.slice(0, start),
    ];

    const attempts = [];

    // Pass 1 — skip providers whose breaker is OPEN (fast, no timeout spent).
    // Pass 2 — if every breaker was open, probe the least-recently-failed one
    //          anyway (half-open) so a recovered provider gets traffic back.
    let candidates = ordered.filter((p) => getBreaker(p.name).canAttempt());
    if (candidates.length === 0) {
        // All breakers open → allow the provider closest to recovery to probe.
        candidates = [...ordered].sort(
            (a, b) => getBreaker(a.name).retryInMs() - getBreaker(b.name).retryInMs()
        ).slice(0, 1);
        logger.warn?.(
            `[ai-failover] all providers have open breakers — probing ${candidates[0].name}`
        );
    }

    for (const provider of candidates) {
        const breaker = getBreaker(provider.name);

        let result;
        try {
            result = await withTimeout(
                provider.run(capability, args, timeoutMs),
                timeoutMs,
                `${provider.name} ${capability}`
            );
        } catch (err) {
            const reason = errorReason(err);
            breaker.recordFailure(err);
            attempts.push({ provider: provider.name, ok: false, error: reason });
            logger.warn?.(
                `[ai-failover] ${provider.name} failed (${reason}) — cycling to next provider`
            );
            continue;
        }

        // Validation (e.g. JSON contract) failure counts as a provider failure.
        if (validate) {
            try {
                result = validate(result);
            } catch (err) {
                const reason = `invalid_response: ${(err?.message || String(err)).slice(0, 100)}`;
                breaker.recordFailure(err);
                attempts.push({ provider: provider.name, ok: false, error: reason });
                logger.warn?.(
                    `[ai-failover] ${provider.name} returned unusable output (${reason}) — cycling`
                );
                continue;
            }
        }

        breaker.recordSuccess();
        attempts.push({ provider: provider.name, ok: true });
        return { result, provider: provider.name, attempts };
    }

    throw new AllProvidersFailedError(capability, attempts);
}

/**
 * Reject if the promise does not settle within `ms`.
 */
function withTimeout(promise, ms, label) {
    return new Promise((resolve, reject) => {
        const timer = setTimeout(
            () => reject(Object.assign(new Error(`${label} timed out after ${ms}ms`), { code: "ECONNABORTED" })),
            ms
        );
        promise.then(
            (v) => { clearTimeout(timer); resolve(v); },
            (e) => { clearTimeout(timer); reject(e); }
        );
    });
}

/** Test helper — reset the rotation counter. */
export function resetRotation() {
    rotation = 0;
}

export default { execute, AllProvidersFailedError, NoProvidersConfiguredError, resetRotation };
