/**
 * Circuit Breaker — per-provider failure isolation for the AI failover router.
 *
 * States:
 *   CLOSED   → provider is healthy, all requests pass through.
 *   OPEN     → provider failed too many times in a row; requests are skipped
 *              instantly (zero latency cost) until the cooldown expires.
 *   HALF-OPEN → cooldown expired; the next request is a probe. Success closes
 *              the breaker, failure re-opens it with a longer (exponential)
 *              cooldown.
 *
 * The breaker is time-based, so no timers are needed — state advances lazily
 * as `canAttempt()` is called. Safe to share across concurrent requests.
 */

const DEFAULTS = {
    failureThreshold: Number(process.env.AI_BREAKER_THRESHOLD || 3),
    baseCooldownMs: Number(process.env.AI_BREAKER_COOLDOWN_MS || 15_000),
    maxCooldownMs: Number(process.env.AI_BREAKER_MAX_COOLDOWN_MS || 10 * 60_000),
};

class CircuitBreaker {
    /**
     * @param {string} name - Provider name (for logs)
     * @param {{failureThreshold?: number, baseCooldownMs?: number, maxCooldownMs?: number}} [options]
     */
    constructor(name, options = {}) {
        this.name = name;
        this.failureThreshold = options.failureThreshold ?? DEFAULTS.failureThreshold;
        this.baseCooldownMs = options.baseCooldownMs ?? DEFAULTS.baseCooldownMs;
        this.maxCooldownMs = options.maxCooldownMs ?? DEFAULTS.maxCooldownMs;

        this.consecutiveFailures = 0;
        this.openUntil = 0;          // epoch ms; 0 = closed
        this.lastError = null;
        this.lastSuccessAt = 0;
        this.totalFailures = 0;
        this.totalSuccesses = 0;
    }

    /**
     * Whether a request may go through right now.
     * @param {number} [now] - epoch ms (injectable for tests)
     * @returns {boolean}
     */
    canAttempt(now = Date.now()) {
        if (this.openUntil <= now) return true; // closed or half-open (probe)
        return false;                            // open — skip instantly
    }

    /** @returns {'closed'|'open'|'half-open'} */
    state(now = Date.now()) {
        if (this.openUntil === 0) return "closed";
        if (this.openUntil > now) return "open";
        return "half-open";
    }

    recordSuccess(now = Date.now()) {
        this.consecutiveFailures = 0;
        this.openUntil = 0;
        this.lastSuccessAt = now;
        this.totalSuccesses += 1;
    }

    recordFailure(error, now = Date.now()) {
        this.consecutiveFailures += 1;
        this.totalFailures += 1;
        this.lastError = error?.message || String(error || "unknown error");

        if (this.consecutiveFailures >= this.failureThreshold) {
            // Exponential cooldown: base × 2^(failures−threshold), capped.
            const over = this.consecutiveFailures - this.failureThreshold;
            const cooldown = Math.min(
                this.baseCooldownMs * 2 ** over,
                this.maxCooldownMs
            );
            this.openUntil = now + cooldown;
        }
    }

    /** Milliseconds until the breaker half-opens (0 if not open). */
    retryInMs(now = Date.now()) {
        return this.openUntil > now ? this.openUntil - now : 0;
    }

    /** Human-safe snapshot for /ai-status (never exposes keys). */
    snapshot(now = Date.now()) {
        return {
            state: this.state(now),
            consecutiveFailures: this.consecutiveFailures,
            openUntil: this.openUntil > now ? new Date(this.openUntil).toISOString() : null,
            retryInMs: this.retryInMs(now),
            lastError: this.lastError,
            lastSuccessAt: this.lastSuccessAt ? new Date(this.lastSuccessAt).toISOString() : null,
            totalSuccesses: this.totalSuccesses,
            totalFailures: this.totalFailures,
        };
    }
}

/**
 * Shared breaker registry — one breaker per provider name for the process.
 * @type {Map<string, CircuitBreaker>}
 */
const breakers = new Map();

/**
 * Get (or lazily create) the breaker for a provider.
 * @param {string} name
 * @param {object} [options]
 * @returns {CircuitBreaker}
 */
export function getBreaker(name, options) {
    let breaker = breakers.get(name);
    if (!breaker) {
        breaker = new CircuitBreaker(name, options);
        breakers.set(name, breaker);
    }
    return breaker;
}

/** Snapshot of every breaker (for the /ai-status endpoint). */
export function snapshotAll() {
    const out = {};
    for (const [name, breaker] of breakers) out[name] = breaker.snapshot();
    return out;
}

/** Test helper — wipe all breaker state. */
export function resetBreakers() {
    breakers.clear();
}

export { CircuitBreaker, DEFAULTS as BREAKER_DEFAULTS };
export default { CircuitBreaker, getBreaker, snapshotAll, resetBreakers };
