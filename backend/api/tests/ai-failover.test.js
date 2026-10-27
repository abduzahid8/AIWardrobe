/**
 * Unit tests for the AI failover architecture:
 *   - circuitBreaker.js  (states, thresholds, exponential cooldown)
 *   - failover.js        (rotation cycling, failover, validation, timeouts)
 *   - jsonUtils.js       (JSON extraction contracts)
 *
 * Providers are injected as plain stubs — no network, no vendor SDKs.
 */
import {
    CircuitBreaker,
    getBreaker,
    resetBreakers,
} from "../services/ai/circuitBreaker.js";
import {
    execute,
    AllProvidersFailedError,
    NoProvidersConfiguredError,
    resetRotation,
} from "../services/ai/failover.js";
import { extractJSON, parseNumberArray } from "../services/ai/jsonUtils.js";

const silentLogger = { info: () => {}, warn: () => {}, error: () => {} };

// ── helpers ──────────────────────────────────────────

/** Build a provider stub. fail: Error | 'timeout' | null */
function makeProvider(name, { fail = null, result = { text: "ok" }, delayMs = 0 } = {}) {
    let calls = 0;
    return {
        name,
        calls: () => calls,
        capabilities: ["text"],
        isConfigured: () => true,
        run: async () => {
            calls += 1;
            if (delayMs) await new Promise((r) => setTimeout(r, delayMs));
            if (fail === "timeout") {
                await new Promise(() => {}); // never settles — engine timeout must reject first
            }
            if (fail) throw fail;
            return result;
        },
    };
}

// ── CircuitBreaker ───────────────────────────────────

describe("CircuitBreaker", () => {
    beforeEach(() => resetBreakers());

    test("stays closed below the failure threshold", () => {
        const b = new CircuitBreaker("p", { failureThreshold: 3 });
        b.recordFailure(new Error("x"));
        b.recordFailure(new Error("x"));
        expect(b.state(1000)).toBe("closed");
        expect(b.canAttempt(1000)).toBe(true);
    });

    test("opens after threshold consecutive failures", () => {
        const b = new CircuitBreaker("p", { failureThreshold: 3, baseCooldownMs: 1000 });
        const t0 = 1_000_000;
        b.recordFailure(new Error("x"), t0);
        b.recordFailure(new Error("x"), t0);
        b.recordFailure(new Error("x"), t0);
        expect(b.state(t0)).toBe("open");
        expect(b.canAttempt(t0)).toBe(false);
        expect(b.retryInMs(t0)).toBe(1000);
    });

    test("half-opens after cooldown and closes on success", () => {
        const b = new CircuitBreaker("p", { failureThreshold: 1, baseCooldownMs: 1000 });
        const t0 = 1_000_000;
        b.recordFailure(new Error("boom"), t0);
        expect(b.canAttempt(t0)).toBe(false);
        expect(b.canAttempt(t0 + 1001)).toBe(true); // half-open probe allowed
        b.recordSuccess(t0 + 1002);
        expect(b.state(t0 + 1002)).toBe("closed");
        expect(b.consecutiveFailures).toBe(0);
    });

    test("uses exponential cooldown capped at max", () => {
        const b = new CircuitBreaker("p", {
            failureThreshold: 1,
            baseCooldownMs: 1000,
            maxCooldownMs: 4000,
        });
        const t0 = 1_000_000;
        b.recordFailure(new Error("x"), t0);            // 1s
        b.recordFailure(new Error("x"), t0 + 1_001);    // 2s
        b.recordFailure(new Error("x"), t0 + 3_003);    // 4s
        b.recordFailure(new Error("x"), t0 + 7_007);    // capped at 4s
        expect(b.retryInMs(t0 + 7_007)).toBe(4000);
    });

    test("registry returns the same instance per name", () => {
        expect(getBreaker("shared")).toBe(getBreaker("shared"));
    });
});

// ── failover.execute ─────────────────────────────────

describe("failover.execute", () => {
    beforeEach(() => {
        resetBreakers();
        resetRotation();
    });

    test("returns the first healthy provider's result", async () => {
        const a = makeProvider("a");
        const b = makeProvider("b");
        const out = await execute("text", {}, { providers: [a, b], logger: silentLogger });
        expect(out.result.text).toBe("ok");
        expect(out.provider).toBe("a");
        expect(a.calls()).toBe(1);
        expect(b.calls()).toBe(0);
    });

    test("cycles to the next provider on failure", async () => {
        const dead = makeProvider("dead", { fail: new Error("http_500") });
        const alive = makeProvider("alive", { result: { text: "rescued" } });
        const out = await execute(
            "text", {},
            { providers: [dead, alive], logger: silentLogger }
        );
        expect(out.provider).toBe("alive");
        expect(out.result.text).toBe("rescued");
        expect(out.attempts).toEqual([
            { provider: "dead", ok: false, error: "http_500" },
            { provider: "alive", ok: true },
        ]);
    });

    test("rotates the start provider across requests (round-robin)", async () => {
        const a = makeProvider("a");
        const b = makeProvider("b");
        const providers = [a, b];

        const first = await execute("text", {}, { providers, logger: silentLogger });
        const second = await execute("text", {}, { providers, logger: silentLogger });
        expect(first.provider).toBe("a");
        expect(second.provider).toBe("b");
    });

    test("skips providers with open breakers immediately", async () => {
        const flaky = makeProvider("flaky", { fail: new Error("http_429") });
        const solid = makeProvider("solid");
        const providers = [flaky, solid];

        // Deterministic order [flaky, solid] — flaky fails 3×, breaker opens.
        for (let i = 0; i < 3; i++) {
            resetRotation();
            await execute("text", {}, { providers, logger: silentLogger });
        }
        expect(flaky.calls()).toBe(3);

        // Next request would start at flaky again, but its breaker is open
        // so the engine must skip it without spending an attempt.
        resetRotation();
        const out = await execute("text", {}, { providers, logger: silentLogger });
        expect(out.provider).toBe("solid");
        expect(flaky.calls()).toBe(3);
    });

    test("treats invalid JSON output as a provider failure", async () => {
        const sloppy = makeProvider("sloppy", { result: { text: "I cannot do that" } });
        const precise = makeProvider("precise", { result: { text: "[95, 70, 85, 60]" } });

        const out = await execute(
            "text", {},
            {
                providers: [sloppy, precise],
                logger: silentLogger,
                validate: (r) => parseNumberArray(r.text, 4),
            }
        );
        expect(out.provider).toBe("precise");
        expect(out.result).toEqual([95, 70, 85, 60]);
        expect(sloppy.calls()).toBe(1);
    });

    test("enforces the per-attempt timeout and cycles onward", async () => {
        const slow = makeProvider("slow", { fail: "timeout" });
        const fast = makeProvider("fast", { result: { text: "quick" } });

        const out = await execute(
            "text", {},
            { providers: [slow, fast], logger: silentLogger, timeoutMs: 50 }
        );
        expect(out.provider).toBe("fast");
        expect(out.attempts[0]).toEqual({ provider: "slow", ok: false, error: "timeout" });
    }, 5000);

    test("throws AllProvidersFailedError with an attempt trail", async () => {
        const a = makeProvider("a", { fail: new Error("http_401") });
        const b = makeProvider("b", { fail: new Error("empty_response") });
        await expect(
            execute("text", {}, { providers: [a, b], logger: silentLogger })
        ).rejects.toBeInstanceOf(AllProvidersFailedError);
    });

    test("throws NoProvidersConfiguredError when nothing is eligible", async () => {
        const off = { name: "off", capabilities: ["text"], isConfigured: () => false, run: async () => {} };
        await expect(
            execute("text", {}, { providers: [off], logger: silentLogger })
        ).rejects.toBeInstanceOf(NoProvidersConfiguredError);
    });

    test("only providers with the requested capability are used", async () => {
        const visionOnly = { name: "visionOnly", capabilities: ["vision"], isConfigured: () => true, run: async () => ({ text: "x" }) };
        await expect(
            execute("text", {}, { providers: [visionOnly], logger: silentLogger })
        ).rejects.toBeInstanceOf(NoProvidersConfiguredError);
    });

    test("probes the breaker closest to recovery when all are open", async () => {
        const a = makeProvider("a", { fail: new Error("x") });
        const b = makeProvider("b", { fail: new Error("x") });
        const providers = [a, b];

        // Both providers fail on every request → both breakers hit the
        // threshold together after 3 attempts.
        for (let i = 0; i < 3; i++) {
            resetRotation();
            await expect(
                execute("text", {}, { providers, logger: silentLogger })
            ).rejects.toBeInstanceOf(AllProvidersFailedError);
        }
        expect(a.calls()).toBe(3);
        expect(b.calls()).toBe(3);

        // Both open — the probe pass must attempt exactly ONE provider
        // (the one closest to recovery), not burn attempts on both.
        const before = a.calls() + b.calls();
        await expect(
            execute("text", {}, { providers, logger: silentLogger })
        ).rejects.toBeInstanceOf(AllProvidersFailedError);
        expect(a.calls() + b.calls()).toBe(before + 1);
    });
});

// ── jsonUtils ────────────────────────────────────────

describe("jsonUtils", () => {
    test("extracts bare JSON objects", () => {
        expect(extractJSON('{"category":"Shirt"}')).toEqual({ category: "Shirt" });
    });

    test("extracts JSON from prose and markdown fences", () => {
        const text = 'Sure! Here is the analysis:\n```json\n{"category":"Dress","style":"casual"}\n```';
        expect(extractJSON(text)).toEqual({ category: "Dress", style: "casual" });
    });

    test("prefers arrays when both appear", () => {
        expect(extractJSON('scores: [1, 2, 3] done')).toEqual([1, 2, 3]);
    });

    test("throws on JSON-less text", () => {
        expect(() => extractJSON("no structure here")).toThrow("no_json_found");
    });

    test("parseNumberArray enforces length and types", () => {
        expect(parseNumberArray("[10, 20, 30]", 3)).toEqual([10, 20, 30]);
        expect(() => parseNumberArray("[10, 20]", 3)).toThrow();
        expect(() => parseNumberArray('{"a":1}', 3)).toThrow();
    });
});
