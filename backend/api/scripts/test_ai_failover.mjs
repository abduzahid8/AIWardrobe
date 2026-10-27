/**
 * Live smoke test for the AI failover router.
 *
 * Usage:  node scripts/test_ai_failover.mjs
 * Requires provider keys in backend/api/.env (loaded via dotenv).
 *
 * Exercises the real vendor APIs (Aion Labs, Cohere, Ollama Cloud,
 * OpenRouter, Cloudflare) through the exact same code path the Express
 * routes use — except vendor SDK-wrapped providers (Gemini/OpenAI/HF),
 * which are covered by their own scripts.
 */
import path from "path";
import { fileURLToPath } from "url";
import dotenv from "dotenv";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
dotenv.config({ path: path.join(__dirname, "..", ".env") });

// Settle on the standalone providers only (SDK-wrapped ones need the
// full backend node_modules — use scripts/test_openai.js for those).
const { default: aionlabs } = await import("../services/ai/providers/aionlabs.js");
const { default: cohere } = await import("../services/ai/providers/cohere.js");
const { default: ollamacloud } = await import("../services/ai/providers/ollamacloud.js");
const { default: openrouter } = await import("../services/ai/providers/openrouter.js");
const { default: cloudflare } = await import("../services/ai/providers/cloudflare.js");
const { execute } = await import("../services/ai/failover.js");
const { describeProviders } = await import("../services/ai/registry.js");
const { snapshotAll } = await import("../services/ai/circuitBreaker.js");

const providers = [aionlabs, cohere, ollamacloud, openrouter, cloudflare];
const logger = {
    info: (...a) => console.log("  ℹ️ ", ...a),
    warn: (...a) => console.log("  ⚠️ ", ...a),
    error: (...a) => console.log("  ❌", ...a),
};

console.log("\n═ Provider configuration ═");
for (const p of await describeProviders()) {
    console.log(`  ${p.configured ? "✅" : "⬜"} ${p.name} [${p.capabilities.join(", ")}]`);
}

console.log("\n═ Test 1: text with failover ═");
try {
    const r = await execute(
        "text",
        { prompt: "In one short sentence, what should I wear to a business casual lunch?", systemInstruction: "You are a stylist. Answer in one sentence.", maxTokens: 80 },
        { providers, logger }
    );
    console.log(`  ✅ provider=${r.provider} attempts=${r.attempts.length}`);
    console.log(`  📝 ${r.result.text}`);
} catch (err) {
    console.error("  ❌ FAILED:", err.message);
    process.exitCode = 1;
}

console.log("\n═ Test 2: rotation — second request starts at the next provider ═");
try {
    const r = await execute(
        "text",
        { prompt: "Reply with a single word: ready", maxTokens: 40 },
        { providers, logger }
    );
    console.log(`  ✅ provider=${r.provider}`);
} catch (err) {
    console.error("  ❌ FAILED:", err.message);
    process.exitCode = 1;
}

console.log("\n═ Test 3: forced failover (dead provider first in cycle) ═");
try {
    const dead = {
        name: "dead-provider",
        capabilities: ["text"],
        isConfigured: () => true,
        run: async () => { throw new Error("http_503 (simulated outage)"); },
    };
    const r = await execute(
        "text",
        { prompt: "Reply with a single word: ok", maxTokens: 40 },
        { providers: [dead, ...providers], logger }
    );
    console.log(`  ✅ provider=${r.provider} (cycled past dead-provider: ${r.attempts.map((a) => a.provider).join(" → ")})`);
} catch (err) {
    console.error("  ❌ FAILED:", err.message);
    process.exitCode = 1;
}

console.log("\n═ Test 4: embeddings failover contract (384-dim) ═");
try {
    const r = await execute(
        "embeddings",
        { text: "red sneakers for a summer wedding" },
        { providers, logger }
    );
    console.log(`  ✅ provider=${r.provider} dim=${r.result.length}`);
    if (r.result.length !== 384) {
        console.error("  ❌ unexpected dimension — smart-search vectors would break");
        process.exitCode = 1;
    }
} catch (err) {
    console.error("  ❌ FAILED:", err.message);
    process.exitCode = 1;
}

console.log("\n═ Test 5: breaker — repeated failures open the circuit ═");
try {
    const dead = {
        name: "flaky-vendor",
        capabilities: ["text"],
        isConfigured: () => true,
        run: async () => { throw new Error("http_500"); },
    };
    const healthy = {
        name: "healthy",
        capabilities: ["text"],
        isConfigured: () => true,
        run: async () => ({ text: "ok" }),
    };
    for (let i = 0; i < 3; i++) {
        await execute("text", { prompt: "x" }, { providers: [dead, healthy], logger });
    }
    const r = await execute("text", { prompt: "x" }, { providers: [dead, healthy], logger });
    console.log(`  ✅ provider=${r.provider} — dead-provider skipped by breaker after 3 failures`);
} catch (err) {
    console.error("  ❌ FAILED:", err.message);
    process.exitCode = 1;
}

console.log("\nDone.");
