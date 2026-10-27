# AI Failover Architecture

> One contract, many providers. If a provider stops working, requests
> automatically cycle to the next one — no user-visible errors, no manual
> intervention, no stuck queues.

## Overview

```
                ┌───────────────────────────────────────────────┐
                │  Express routes (existing API surface)        │
                │  /api/ai-chat        /api/generate-outfits    │
                │  /api/openai/analyze-clothing                 │
                │  /api/gemini/*       /api/smart-search        │
                │  /api/process-upload GET /ai-status           │
                └───────────────────────┬───────────────────────┘
                                        │
                ┌───────────────────────▼───────────────────────┐
                │  services/ai/index.js (facade)                │
                │  generateText / chatWithHistory / generateJSON│
                │  analyzeImage / analyzeClothingImage          │
                │  generateEmbedding / status                   │
                └───────────────────────┬───────────────────────┘
                                        │
                ┌───────────────────────▼───────────────────────┐
                │  failover.execute(capability)                 │
                │  • round-robin rotation (cycle start index)   │
                │  • per-attempt timeout                        │
                │  • validate() → bad output = provider failure │
                └───────┬───────┬───────┬───────┬───────┬───────┘
                        ▼       ▼       ▼       ▼       ▼
              ┌─────────────────────────────────────────────┐
              │  Circuit breakers (one per provider)        │
              │  closed → open(3 fails) → half-open probe   │
              └─────────────────────────────────────────────┘
                        ▼       ▼       ▼       ▼       ▼
   gemini  openai  aionlabs  cohere  ollamacloud  openrouter  cloudflare  huggingface
   (text+  (text+  (text)    (text+  (text)       (text+      (text)      (text+
   vision) vision)           embeds)               vision)                 embeds)
```

## Files

| Path | Role |
|---|---|
| `backend/api/services/ai/index.js` | Public facade — routes call **only** this |
| `backend/api/services/ai/failover.js` | Engine: rotation + failover + timeouts + validation |
| `backend/api/services/ai/circuitBreaker.js` | Per-provider breaker state |
| `backend/api/services/ai/registry.js` | Builds the chain from env, fault-isolated loading |
| `backend/api/services/ai/jsonUtils.js` | JSON extraction / output contracts |
| `backend/api/services/ai/providers/*` | One adapter per vendor |
| `backend/api/tests/ai-failover.test.js` | 20 unit tests (rotation, breakers, failover) |
| `backend/api/scripts/test_ai_failover.mjs` | Live smoke test against real vendor APIs |

## How a request flows

1. **Capability resolution** — the route asks for `text`, `vision` or
   `embeddings`. Only providers that are *configured* (key present) *and*
   *support the capability* are eligible.
2. **Rotation (cycling)** — each request starts from the next provider in
   the cycle (`rotation++ % n`). Load spreads across providers, and when one
   dies, subsequent requests naturally begin with the next healthy one.
3. **Breaker check** — providers whose breaker is OPEN are skipped with zero
   latency cost. If *all* breakers are open, the one closest to recovery is
   probed (half-open) so recovered providers rejoin quickly.
4. **Attempt** — one call with a per-attempt timeout (30s text / 45s vision /
   20s embeddings).
5. **Validation** — for structured tasks the result must pass a contract
   (e.g. valid JSON clothing schema, exact-length score array). A malformed
   output counts as a provider failure → cycle to the next provider.
6. **Success** — the breaker closes; the response carries `provider` so
   clients/logs show who answered.
7. **Total failure** — `AllProvidersFailedError` with a full attempt trail;
   routes return 503 (or fall back to deterministic logic, e.g. keyword
   outfit matching, HF CLIP classification).

## Circuit breaker

- Opens after **3 consecutive failures** (`AI_BREAKER_THRESHOLD`).
- Cooldown starts at **15s** and doubles per extra failure, capped at
  **10 min** (exponential backoff — a flapping provider stops receiving
  the majority of traffic).
- After the cooldown the next request is a **probe**: success closes the
  breaker, failure re-opens it with a longer cooldown.
- State is in-process and time-based (no timers, no locks; safe under
  concurrency).

## Provider adapters

Every adapter implements the same interface:

```js
{
  name: "cohere",
  capabilities: ["text", "embeddings"],
  isConfigured: () => Boolean(API_KEY),
  run: async (capability, args, timeoutMs) => ({ text } | number[]),
}
```

- **openaiCompat factory** — one factory powers Aion Labs, Ollama Cloud and
  OpenRouter (all OpenAI-compatible). It also handles reasoning models
  (Aion, gpt-oss): extra token headroom (`tokenBuffer`) and empty
  `content` with populated `reasoning` is treated as a failure.
- **Wrappers** — Gemini/OpenAI/HuggingFace adapters wrap the *existing*
  services (`services/gemini.js`, `services/openai.js`,
  `services/huggingface.js`) instead of reimplementing vendor calls.
- **Fault isolation** — the registry loads providers lazily via dynamic
  import; a missing vendor SDK only disables that provider, never the API.

## Embeddings contract (dimension-locked)

`smart-search` stores 384-dim vectors (HF MiniLM). Cohere's
`embed-english-light-v3.0` is also 384-dim, so it is a drop-in fallback.
The adapter hard-fails on any other dimension (`embedding_dim_mismatch`)
rather than silently corrupting the similarity index.

## Adding a new provider

1. Create `providers/<name>.js` (use the `openaiCompat` factory if the
   vendor speaks OpenAI's contract).
2. Register it in `registry.js` `LOADERS`.
3. Add `<NAME>_API_KEY` (+ optional model overrides) to:
   - `backend/api/.env` (local)
   - `backend/api/.env.example` (documentation)
   - Render env vars / `render.yaml` (`sync: false`)
4. Optionally set its position via `AI_PROVIDER_ORDER`.

No route changes required — the chain picks it up automatically.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `AI_PROVIDER_ORDER` | gemini,openai,aionlabs,cohere,ollamacloud,openrouter,cloudflare,huggingface | Chain order |
| `AI_TIMEOUT_TEXT_MS` / `_VISION_` / `_EMBEDDINGS_` | 30000 / 45000 / 20000 | Per-attempt timeouts |
| `AI_BREAKER_THRESHOLD` | 3 | Consecutive failures before open |
| `AI_BREAKER_COOLDOWN_MS` | 15000 | Base cooldown (doubles per failure) |
| `AI_BREAKER_MAX_COOLDOWN_MS` | 600000 | Cooldown cap |
| `AIONLABS_API_KEY` / `AIONLABS_MODEL` | — / aion-labs/aion-3.5-mini | Aion Labs |
| `COHERE_API_KEY` / `COHERE_MODEL` / `COHERE_EMBED_MODEL` | — / command-r7b-12-2024 / embed-english-light-v3.0 | Cohere |
| `OLLAMA_API_KEY` / `OLLAMA_MODEL` | — / gpt-oss:20b | Ollama Cloud |
| `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` | — / google/gemini-2.5-flash | OpenRouter |
| `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` | — | Cloudflare Workers AI (both required) |

## Observability

- `GET /ai-status` — JSON snapshot of every provider: configured?,
  capabilities, breaker state, consecutive failures, retry timer,
  last error, success/failure totals. No secrets.
- Every attempt is logged: `[ai-failover] <provider> failed (<reason>) — cycling to next provider`.
- Every response includes the `provider` field that actually served it.

## Supabase edge functions

`ai-chat` mirrors the same pattern (OpenAI → Gemini → NVIDIA → Aion Labs →
Cohere → Ollama Cloud → OpenRouter) with per-provider fetch helpers.
New provider keys are set via `supabase secrets set`:

```sh
supabase secrets set AIONLABS_API_KEY=... COHERE_API_KEY=... \
  OLLAMA_API_KEY=... OPENROUTER_API_KEY=...
```

`generate-outfits` intentionally remains NVIDIA-only with a deterministic
fallback (documented in that function) — outfit generation needs strict
JSON structure, and the deterministic builder is the designed fallback.

## Testing

```sh
cd backend/api
NODE_OPTIONS="--experimental-vm-modules" \
  node ../../node_modules/.bin/jest --config jest.config.cjs   # 20 unit tests
node scripts/test_ai_failover.mjs                              # live vendor smoke test
```
