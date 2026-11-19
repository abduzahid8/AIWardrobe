/**
 * Stylist 6 — Multi-Provider AI Router with Circuit Breaker, Adaptive Concurrency
 * Semaphore, Cross-Lab Model Selection, and Per-Call Telemetry.
 *
 * Replaces single-provider NVIDIA SPOF and process-local `inflight >= 2` brick walls
 * with a scalable, deadline-aware router supporting:
 *   - NVIDIA NIM (Nemotron 3.5 Super VL 120B, Nemotron 3.5 Lightning 30B,
 *     Kimi K3.1 Pro, Gemma 4 31B)
 *   - OpenRouter (Gemini 2.5 Flash fast vision+text + configurable fallbacks)
 *
 * Security & correctness invariants:
 *   - `redirect: 'error'` on all outbound provider calls.
 *   - Bounded streaming body reads (never trusts hostile Content-Length).
 *   - Never leaks API keys, JWTs, or raw image bytes in telemetry or errors.
 *   - Preserves Kimi K3.1 Pro fixed nucleus sampler constraint (`top_p` omitted).
 *   - Never fabricates provider outcomes; returns honest error diagnostics.
 */

export const AI_PROVIDER_ROUTER_VERSION = 's6-ai-provider-router-v1';

export const PROVIDER_ENDPOINTS = Object.freeze({
  nvidia: 'https://integrate.api.nvidia.com/v1/chat/completions',
  openrouter: 'https://openrouter.ai/api/v1/chat/completions',
});

export const DEFAULT_MAX_CONCURRENCY = (() => {
  const parsed = Number(process.env.STYLIST6_AI_MAX_CONCURRENCY);
  return Number.isInteger(parsed) && parsed >= 1 && parsed <= 128 ? parsed : 24;
})();

export const DEFAULT_OPENROUTER_VISION_MODEL = 'google/gemini-2.5-flash';
export const DEFAULT_NVIDIA_VISION_MODEL = 'nvidia/nemotron-3.5-super-vl-120b-a12b';
export const DEFAULT_NVIDIA_TEXT_MODEL = 'nvidia/nemotron-3.5-lightning-30b-a3b';
export const DEFAULT_KIMI_VISION_MODEL = 'moonshotai/kimi-k3.1-pro';
export const DEFAULT_GEMMA_VISION_MODEL = 'google/gemma-4-31b-it';

const DEFAULT_MAX_RESPONSE_BYTES = 128 * 1024;
const MAX_ALLOWED_RESPONSE_BYTES = 512 * 1024;
const LATENCY_WINDOW_SIZE = 100;

/**
 * Canonical model capabilities and provider quirks.
 * Cost estimates are approximate USD per 1K tokens for operational budgeting
 * (never presented as billing invoices).
 */
export const MODEL_REGISTRY = Object.freeze({
  'nvidia/nemotron-3.5-super-vl-120b-a12b': Object.freeze({
    model: 'nvidia/nemotron-3.5-super-vl-120b-a12b',
    provider: 'nvidia',
    family: 'nemotron',
    lab: 'nvidia',
    supportsVision: true,
    supportsText: true,
    supportsJsonSchema: true,
    supportsThinkingToggle: true,
    fixedTopP: false,
    costPer1kPromptUsd: 0.0006,
    costPer1kCompletionUsd: 0.0018,
    defaultTimeoutMs: 18_000,
    crossLabRank: 30,
  }),
  'nvidia/nemotron-3.5-lightning-30b-a3b': Object.freeze({
    model: 'nvidia/nemotron-3.5-lightning-30b-a3b',
    provider: 'nvidia',
    family: 'nemotron',
    lab: 'nvidia',
    supportsVision: false,
    supportsText: true,
    supportsJsonSchema: true,
    supportsThinkingToggle: true,
    fixedTopP: false,
    costPer1kPromptUsd: 0.00015,
    costPer1kCompletionUsd: 0.00045,
    defaultTimeoutMs: 8_000,
    crossLabRank: 35,
  }),
  'moonshotai/kimi-k3.1-pro': Object.freeze({
    model: 'moonshotai/kimi-k3.1-pro',
    provider: 'nvidia',
    family: 'kimi',
    lab: 'moonshot',
    supportsVision: true,
    supportsText: true,
    supportsJsonSchema: false,
    supportsThinkingToggle: true,
    // Verified live 2026-11-18 (commit caed85dd): NVIDIA NIM Kimi K3.1 Pro fixes
    // top_p=0.95 and returns HTTP 400 when a caller passes top_p=0.9.
    fixedTopP: true,
    costPer1kPromptUsd: 0.0006,
    costPer1kCompletionUsd: 0.002,
    defaultTimeoutMs: 14_000,
    crossLabRank: 15,
  }),
  'google/gemma-4-31b-it': Object.freeze({
    model: 'google/gemma-4-31b-it',
    provider: 'nvidia',
    family: 'gemma',
    lab: 'google',
    supportsVision: true,
    supportsText: true,
    supportsJsonSchema: false,
    supportsThinkingToggle: false,
    fixedTopP: false,
    costPer1kPromptUsd: 0.0002,
    costPer1kCompletionUsd: 0.0006,
    defaultTimeoutMs: 12_000,
    crossLabRank: 20,
  }),
  'google/gemini-2.5-flash': Object.freeze({
    model: 'google/gemini-2.5-flash',
    provider: 'openrouter',
    family: 'gemini',
    lab: 'google',
    supportsVision: true,
    supportsText: true,
    supportsJsonSchema: true,
    supportsThinkingToggle: false,
    fixedTopP: false,
    costPer1kPromptUsd: 0.00015,
    costPer1kCompletionUsd: 0.0006,
    defaultTimeoutMs: 8_000,
    crossLabRank: 10,
  }),
});

function inferFamilyAndLab(modelId, provider) {
  const lower = String(modelId || '').toLowerCase();
  if (lower.includes('nemotron')) return { family: 'nemotron', lab: 'nvidia' };
  if (lower.includes('kimi') || lower.startsWith('moonshotai/')) return { family: 'kimi', lab: 'moonshot' };
  if (lower.includes('gemini')) return { family: 'gemini', lab: 'google' };
  if (lower.includes('gemma')) return { family: 'gemma', lab: 'google' };
  if (lower.includes('claude') || lower.startsWith('anthropic/')) return { family: 'claude', lab: 'anthropic' };
  if (lower.includes('gpt-') || lower.startsWith('openai/')) return { family: 'gpt', lab: 'openai' };
  if (lower.includes('qwen')) return { family: 'qwen', lab: 'alibaba' };
  if (lower.includes('mistral')) return { family: 'mistral', lab: 'mistral' };
  if (lower.includes('llama') || lower.startsWith('meta/')) return { family: 'llama', lab: 'meta' };
  return { family: lower.split('/').pop() || 'unknown', lab: provider || 'unknown' };
}

/**
 * Resolve model metadata even for custom/fallback OpenRouter or NIM models.
 */
export function resolveModelSpec(modelId, providerHint = null) {
  const key = String(modelId || '').trim();
  const known = MODEL_REGISTRY[key];
  if (known) {
    if (providerHint && providerHint !== known.provider) {
      return { ...known, provider: providerHint };
    }
    return known;
  }
  const provider = providerHint
    || (key.startsWith('nvidia/') || key.startsWith('moonshotai/') ? 'nvidia' : 'openrouter');
  const { family, lab } = inferFamilyAndLab(key, provider);
  const fixedTopP = /kimi-k3/i.test(key);
  return {
    model: key,
    provider,
    family,
    lab,
    supportsVision: true,
    supportsText: true,
    supportsJsonSchema: provider === 'openrouter' && !/gemma|kimi/i.test(key),
    supportsThinkingToggle: provider === 'nvidia' && (family === 'nemotron' || family === 'kimi'),
    fixedTopP,
    costPer1kPromptUsd: null,
    costPer1kCompletionUsd: null,
    defaultTimeoutMs: 14_000,
    crossLabRank: 25,
  };
}

/**
 * Normalize an OpenAI-compatible chat/completions payload for a specific
 * model & provider while respecting known provider quirks (especially Kimi K3.1
 * Pro's fixed top_p=0.95 and OpenRouter's lack of NVIDIA chat_template_kwargs).
 */
export function normalizeProviderPayload(modelId, provider, payload = {}, { preserveRequestShape = false } = {}) {
  const spec = resolveModelSpec(modelId, provider);
  const out = { ...payload, model: spec.model, stream: false };

  // CRITICAL (commit caed85dd): Kimi K3.1 Pro rejects explicit top_p != 0.95
  // with HTTP 400. Always omit top_p for fixedTopP models.
  if (spec.fixedTopP) {
    delete out.top_p;
  }

  // Strip NVIDIA-only chat_template_kwargs when routing to OpenRouter or
  // non-thinking-toggle models (like Gemma 4 or Gemini 2.5 Flash).
  if (!spec.supportsThinkingToggle || spec.provider === 'openrouter') {
    delete out.chat_template_kwargs;
  } else if (out.chat_template_kwargs === undefined && !preserveRequestShape) {
    out.chat_template_kwargs = { enable_thinking: false };
  }

  // Strip bare '/no_think' system sentinel when targeting non-Nemotron models.
  if (spec.family !== 'nemotron' && Array.isArray(out.messages)) {
    const filtered = out.messages.filter(
      (msg) => !(msg?.role === 'system' && typeof msg?.content === 'string' && msg.content.trim() === '/no_think'),
    );
    if (filtered.length > 0) out.messages = filtered;
  }

  // Downgrade strict json_schema to json_object on models that reject json_schema.
  if (!preserveRequestShape && out.response_format?.type === 'json_schema' && !spec.supportsJsonSchema) {
    out.response_format = { type: 'json_object' };
  }

  return out;
}

/**
 * Adaptive concurrency semaphore with smooth deadline-aware FIFO queueing
 * and AIMD backoff on HTTP 429 rate limits.
 */
export function createAdaptiveSemaphore({
  maxConcurrency = DEFAULT_MAX_CONCURRENCY,
  minConcurrency = 2,
  maxQueueSize = null,
  recoverySuccessThreshold = 8,
  nowFn = () => Date.now(),
} = {}) {
  const hardMax = Math.max(1, Math.min(128, Math.floor(Number(maxConcurrency) || DEFAULT_MAX_CONCURRENCY)));
  const hardMin = Math.max(1, Math.min(hardMax, Math.floor(Number(minConcurrency) || 2)));
  const queueCap = Number.isInteger(maxQueueSize) && maxQueueSize >= 0 ? maxQueueSize : hardMax * 4;

  let effectiveLimit = hardMax;
  let inFlight = 0;
  let peakInFlight = 0;
  let totalAcquired = 0;
  let totalQueued = 0;
  let totalQueueTimeouts = 0;
  let totalQueueRejected = 0;
  let consecutiveSuccessSinceBackoff = 0;
  const waiters = [];

  function dispatchNext() {
    while (inFlight < effectiveLimit && waiters.length > 0) {
      const waiter = waiters.shift();
      if (waiter.settled) continue;
      waiter.settled = true;
      waiter.cleanup();
      inFlight++;
      totalAcquired++;
      if (inFlight > peakInFlight) peakInFlight = inFlight;
      waiter.resolve({
        queueMs: Math.max(0, nowFn() - waiter.enqueuedAt),
        release: makeRelease(),
      });
    }
  }

  function makeRelease() {
    let released = false;
    return function release() {
      if (released) return;
      released = true;
      inFlight = Math.max(0, inFlight - 1);
      dispatchNext();
    };
  }

  function acquire({ signal = null, timeoutMs = null } = {}) {
    if (signal?.aborted) {
      const err = new Error('aborted');
      err.code = 'aborted';
      return Promise.reject(err);
    }
    if (inFlight < effectiveLimit && waiters.length === 0) {
      inFlight++;
      totalAcquired++;
      if (inFlight > peakInFlight) peakInFlight = inFlight;
      return Promise.resolve({ queueMs: 0, release: makeRelease() });
    }
    if (waiters.length >= queueCap) {
      totalQueueRejected++;
      const err = new Error('concurrency_queue_full');
      err.code = 'concurrency_queue_full';
      return Promise.reject(err);
    }
    totalQueued++;
    return new Promise((resolve, reject) => {
      const waiter = {
        enqueuedAt: nowFn(),
        settled: false,
        resolve,
        reject,
        cleanup: () => {},
      };
      let timer = null;
      const removeFromQueue = () => {
        const idx = waiters.indexOf(waiter);
        if (idx >= 0) waiters.splice(idx, 1);
      };
      const onAbort = () => {
        if (waiter.settled) return;
        waiter.settled = true;
        waiter.cleanup();
        removeFromQueue();
        const err = new Error('aborted');
        err.code = 'aborted';
        reject(err);
      };
      const onTimeout = () => {
        if (waiter.settled) return;
        waiter.settled = true;
        waiter.cleanup();
        removeFromQueue();
        totalQueueTimeouts++;
        const err = new Error('queue_deadline');
        err.code = 'queue_deadline';
        reject(err);
      };
      waiter.cleanup = () => {
        if (timer !== null) clearTimeout(timer);
        if (signal) signal.removeEventListener('abort', onAbort);
      };
      if (signal) signal.addEventListener('abort', onAbort, { once: true });
      if (Number.isFinite(timeoutMs) && timeoutMs > 0) {
        timer = setTimeout(onTimeout, Math.max(1, Math.floor(timeoutMs)));
      }
      waiters.push(waiter);
    });
  }

  function onRateLimit() {
    consecutiveSuccessSinceBackoff = 0;
    effectiveLimit = Math.max(hardMin, Math.floor(effectiveLimit * 0.7));
  }

  function onSuccess() {
    if (effectiveLimit >= hardMax) return;
    consecutiveSuccessSinceBackoff++;
    if (consecutiveSuccessSinceBackoff >= recoverySuccessThreshold) {
      consecutiveSuccessSinceBackoff = 0;
      effectiveLimit = Math.min(hardMax, effectiveLimit + 1);
      dispatchNext();
    }
  }

  function stats() {
    return {
      maxConcurrency: hardMax,
      effectiveLimit,
      inFlight,
      queued: waiters.length,
      peakInFlight,
      totalAcquired,
      totalQueued,
      totalQueueTimeouts,
      totalQueueRejected,
    };
  }

  return { acquire, onRateLimit, onSuccess, stats };
}

/**
 * Per-provider circuit breaker (`closed` -> `open` -> `half_open` -> `closed`).
 */
export function createCircuitBreaker({
  failureThreshold = 3,
  cooldownMs = 30_000,
  nowFn = () => Date.now(),
} = {}) {
  const threshold = Math.max(1, Math.floor(Number(failureThreshold) || 3));
  const cooldown = Math.max(100, Math.floor(Number(cooldownMs) || 30_000));

  let state = 'closed';
  let consecutiveFailures = 0;
  let openedAt = 0;
  let halfOpenProbing = false;
  let lastError = null;
  let lastFailureAt = null;
  let lastSuccessAt = null;

  function evaluateState() {
    if (state === 'open' && nowFn() - openedAt >= cooldown) {
      state = 'half_open';
      halfOpenProbing = false;
    }
    return state;
  }

  function canExecute() {
    const current = evaluateState();
    if (current === 'closed') return { allowed: true, state: 'closed' };
    if (current === 'half_open') {
      if (halfOpenProbing) return { allowed: false, state: 'half_open' };
      halfOpenProbing = true;
      return { allowed: true, state: 'half_open', probe: true };
    }
    return {
      allowed: false,
      state: 'open',
      cooldownRemainingMs: Math.max(0, cooldown - (nowFn() - openedAt)),
    };
  }

  function recordSuccess() {
    state = 'closed';
    consecutiveFailures = 0;
    halfOpenProbing = false;
    openedAt = 0;
    lastSuccessAt = nowFn();
  }

  function recordFailure(errorCode = 'provider_error', { tripImmediately = false } = {}) {
    const now = nowFn();
    lastError = String(errorCode || 'provider_error').slice(0, 80);
    lastFailureAt = now;
    if (state === 'half_open') {
      state = 'open';
      openedAt = now;
      halfOpenProbing = false;
      consecutiveFailures = Math.max(consecutiveFailures + 1, threshold);
      return;
    }
    consecutiveFailures++;
    if (tripImmediately || consecutiveFailures >= threshold) {
      state = 'open';
      openedAt = now;
      halfOpenProbing = false;
    }
  }

  function releaseProbe() {
    if (state === 'half_open') halfOpenProbing = false;
  }

  function snapshot() {
    const current = evaluateState();
    return {
      state: current,
      consecutiveFailures,
      cooldownRemainingMs: current === 'open' ? Math.max(0, cooldown - (nowFn() - openedAt)) : 0,
      lastError,
      lastSuccessAt,
      lastFailureAt,
    };
  }

  function reset() {
    state = 'closed';
    consecutiveFailures = 0;
    openedAt = 0;
    halfOpenProbing = false;
    lastError = null;
    lastFailureAt = null;
    lastSuccessAt = null;
  }

  return { canExecute, recordSuccess, recordFailure, releaseProbe, snapshot, reset };
}

function parseFallbackModelList(raw) {
  if (Array.isArray(raw)) {
    return raw.map((x) => String(x || '').trim()).filter(Boolean);
  }
  if (typeof raw === 'string' && raw.trim()) {
    return raw.split(',').map((x) => x.trim()).filter(Boolean);
  }
  return [];
}

function computeP95(samples) {
  if (!samples.length) return 0;
  const sorted = [...samples].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.ceil(sorted.length * 0.95) - 1);
  return Math.round(sorted[idx]);
}

async function readBoundedResponse(response, maxBytes, signal) {
  const cap = Math.max(1024, Math.min(MAX_ALLOWED_RESPONSE_BYTES, maxBytes || DEFAULT_MAX_RESPONSE_BYTES));
  const contentLength = Number(response.headers?.get?.('content-length'));
  if (Number.isFinite(contentLength) && contentLength > cap) {
    void response.body?.cancel?.().catch(() => {});
    const err = new Error('provider_body_too_large');
    err.code = 'provider_body_too_large';
    throw err;
  }
  if (!response.body?.getReader) {
    const text = await response.text();
    if (Buffer.byteLength(text, 'utf8') > cap) {
      const err = new Error('provider_body_too_large');
      err.code = 'provider_body_too_large';
      throw err;
    }
    return Buffer.from(text, 'utf8');
  }
  const reader = response.body.getReader();
  const cancel = () => { void reader.cancel().catch(() => {}); };
  signal?.addEventListener?.('abort', cancel, { once: true });
  let size = 0;
  const chunks = [];
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      if (!value?.byteLength) continue;
      size += value.byteLength;
      if (size > cap) {
        const err = new Error('provider_body_too_large');
        err.code = 'provider_body_too_large';
        throw err;
      }
      chunks.push(value);
    }
  } finally {
    signal?.removeEventListener?.('abort', cancel);
    cancel();
    reader.releaseLock();
  }
  return Buffer.concat(chunks, size);
}

function abortable(promise, signal) {
  if (!signal) return promise;
  if (signal.aborted) return Promise.race([Promise.reject(signal.reason || new Error('aborted')), promise]);
  let onAbort;
  const aborted = new Promise((_, reject) => {
    onAbort = () => reject(signal.reason || new Error('aborted'));
    signal.addEventListener('abort', onAbort, { once: true });
  });
  return Promise.race([promise, aborted]).finally(() => {
    signal.removeEventListener('abort', onAbort);
  });
}

function buildMessagesFromRequest(req) {
  if (Array.isArray(req.messages) && req.messages.length > 0) {
    return req.messages;
  }
  const messages = [];
  if (typeof req.systemPrompt === 'string' && req.systemPrompt.trim()) {
    messages.push({ role: 'system', content: req.systemPrompt });
  }
  const images = Array.isArray(req.images) ? req.images : [];
  if (images.length === 0) {
    messages.push({ role: 'user', content: String(req.prompt || '') });
    return messages;
  }
  const content = [];
  if (req.prompt) content.push({ type: 'text', text: String(req.prompt) });
  for (const img of images) {
    if (!img) continue;
    if (img.label) content.push({ type: 'text', text: String(img.label) });
    const mimeType = img.mimeType || img.mediaType || 'image/jpeg';
    let url = img.url;
    if (!url) {
      const b64 = typeof img.data === 'string'
        ? img.data
        : typeof img.base64 === 'string'
          ? img.base64
          : Buffer.isBuffer(img.bytes)
            ? img.bytes.toString('base64')
            : Buffer.isBuffer(img.buffer)
              ? img.buffer.toString('base64')
              : null;
      if (b64) url = `data:${mimeType};base64,${b64}`;
    }
    if (url) content.push({ type: 'image_url', image_url: { url } });
  }
  messages.push({ role: 'user', content });
  return messages;
}

function parseJsonLoose(content) {
  if (typeof content !== 'string') return null;
  const trimmed = content.trim().replace(/^```(?:json)?\s*([\s\S]*?)\s*```$/i, '$1').trim();
  try {
    return JSON.parse(trimmed);
  } catch {
    return null;
  }
}

function normalizeExcludeSet(excludeFamily) {
  const list = Array.isArray(excludeFamily)
    ? excludeFamily
    : excludeFamily ? [excludeFamily] : [];
  const excludedFamilies = new Set();
  const excludedLabs = new Set();
  const excludedModels = new Set();
  for (const raw of list) {
    const val = String(raw || '').trim().toLowerCase();
    if (!val) continue;
    excludedModels.add(val);
    if (val.includes('/')) {
      const spec = resolveModelSpec(val);
      excludedFamilies.add(spec.family.toLowerCase());
      excludedLabs.add(spec.lab.toLowerCase());
    } else {
      excludedFamilies.add(val);
      if (val === 'nemotron') excludedLabs.add('nvidia');
      if (val === 'kimi') excludedLabs.add('moonshot');
      if (val === 'gemini' || val === 'gemma') excludedLabs.add('google');
      if (val === 'nvidia' || val === 'google' || val === 'moonshot' || val === 'openai' || val === 'anthropic') {
        excludedLabs.add(val);
      }
    }
  }
  return { excludedFamilies, excludedLabs, excludedModels };
}

/**
 * Create a configured multi-provider router instance.
 */
export function createProviderRouter(config = {}) {
  const env = config.env || process.env;
  const fetchImpl = config.fetchImpl || globalThis.fetch;
  const nowFn = config.nowFn || (() => Date.now());
  const nvidiaApiKey = config.nvidiaApiKey ?? env?.NVIDIA_API_KEY ?? '';
  const openrouterApiKey = config.openrouterApiKey ?? env?.OPENROUTER_API_KEY ?? '';
  const nvidiaEndpoint = config.nvidiaEndpoint || PROVIDER_ENDPOINTS.nvidia;
  const openrouterEndpoint = config.openrouterEndpoint || PROVIDER_ENDPOINTS.openrouter;
  const openrouterReferer = config.openrouterReferer || env?.OPENROUTER_SITE_URL || 'https://aiwardrobe.app';
  const openrouterTitle = config.openrouterTitle || 'AIWardrobe-Stylist6';

  const openrouterVisionModel = config.openrouterVisionModel
    || env?.STYLIST6_OPENROUTER_VISION_MODEL
    || DEFAULT_OPENROUTER_VISION_MODEL;
  const openrouterFallbackVisionModels = parseFallbackModelList(
    config.openrouterFallbackVisionModels ?? env?.OPENROUTER_FALLBACK_VISION_MODELS,
  );
  const openrouterFallbackTextModels = parseFallbackModelList(
    config.openrouterFallbackTextModels ?? env?.OPENROUTER_FALLBACK_TEXT_MODELS,
  );

  const maxConcurrency = config.maxConcurrency
    ?? (Number(env?.STYLIST6_AI_MAX_CONCURRENCY) || DEFAULT_MAX_CONCURRENCY);

  const semaphore = createAdaptiveSemaphore({
    maxConcurrency,
    minConcurrency: config.minConcurrency ?? 2,
    maxQueueSize: config.maxQueueSize ?? null,
    nowFn,
  });

  const breakerOpts = {
    failureThreshold: config.circuitFailureThreshold ?? 3,
    cooldownMs: config.circuitCooldownMs ?? 30_000,
    nowFn,
  };

  const providers = {
    nvidia: {
      name: 'nvidia',
      endpoint: nvidiaEndpoint,
      apiKey: typeof nvidiaApiKey === 'string' ? nvidiaApiKey.trim() : '',
      breaker: createCircuitBreaker(breakerOpts),
      metrics: {
        totalCalls: 0,
        totalSuccesses: 0,
        totalFailures: 0,
        totalRateLimits: 0,
        totalPromptTokens: 0,
        totalCompletionTokens: 0,
        estimatedCostUsd: 0,
        latencies: [],
      },
    },
    openrouter: {
      name: 'openrouter',
      endpoint: openrouterEndpoint,
      apiKey: typeof openrouterApiKey === 'string' ? openrouterApiKey.trim() : '',
      breaker: createCircuitBreaker(breakerOpts),
      metrics: {
        totalCalls: 0,
        totalSuccesses: 0,
        totalFailures: 0,
        totalRateLimits: 0,
        totalPromptTokens: 0,
        totalCompletionTokens: 0,
        estimatedCostUsd: 0,
        latencies: [],
      },
    },
  };

  let totalFailovers = 0;

  function isProviderConfigured(providerName) {
    const p = providers[providerName];
    return Boolean(p && typeof p.apiKey === 'string' && p.apiKey.length >= 8);
  }

  function recordLatency(metrics, ms) {
    metrics.latencies.push(ms);
    if (metrics.latencies.length > LATENCY_WINDOW_SIZE) metrics.latencies.shift();
  }

  function availableModelsForCapability(capability = 'vision') {
    const list = [];
    const seen = new Set();
    const pushSpec = (modelId, providerName) => {
      if (!isProviderConfigured(providerName)) return;
      const spec = resolveModelSpec(modelId, providerName);
      if (capability === 'vision' && !spec.supportsVision) return;
      if (capability === 'text' && !spec.supportsText) return;
      const dedupeKey = `${providerName}:${spec.model}`;
      if (seen.has(dedupeKey)) return;
      seen.add(dedupeKey);
      list.push(spec);
    };

    // OpenRouter fast primary & configured fallbacks
    pushSpec(openrouterVisionModel, 'openrouter');
    for (const m of openrouterFallbackVisionModels) pushSpec(m, 'openrouter');
    if (capability === 'text') {
      for (const m of openrouterFallbackTextModels) pushSpec(m, 'openrouter');
    }

    // NVIDIA NIM live verified models
    if (capability === 'vision') {
      pushSpec(DEFAULT_KIMI_VISION_MODEL, 'nvidia');
      pushSpec(DEFAULT_GEMMA_VISION_MODEL, 'nvidia');
      pushSpec(DEFAULT_NVIDIA_VISION_MODEL, 'nvidia');
    } else {
      pushSpec(DEFAULT_NVIDIA_TEXT_MODEL, 'nvidia');
      pushSpec(DEFAULT_KIMI_VISION_MODEL, 'nvidia');
      pushSpec(DEFAULT_GEMMA_VISION_MODEL, 'nvidia');
      pushSpec(DEFAULT_NVIDIA_VISION_MODEL, 'nvidia');
    }
    return list;
  }

  /**
   * Select a model from a DIFFERENT lab/family than `excludeFamily` whenever
   * available so the visual judge / jury is independent of the generator.
   */
  function selectCrossLabModel(excludeFamily, {
    capability = 'vision',
    preferProvider = null,
    excludeModels = [],
    allowSameLabFallback = false,
  } = {}) {
    const { excludedFamilies, excludedLabs, excludedModels } = normalizeExcludeSet(excludeFamily);
    for (const m of excludeModels || []) excludedModels.add(String(m || '').toLowerCase());

    const candidates = availableModelsForCapability(capability);
    if (!candidates.length) return null;

    const scoreCandidate = (spec) => {
      const snap = providers[spec.provider].breaker.snapshot();
      const healthScore = snap.state === 'closed' ? 0 : snap.state === 'half_open' ? 100 : 1000;
      const providerBias = preferProvider && spec.provider === preferProvider ? -5 : 0;
      return healthScore + providerBias + (spec.crossLabRank ?? 25);
    };

    const crossLab = candidates.filter((spec) =>
      !excludedModels.has(spec.model.toLowerCase())
      && !excludedFamilies.has(spec.family.toLowerCase())
      && !excludedLabs.has(spec.lab.toLowerCase()),
    ).sort((a, b) => scoreCandidate(a) - scoreCandidate(b));

    const healthyCrossLab = crossLab.filter((spec) => providers[spec.provider].breaker.snapshot().state !== 'open');
    const orderedCrossLab = healthyCrossLab.length > 0 ? healthyCrossLab : crossLab;

    if (orderedCrossLab.length > 0) {
      const [primary, ...fallbacks] = orderedCrossLab;
      return {
        model: primary.model,
        provider: primary.provider,
        family: primary.family,
        lab: primary.lab,
        crossLab: true,
        fallbacks: fallbacks.map((f) => ({
          model: f.model,
          provider: f.provider,
          family: f.family,
          lab: f.lab,
        })),
      };
    }

    if (!allowSameLabFallback) return null;
    const sameLab = candidates
      .filter((spec) => !excludedModels.has(spec.model.toLowerCase()))
      .sort((a, b) => scoreCandidate(a) - scoreCandidate(b));
    if (!sameLab.length) return null;
    const [primary, ...fallbacks] = sameLab;
    return {
      model: primary.model,
      provider: primary.provider,
      family: primary.family,
      lab: primary.lab,
      crossLab: false,
      fallbacks: fallbacks.map((f) => ({
        model: f.model,
        provider: f.provider,
        family: f.family,
        lab: f.lab,
      })),
    };
  }

  function buildAttemptChain(req, capability) {
    const chain = [];
    const seen = new Set();
    const pushTarget = (modelId, providerHint = null) => {
      if (!modelId) return;
      const spec = resolveModelSpec(modelId, providerHint);
      if (!isProviderConfigured(spec.provider)) return;
      const key = `${spec.provider}:${spec.model}`;
      if (seen.has(key)) return;
      seen.add(key);
      chain.push(spec);
    };

    const preferFastCrossLabVision = capability === 'vision'
      && (req.preferCrossLabFastVision === true || req.fastVisualMode === true)
      && isProviderConfigured('openrouter');

    if (preferFastCrossLabVision) {
      // Fast cross-lab primary vision (~1.8–2.5s on OpenRouter Gemini 2.5 Flash)
      // with automatic circuit-breaker failover to NVIDIA VL / Gemma 4 / Kimi.
      pushTarget(openrouterVisionModel, 'openrouter');
      if (req.model && req.model !== openrouterVisionModel) {
        pushTarget(req.model, req.provider || null);
      }
    } else if (req.excludeFamily && !req.model) {
      const cross = selectCrossLabModel(req.excludeFamily, {
        capability,
        preferProvider: req.preferProvider || null,
        allowSameLabFallback: req.allowSameLabFallback === true,
      });
      if (cross) {
        pushTarget(cross.model, cross.provider);
        for (const fb of cross.fallbacks) pushTarget(fb.model, fb.provider);
      }
    } else if (req.model) {
      pushTarget(req.model, req.provider || null);
    } else if (req.provider === 'openrouter') {
      pushTarget(openrouterVisionModel, 'openrouter');
    } else if (req.provider === 'nvidia') {
      pushTarget(capability === 'vision' ? DEFAULT_NVIDIA_VISION_MODEL : DEFAULT_NVIDIA_TEXT_MODEL, 'nvidia');
    } else {
      // Auto-select best configured model for capability
      if (capability === 'vision') {
        pushTarget(DEFAULT_NVIDIA_VISION_MODEL, 'nvidia');
        pushTarget(openrouterVisionModel, 'openrouter');
      } else {
        pushTarget(DEFAULT_NVIDIA_TEXT_MODEL, 'nvidia');
        pushTarget(openrouterVisionModel, 'openrouter');
      }
    }

    // Add explicit caller fallbacks
    if (Array.isArray(req.fallbacks)) {
      for (const fb of req.fallbacks) {
        if (typeof fb === 'string') pushTarget(fb, null);
        else if (fb && typeof fb === 'object') pushTarget(fb.model, fb.provider || null);
      }
    }

    // Add automatic multi-provider failover targets when allowed (default true)
    if (req.allowFailover !== false) {
      const all = availableModelsForCapability(capability);
      // Prefer a different provider first on failover, then different model on same provider
      const primaryProvider = chain[0]?.provider;
      const ordered = [
        ...all.filter((s) => s.provider !== primaryProvider),
        ...all.filter((s) => s.provider === primaryProvider),
      ];
      for (const spec of ordered) pushTarget(spec.model, spec.provider);
    }

    return chain;
  }

  async function executeSingleTarget(spec, req, {
    deadlineAt,
    callerSignal,
    maxTransientRetries,
    preserveRequestShape,
    parseJson,
    maxResponseBytes,
    providerTimeoutMs,
  }) {
    const providerState = providers[spec.provider];
    const metrics = providerState.metrics;
    const gate = providerState.breaker.canExecute();
    if (!gate.allowed) {
      return {
        ok: false,
        provider: spec.provider,
        model: spec.model,
        family: spec.family,
        lab: spec.lab,
        error: 'circuit_open',
        httpStatus: null,
        circuitState: gate.state,
        telemetry: { queueMs: 0, providerMs: 0, totalMs: 0, retried: false },
      };
    }

    const baseMessages = buildMessagesFromRequest(req);
    const rawPayload = {
      messages: baseMessages,
      ...(req.temperature !== undefined ? { temperature: req.temperature } : { temperature: 0.2 }),
      ...(req.top_p !== undefined ? { top_p: req.top_p } : {}),
      ...(req.max_tokens !== undefined ? { max_tokens: req.max_tokens } : { max_tokens: 700 }),
      ...(req.response_format !== undefined ? { response_format: req.response_format } : {}),
      ...(req.chat_template_kwargs !== undefined ? { chat_template_kwargs: req.chat_template_kwargs } : {}),
    };
    const normalizedPayload = normalizeProviderPayload(spec.model, spec.provider, rawPayload, {
      preserveRequestShape,
    });
    const headers = spec.provider === 'openrouter'
      ? {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${providerState.apiKey}`,
          'HTTP-Referer': openrouterReferer,
          'X-Title': openrouterTitle,
        }
      : {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${providerState.apiKey}`,
        };

    const remainingBeforeQueue = deadlineAt ? deadlineAt - nowFn() : null;
    if (remainingBeforeQueue !== null && remainingBeforeQueue <= 0) {
      providerState.breaker.releaseProbe();
      return {
        ok: false,
        provider: spec.provider,
        model: spec.model,
        family: spec.family,
        lab: spec.lab,
        error: 'deadline_exceeded',
        httpStatus: null,
        telemetry: { queueMs: 0, providerMs: 0, totalMs: 0, retried: false },
      };
    }

    let permit;
    try {
      permit = await semaphore.acquire({
        signal: callerSignal,
        timeoutMs: remainingBeforeQueue ?? (req.queueTimeoutMs ?? 15_000),
      });
    } catch (err) {
      providerState.breaker.releaseProbe();
      return {
        ok: false,
        provider: spec.provider,
        model: spec.model,
        family: spec.family,
        lab: spec.lab,
        error: err?.code || 'queue_aborted',
        httpStatus: null,
        telemetry: { queueMs: 0, providerMs: 0, totalMs: 0, retried: false },
      };
    }

    const startedAt = nowFn();
    let retried = false;
    let httpStatus = null;
    let headersMs = null;
    let lastError = 'provider_transport';

    try {
      for (let attempt = 0; attempt <= maxTransientRetries; attempt++) {
        const remaining = deadlineAt ? deadlineAt - nowFn() : null;
        if (remaining !== null && remaining <= 0) {
          lastError = attempt === 0 ? 'deadline_exceeded' : 'deadline_before_retry';
          break;
        }
        const perCallMs = Math.min(
          providerTimeoutMs || spec.defaultTimeoutMs || 15_000,
          remaining !== null ? Math.max(50, remaining) : Number.POSITIVE_INFINITY,
        );
        const callController = new AbortController();
        const callTimer = setTimeout(() => callController.abort(), perCallMs);
        const attemptSignal = callerSignal
          ? AbortSignal.any([callerSignal, callController.signal])
          : callController.signal;

        const dispatchAt = nowFn();
        metrics.totalCalls++;
        let response;
        try {
          response = await abortable(fetchImpl(providerState.endpoint, {
            method: 'POST',
            signal: attemptSignal,
            redirect: 'error',
            headers,
            body: JSON.stringify(normalizedPayload),
          }), attemptSignal);
        } catch (err) {
          clearTimeout(callTimer);
          metrics.totalFailures++;
          const elapsed = Math.max(0, nowFn() - dispatchAt);
          recordLatency(metrics, elapsed);
          if (callerSignal?.aborted) {
            providerState.breaker.releaseProbe();
            return {
              ok: false,
              provider: spec.provider,
              model: spec.model,
              family: spec.family,
              lab: spec.lab,
              error: 'aborted',
              httpStatus: null,
              telemetry: { queueMs: permit.queueMs, providerMs: elapsed, totalMs: Math.max(0, nowFn() - startedAt) + permit.queueMs, retried },
            };
          }
          if (callController.signal.aborted) {
            lastError = 'provider_deadline';
            providerState.breaker.recordFailure('provider_deadline');
            break;
          }
          lastError = 'provider_transport';
          if (attempt < maxTransientRetries && (!deadlineAt || deadlineAt - nowFn() > 50)) {
            retried = true;
            continue;
          }
          providerState.breaker.recordFailure('provider_transport');
          break;
        }

        headersMs = Math.max(0, nowFn() - dispatchAt);
        httpStatus = response.status;

        if (!response.ok) {
          clearTimeout(callTimer);
          void response.body?.cancel?.().catch(() => {});
          metrics.totalFailures++;
          recordLatency(metrics, headersMs);

          if (httpStatus >= 500) {
            lastError = 'provider_http_5xx';
            if (attempt < maxTransientRetries && !callerSignal?.aborted && (!deadlineAt || deadlineAt - nowFn() > 50)) {
              retried = true;
              continue;
            }
            providerState.breaker.recordFailure(`http_${httpStatus}`);
            break;
          }
          if (httpStatus === 429) {
            metrics.totalRateLimits++;
            semaphore.onRateLimit();
            lastError = 'rate_limited';
            providerState.breaker.recordFailure('rate_limited');
            break;
          }
          if (httpStatus === 402) {
            lastError = 'provider_payment_required';
            providerState.breaker.recordFailure('provider_payment_required', { tripImmediately: true });
            break;
          }
          if (httpStatus === 401 || httpStatus === 403) {
            lastError = 'provider_access_denied';
            providerState.breaker.recordFailure('provider_access_denied', { tripImmediately: true });
            break;
          }
          if (httpStatus === 404 || httpStatus === 410) {
            lastError = 'model_unavailable';
            // Single retired model (e.g. 410) shouldn't permanently break healthy sibling models on same host
            providerState.breaker.releaseProbe();
            break;
          }
          lastError = 'provider_http_error';
          providerState.breaker.releaseProbe();
          break;
        }

        let bodyBytes;
        try {
          bodyBytes = await abortable(readBoundedResponse(response, maxResponseBytes, attemptSignal), attemptSignal);
        } catch (err) {
          clearTimeout(callTimer);
          metrics.totalFailures++;
          const elapsed = Math.max(0, nowFn() - dispatchAt);
          recordLatency(metrics, elapsed);
          lastError = callerSignal?.aborted ? 'aborted'
            : callController.signal.aborted ? 'provider_deadline'
              : err?.code || 'provider_body_read_failed';
          if (lastError === 'provider_deadline') providerState.breaker.recordFailure(lastError);
          else providerState.breaker.releaseProbe();
          break;
        }
        clearTimeout(callTimer);

        const providerMs = Math.max(0, nowFn() - dispatchAt);
        recordLatency(metrics, providerMs);
        const rawText = bodyBytes.toString('utf8');
        let rawData;
        try {
          rawData = JSON.parse(rawText);
        } catch {
          metrics.totalFailures++;
          providerState.breaker.releaseProbe();
          return {
            ok: false,
            provider: spec.provider,
            model: spec.model,
            family: spec.family,
            lab: spec.lab,
            error: 'invalid_provider_json',
            httpStatus,
            telemetry: { queueMs: permit.queueMs, headersMs, providerMs, totalMs: permit.queueMs + providerMs, retried },
          };
        }

        const choice = rawData?.choices?.[0];
        const finishReason = ['stop', 'length', 'content_filter', 'tool_calls'].includes(choice?.finish_reason)
          ? choice.finish_reason
          : (choice?.finish_reason ? 'unknown' : null);
        const refusal = choice?.message?.refusal || null;
        const content = typeof choice?.message?.content === 'string' ? choice.message.content : null;

        const promptTokens = Number.isSafeInteger(rawData?.usage?.prompt_tokens) ? rawData.usage.prompt_tokens : 0;
        const completionTokens = Number.isSafeInteger(rawData?.usage?.completion_tokens) ? rawData.usage.completion_tokens : 0;
        const totalTokens = Number.isSafeInteger(rawData?.usage?.total_tokens)
          ? rawData.usage.total_tokens
          : (promptTokens + completionTokens);
        const usage = Object.fromEntries(
          [['prompt_tokens', rawData?.usage?.prompt_tokens],
            ['completion_tokens', rawData?.usage?.completion_tokens],
            ['total_tokens', rawData?.usage?.total_tokens]]
            .filter(([, count]) => Number.isSafeInteger(count) && count >= 0),
        );

        const estimatedCostUsd = (spec.costPer1kPromptUsd != null && spec.costPer1kCompletionUsd != null && (promptTokens > 0 || completionTokens > 0))
          ? Number((((promptTokens / 1000) * spec.costPer1kPromptUsd) + ((completionTokens / 1000) * spec.costPer1kCompletionUsd)).toFixed(6))
          : null;

        metrics.totalPromptTokens += promptTokens;
        metrics.totalCompletionTokens += completionTokens;
        if (estimatedCostUsd != null) metrics.estimatedCostUsd = Number((metrics.estimatedCostUsd + estimatedCostUsd).toFixed(6));

        const parsedJson = parseJson ? parseJsonLoose(content) : null;
        const responseValid = content !== null && !refusal && (!finishReason || finishReason === 'stop') && (!parseJson || parsedJson !== null);

        if (!responseValid) {
          metrics.totalFailures++;
          providerState.breaker.releaseProbe();
          return {
            ok: false,
            provider: spec.provider,
            model: spec.model,
            family: spec.family,
            lab: spec.lab,
            content,
            rawText,
            rawData,
            parsedJson,
            finishReason,
            refusal,
            httpStatus,
            usage,
            error: refusal ? 'provider_refusal'
              : finishReason === 'length' ? 'provider_truncated'
                : finishReason === 'content_filter' ? 'provider_content_filter'
                  : content === null ? 'missing_content'
                    : 'invalid_json_content',
            telemetry: {
              queueMs: permit.queueMs,
              headersMs,
              providerMs,
              totalMs: permit.queueMs + providerMs,
              promptTokens,
              completionTokens,
              totalTokens,
              estimatedCostUsd,
              retried,
            },
          };
        }

        metrics.totalSuccesses++;
        providerState.breaker.recordSuccess();
        semaphore.onSuccess();
        return {
          ok: true,
          provider: spec.provider,
          model: spec.model,
          family: spec.family,
          lab: spec.lab,
          content,
          rawText,
          rawData,
          parsedJson,
          finishReason,
          refusal: null,
          httpStatus,
          usage,
          error: null,
          telemetry: {
            queueMs: permit.queueMs,
            headersMs,
            providerMs,
            totalMs: permit.queueMs + providerMs,
            promptTokens,
            completionTokens,
            totalTokens,
            estimatedCostUsd,
            retried,
          },
        };
      }

      return {
        ok: false,
        provider: spec.provider,
        model: spec.model,
        family: spec.family,
        lab: spec.lab,
        error: lastError,
        httpStatus,
        telemetry: {
          queueMs: permit.queueMs,
          ...(headersMs !== null ? { headersMs } : {}),
          providerMs: Math.max(0, nowFn() - startedAt),
          totalMs: permit.queueMs + Math.max(0, nowFn() - startedAt),
          retried,
        },
      };
    } finally {
      permit.release();
    }
  }

  async function dispatch(req = {}, capability = 'text') {
    const chain = buildAttemptChain(req, capability);
    if (!chain.length) {
      return {
        ok: false,
        provider: null,
        model: null,
        family: null,
        lab: null,
        content: null,
        parsedJson: null,
        httpStatus: null,
        attempts: [],
        failedOver: false,
        error: 'no_provider_configured',
        telemetry: { queueMs: 0, providerMs: 0, totalMs: 0, retried: false },
      };
    }

    const totalTimeoutMs = Number.isFinite(req.timeoutMs) && req.timeoutMs > 0
      ? Math.floor(req.timeoutMs)
      : (capability === 'vision' ? 20_000 : 12_000);
    const deadlineAt = nowFn() + totalTimeoutMs;
    const maxTransientRetries = Number.isInteger(req.maxTransientRetries) && req.maxTransientRetries >= 0
      ? Math.min(2, req.maxTransientRetries)
      : 1;
    const attempts = [];

    for (let i = 0; i < chain.length; i++) {
      if (req.signal?.aborted) {
        return {
          ok: false,
          provider: chain[i].provider,
          model: chain[i].model,
          family: chain[i].family,
          lab: chain[i].lab,
          content: null,
          parsedJson: null,
          httpStatus: null,
          attempts,
          failedOver: i > 0,
          error: 'aborted',
          telemetry: { queueMs: 0, providerMs: 0, totalMs: Math.max(0, nowFn() - (deadlineAt - totalTimeoutMs)), retried: false },
        };
      }
      const spec = chain[i];
      const attemptProviderTimeoutMs = (
        i === 0 && chain.length > 1 && Number.isFinite(req.primaryTimeoutMs) && req.primaryTimeoutMs > 0
      )
        ? Math.min(req.providerTimeoutMs || req.primaryTimeoutMs, req.primaryTimeoutMs)
        : (req.providerTimeoutMs || null);
      const result = await executeSingleTarget(spec, req, {
        deadlineAt,
        callerSignal: req.signal || null,
        maxTransientRetries,
        preserveRequestShape: req.preserveRequestShape === true && i === 0,
        parseJson: req.parseJson === true,
        maxResponseBytes: req.maxResponseBytes || DEFAULT_MAX_RESPONSE_BYTES,
        providerTimeoutMs: attemptProviderTimeoutMs,
      });

      attempts.push({
        provider: spec.provider,
        model: spec.model,
        ok: result.ok,
        error: result.error,
        httpStatus: result.httpStatus,
        latencyMs: result.telemetry?.providerMs ?? 0,
        retried: result.telemetry?.retried === true,
      });

      if (result.ok) {
        if (i > 0) totalFailovers++;
        return { ...result, attempts, failedOver: i > 0 };
      }

      if (result.error === 'aborted' || result.error === 'deadline_exceeded') {
        return { ...result, attempts, failedOver: i > 0 };
      }
    }

    const last = attempts.at(-1);
    return {
      ok: false,
      provider: last?.provider || chain[0].provider,
      model: last?.model || chain[0].model,
      family: chain.at(-1)?.family || null,
      lab: chain.at(-1)?.lab || null,
      content: null,
      parsedJson: null,
      httpStatus: last?.httpStatus ?? null,
      attempts,
      failedOver: attempts.length > 1,
      error: last?.error || 'all_providers_failed',
      telemetry: {
        queueMs: 0,
        providerMs: attempts.reduce((sum, a) => sum + (a.latencyMs || 0), 0),
        totalMs: Math.max(0, nowFn() - (deadlineAt - totalTimeoutMs)),
        retried: attempts.some((a) => a.retried),
      },
    };
  }

  function callVisionModel(req = {}) {
    return dispatch(req, 'vision');
  }

  function callTextModel(req = {}) {
    return dispatch(req, 'text');
  }

  function getProviderHealth() {
    const providerSnapshots = {};
    let totalCalls = 0;
    let totalSuccesses = 0;
    let totalFailures = 0;
    let totalPromptTokens = 0;
    let totalCompletionTokens = 0;
    let estimatedCostUsd = 0;

    for (const [name, p] of Object.entries(providers)) {
      const m = p.metrics;
      const avgLatencyMs = m.latencies.length
        ? Math.round(m.latencies.reduce((a, b) => a + b, 0) / m.latencies.length)
        : 0;
      providerSnapshots[name] = {
        configured: isProviderConfigured(name),
        endpoint: p.endpoint,
        ...p.breaker.snapshot(),
        totalCalls: m.totalCalls,
        totalSuccesses: m.totalSuccesses,
        totalFailures: m.totalFailures,
        totalRateLimits: m.totalRateLimits,
        avgLatencyMs,
        p95LatencyMs: computeP95(m.latencies),
        totalPromptTokens: m.totalPromptTokens,
        totalCompletionTokens: m.totalCompletionTokens,
        estimatedCostUsd: m.estimatedCostUsd,
      };
      totalCalls += m.totalCalls;
      totalSuccesses += m.totalSuccesses;
      totalFailures += m.totalFailures;
      totalPromptTokens += m.totalPromptTokens;
      totalCompletionTokens += m.totalCompletionTokens;
      estimatedCostUsd = Number((estimatedCostUsd + m.estimatedCostUsd).toFixed(6));
    }

    return {
      version: AI_PROVIDER_ROUTER_VERSION,
      providers: providerSnapshots,
      semaphore: semaphore.stats(),
      totals: {
        calls: totalCalls,
        successes: totalSuccesses,
        failures: totalFailures,
        failovers: totalFailovers,
        promptTokens: totalPromptTokens,
        completionTokens: totalCompletionTokens,
        estimatedCostUsd,
      },
    };
  }

  function reset() {
    totalFailovers = 0;
    for (const p of Object.values(providers)) {
      p.breaker.reset();
      p.metrics.totalCalls = 0;
      p.metrics.totalSuccesses = 0;
      p.metrics.totalFailures = 0;
      p.metrics.totalRateLimits = 0;
      p.metrics.totalPromptTokens = 0;
      p.metrics.totalCompletionTokens = 0;
      p.metrics.estimatedCostUsd = 0;
      p.metrics.latencies.length = 0;
    }
  }

  return {
    version: AI_PROVIDER_ROUTER_VERSION,
    callVisionModel,
    callTextModel,
    selectCrossLabModel,
    getProviderHealth,
    isProviderConfigured,
    semaphore,
    reset,
  };
}

let defaultRouter = null;

function getDefaultRouter() {
  if (!defaultRouter) defaultRouter = createProviderRouter();
  return defaultRouter;
}

export function resetDefaultProviderRouter() {
  defaultRouter = null;
}

export function callVisionModel(req = {}) {
  const router = req?.router || getDefaultRouter();
  return router.callVisionModel(req);
}

export function callTextModel(req = {}) {
  const router = req?.router || getDefaultRouter();
  return router.callTextModel(req);
}

export function selectCrossLabModel(excludeFamily, options = {}) {
  const router = options?.router || getDefaultRouter();
  return router.selectCrossLabModel(excludeFamily, options);
}

export function getProviderHealth(router = null) {
  return (router || getDefaultRouter()).getProviderHealth();
}
