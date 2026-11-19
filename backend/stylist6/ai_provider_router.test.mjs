import test from 'node:test';
import assert from 'node:assert/strict';
import {
  AI_PROVIDER_ROUTER_VERSION,
  DEFAULT_KIMI_VISION_MODEL,
  DEFAULT_NVIDIA_VISION_MODEL,
  DEFAULT_NVIDIA_TEXT_MODEL,
  DEFAULT_OPENROUTER_VISION_MODEL,
  DEFAULT_GEMMA_VISION_MODEL,
  PROVIDER_ENDPOINTS,
  createProviderRouter,
  createCircuitBreaker,
  createAdaptiveSemaphore,
  normalizeProviderPayload,
  callVisionModel,
  callTextModel,
  selectCrossLabModel,
  getProviderHealth,
  resetDefaultProviderRouter,
} from './ai_provider_router.mjs';

const NVIDIA_KEY = 'nvapi-test-router-key-0123456789abcdef';
const OPENROUTER_KEY = 'sk-or-v1-test-router-key-0123456789abcdef';

const okJson = (content, usage = { prompt_tokens: 400, completion_tokens: 120, total_tokens: 520 }) =>
  Response.json({
    choices: [{ finish_reason: 'stop', message: { content: typeof content === 'string' ? content : JSON.stringify(content) } }],
    usage,
  });

test('normalizeProviderPayload preserves Kimi K3.1 Pro fixed top_p constraint and OpenRouter cleanliness', () => {
  assert.match(AI_PROVIDER_ROUTER_VERSION, /ai-provider-router/);
  const kimiPayload = normalizeProviderPayload(DEFAULT_KIMI_VISION_MODEL, 'nvidia', {
    messages: [{ role: 'system', content: '/no_think' }, { role: 'user', content: 'Judge look' }],
    temperature: 0.1,
    top_p: 0.9,
    max_tokens: 280,
    response_format: { type: 'json_schema', json_schema: { name: 'test', strict: true, schema: { type: 'object' } } },
  });
  assert.equal(Object.hasOwn(kimiPayload, 'top_p'), false, 'Kimi K3.1 Pro rejects explicit top_p != 0.95 with HTTP 400');
  assert.equal(kimiPayload.model, DEFAULT_KIMI_VISION_MODEL);
  assert.deepEqual(kimiPayload.response_format, { type: 'json_object' });
  assert.deepEqual(kimiPayload.chat_template_kwargs, { enable_thinking: false });
  assert.equal(kimiPayload.messages.length, 1, 'bare /no_think sentinel stripped for non-Nemotron');

  const openrouterPayload = normalizeProviderPayload(DEFAULT_OPENROUTER_VISION_MODEL, 'openrouter', {
    messages: [{ role: 'system', content: '/no_think' }, { role: 'user', content: 'Judge look' }],
    temperature: 0.2,
    top_p: 0.95,
    chat_template_kwargs: { enable_thinking: false },
  });
  assert.equal(Object.hasOwn(openrouterPayload, 'chat_template_kwargs'), false);
  assert.equal(openrouterPayload.top_p, 0.95);
  assert.equal(openrouterPayload.messages.length, 1);

  const nemotronPayload = normalizeProviderPayload(DEFAULT_NVIDIA_VISION_MODEL, 'nvidia', {
    messages: [{ role: 'system', content: '/no_think' }, { role: 'user', content: 'Judge look' }],
    top_p: 0.95,
    chat_template_kwargs: { enable_thinking: false },
  });
  assert.equal(nemotronPayload.top_p, 0.95);
  assert.equal(nemotronPayload.messages.length, 2);
});

test('multi-provider failover routes from failing NVIDIA NIM to OpenRouter Gemini 2.5 Flash with telemetry', async () => {
  const seen = [];
  const router = createProviderRouter({
    nvidiaApiKey: NVIDIA_KEY,
    openrouterApiKey: OPENROUTER_KEY,
    fetchImpl: async (url, init) => {
      const body = JSON.parse(init.body);
      seen.push({ url, model: body.model, auth: init.headers.Authorization, redirect: init.redirect });
      if (url === PROVIDER_ENDPOINTS.nvidia) {
        return new Response('upstream overloaded', { status: 503 });
      }
      assert.equal(url, PROVIDER_ENDPOINTS.openrouter);
      assert.equal(init.headers['X-Title'], 'AIWardrobe-Stylist6');
      return okJson({ score: 8.6, ok: true }, { prompt_tokens: 1000, completion_tokens: 200, total_tokens: 1200 });
    },
  });

  const res = await callVisionModel({
    router,
    model: DEFAULT_NVIDIA_VISION_MODEL,
    prompt: 'Evaluate this 4-item studio look board.',
    images: [{ mimeType: 'image/jpeg', data: Buffer.from('fake-jpeg').toString('base64'), label: 'Board:' }],
    parseJson: true,
    maxTransientRetries: 0,
  });

  assert.equal(res.ok, true);
  assert.equal(res.failedOver, true);
  assert.equal(res.provider, 'openrouter');
  assert.equal(res.model, DEFAULT_OPENROUTER_VISION_MODEL);
  assert.deepEqual(res.parsedJson, { score: 8.6, ok: true });
  assert.equal(res.attempts.length, 2);
  assert.equal(res.telemetry.promptTokens, 1000);
  assert.equal(res.telemetry.completionTokens, 200);
  assert.ok(res.telemetry.estimatedCostUsd > 0);
  assert.ok(seen.every((s) => s.redirect === 'error'));

  const health = getProviderHealth(router);
  assert.equal(health.providers.nvidia.totalFailures, 1);
  assert.equal(health.providers.openrouter.totalSuccesses, 1);
  assert.equal(health.totals.failovers, 1);
  assert.doesNotMatch(JSON.stringify(health), new RegExp(`${NVIDIA_KEY}|${OPENROUTER_KEY}`));
});

test('per-provider circuit breaker transitions closed -> open -> half_open -> closed', async () => {
  let now = 10_000;
  const breaker = createCircuitBreaker({ failureThreshold: 2, cooldownMs: 5000, nowFn: () => now });
  assert.equal(breaker.canExecute().allowed, true);
  breaker.recordFailure('http_503');
  assert.equal(breaker.snapshot().state, 'closed');
  breaker.recordFailure('http_503');
  assert.equal(breaker.snapshot().state, 'open');
  assert.equal(breaker.canExecute().allowed, false);
  assert.equal(breaker.snapshot().cooldownRemainingMs, 5000);

  now += 5001;
  const probe = breaker.canExecute();
  assert.equal(probe.allowed, true);
  assert.equal(probe.state, 'half_open');
  assert.equal(probe.probe, true);
  // Second concurrent probe blocked while first probe in flight
  assert.equal(breaker.canExecute().allowed, false);

  // Probe fails -> immediately reopens
  breaker.recordFailure('http_503');
  assert.equal(breaker.snapshot().state, 'open');

  // Cooldown passes again -> probe succeeds -> closed
  now += 5001;
  assert.equal(breaker.canExecute().allowed, true);
  breaker.recordSuccess();
  assert.equal(breaker.snapshot().state, 'closed');
  assert.equal(breaker.snapshot().consecutiveFailures, 0);
});

test('open circuit short-circuits failing provider with zero outbound HTTP calls', async () => {
  let nvidiaHits = 0;
  let openrouterHits = 0;
  let now = 1_000;
  const router = createProviderRouter({
    nvidiaApiKey: NVIDIA_KEY,
    openrouterApiKey: OPENROUTER_KEY,
    circuitFailureThreshold: 2,
    circuitCooldownMs: 20_000,
    nowFn: () => now,
    fetchImpl: async (url) => {
      if (url === PROVIDER_ENDPOINTS.nvidia) {
        nvidiaHits++;
        return new Response('down', { status: 502 });
      }
      openrouterHits++;
      return okJson({ winner: 'F01' });
    },
  });

  // Two failures trip NVIDIA breaker
  await callTextModel({ router, model: DEFAULT_NVIDIA_TEXT_MODEL, prompt: 'test 1', maxTransientRetries: 0 });
  await callTextModel({ router, model: DEFAULT_NVIDIA_TEXT_MODEL, prompt: 'test 2', maxTransientRetries: 0 });
  assert.equal(nvidiaHits, 2);
  assert.equal(router.getProviderHealth().providers.nvidia.state, 'open');

  // Third call skips NVIDIA completely without hitting fetch for NVIDIA
  const third = await callTextModel({ router, model: DEFAULT_NVIDIA_TEXT_MODEL, prompt: 'test 3', maxTransientRetries: 0 });
  assert.equal(third.ok, true);
  assert.equal(third.provider, 'openrouter');
  assert.equal(nvidiaHits, 2, 'open circuit must not dispatch HTTP to failing provider');
  assert.equal(openrouterHits, 3);
  assert.equal(third.attempts[0].error, 'circuit_open');
});

test('adaptive concurrency semaphore queues smoothly up to maxConcurrency, respects deadlines, and backs off on 429', async () => {
  const sem = createAdaptiveSemaphore({ maxConcurrency: 4, minConcurrency: 2, recoverySuccessThreshold: 2 });
  const permits = await Promise.all([sem.acquire(), sem.acquire(), sem.acquire(), sem.acquire()]);
  assert.equal(sem.stats().inFlight, 4);
  assert.equal(sem.stats().peakInFlight, 4);

  // Fifth caller queues smoothly and succeeds once a permit releases
  const queued = sem.acquire({ timeoutMs: 500 });
  assert.equal(sem.stats().queued, 1);

  // Sixth caller with tight deadline times out cleanly without leaking
  await assert.rejects(() => sem.acquire({ timeoutMs: 15 }), /queue_deadline/);
  assert.equal(sem.stats().totalQueueTimeouts, 1);
  assert.equal(sem.stats().queued, 1);

  permits[0].release();
  const fifth = await queued;
  assert.equal(sem.stats().inFlight, 4);
  fifth.release();
  permits.slice(1).forEach((p) => p.release());
  assert.equal(sem.stats().inFlight, 0);

  // AIMD backoff on 429 and gradual recovery on successes
  sem.onRateLimit();
  assert.equal(sem.stats().effectiveLimit, 2);
  sem.onSuccess();
  sem.onSuccess();
  assert.equal(sem.stats().effectiveLimit, 3);
  sem.onSuccess();
  sem.onSuccess();
  assert.equal(sem.stats().effectiveLimit, 4);
});

test('selectCrossLabModel chooses a different lab/family than the generator and respects circuit health', async () => {
  const router = createProviderRouter({
    nvidiaApiKey: NVIDIA_KEY,
    openrouterApiKey: OPENROUTER_KEY,
    openrouterFallbackVisionModels: ['anthropic/claude-3.5-haiku'],
    circuitFailureThreshold: 1,
    fetchImpl: async (url) => (url === PROVIDER_ENDPOINTS.openrouter
      ? new Response('limit', { status: 429 })
      : okJson({ ok: true })),
  });

  const pick1 = selectCrossLabModel('nemotron', { router, capability: 'vision' });
  assert.ok(pick1);
  assert.equal(pick1.crossLab, true);
  assert.notEqual(pick1.family, 'nemotron');
  assert.notEqual(pick1.lab, 'nvidia');
  assert.equal(pick1.model, DEFAULT_OPENROUTER_VISION_MODEL);
  assert.ok(pick1.fallbacks.some((f) => f.model === DEFAULT_KIMI_VISION_MODEL));
  assert.ok(pick1.fallbacks.some((f) => f.model === DEFAULT_GEMMA_VISION_MODEL));

  // If OpenRouter trips its circuit, cross-lab automatically promotes Kimi K3.1 Pro (Moonshot)
  await router.callVisionModel({ model: DEFAULT_OPENROUTER_VISION_MODEL, prompt: 'probe', allowFailover: false, maxTransientRetries: 0 });
  assert.equal(router.getProviderHealth().providers.openrouter.state, 'open');

  const pick2 = router.selectCrossLabModel(DEFAULT_NVIDIA_VISION_MODEL, { capability: 'vision' });
  assert.equal(pick2.model, DEFAULT_KIMI_VISION_MODEL);
  assert.equal(pick2.lab, 'moonshot');
  assert.equal(pick2.provider, 'nvidia');

  // With only NVIDIA configured, excluding nemotron still finds Kimi and Gemma
  const nvidiaOnly = createProviderRouter({ nvidiaApiKey: NVIDIA_KEY, openrouterApiKey: '' });
  const pick3 = nvidiaOnly.selectCrossLabModel('nvidia/nemotron-3.5-super-vl-120b-a12b', { capability: 'vision' });
  assert.equal(pick3.model, DEFAULT_KIMI_VISION_MODEL);
  assert.ok(pick3.fallbacks.some((f) => f.model === DEFAULT_GEMMA_VISION_MODEL));
});

test('router rejects oversized response bodies and honors abort signal cleanly', async () => {
  const router = createProviderRouter({
    nvidiaApiKey: NVIDIA_KEY,
    openrouterApiKey: '',
    fetchImpl: async () => new Response('x'.repeat(20_000), { status: 200 }),
  });
  const oversized = await router.callTextModel({
    model: DEFAULT_NVIDIA_TEXT_MODEL,
    prompt: 'hello',
    maxResponseBytes: 4096,
    allowFailover: false,
  });
  assert.equal(oversized.ok, false);
  assert.equal(oversized.error, 'provider_body_too_large');

  const controller = new AbortController();
  controller.abort();
  const aborted = await router.callTextModel({
    model: DEFAULT_NVIDIA_TEXT_MODEL,
    prompt: 'hello',
    signal: controller.signal,
    allowFailover: false,
  });
  assert.equal(aborted.ok, false);
  assert.equal(aborted.error, 'aborted');
  resetDefaultProviderRouter();
});

test('preferCrossLabFastVision routes primary vision to OpenRouter Gemini 2.5 Flash and fails over cleanly to NVIDIA VL / Gemma with circuit breaker', async () => {
  const seen = [];
  let openrouterDown = false;
  const router = createProviderRouter({
    nvidiaApiKey: NVIDIA_KEY,
    openrouterApiKey: OPENROUTER_KEY,
    circuitFailureThreshold: 2,
    fetchImpl: async (url, init) => {
      const body = JSON.parse(init.body);
      seen.push({ url, model: body.model });
      if (url === PROVIDER_ENDPOINTS.openrouter) {
        if (openrouterDown) return new Response('overloaded', { status: 503 });
        return okJson({ verdict: 'fast_openrouter_ok' });
      }
      assert.equal(url, PROVIDER_ENDPOINTS.nvidia);
      return okJson({ verdict: 'nvidia_failover_ok', model: body.model });
    },
  });

  // 1. Healthy OpenRouter -> Gemini 2.5 Flash wins immediately as primary
  const fast = await router.callVisionModel({
    model: DEFAULT_NVIDIA_VISION_MODEL,
    preferCrossLabFastVision: true,
    prompt: 'Inspect 1024x1024 studio board',
    parseJson: true,
    maxTransientRetries: 0,
  });
  assert.equal(fast.ok, true);
  assert.equal(fast.provider, 'openrouter');
  assert.equal(fast.model, DEFAULT_OPENROUTER_VISION_MODEL);
  assert.equal(fast.failedOver, false);

  // 2. OpenRouter fails -> automatic failover to NVIDIA 120B VL; 2 failures trip OpenRouter circuit
  openrouterDown = true;
  const fo1 = await router.callVisionModel({
    model: DEFAULT_NVIDIA_VISION_MODEL,
    preferCrossLabFastVision: true,
    prompt: 'Inspect board 2',
    parseJson: true,
    maxTransientRetries: 0,
  });
  assert.equal(fo1.ok, true);
  assert.equal(fo1.failedOver, true);
  assert.equal(fo1.provider, 'nvidia');
  assert.equal(fo1.model, DEFAULT_NVIDIA_VISION_MODEL);

  const fo2 = await router.callVisionModel({
    model: DEFAULT_NVIDIA_VISION_MODEL,
    preferCrossLabFastVision: true,
    prompt: 'Inspect board 3',
    parseJson: true,
    maxTransientRetries: 0,
  });
  assert.equal(fo2.ok, true);
  assert.equal(router.getProviderHealth().providers.openrouter.state, 'open');

  // 3. With OpenRouter circuit open, third call skips OpenRouter HTTP completely (0ms short-circuit!)
  const openrouterCallsBefore = seen.filter((s) => s.url === PROVIDER_ENDPOINTS.openrouter).length;
  const fo3 = await router.callVisionModel({
    model: DEFAULT_NVIDIA_VISION_MODEL,
    preferCrossLabFastVision: true,
    prompt: 'Inspect board 4',
    parseJson: true,
    maxTransientRetries: 0,
  });
  assert.equal(fo3.ok, true);
  assert.equal(fo3.provider, 'nvidia');
  assert.equal(fo3.attempts[0].error, 'circuit_open');
  assert.equal(
    seen.filter((s) => s.url === PROVIDER_ENDPOINTS.openrouter).length,
    openrouterCallsBefore,
    'open circuit must not hit OpenRouter HTTP endpoint',
  );
});
