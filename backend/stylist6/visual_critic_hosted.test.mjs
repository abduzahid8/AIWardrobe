import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createHostedVisualCritic, NVIDIA_VISION_MODEL } from './visual_critic_hosted.mjs';

const apiKey = 'test-api-key-0123456789abcdef';
function asset(fill = 7, width = 16) {
  const bytes = Buffer.alloc(256, fill);
  Buffer.from([0xff, 0xd8, 0xff, 0xc0, 0, 8, 8, 0, 16, width >> 8, width & 255, 3]).copy(bytes);
  bytes.writeUInt16BE(0xffd9, bytes.length - 2);
  return { bytes, sha256: createHash('sha256').update(bytes).digest('hex') };
}

const candidate = {
  id: 's6_abc123', referenceId: 'rt_001',
  products: ['top', 'outerwear', 'bottom', 'shoes'].map((slot) => ({
    id: slot, slot, imageUrl: 'https://127.0.0.1/private',
  })),
};

const verdict = {
  slotObservations: { top: 'light striped shirt', outerwear: 'short dark jacket', bottom: 'grey straight trousers', shoes: 'dark round toe shoes' },
  candidateId: candidate.id, visualCohesion: 0.9, referenceFidelity: 0.87, styleFit: 0.88,
  occasionFit: 0.91, formalityFit: 0.93, confidence: 0.95, reject: false,
  relationships: [
          { kind: 'outfit', slots: ['top', 'outerwear'], observation: 'mock top and layer match' },
          { kind: 'outfit', slots: ['bottom', 'shoes'], observation: 'mock footwear repeats bottom color' },
          { kind: 'reference', slots: ['top'], observation: 'mock reference repeats the top pattern' },
        ],
};
const nvidiaResponse = (row, finish_reason = 'stop') => Response.json({
  choices: [{ finish_reason, message: { content: JSON.stringify(row) } }],
});

test('configuration requires key and trusted byte resolvers', () => {
  assert.throws(() => createHostedVisualCritic({ apiKey: 'short' }), /hosted_visual_critic_not_configured/);
  assert.throws(() => createHostedVisualCritic({ apiKey }), /trusted_media_resolvers_required/);
});

test('NVIDIA request sends only style context and ordered, pinned outfit images', async () => {
  const reference = asset(3), photos = candidate.products.map((_, i) => asset(i + 4));
  const calls = [];
  const critic = createHostedVisualCritic({
    provider: 'nvidia', apiKey,
    resolveReference: () => reference,
    resolveProduct: ({ slot }) => photos[candidate.products.findIndex((p) => p.slot === slot)],
    fetchImpl: async (url, init) => {
      calls.push({ url, init });
      return nvidiaResponse(verdict);
    },
  });
  const result = await critic([candidate], { style: 'classic', occasion: 'Work', brief: 'Soft tailoring', jwt: 'private-data' });
  assert.equal(result[0].referenceFidelity, verdict.referenceFidelity);
  assert.equal(result[0].estimatedCostUsd, null);
  assert.equal(calls.length, 1);
  const [{ url, init }] = calls;
  assert.equal(url, 'https://integrate.api.nvidia.com/v1/chat/completions');
  assert.equal(init.redirect, 'error');
  assert.equal(init.headers.Authorization, 'Bearer ' + apiKey);
  const body = JSON.parse(init.body);
  assert.equal(body.model, NVIDIA_VISION_MODEL);
  assert.equal(body.stream, false);
  assert.deepEqual(body.response_format, { type: 'json_object' });
  assert.equal(body.chat_template_kwargs.enable_thinking, false);
  const [prompt, ...mediaParts] = body.messages[0].content;
  const images = mediaParts.filter((part) => part.type === 'image_url');
  assert.deepEqual(mediaParts.filter((part) => part.type === 'text').map(({ text }) => text), [
    'Reference outfit image:', ...candidate.products.map(({ slot }) => `Candidate ${slot} image:`),
  ]);
  assert.ok(mediaParts.every((part, index) => part.type === (index % 2 === 0 ? 'text' : 'image_url')));
  assert.match(prompt.text, /Soft tailoring/);
  assert.doesNotMatch(init.body, /127\.0\.0\.1|private-data/);
  assert.deepEqual(images.map(({ image_url }) => image_url.url),
    [reference, ...photos].map(({ bytes }) => 'data:image/jpeg;base64,' + bytes.toString('base64')));
});

test('Gemini 3.8 retains multimodal JSON contract without removed sampling parameters', async () => {
  let sent;
  const critic = createHostedVisualCritic({
    apiKey, resolveReference: () => asset(), resolveProduct: () => asset(),
    fetchImpl: async (url, init) => {
      assert.match(url, /gemini-3\.8-flash:generateContent$/);
      sent = JSON.parse(init.body);
      return Response.json({ candidates: [{ content: { parts: [{ text: JSON.stringify(verdict) }] } }] });
    },
  });
  assert.equal((await critic([candidate])).length, 1);
  assert.deepEqual(sent.generationConfig, { responseMimeType: 'application/json', maxOutputTokens: 600 });
  assert.equal(sent.contents[0].parts.filter((p) => p.inlineData).length, 5);
});

test('missing, mismatched and oversized media abstains with zero outbound I/O', async () => {
  for (const product of [null, { ...asset(), sha256: '0'.repeat(64) }, asset(8, 3000), { bytes: Buffer.alloc(524289) }]) {
    const events = [];
    let calls = 0;
    const critic = createHostedVisualCritic({
      provider: 'nvidia', apiKey, resolveReference: () => asset(), resolveProduct: () => product,
      fetchImpl: async () => { calls++; throw Error('unexpected outbound request'); },
      onEvent: (event) => events.push(event),
    });
    assert.deepEqual(await critic([candidate]), []);
    assert.equal(calls, 0);
    assert.deepEqual(events, [{ candidateId: candidate.id, outcome: 'abstain', dispatched: false, reason: 'product_image_unresolved' }]);
  }
});

test('invalid, mismatched, truncated and oversized provider replies abstain without retry', async () => {
  for (const respond of [
    () => new Response('bad JSON'),
    () => new Response('x'.repeat(65537)),
    () => new Response(null, { status: 503 }),
    () => nvidiaResponse({ ...verdict, candidateId: 'other' }),
    () => nvidiaResponse({ ...verdict, referenceFidelity: undefined }),
    () => nvidiaResponse({ ...verdict, reject: undefined }),
    () => nvidiaResponse(verdict, 'length'),
  ]) {
    let calls = 0;
    const critic = createHostedVisualCritic({
      provider: 'nvidia', apiKey, resolveReference: () => asset(), resolveProduct: () => asset(),
      fetchImpl: async () => { calls++; return respond(); },
    });
    assert.deepEqual(await critic([candidate]), []);
    assert.equal(calls, 1);
  }
});

test('429 stops further dispatch and never exceeds two in flight', async () => {
  let calls = 0, inflight = 0, peak = 0, references = 0;
  const critic = createHostedVisualCritic({
    provider: 'nvidia', apiKey, resolveReference: () => { references++; return asset(); }, resolveProduct: () => asset(),
    fetchImpl: async () => {
      calls++;
      peak = Math.max(peak, ++inflight);
      await new Promise((resolve) => setTimeout(resolve, 5));
      inflight--;
      return new Response(null, { status: 429 });
    },
  });
  const candidates = Array.from({ length: 4 }, (_, i) => ({ ...candidate, id: 'look-' + i }));
  assert.deepEqual(await critic(candidates), []);
  assert.equal(references, 1);
  assert.equal(calls, 2);
  assert.equal(peak, 2);
  await assert.rejects(critic([...candidates, candidate]), /invalid_visual_batch/);
  await assert.rejects(critic([candidate, candidate]), /invalid_visual_batch/);
});

test('the shared deadline aborts in-flight requests and prevents later dispatch', async () => {
  const events = [];
  let calls = 0;
  const critic = createHostedVisualCritic({
    provider: 'nvidia', apiKey, timeoutMs: 20, resolveReference: () => asset(), resolveProduct: () => asset(),
    fetchImpl: async (url, { signal }) => {
      calls++;
      await new Promise((resolve) => setTimeout(resolve, 30));
      signal.throwIfAborted();
    },
    onEvent: (event) => events.push(event),
  });
  assert.deepEqual(await critic(Array.from({ length: 4 }, (_, i) => ({ ...candidate, id: 'deadline-' + i }))), []);
  assert.equal(calls, 2);
  assert.equal(events.length, 4);
  assert.ok(events.every(({ reason }) => reason === 'deadline'));
});

test('a stalled media resolver is bounded and never dispatches', async () => {
  const events = [];
  let calls = 0;
  const critic = createHostedVisualCritic({
    provider: 'nvidia', apiKey, timeoutMs: 10,
    resolveReference: () => new Promise(() => {}), resolveProduct: () => asset(),
    fetchImpl: async () => { calls++; throw Error('unexpected outbound request'); },
    onEvent: (event) => events.push(event),
  });
  assert.deepEqual(await critic([candidate]), []);
  assert.equal(calls, 0);
  assert.equal(events[0].reason, 'deadline');
});

test('payment-required responses block any remaining dispatches without retry', async () => {
  const events = [];
  let calls = 0;
  const critic = createHostedVisualCritic({
    provider: 'nvidia', apiKey, resolveReference: () => asset(), resolveProduct: () => asset(),
    fetchImpl: async () => { calls++; return new Response(null, { status: 402 }); },
    onEvent: (event) => events.push(event),
  });
  assert.deepEqual(await critic(Array.from({ length: 4 }, (_, i) => ({ ...candidate, id: 'payment-' + i }))), []);
  assert.ok(calls >= 1 && calls <= 2);
  assert.equal(events.filter((event) => event.httpStatus === 402).length, calls);
  assert.ok(events.every(({ reason }) => reason === 'provider_payment_required'));
});

test('usage telemetry retains safe token counts even when a successful response fails verdict validation', async () => {
  const events = [];
  const critic = createHostedVisualCritic({
    provider: 'nvidia', apiKey, resolveReference: () => asset(), resolveProduct: () => asset(),
    fetchImpl: async () => Response.json({
      usage: { prompt_tokens: 120, completion_tokens: 2, total_tokens: 'invalid', other: 'private' },
      choices: [{ message: { content: 'bad verdict' } }],
    }),
    onEvent: (event) => events.push(event),
  });
  assert.deepEqual(await critic([candidate]), []);
  assert.equal(events[0].httpStatus, 200);
  assert.deepEqual(events[0].usage, { prompt_tokens: 120, completion_tokens: 2 });
  assert.ok(events[0].latencyMs >= 0);
});

test('a provider that did not visibly resolve every supplied slot must abstain', async () => {
  for (const slotObservations of [undefined, { ...verdict.slotObservations, outerwear: 'No outerwear image is supplied' },
    { ...verdict.slotObservations, outerwear: null }, { ...verdict.slotObservations, coat: 'unknown garment' }]) {
    const events = [];
    const critic = createHostedVisualCritic({
      provider: 'nvidia', apiKey, resolveReference: () => asset(), resolveProduct: () => asset(),
      fetchImpl: async () => nvidiaResponse({ ...verdict, slotObservations }),
      onEvent: (event) => events.push(event),
    });
    assert.deepEqual(await critic([candidate]), []);
    assert.equal(events[0].reason, 'incomplete_visual_coverage');
  }
});

test('well-formed scores cannot pass with single-garment or contradictory outfit evidence', async () => {
  for (const relationships of [
    [{ ...verdict.relationships[0], slots: ['top'] }, ...verdict.relationships.slice(1)],
    [{ ...verdict.relationships[0], observation: 'No outerwear image was supplied.' }, ...verdict.relationships.slice(1)],
    [verdict.relationships[0], verdict.relationships[0], verdict.relationships[2]],
  ]) {
    const events = [];
    const critic = createHostedVisualCritic({
      provider: 'nvidia', apiKey, resolveReference: () => asset(), resolveProduct: () => asset(),
      fetchImpl: async () => nvidiaResponse({ ...verdict, relationships }),
      onEvent: (event) => events.push(event),
    });
    assert.deepEqual(await critic([candidate]), []);
    assert.equal(events[0].reason, 'incomplete_outfit_evidence');
  }
});
