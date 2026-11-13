import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createHostedVisualCritic, NVIDIA_VISION_MODEL, validVisualVerdict } from './visual_critic_hosted.mjs';

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

test('unconfigured owner reference bucket uses product-only critique without fake reference claims',async()=>{
  const captures=[];
  const critic=createHostedVisualCritic({
    provider:'nvidia',apiKey,
    referenceMediaConfigured:false,
    resolveReference:()=>{throw new Error('must not fetch absent owner reference bucket')},
    resolveProduct:()=>asset(),
    fetchImpl:async(url, init)=>{
      captures.push(JSON.parse(init.body));
      return nvidiaResponse(referencelessVerdict(candidate.id));
    },
  });
  const result=await critic([candidate],{style:'classic',occasion:'Work'});
  assert.equal(result.length,1);
  assert.equal(result[0].referenceless,true);
  assert.equal(result[0].referenceFidelity,undefined);
  assert.equal(captures.length,1);
  const images=captures[0].messages[0].content.filter(item=>item.type==='image_url');
  assert.equal(images.length,4,'products only; no missing reference slot');
  assert.match(captures[0].messages[0].content[0].text,/No reference image is supplied/);
});

test('configured owner reference bucket failure never silently downgrades to product-only',async()=>{
  let calls=0;const events=[];
  const critic=createHostedVisualCritic({
    provider:'nvidia',apiKey,referenceMediaConfigured:true,
    resolveReference:()=>{throw new Error('staged_owner_reference_missing')},
    resolveProduct:()=>asset(),
    onEvent:e=>events.push(e),
    fetchImpl:async()=>{calls++;throw Error('must_not_call_provider')},
  });
  assert.deepEqual(await critic([candidate]),[]);
  assert.equal(calls,0);
  assert.equal(events[0].reason,'reference_unresolved');
  assert.equal(events[0].dispatched,false);
  assert.throws(()=>createHostedVisualCritic({
    apiKey,resolveReference:()=>asset(),resolveProduct:()=>asset(),
    referenceMediaConfigured:'false',
  }),/invalid_reference_media_configuration/);
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
    const [{ mediaMs, queueMs, ...event }] = events;
    assert.ok(mediaMs >= 0 && queueMs >= 0);
    assert.deepEqual(event, { candidateId: candidate.id, outcome: 'abstain', dispatched: false, reason: 'product_image_unresolved' });
  }
});

test('invalid, mismatched, truncated and oversized provider replies abstain without retry', async () => {
  for (const respond of [
    () => new Response('bad JSON'),
    () => new Response('x'.repeat(65537)),
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

test('a transient 5xx or dropped connection retries exactly once with the same provider deadline', async () => {
  for (const [index, firstFailure] of [
    [0, () => new Response(null, { status: 503 })],
    [1, () => { throw new TypeError('terminated'); }],
  ]) {
    const events = [];
    let calls = 0;
    // Unique bytes per iteration: a retry that succeeds judges (and caches)
    // a verdict, so each case needs its own cache key.
    const critic = createHostedVisualCritic({
      provider: 'nvidia', apiKey,
      resolveReference: () => asset(100 + index * 2),
      resolveProduct: () => asset(101 + index * 2),
      fetchImpl: async () => {
        calls++;
        return calls === 1 ? firstFailure() : nvidiaResponse(verdict);
      },
      onEvent: (event) => events.push(event),
    });
    const [judged] = await critic([candidate]);
    assert.equal(calls, 2);
    assert.equal(judged.candidateId, candidate.id);
    assert.equal(events[0].retried, true);
    assert.equal(events[0].outcome, 'judged');
  }
});

test('a 5xx that fails twice abstains with the retry recorded', async () => {
  const events = [];
  let calls = 0;
  const critic = createHostedVisualCritic({
    provider: 'nvidia', apiKey, resolveReference: () => asset(), resolveProduct: () => asset(),
    fetchImpl: async () => { calls++; return new Response(null, { status: 502 }); },
    onEvent: (event) => events.push(event),
  });
  assert.deepEqual(await critic([candidate]), []);
  assert.equal(calls, 2);
  assert.equal(events[0].retried, true);
  assert.equal(events[0].outcome, 'abstain');
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
  assert.equal(events.filter(({ reason }) => reason === 'batch_deadline_in_flight').length, 2);
  assert.equal(events.filter(({ reason }) => reason === 'batch_deadline_queued').length, 2);
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
  assert.equal(events[0].reason, 'batch_deadline_media');
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

test('separate provider deadline releases a worker with no retry and every candidate traced', async () => {
  const calls = [], events = [];
  // Distinct image bytes per candidate: the in-process verdict cache must
  // not mask the per-candidate dispatch tracing this test proves.
  const payload = Array.from({ length: 4 }, (_, i) => ({
    ...candidate, id: 'look-' + i,
    products: candidate.products.map((p) => ({ ...p, id: `${p.slot}-${i}` })),
  }));
  const critic = createHostedVisualCritic({
    provider: 'nvidia', apiKey, timeoutMs: 250, providerTimeoutMs: 25,
    resolveReference: () => asset(9), resolveProduct: ({ id }) => asset(10 + Number(id.split('-')[1])),
    fetchImpl: async (url, { body }) => {
      const id = JSON.parse(body).messages[0].content[0].text.match(/exactly "(look-\d)"/)[1];
      calls.push(id);
      return id === 'look-0' ? new Promise(() => {}) : nvidiaResponse({ ...verdict, candidateId: id });
    },
    onEvent: (event) => events.push(event),
  });
  assert.equal((await critic(payload)).length, 3);
  assert.equal(new Set(calls).size, 4);
  assert.equal(events.length, 4);
  const slow = events.find(({ candidateId }) => candidateId === 'look-0');
  assert.equal(slow.reason, 'provider_deadline');
  assert.ok(slow.providerMs >= 0 && slow.mediaMs >= 0 && slow.queueMs >= 0);
  assert.ok(events.filter(({ outcome }) => outcome === 'judged').every(({ headersMs, parseMs, validationMs }) =>
    headersMs >= 0 && parseMs >= 0 && validationMs >= 0));
});

test('caller cancellation labels queued candidates and aborts active calls', async () => {
  const controller = new AbortController(), events = [];
  let calls = 0;
  const critic = createHostedVisualCritic({
    provider: 'nvidia', apiKey, signal: controller.signal,
    resolveReference: () => asset(19),
    resolveProduct: ({ id }) => asset(20 + Number(id.split('-')[1])),
    fetchImpl: async () => {
      if (++calls === 2) controller.abort();
      return new Promise(() => {});
    },
    onEvent: (event) => events.push(event),
  });
  await critic(Array.from({ length: 4 }, (_, i) => ({
    ...candidate, id: 'cancel-' + i,
    products: candidate.products.map((p) => ({ ...p, id: `${p.slot}-${i}` })),
  })));
  assert.equal(calls, 2);
  assert.equal(events.length, 4);
  assert.ok(events.every(({ reason }) => reason === 'cancelled'));
});

test('verified exact reference photos cannot be counted as model-observed differences', async () => {
  const ref = asset(1);
  let photo = asset(2);
  const events = [];
  let content, result = verdict;
  const critic = createHostedVisualCritic({
    provider: 'nvidia', apiKey,
    resolveReference: () => ({ ...ref, components: { referenceSha256: ref.sha256, slotSha256: { outerwear: photo.sha256 } } }),
    resolveProduct: () => photo,
    fetchImpl: async (url, { body }) => {
      content = JSON.parse(body).messages[0].content;
      return nvidiaResponse(result);
    },
    onEvent: (event) => events.push(event),
  });
  const [judged] = await critic([candidate]);
  assert.deepEqual(judged.verifiedSharedSlots, ['outerwear']);
  assert.match(content[0].text, /"verifiedIdenticalReferenceSlots":\["outerwear"\]/);
  assert.doesNotMatch(content[0].text, new RegExp(photo.sha256));
  // Fresh bytes for the second phase so the verdict cache (keyed on image
  // digests) does not replay the first phase's judged verdict.
  photo = asset(22);
  result = { ...verdict, relationships: verdict.relationships.map((link) =>
    link.kind === 'reference' ? { ...link, slots: ['outerwear'], observation: 'The candidate adds trim absent from the reference.' } : link) };
  assert.deepEqual(await critic([candidate]), []);
  assert.equal(events.at(-1).reason, 'incomplete_outfit_evidence');
});

test('an unbound reference manifest cannot declare any identical slots', async () => {
  const ref = asset(31), photo = asset(32);
  const critic = createHostedVisualCritic({
    provider: 'nvidia', apiKey,
    resolveReference: () => ({ ...ref, components: { referenceSha256: photo.sha256, slotSha256: { top: photo.sha256 } } }),
    resolveProduct: () => photo,
    fetchImpl: async () => nvidiaResponse(verdict),
  });
  const [judged] = await critic([candidate]);
  assert.deepEqual(judged.verifiedSharedSlots, []);
});

test('reference-only outerwear is an outfit difference even when all three candidate photos match', async () => {
  const photo = asset(2), ref = asset(1);
  const three = { ...candidate, products: candidate.products.filter(({ slot }) => slot !== 'outerwear') };
  const { outerwear, ...slotObservations } = verdict.slotObservations;
  const links = [
    { kind: 'outfit', slots: ['top', 'bottom'], observation: 'Light top contrasts with dark trousers.' },
    verdict.relationships[1],
    { kind: 'reference', slots: [], observation: 'These use the same three photographs.' },
  ];
  let payload = { ...verdict, slotObservations, relationships: links };
  const critic = createHostedVisualCritic({
    provider: 'nvidia', apiKey,
    resolveReference: () => ({ ...ref, components: {
      referenceSha256: ref.sha256,
      slotSha256: { top: photo.sha256, bottom: photo.sha256, shoes: photo.sha256, outerwear: asset(3).sha256 },
    } }),
    resolveProduct: () => photo,
    fetchImpl: async () => nvidiaResponse(payload),
  });
  assert.deepEqual(await critic([three]), []);
  payload = { ...payload, relationships: [...links.slice(0, 2),
    { kind: 'reference', slots: ['outerwear'], observation: 'Omitting the long coat makes the outfit less layered.' }] };
  const [judged] = await critic([three]);
  assert.deepEqual(judged.verifiedSharedSlots, ['top', 'bottom', 'shoes']);
  assert.deepEqual(judged.verifiedReferenceOnlySlots, ['outerwear']);
  assert.match(judged.evidence.at(-1), /Omitting the long coat/);
});

test('access denial stops later scheduling while recording every candidate', async () => {
  for (const [status, refFill, photoFill] of [[401, 41, 42], [403, 43, 44]]) {
    const events = [];
    let calls = 0;
    const critic = createHostedVisualCritic({
      provider: 'nvidia', apiKey, resolveReference: () => asset(refFill), resolveProduct: () => asset(photoFill),
      fetchImpl: async () => { calls++; return new Response(null, { status }); },
      onEvent: (event) => events.push(event),
    });
    assert.deepEqual(await critic(Array.from({ length: 4 }, (_, i) => ({ ...candidate, id: 'deny-' + i }))), []);
    assert.ok(calls >= 1 && calls <= 2);
    assert.equal(events.length, 4);
    assert.equal(events.filter(({ dispatched }) => dispatched).length, calls);
    assert.ok(events.every(({ reason }) => reason === 'provider_access_denied'));
  }
});

const referencelessVerdict = (id) => ({
  slotObservations: { top: 'light striped shirt', outerwear: 'short dark jacket', bottom: 'grey straight trousers', shoes: 'dark round toe shoes' },
  candidateId: id, visualCohesion: 0.9, styleFit: 0.88, occasionFit: 0.91, formalityFit: 0.93,
  confidence: 0.95, reject: false,
  relationships: [
    { kind: 'outfit', slots: ['top', 'outerwear'], observation: 'mock top and layer match' },
    { kind: 'outfit', slots: ['bottom', 'shoes'], observation: 'mock footwear repeats bottom color' },
    { kind: 'outfit', slots: ['top', 'shoes'], observation: 'mock belt line echoes the shoe tone' },
  ],
});

test('a candidate without a reference id is judged purely on its product images', async () => {
  const events = [];
  const photos = candidate.products.map((_, i) => asset(60 + i));
  let sent;
  const critic = createHostedVisualCritic({
    provider: 'nvidia', apiKey,
    resolveReference: () => { throw new Error('reference resolver must never run without a reference id'); },
    resolveProduct: ({ slot }) => photos[candidate.products.findIndex((p) => p.slot === slot)],
    fetchImpl: async (url, init) => {
      sent = JSON.parse(init.body);
      return nvidiaResponse(referencelessVerdict(candidate.id));
    },
    onEvent: (event) => events.push(event),
  });
  const [judged] = await critic([{ ...candidate, referenceId: undefined }], { style: 'casual', occasion: 'Work' });
  assert.equal(judged.referenceless, true);
  assert.equal(judged.referenceFidelity, undefined);
  assert.ok(validVisualVerdict(judged, candidate.id));
  assert.equal(sent.messages[0].content.filter((part) => part.type === 'image_url').length, 4);
  assert.match(sent.messages[0].content[0].text, /No reference image is supplied/);
  assert.doesNotMatch(sent.messages[0].content[0].text, /Reference outfit image/);
  assert.deepEqual(events[0].images, 4);
  assert.equal(events[0].outcome, 'judged');
  // Every relationship is outfit-internal: no reference comparison exists.
  assert.ok(judged.relationships.every((link) => link.kind === 'outfit'));
  assert.ok(judged.evidence.every((line) => !/\breferences?\b/i.test(line)));
});

test('reference-less verdicts never claim a reference comparison', async () => {
  const photos = candidate.products.map((_, i) => asset(70 + i));
  for (const badRow of [
    { ...referencelessVerdict(candidate.id), relationships: [
        { kind: 'outfit', slots: ['top', 'bottom'], observation: 'mock top and trouser match' },
        { kind: 'outfit', slots: ['bottom', 'shoes'], observation: 'mock footwear repeats bottom color' },
        { kind: 'reference', slots: ['top'], observation: 'palette follows the reference' },
      ] },
    { ...referencelessVerdict(candidate.id), relationships: [
        { kind: 'outfit', slots: ['top', 'outerwear'], observation: 'mock top and layer match' },
        { kind: 'outfit', slots: ['bottom', 'shoes'], observation: 'mock footwear repeats bottom color' },
        { kind: 'outfit', slots: ['top', 'shoes'], observation: 'the shoe tone mirrors the reference palette' },
      ] },
  ]) {
    const critic = createHostedVisualCritic({
      provider: 'nvidia', apiKey,
      resolveReference: () => { throw new Error('never called'); },
      resolveProduct: ({ slot }) => photos[candidate.products.findIndex((p) => p.slot === slot)],
      fetchImpl: async () => nvidiaResponse(badRow),
    });
    assert.deepEqual(await critic([{ ...candidate, referenceId: null }]), []);
  }
  // An unsolicited referenceFidelity score from the model is dropped: the
  // reference-less verdict carries no reference dimension at all.
  const unsolicited = createHostedVisualCritic({
    provider: 'nvidia', apiKey,
    resolveReference: () => { throw new Error('never called'); },
    resolveProduct: ({ slot }) => photos[candidate.products.findIndex((p) => p.slot === slot)],
    fetchImpl: async () => nvidiaResponse({ ...referencelessVerdict(candidate.id), referenceFidelity: 0.9 }),
  });
  const [row] = await unsolicited([{ ...candidate, referenceId: null }]);
  assert.equal(row.referenceless, true);
  assert.equal(row.referenceFidelity, undefined);
});

test('validVisualVerdict accepts reference-less rows and still requires referenceFidelity otherwise', () => {
  const row = { ...referencelessVerdict(candidate.id), referenceless: true, evidence: referencelessVerdict(candidate.id).relationships.map((l) => l.observation) };
  delete row.relationships;
  assert.equal(validVisualVerdict(row, candidate.id), true);
  assert.equal(validVisualVerdict({ ...row, referenceless: false }, candidate.id), false);
  assert.equal(validVisualVerdict({ ...row, referenceFidelity: 0.9, referenceless: false }, candidate.id), true);
});

test('the verdict cache reuses a judged look without another provider dispatch', async () => {
  const events = [];
  const photos = candidate.products.map((_, i) => asset(80 + i));
  let calls = 0;
  const critic = createHostedVisualCritic({
    provider: 'nvidia', apiKey,
    resolveReference: () => asset(89),
    resolveProduct: ({ slot }) => photos[candidate.products.findIndex((p) => p.slot === slot)],
    fetchImpl: async () => { calls++; return nvidiaResponse({ ...verdict, candidateId: 'fresh-id' }); },
    onEvent: (event) => events.push(event),
  });
  const other = { ...candidate, id: 'fresh-id' };
  const [first] = await critic([other]);
  assert.equal(calls, 1);
  const [second] = await critic([other]);
  assert.equal(calls, 1);
  assert.equal(second.candidateId, 'fresh-id');
  assert.deepEqual(second.evidence, first.evidence);
  assert.equal(second.latencyMs <= first.latencyMs + 5, true);
  const hit = events.find(({ cacheHit }) => cacheHit === true);
  assert.equal(hit.outcome, 'judged');
  assert.equal(hit.dispatched, false);
  // A changed reference id or changed image bytes never hits the cache.
  const changedReference = { ...candidate, id: 'fresh-id', referenceId: 'rt_002' };
  await critic([changedReference]);
  assert.equal(calls, 2);
});

test('abstentions and failures are never cached; only valid judged verdicts are', async () => {
  let calls = 0;
  const critic = createHostedVisualCritic({
    provider: 'nvidia', apiKey,
    resolveReference: () => asset(91), resolveProduct: () => asset(92),
    fetchImpl: async () => { calls++; return nvidiaResponse({ ...verdict, candidateId: 'other' }); },
  });
  assert.deepEqual(await critic([candidate]), []);
  assert.equal(calls, 1);
  assert.deepEqual(await critic([candidate]), []);
  assert.equal(calls, 2, 'an abstained candidate must not poison the verdict cache');
});

