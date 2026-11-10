import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { signStylistTransport } from '../transport/signature.mjs';
import route from '../api/stylist6-vision-test.ts';

const SECRET = 'test-transport-secret-0123456789abcdef';
const KEY = 'test-nvidia-key-0123456789abcdef';
const AUTH = 'Bearer test-caller-token';

const jpeg = Buffer.from([0xff, 0xd8, 0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, ...new Array(2040).fill(7)]);
const REF_SHA = createHash('sha256').update(jpeg).digest('hex');

function body(over = {}) {
  return JSON.stringify({
    protocol: 'aiw-stylist6-vision-test-v1',
    confirmSpend: true,
    rightsConfirmed: true,
    maxCalls: 1,
    candidate: {
      id: 'oneshot-nvidia-1',
      briefId: 'rt_005',
      concept: 'Brown tonal layering',
      referenceId: 'rt_005',
      reference: { base64: jpeg.toString('base64'), mimeType: 'image/jpeg', sha256: REF_SHA },
      products: [
        { id: 'top-1', imageUrl: 'https://example.com/p/top-1.jpg', name: 'Mockneck' },
        { id: 'bot-1', imageUrl: 'https://example.com/p/bot-1.jpg', name: 'Trousers' },
      ],
    },
    request: { style: 'old_money', occasion: 'Work' },
    ...over,
  });
}

async function signedCall(text, { key = KEY, fetchImpl = null } = {}) {
  const prevKey = process.env.NVIDIA_API_KEY;
  const prevSecret = process.env.STYLIST_NODE_SHARED_SECRET;
  const prevFetch = globalThis.fetch;
  process.env.STYLIST_NODE_SHARED_SECRET = SECRET;
  if (key === null) delete process.env.NVIDIA_API_KEY;
  else process.env.NVIDIA_API_KEY = key;
  if (fetchImpl) globalThis.fetch = fetchImpl;
  try {
    const headers = await signStylistTransport({ secret: SECRET, body: text, authorization: AUTH });
    const response = await route.fetch(new Request('https://stylist.internal/api/stylist6-vision-test', {
      method: 'POST',
      headers: { ...headers, authorization: AUTH, 'content-type': 'application/json' },
      body: text,
    }));
    const payload = await response.json();
    return { status: response.status, payload };
  } finally {
    if (prevSecret === undefined) delete process.env.STYLIST_NODE_SHARED_SECRET;
    else process.env.STYLIST_NODE_SHARED_SECRET = prevSecret;
    if (prevKey === undefined) delete process.env.NVIDIA_API_KEY;
    else process.env.NVIDIA_API_KEY = prevKey;
    globalThis.fetch = prevFetch;
  }
}

test('GET is refused without touching auth or provider', async () => {
  const response = await route.fetch(new Request('https://stylist.internal/api/stylist6-vision-test'));
  assert.equal(response.status, 405);
});

test('unsigned POST is rejected before any provider traffic', async () => {
  let calls = 0;
  const counting = async () => { calls += 1; throw new Error('must not fetch'); };
  const prev = globalThis.fetch;
  const prevSecret = process.env.STYLIST_NODE_SHARED_SECRET;
  process.env.STYLIST_NODE_SHARED_SECRET = SECRET;
  globalThis.fetch = counting;
  try {
    const response = await route.fetch(new Request('https://stylist.internal/api/stylist6-vision-test', {
      method: 'POST',
      headers: { authorization: AUTH, 'content-type': 'application/json' },
      body: body(),
    }));
    assert.equal(response.status, 401);
    assert.equal(calls, 0);
  } finally {
    globalThis.fetch = prev;
    if (prevSecret === undefined) delete process.env.STYLIST_NODE_SHARED_SECRET;
    else process.env.STYLIST_NODE_SHARED_SECRET = prevSecret;
  }
});

test('signed but unapproved call is rejected with zero provider traffic', async () => {
  let calls = 0;
  const { status } = await signedCall(body({ confirmSpend: false }), {
    fetchImpl: (async () => { calls += 1; throw new Error('must not fetch'); }),
  });
  assert.equal(status, 403);
  assert.equal(calls, 0);
});

test('approved call without server key is rejected without provider traffic', async () => {
  let calls = 0;
  const { status, payload } = await signedCall(body(), {
    key: null,
    fetchImpl: (async () => { calls += 1; throw new Error('must not fetch'); }),
  });
  assert.equal(status, 503);
  assert.equal(payload.code, 'vision_unconfigured');
  assert.equal(calls, 0);
});

test('identical replay on the same instance is rejected without provider traffic', async () => {
  const prevSecret = process.env.STYLIST_NODE_SHARED_SECRET;
  const prevKey = process.env.NVIDIA_API_KEY;
  process.env.STYLIST_NODE_SHARED_SECRET = SECRET;
  delete process.env.NVIDIA_API_KEY;
  let calls = 0;
  const prev = globalThis.fetch;
  globalThis.fetch = (async () => { calls += 1; throw new Error('must not fetch'); });
  try {
    const text = body();
    const headers = await signStylistTransport({ secret: SECRET, body: text, authorization: AUTH });
    const once = async () => route.fetch(new Request('https://stylist.internal/api/stylist6-vision-test', {
      method: 'POST',
      headers: { ...headers, authorization: AUTH, 'content-type': 'application/json' },
      body: text,
    }));
    const first = await once();
    assert.equal(first.status, 503, 'first delivery reaches the key check');
    const second = await once();
    assert.equal(second.status, 409, 'same-instance replay is rejected');
    assert.equal((await second.json()).code, 'duplicate_replay');
    assert.equal(calls, 0);
  } finally {
    globalThis.fetch = prev;
    if (prevSecret === undefined) delete process.env.STYLIST_NODE_SHARED_SECRET;
    else process.env.STYLIST_NODE_SHARED_SECRET = prevSecret;
    if (prevKey === undefined) delete process.env.NVIDIA_API_KEY;
    else process.env.NVIDIA_API_KEY = prevKey;
  }
});

test('approved call returns one redacted NVIDIA verdict for exactly one provider POST', async () => {
  let providerCalls = 0;
  const seenModels = [];
  const fetchImpl = async (url, init) => {
    if (String(url).includes('integrate.api.nvidia.com')) {
      providerCalls += 1;
      const parsed = JSON.parse(init.body);
      seenModels.push(parsed.model);
      assert.equal(parsed.stream, false);
      const parts = parsed.messages[0].content;
      const images = parts.filter((p) => p.type === 'image_url');
      assert.equal(images.length, 3, 'reference + 2 product photos');
      assert.ok(images.every((p) => p.image_url.url.startsWith('data:image/')));
      const id = 'oneshot-nvidia-1';
      return {
        ok: true, status: 200,
        json: async () => ({
          choices: [{ message: { content: JSON.stringify({
            candidateId: id, visualCohesion: 0.9, styleFit: 0.88, occasionFit: 0.91,
            formalityFit: 0.93, confidence: 0.95, reject: false,
            evidence: ['tonal register holds', 'coat weight suits autumn', 'boot formality fits'],
          }) } }],
        }),
      };
    }
    return { ok: true, status: 200, arrayBuffer: async () => jpeg };
  };
  const { status, payload } = await signedCall(body(), { fetchImpl });
  assert.equal(status, 200);
  assert.equal(providerCalls, 1);
  assert.deepEqual(seenModels, ['nvidia/nemotron-3.5-super-vl-120b-a12b']);
  assert.equal(payload.ok, true);
  assert.match(payload.model, /hosted-visual-critic-v1\/nvidia\/nvidia\//);
  assert.equal(payload.verdict.candidateId, 'oneshot-nvidia-1');
  assert.equal(payload.verdict.evidence.length, 3);
  const serialized = JSON.stringify(payload);
  assert.ok(!serialized.includes(KEY), 'server key must never appear in the response');
  assert.ok(!serialized.includes(jpeg.toString('base64').slice(0, 64)), 'reference bytes must not be echoed');
});
