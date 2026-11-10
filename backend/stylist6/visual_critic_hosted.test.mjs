import test from 'node:test';
import assert from 'node:assert/strict';
import { createHostedVisualCritic, HOSTED_VISUAL_CRITIC_VERSION } from './visual_critic_hosted.mjs';

const KEY = 'test-api-key-0123456789abcdef';
const candidate = {
  id: 's6_abc123',
  briefId: 'rt_001',
  concept: 'Grey tailored utility',
  referenceId: 'rt_001',
  products: [
    { id: 'top-1', name: 'Ivory Mockneck', imageUrl: 'https://cdn.example.com/t.jpg' },
    { id: 'bot-1', name: 'Black Trousers', imageUrl: 'https://cdn.example.com/b.jpg' },
  ],
};

const jpeg = (fill) => {
  const bytes = Buffer.alloc(2048, fill);
  bytes[0] = 0xff; bytes[1] = 0xd8;
  return bytes;
};

const verdictJson = JSON.stringify({
  candidateId: 's6_abc123',
  visualCohesion: 0.9, styleFit: 0.88, occasionFit: 0.91, formalityFit: 0.93,
  confidence: 0.95, reject: false,
  evidence: ['mockneck echoes reference high neckline', 'charcoal palette stays tonal', 'trouser formality matches work register'],
});

function stubFetch({ verdict = verdictJson, imageStatus = 200, providerStatus = 200 } = {}) {
  const calls = [];
  const impl = async (url, init) => {
    calls.push({ url, body: init?.body });
    if (String(url).startsWith('https://cdn.example.com/')) {
      return { ok: imageStatus === 200, status: imageStatus, arrayBuffer: async () => jpeg(7) };
    }
    return {
      ok: providerStatus === 200, status: providerStatus,
      json: async () => ({ candidates: [{ content: { parts: [{ text: verdict }] } }] }),
    };
  };
  return { impl, calls };
}

test('factory refuses to build an unconfigured critic', () => {
  assert.throws(() => createHostedVisualCritic({}), /hosted_visual_critic_not_configured/);
  assert.throws(() => createHostedVisualCritic({ apiKey: 'short' }), /hosted_visual_critic_not_configured/);
  assert.equal(HOSTED_VISUAL_CRITIC_VERSION, 'hosted-visual-critic-v1');
});

test('recorded provider verdict parses to structured evidence', async () => {
  const { impl, calls } = stubFetch();
  const events = [];
  const critic = createHostedVisualCritic({
    apiKey: KEY,
    model: 'gemini-2.0-flash-test',
    fetchImpl: impl,
    onEvent: (e) => events.push(e),
    resolveReference: async (id) => {
      assert.equal(id, 'rt_001');
      return { base64: jpeg(9).toString('base64'), mimeType: 'image/jpeg', sha256: '0'.repeat(64) };
    },
  });
  const out = await critic([candidate], { style: 'old_money', occasion: 'Work' });
  assert.equal(out.length, 1);
  assert.equal(out[0].candidateId, 's6_abc123');
  assert.equal(out[0].evidence.length, 3);
  assert.match(out[0].criticVersion, /hosted-visual-critic-v1\/gemini-2\.0-flash-test/);
  assert.ok(out[0].latencyMs >= 0);
  // Prompt carried the reference plus both product photos to the provider.
  const providerCall = calls.find((c) => String(c.url).includes('generateContent'));
  const sent = JSON.parse(providerCall.body);
  const images = sent.contents[0].parts.filter((p) => p.inlineData);
  assert.equal(images.length, 3);
  assert.deepEqual(events, [{ candidateId: 's6_abc123', outcome: 'judged', latencyMs: events[0].latencyMs }]);
});

test('unparseable provider text abstains instead of scoring', async () => {
  const { impl } = stubFetch({ verdict: 'no json here' });
  const events = [];
  const critic = createHostedVisualCritic({
    apiKey: KEY, fetchImpl: impl, onEvent: (e) => events.push(e),
    resolveReference: async () => ({ base64: jpeg(1).toString('base64') }),
  });
  assert.deepEqual(await critic([candidate], {}), []);
  assert.equal(events[0].outcome, 'abstain');
});

test('candidate id mismatch abstains for that candidate only', async () => {
  const other = { ...candidate, id: 's6_other' };
  const { impl } = stubFetch();
  const critic = createHostedVisualCritic({
    apiKey: KEY, fetchImpl: impl,
    resolveReference: async () => ({ base64: jpeg(1).toString('base64') }),
  });
  assert.deepEqual(await critic([other], {}), []);
});

test('missing reference bytes abstain without calling the provider', async () => {
  const { impl, calls } = stubFetch();
  const critic = createHostedVisualCritic({
    apiKey: KEY, fetchImpl: impl, resolveReference: async () => null,
  });
  assert.deepEqual(await critic([candidate], {}), []);
  assert.ok(calls.every((c) => !String(c.url).includes('generateContent')));
});

test('unfetchable product photo abstains without a fake verdict', async () => {
  const { impl } = stubFetch({ imageStatus: 404 });
  const events = [];
  const critic = createHostedVisualCritic({
    apiKey: KEY, fetchImpl: impl, onEvent: (e) => events.push(e),
    resolveReference: async () => ({ base64: jpeg(1).toString('base64') }),
  });
  assert.deepEqual(await critic([candidate], {}), []);
  assert.match(events[0].reason, /image_http_404/);
});
