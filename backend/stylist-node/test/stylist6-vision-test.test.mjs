import test from 'node:test';
import assert from 'node:assert/strict';
import { signStylistTransport } from '../transport/signature.mjs';
import first from '../api/stylist6-vision-test.ts';

test('retired diagnostic refuses signed replays across separate module instances before reading the body', async () => {
  const { default: second } = await import('../api/stylist6-vision-test.ts?instance=second');
  const authorization = 'Bearer test-caller';
  const body = JSON.stringify({ candidate: { products: [{ imageUrl: 'https://127.0.0.1/private' }] } });
  const headers = await signStylistTransport({
    secret: 'test-transport-secret-0123456789abcdef', body, authorization,
  });
  const requests = Array.from({ length: 100 }, () => new Request('https://stylist.internal/api/stylist6-vision-test', {
    method: 'POST', headers: { ...headers, authorization }, body,
  }));
  const previous = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async () => { calls++; throw new Error('unexpected fetch'); };
  try {
    const responses = await Promise.all(requests.map((r, i) => (i % 2 ? first : second).fetch(r)));
    assert.ok(responses.every((r) => r.status === 410));
    assert.ok(requests.every((r) => !r.bodyUsed));
    assert.equal(calls, 0);
  } finally {
    globalThis.fetch = previous;
  }
});

test('GET is method-not-allowed', () => {
  assert.equal(first.fetch(new Request('https://stylist.internal/api/stylist6-vision-test')).status, 405);
});
