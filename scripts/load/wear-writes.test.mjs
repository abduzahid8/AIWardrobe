import test from 'node:test';
import assert from 'node:assert/strict';
import { runWearWrites } from './wear-writes.mjs';
const users = [{ id: 'owner', accessToken: 'test-only', itemId: 'garment', logIds: ['log-1', 'log-2'] }];
const options = { baseUrl: 'https://example.invalid', apiKey: 'test-only', users, rps: 20, durationSec: .2 };
test('repeated requests reuse IDs and validate ordinary owner write confirmations', async () => {
  const sent = [];
  const result = await runWearWrites({ ...options, fetchImpl: async (_url, request) => {
    const row = JSON.parse(request.body); sent.push(row.id);
    assert.equal(request.headers.Authorization, 'Bearer test-only');
    return new Response(JSON.stringify([row]), { status: 201 });
  } });
  assert.equal(result.passed, true);
  assert.deepEqual(sent, ['log-1', 'log-1', 'log-2', 'log-2']);
});
for (const scenario of ['rate_limit', 'wrong_owner', 'invalid_json', 'transport']) test(`fails ${scenario}`, async () => {
  const result = await runWearWrites({ ...options, fetchImpl: async (_url, request) => {
    if (scenario === 'transport') throw new TypeError('private transport message');
    if (scenario === 'rate_limit') return new Response('{}', { status: 429 });
    if (scenario === 'invalid_json') return new Response('invalid', { status: 200 });
    return new Response(JSON.stringify([{ ...JSON.parse(request.body), user_id: 'other' }]), { status: 200 });
  } });
  assert.equal(result.passed, false); assert.equal(result.failed, 4);
  assert.equal(JSON.stringify(result).includes('private transport message'), false);
});
test('an aborted workload cannot pass', async () => {
  const controller = new AbortController(); controller.abort();
  const result = await runWearWrites({ ...options, signal: controller.signal });
  assert.equal(result.passed, false); assert.equal(result.completed, 0);
});
test('write failure callback stops scheduling further writes', async () => {
  const controller = new AbortController();
  let calls = 0;
  const result = await runWearWrites({ ...options, signal: controller.signal, onFailure: () => controller.abort(),
    fetchImpl: async () => { calls++; return new Response('{}', { status: 429 }); } });
  assert.equal(calls, 1);
  assert.equal(result.aborted, true);
  assert.equal(result.passed, false);
});

test('dropped scheduled writes prevent capacity success', async () => {
  const result = await runWearWrites({ ...options, users: [{ ...users[0], logIds: Array.from({ length: 10 }, (_, i) => `log-${i}`) }],
    rps: 200, durationSec: .05, maxInflight: 1, fetchImpl: async (_url, request) => {
      await new Promise(resolve => setTimeout(resolve, 30));
      return new Response(JSON.stringify([JSON.parse(request.body)]), { status: 201 });
    } });
  assert.equal(result.passed, false); assert.ok(result.dropped > 0);
});

test('transport diagnostics retain safe cause codes without headers or error messages', async () => {
  const result = await runWearWrites({ ...options, fetchImpl: async () => {
    throw new TypeError('private-token-and-url', { cause: { code: 'ECONNRESET', message: 'private-cause' } });
  } });
  assert.equal(result.diagnostics.errorTypes.TypeError, 4);
  const {startedAt,...detail}=result.failureDetails[0];
  assert.ok(Number.isFinite(Date.parse(startedAt)));
  assert.deepEqual(detail, { requestIndex: 0, stage: 'fetch', errorType: 'TypeError', causeCode: 'ECONNRESET' });
  assert.equal(JSON.stringify(result).includes('private-'), false);
  assert.equal(JSON.stringify(result).includes('test-only'), false);
});
