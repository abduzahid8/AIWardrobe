import test from 'node:test';
import assert from 'node:assert/strict';
import health from '../api/health.ts';

const required = ['STYLIST_NODE_SHARED_SECRET', 'SUPABASE_URL', 'SUPABASE_ANON_KEY', 'SUPABASE_SERVICE_ROLE_KEY', 'NVIDIA_API_KEY'];

test('liveness is not mistaken for compute readiness without backend secrets', async () => {
  const saved = Object.fromEntries(required.map(k => [k, process.env[k]]));
  try {
    for (const key of required) delete process.env[key];
    const r = health.fetch();
    assert.equal(r.status, 200);
    assert.deepEqual(await r.json(), { status: 'ok', computeConfigured: false });
    assert.equal(r.headers.get('Cache-Control'), 'no-store');
  } finally {
    for (const key of required) saved[key] === undefined ? delete process.env[key] : (process.env[key] = saved[key]);
  }
});

test('readiness becomes true only when all required server env are provided', async () => {
  const saved = Object.fromEntries(required.map(k => [k, process.env[k]]));
  try {
    for (const key of required) process.env[key] = 'test-preview-only-key-value';
    process.env.STYLIST_NODE_SHARED_SECRET = 'test-preview-only-shared-secret-more-than-32-chars';
    const r = health.fetch();
    assert.deepEqual(await r.json(), { status: 'ok', computeConfigured: true });
  } finally {
    for (const key of required) saved[key] === undefined ? delete process.env[key] : (process.env[key] = saved[key]);
  }
});
