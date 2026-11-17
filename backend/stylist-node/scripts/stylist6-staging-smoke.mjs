/**
 * Opt-in, local-only Stylist 6 staging acceptance check. Never schedule in push CI.
 * Requires an authorized dedicated paid TEST account and read-only snapshots
 * via a server-side Supabase service key. Nothing sensitive is printed.
 * Calls ONLY the isolated aiw-stylist6-node-staging Edge Function (not prod,
 * not the Stylist 5 staging proxy).
 *
 * Unlike the Stylist 5 smoke, this asserts an actual `stylist-6-*` engine
 * version, server-side inventory, locked anchors, UZS budget enforcement,
 * honest currencies and the genuine NVIDIA status vocabulary. A Stylist 5
 * response does NOT satisfy these assertions.
 *
 * Quota is consumed on admission, including any later planning/provider
 * failure. Read backend/stylist-node/README.md before using.
 */
import assert from 'node:assert/strict';
import { open, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { canonicalRequestFingerprint } from '../../stylist6/cloud_handler.mjs';
import { performance } from 'node:perf_hooks';
import { createClient } from '@supabase/supabase-js';

if (process.argv.slice(2).join(' ') !== '--single-request') {
  throw new Error('Refusing metered requests: only --single-request is supported; the old three-scenario run is disabled.');
}
const stableRequestId = process.env.AIW_S6_REQUEST_ID || '';
assert.match(stableRequestId, /^s6-smoke-[A-Za-z0-9_-]{8,110}$/, 'Set one stable AIW_S6_REQUEST_ID; never change it after dispatch');
const confirmation = 'I_ACCEPT_S6_QUOTA_DEBITS';
if (process.env.AIW_S6_STAGING_SMOKE_ACK !== confirmation) {
  throw new Error('Refusing metered requests. Explicit AIW_S6_STAGING_SMOKE_ACK required.');
}
const required = ['SUPABASE_URL', 'SUPABASE_ANON_KEY', 'AIW_TEST_SERVICE_ROLE_KEY'];
for (const key of required) {
  if (!process.env[key]) throw new Error('Missing required local-only configuration: ' + key);
}
const origin = new URL(process.env.SUPABASE_URL);
assert.equal(origin.protocol, 'https:', 'Require HTTPS Supabase');
assert.equal(origin.hostname, 'fyqpifmrsftsfqibhwhy.supabase.co', 'Refusing unexpected project');
const stage = new URL('/functions/v1/aiw-stylist6-node-staging', origin).toString();
const publicKey = process.env.SUPABASE_ANON_KEY;
const adminKey = process.env.AIW_TEST_SERVICE_ROLE_KEY;
const authClient = createClient(origin.origin, publicKey, {
  auth: { persistSession: false, autoRefreshToken: false },
});
const admin = createClient(origin.origin, adminKey, {
  auth: { persistSession: false, autoRefreshToken: false },
});
let token = process.env.AIW_TEST_ACCESS_TOKEN || '';
if (!token) {
  const email = process.env.AIW_TEST_EMAIL;
  const password = process.env.AIW_TEST_PASSWORD;
  if (!email || !password) throw new Error('Use local AIW_TEST_ACCESS_TOKEN or AIW_TEST_EMAIL/PASSWORD');
  const { data, error } = await authClient.auth.signInWithPassword({ email, password });
  if (error || !data?.session?.access_token) throw new Error('Dedicated test account login failed');
  token = data.session.access_token;
}
const { data: identity, error: identityError } = await authClient.auth.getUser(token);
if (identityError || !identity?.user?.id) throw new Error('Supabase did not validate test JWT');
const userId = identity.user.id;
assert.equal(userId, process.env.AIW_TEST_USER_ID, 'Explicit dedicated test user ID must match validated login');
const { data: tier, error: tierError } = await admin.rpc('effective_subscription_tier', { p_user: userId });
assert.equal(tierError, null, 'Unable to verify legitimate server entitlements');
assert.ok(['lite', 'premium', 'vip'].includes(tier), 'No legitimate outfit-generation quota; do not change entitlements');
const feature = 'edge:aiOutfits';
const day = () => new Date().toISOString().slice(0, 10);
async function usedFor(date) {
  const { data, error } = await admin.from('usage_counters')
    .select('used').eq('user_id', userId).eq('feature', feature)
    .eq('day', date).maybeSingle();
  if (error) throw new Error('Unable to read isolated quota counter for test user');
  return data?.used ?? 0;
}
const first = { date: day(), used: await usedFor(day()) };
// Negative admission probes: rejected without quota or upstream work.
const negative = await fetch(stage, {
  method: 'POST',
  headers: { apikey: publicKey, 'content-type': 'application/json' },
  body: JSON.stringify({ request: { style: 'classic', occasion: 'Work' } }),
  signal: AbortSignal.timeout(15000),
});
assert.equal(negative.status, 401, 'Anonymous POST should fail 401');
assert.equal(await usedFor(first.date), first.used, 'Anonymous probe unexpectedly changed quota');
const oversize = await fetch(stage, {
  method: 'POST',
  headers: { apikey: publicKey, authorization: 'Bearer ' + token, 'content-type': 'application/json' },
  body: JSON.stringify({ request: { style: 'classic', occasion: 'Work' }, pad: 'x'.repeat(1024 * 1024 + 1) }),
  signal: AbortSignal.timeout(15000),
});
assert.ok([401, 413].includes(oversize.status), 'Oversize body should fail closed');
console.log('negative_admission_verified: anon=401 quota_delta=0');

// One scenario, one paid dispatch. No three-scenario mode or automatic retry.
const scenarios = [
  { name: 'Classic + Work, UZS 25M cap', style: 'classic', occasion: 'Work', sourceMode: 'catalog', budget: 25_000_000 },
];
assert.equal(scenarios.length, 1);
// Valid-user, invalid-anchor admission proves gateway + protection + HMAC + Auth,
// engine identity and real inventory before the single paid request.
const probe = await fetch(stage, {
  method: 'POST', redirect: 'error',
  headers: { apikey: publicKey, authorization: 'Bearer ' + token, 'content-type': 'application/json' },
  body: JSON.stringify({ request: { style: 'classic', occasion: 'Work', sourceMode: 'catalog', requiredIds: ['s6-nonexistent-admission-probe'] } }),
  signal: AbortSignal.timeout(55000),
});
const probeBody = await probe.json();
assert.equal(probe.status, 422, 'Protected authenticated admission must reject unknown anchor pre-debit');
assert.equal(probeBody.generationVersion, 'stylist-6-cloud-staging-1');
assert.equal(probeBody.code, 'anchor_not_found');
assert.equal(await usedFor(first.date), first.used, 'Preflight changed quota');
console.log('protected_authenticated_admission_verified: quota_delta=0');
const validHttps = (value) => {
  try { const u = new URL(value); return u.protocol === 'https:' && !!u.hostname; } catch { return false; }
};
const results = [];
for (let index = 0; index < scenarios.length; index++) {
  const scenario = scenarios[index];
  const before = await usedFor(first.date);
  const request = {
    style: scenario.style,
    occasion: scenario.occasion,
    brief: `${scenario.occasion}, ${scenario.style}`,
    sourceMode: scenario.sourceMode,
    preferences: {},
    recentSignatures: [],
    excludeItemIds: [],
    requiredIds: [],
    generationRequestId: stableRequestId,
    enableVisual: false,
    ...(scenario.budget ? { budget: { amount: scenario.budget, currency: 'UZS' } } : {}),
  };
  assert.equal(day(), first.date, 'UTC day crossed; stop before dispatch');
  const { data: prior, error: priorError } = await admin.from('stylist6_request_claims')
    .select('request_id').eq('user_id', userId).eq('request_id', stableRequestId).maybeSingle();
  assert.equal(priorError, null, 'Durable idempotency storage must be readable before dispatch');
  assert.equal(prior, null, 'Request already claimed; reconcile it, never generate again');
  const ledgerPath = '/tmp/aiw-s6-single-' + createHash('sha256').update(userId + ':' + stableRequestId).digest('hex') + '.json';
  // Exclusive creation is the local one-shot fuse. Keep it even on timeout or failure.
  const ledger = await open(ledgerPath, 'wx', 0o600);
  try { await ledger.writeFile(JSON.stringify({ requestId: stableRequestId, date: first.date, initialUsed: before, state: 'dispatch_started' })); }
  finally { await ledger.close(); }
  const started = performance.now();
  let response;
  try {
    response = await fetch(stage, {
      method: 'POST', redirect: 'error',
      headers: {
        apikey: publicKey,
        authorization: 'Bearer ' + token,
        'content-type': 'application/json',
      },
      body: JSON.stringify({ request }),
      signal: AbortSignal.timeout(55000),
    });
  } catch {
    throw new Error('Staging transport failed or timed out; no automatic retry (avoids extra quota charge)');
  }
  const clientWallMs = Math.round(performance.now() - started);
  let payload;
  try { payload = await response.json(); } catch { throw new Error('Staging did not return JSON'); }
  const after = await usedFor(first.date);
  await writeFile(ledgerPath, JSON.stringify({ requestId: stableRequestId, date: first.date, initialUsed: before, finalUsed: after, status: response.status, payload }), { mode: 0o600 });
  assert.equal(day(), first.date, 'UTC day crossed; reconcile both counter days');
  assert.equal(after - before, 1, 'Quota must increment exactly once per admitted request');
  if (!response.ok) {
    throw new Error(scenario.name + ': staging failure HTTP ' + response.status +
      ' code=' + String(payload.code || payload.error || ''));
  }
  // True Stylist 6 contract: a Stylist 5 response does NOT satisfy this.
  assert.equal(payload.success, true);
  assert.equal(payload.generationVersion, 'stylist-6-cloud-staging-1');
  assert.doesNotMatch(String(payload.generationVersion || ''), /^stylist-5-/);
  assert.equal(payload.serverInventory, true);
  assert.ok(Array.isArray(payload.outfits) && payload.outfits.length >= 1, 'No accepted outfits');
  assert.ok(['NVIDIA_APPLIED', 'NVIDIA_EVALUATED_NO_CHANGE', 'NVIDIA_ABSTAINED', 'NVIDIA_NOT_RUN']
    .includes(payload.visualStatus), 'Missing genuine NVIDIA status');
  assert.equal(payload.source, 'stylist-6-cloud-staging');
  assert.equal(payload.visualStatus, 'NVIDIA_NOT_RUN', 'No inference was authorized for this smoke');
  let items = 0;
  let allUzs = true;
  for (const outfit of payload.outfits) {
    assert.ok(Array.isArray(outfit.items) && outfit.items.length >= 3);
    for (const item of outfit.items) {
      items++;
      assert.ok(item.id && typeof item.id === 'string', 'Missing product ID');
      assert.ok(validHttps(item.imageUrl), 'Unusable product image URL');
      assert.ok(validHttps(item.shopUrl), 'Unusable product URL');
      assert.ok(Number.isFinite(Number(item.price)) && Number(item.price) > 0, 'Missing positive catalog price');
      assert.match(String(item.currency), /^[A-Z]{3}$/);
      assert.equal(item.currency, 'UZS', 'Budgeted catalog result must use budget currency');
      if (scenario.sourceMode === 'catalog') assert.equal(item.isShopItem, true, 'Shop request leaked wardrobe inventory');
      if (scenario.sourceMode === 'wardrobe') assert.equal(item.isShopItem, false, 'Wardrobe request leaked shop inventory');
      if (item.currency !== 'UZS') allUzs = false;
    }
    if (scenario.budget && allUzs) {
      const subtotal = outfit.items
        .filter((item) => item.currency === 'UZS' && Number.isFinite(Number(item.price)))
        .reduce((sum, item) => sum + Number(item.price), 0);
      assert.ok(subtotal <= scenario.budget, 'Hard UZS budget exceeded: ' + subtotal);
    }
  }
  const outcome = {
    index: index + 1, scenario: scenario.name, status: response.status,
    generationVersion: payload.generationVersion, outfits: payload.outfits.length,
    items, visualStatus: payload.visualStatus, clientWallMs, quotaDelta: 1,
  };
  const { data: claim, error: claimError } = await admin.from('stylist6_request_claims')
    .select('request_fingerprint,status_code,outcome_code').eq('user_id', userId).eq('request_id', stableRequestId).single();
  assert.equal(claimError, null, 'Must verify durable completed claim before replay');
  assert.equal(claim.request_fingerprint, canonicalRequestFingerprint(request));
  assert.equal(claim.status_code, 200);
  // Explicit post-success replay only: identical bytes/ID with a verified claim.
  // This is never run after timeout, failure, or ambiguous first outcome.
  if (process.env.AIW_S6_VERIFY_REPLAY === '1') {
    const replay = await fetch(stage, { method: 'POST', redirect: 'error',
      headers: { apikey: publicKey, authorization: 'Bearer ' + token, 'content-type': 'application/json' },
      body: JSON.stringify({ request }), signal: AbortSignal.timeout(55000) });
    const replayBody = await replay.json();
    assert.equal(replay.status, 409);
    assert.equal(replayBody.code, 'duplicate_request');
    assert.equal(replayBody.replayProtection, 'durable');
    assert.equal(await usedFor(first.date), after, 'Replay must not debit');
    outcome.replayQuotaDelta = 0;
  }
  results.push(outcome);
  console.log(JSON.stringify(outcome));
}
console.log(JSON.stringify({
  summary: 'completed_isolated_stylist6_staging_calls',
  calls: results.length, totalQuotaDebits: results.length,
  productionCutoverAuthorized: false,
}));
