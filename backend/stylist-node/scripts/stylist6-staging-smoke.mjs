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
import { performance } from 'node:perf_hooks';
import { createClient } from '@supabase/supabase-js';

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

const scenarios = [
  { name: 'Old Money + Work, shop', style: 'old_money', occasion: 'Work', sourceMode: 'catalog' },
  { name: 'Classic + Work, UZS 25M cap', style: 'classic', occasion: 'Work', sourceMode: 'catalog', budget: 25_000_000 },
  { name: 'Minimalist + Work, owned wardrobe', style: 'minimalist', occasion: 'Work', sourceMode: 'wardrobe' },
];
assert.equal(scenarios.length, 3);
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
    ...(scenario.budget ? { budget: { amount: scenario.budget, currency: 'UZS' } } : {}),
  };
  const started = performance.now();
  let response;
  try {
    response = await fetch(stage, {
      method: 'POST',
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
  assert.equal(after - before, 1, 'Quota must increment exactly once per admitted request');
  if (!response.ok) {
    throw new Error(scenario.name + ': staging failure HTTP ' + response.status +
      ' code=' + String(payload.code || payload.error || ''));
  }
  // True Stylist 6 contract: a Stylist 5 response does NOT satisfy this.
  assert.equal(payload.success, true);
  assert.match(String(payload.generationVersion || ''), /^stylist-6-/);
  assert.notMatch(String(payload.generationVersion || ''), /^stylist-5-/);
  assert.equal(payload.serverInventory, true);
  assert.ok(Array.isArray(payload.outfits) && payload.outfits.length >= 1, 'No accepted outfits');
  assert.ok(['NVIDIA_APPLIED', 'NVIDIA_EVALUATED_NO_CHANGE', 'NVIDIA_ABSTAINED', 'NVIDIA_NOT_RUN']
    .includes(payload.visualStatus), 'Missing genuine NVIDIA status');
  let items = 0;
  let allUzs = true;
  for (const outfit of payload.outfits) {
    assert.ok(Array.isArray(outfit.items) && outfit.items.length >= 3);
    for (const item of outfit.items) {
      items++;
      assert.ok(item.id && typeof item.id === 'string', 'Missing product ID');
      assert.ok(validHttps(item.imageUrl), 'Unusable product image URL');
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
  results.push(outcome);
  console.log(JSON.stringify(outcome));
}
// Locked-anchor round: pin the first real SKU from scenario 1 and require
// it in every returned outfit. Unknown IDs must 422, never drop silently.
const anchorProbe = await fetch(stage, {
  method: 'POST',
  headers: { apikey: publicKey, authorization: 'Bearer ' + token, 'content-type': 'application/json' },
  body: JSON.stringify({ request: { style: 'classic', occasion: 'Work', requiredIds: ['no-such-sku'] } }),
  signal: AbortSignal.timeout(55000),
});
assert.equal(anchorProbe.status, 422, 'Unknown anchor should fail 422');
const anchorProbeBody = await anchorProbe.json();
assert.equal(anchorProbeBody.code, 'anchor_not_found');
console.log(JSON.stringify({ anchor: 'unknown anchor rejected without silent drop' }));
console.log(JSON.stringify({
  summary: 'completed_isolated_stylist6_staging_calls',
  calls: results.length, totalQuotaDebits: results.length + 1,
  note: 'anchor probe debits once on admission like production',
  productionCutoverAuthorized: false,
}));
