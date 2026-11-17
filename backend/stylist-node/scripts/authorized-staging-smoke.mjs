/**
 * Opt-in, local-only paid staging acceptance check. Never schedule in push CI.
 * Requires an authorized dedicated paid TEST account and read-only snapshots
 * via a server-side Supabase service key. Nothing sensitive is printed.
 * Calls ONLY the isolated aiw-stylist-node-staging Edge Function (not prod).
 *
 * Read backend/stylist-node/README.md before using. Quota is consumed on
 * admission, including any later planning/provider failure.
 */
import assert from 'node:assert/strict';
import { performance } from 'node:perf_hooks';
import { createClient } from '@supabase/supabase-js';

const confirmation = 'I_ACCEPT_5_QUOTA_DEBITS';
if (process.env.AIW_STAGING_SMOKE_ACK !== confirmation) {
  throw new Error('Refusing metered requests. Explicit AIW_STAGING_SMOKE_ACK required.');
}
const required = ['SUPABASE_URL', 'SUPABASE_ANON_KEY', 'AIW_TEST_SERVICE_ROLE_KEY'];
for (const key of required) {
  if (!process.env[key]) throw new Error('Missing required local-only configuration: ' + key);
}
const origin = new URL(process.env.SUPABASE_URL);
assert.equal(origin.protocol, 'https:', 'Require HTTPS Supabase');
assert.equal(origin.hostname, 'fyqpifmrsftsfqibhwhy.supabase.co', 'Refusing unexpected project');
const stage = new URL('/functions/v1/aiw-stylist-node-staging', origin).toString();
const version = 'stylist-5-editorial-v2-rc-20261109-r2';
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
async function quotaSnapshot() {
  const date = day();
  return { date, used: await usedFor(date) };
}
const first = await quotaSnapshot();
// Negative admission probe: no Authorization; must be rejected without quota.
const negative = await fetch(stage, {
  method: 'POST',
  headers: { apikey: publicKey, 'content-type': 'application/json' },
  body: '{}',
  signal: AbortSignal.timeout(15000),
});
assert.equal(negative.status, 401, 'Anonymous POST should fail 401');
const negativeAfter = await quotaSnapshot();
assert.equal(negativeAfter.date, first.date, 'UTC day crossed during quota baseline');
assert.equal(negativeAfter.used, first.used, 'Anonymous probe unexpectedly changed quota');
console.log('negative_admission_verified: status=401 quota_delta=0');

const scenarios = [
  { name: 'Minimalist + Work', style: 'minimalist', occasion: 'Work' },
  { name: 'Old Money + Work', style: 'old_money', occasion: 'Work' },
  { name: 'Classic + Date night', style: 'classic', occasion: 'Date night' },
  { name: 'New clothing, UZS 20M cap', style: 'classic', occasion: 'Work', budget: 20_000_000 },
  { name: 'New clothing, UZS 1.5M cap', style: 'minimalist', occasion: 'Work', budget: 1_500_000, mayRejectBudget: true },
];
assert.equal(scenarios.length, 5);
const results = [];
const validHttps = value => {
  try { const u = new URL(value); return u.protocol === 'https:' && !!u.hostname; }
  catch { return false; }
};
const expectedSlots = ['top', 'bottom', 'shoes'];
function checkOutfits(payload, scenario) {
  assert.equal(payload.success, true);
  assert.equal(payload.generationVersion, version, 'Runtime source is not expected Stylist r2');
  assert.ok(Array.isArray(payload.outfits) && payload.outfits.length >= 1, 'No accepted outfits');
  assert.ok(['ai', 'hybrid', 'deterministic_fallback'].includes(payload.source), 'Missing source provenance');
  let items = 0;
  let costsSameCurrency = true;
  let maximumSameCurrencyTotal = 0;
  const shoes = [];
  for (const outfit of payload.outfits) {
    assert.ok(Array.isArray(outfit.items) && outfit.items.length >= 3);
    assert.equal(outfit.qualityAssessment?.acceptance?.passed, true, 'Unaccepted look returned');
    assert.ok(typeof outfit.concept?.title === 'string' && outfit.concept.title.trim(), 'Missing concept');
    const slots = new Set(outfit.items.map(item => item.macroCategory));
    for (const slot of expectedSlots) assert.ok(slots.has(slot), 'Missing required slot: ' + slot);
    let subtotal = 0;
    for (const item of outfit.items) {
      items++;
      assert.ok(item.id && typeof item.id === 'string', 'Missing product ID');
      assert.equal(item.isShopItem, true, 'Catalog-only request included wardrobe inventory');
      assert.ok(validHttps(item.imageUrl), 'Unusable product image URL');
      assert.ok(validHttps(item.shopUrl), 'Unusable product source URL');
      assert.ok(Number.isFinite(Number(item.price)) && Number(item.price) > 0, 'Missing positive catalog price');
      assert.ok(typeof item.currency === 'string' && item.currency.length === 3, 'Missing currency');
      if (item.macroCategory === 'shoes') shoes.push(item.id);
      if (item.currency !== 'UZS') costsSameCurrency = false;
      else subtotal += Number(item.price);
    }
    maximumSameCurrencyTotal = Math.max(maximumSameCurrencyTotal, subtotal);
    if (scenario.budget && costsSameCurrency) {
      assert.ok(subtotal <= scenario.budget, 'Hard budget exceeded within UZS');
    }
  }
  return {
    outfits: payload.outfits.length,
    items,
    distinctShoes: new Set(shoes).size,
    provenance: payload.source,
    budgetCheck: !scenario.budget ? 'not_applicable' :
      costsSameCurrency ? 'same_currency_verified' : 'needs_pinned_FX_cross_currency_check',
    maximumSameCurrencyTotal: costsSameCurrency && scenario.budget ? maximumSameCurrencyTotal : undefined,
  };
}
for (let index = 0; index < scenarios.length; index++) {
  const scenario = scenarios[index];
  const before = await quotaSnapshot();
  assert.equal(before.date, first.date, 'UTC day crossed; restart smoke to avoid misleading deltas');
  const request = {
    stylePreferences: scenario.style,
    occasion: scenario.occasion,
    prompt: scenario.occasion,
    catalogOnly: true,
    wardrobeOnly: false,
    limit: 3,
    deckTarget: 3,
    allowPartialResults: true,
    compositionProtocol: 'optional-layer-v1',
    generationRequestId: 'staging-smoke-' + crypto.randomUUID(),
    ...(scenario.budget ? { preferences: { maxShopBudget: { amount: scenario.budget, currency: 'UZS' } } } : {}),
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
      body: JSON.stringify(request),
      signal: AbortSignal.timeout(55_000),
    });
  } catch {
    throw new Error('Staging transport failed or timed out; no automatic retry (avoids extra quota charge)');
  }
  const clientWallMs = Math.round(performance.now() - started);
  const serverTiming = response.headers.get('server-timing') || '';
  const cpu = /aiw_node_cpu;dur=(\d+(?:\.\d+)?)/.exec(serverTiming);
  const wall = /aiw_node_wall;dur=(\d+(?:\.\d+)?)/.exec(serverTiming);
  let payload;
  try { payload = await response.json(); } catch { throw new Error('Staging did not return JSON'); }
  const after = await quotaSnapshot();
  assert.equal(after.date, before.date, 'UTC day crossed during request; rerun on same UTC day');
  const delta = after.used - before.used;
  const code = String(payload.code || payload.error || '');
  const allowedBudgetRejection = scenario.mayRejectBudget && response.status === 422 &&
    code === 'no_outfit_within_budget';
  if (!response.ok && !allowedBudgetRejection) {
    throw new Error(scenario.name + ': staging failure HTTP ' + response.status +
      ' code=' + code + ' quotaDelta=' + delta + '; stopping, no retries');
  }
  assert.equal(delta, 1, 'Quota must increment exactly once per admitted request');
  assert.ok(cpu && wall, 'Node CPU/wall timing not propagated');
  const validation = response.ok ? checkOutfits(payload, scenario) :
    { outcomes: 'budget_rejected_as_expected', budgetCheck: 'server_rejection_only' };
  const outcome = {
    index: index + 1, scenario: scenario.name,
    status: response.status, code: code || undefined,
    clientWallMs,
    serverCpuMs: Number(cpu[1]), serverWallMs: Number(wall[1]),
    quotaDelta: delta, ...validation,
  };
  results.push(outcome);
  console.log(JSON.stringify(outcome));
}
console.log(JSON.stringify({
  summary: 'completed_isolated_staging_calls',
  calls: results.length, totalQuotaDebits: results.reduce((sum, r) => sum + r.quotaDelta, 0),
  maxClientWallMs: Math.max(...results.map(r => r.clientWallMs)),
  crossCurrencyBudgetCasesRequireIndependentPinnedFxAudit:
    results.filter(r => r.budgetCheck === 'needs_pinned_FX_cross_currency_check').length,
  productionCutoverAuthorized: false,
}));
