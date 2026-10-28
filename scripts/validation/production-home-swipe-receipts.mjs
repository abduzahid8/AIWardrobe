/** Real store, ordinary-account RPCs, lost replies and concurrent receipt replay. */
import 'dotenv/config';
import { createClient } from '@supabase/supabase-js';
import { randomUUID, createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import ts from 'typescript';
const require = createRequire(import.meta.url);
const base = process.env.EXPO_PUBLIC_SUPABASE_URL;
const anon = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
const key = process.env.SUPABASE_SERVICE_KEY;
if (!key) throw Error('Setup credential required');
const options = { auth: { persistSession: false, autoRefreshToken: false } };
const admin = createClient(base, key, options);
const fixtures = [];
const report = { measuredAt: new Date().toISOString(), checks: [], fixtures: [],
  scope: 'Actual quota store with controlled storage/plan selectors, live authenticated RPCs. Lost response injected only after a successful server response. Not native gesture or 10kDAU certification.' };
const check = (name, passed) => {
  report.checks.push({ name, passed: !!passed });
  if (!passed) throw Error(`Assertion: ${name}`);
};
const compile = source => ts.transpileModule(source, { compilerOptions: {
  module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022,
} }).outputText;
const source = await readFile('store/dailyUsageStore.ts', 'utf8');
report.sourceSha256 = createHash('sha256').update(source).digest('hex');
const deadline = {};
new Function('exports', compile(await readFile('src/lib/requestDeadline.ts', 'utf8')))(deadline);
async function fixture() {
  const suffix = randomUUID(), email = `readiness-${suffix}@aiwardrobe.club`, password = randomUUID() + 'Aa!9';
  const created = await admin.auth.admin.createUser({ email, password, email_confirm: true,
    app_metadata: { readiness_audit: suffix }, user_metadata: { readiness_test: true } });
  if (created.error) throw Error('Fixture creation failed');
  const owner = created.data.user.id, client = createClient(base, anon, options);
  const entry = { owner, client }; fixtures.push(entry); report.fixtures.push(owner);
  if ((await client.auth.signInWithPassword({ email, password })).error) throw Error('Fixture login failed');
  const profile = await admin.from('profiles').select('trial_started_at,subscription_tier').eq('id', owner).single();
  check('free_fixture_without_trial', !profile.error && profile.data.subscription_tier === 'free' && profile.data.trial_started_at === null);
  return entry;
}
function actualStore(entry, loseReplyFor) {
  const api = {}, disk = new Map(); let lost = false;
  const transport = { rpc: (name, params) => {
    const query = entry.client.rpc(name, params);
    if (params.p_card_key !== loseReplyFor || lost) return query;
    return { abortSignal: async signal => {
      const response = await query.abortSignal(signal);
      if (!response.error) { lost = true; throw Error('Injected lost successful reply'); }
      return response;
    } };
  } };
  new Function('require', 'exports', compile(source))(name => {
    if (name === 'zustand') return require(name);
    if (name.includes('async-storage')) return { __esModule: true, default: {
      setItem: async (k, v) => disk.set(k, v), getItem: async k => disk.get(k) ?? null,
    } };
    if (name.includes('subscriptionStore')) return { __esModule: true,
      default: { getState: () => ({ refreshTimeState: () => {}, effectiveTier: 'free' }) },
      TIER_FEATURES: { free: { homeSwipes: 3 } } };
    if (name.includes('lib/supabase')) return { supabase: transport };
    if (name.includes('authEvents')) return { authEvents: { getUserId: () => entry.owner, getRevision: () => 1 } };
    if (name.includes('requestDeadline')) return deadline;
    throw Error('Unexpected store dependency');
  }, api);
  return api.default;
}
const authorize = (store, card, signal) => store.getState().consume('homeSwipes', 1,
  { requireServer: true, homeCardKey: card, cancelSignal: signal });
const used = async entry => {
  const value = await entry.client.rpc('get_usage_today', { p_feature: 'homeSwipes' });
  if (value.error) throw Error('Counter read failed'); return value.data;
};
try {
  const a = await fixture(), b = await fixture(), storeA = actualStore(a, 'lost-card');
  const duplicates = await Promise.all(Array.from({ length: 20 }, () => authorize(storeA, 'shared-card')));
  check('twenty_simultaneous_same_card_all_approved_once', duplicates.every(x => x.allowed) && await used(a) === 1);
  let lost = false;
  try { await authorize(storeA, 'lost-card'); } catch (error) { lost = error.name === 'UsageUnavailableError'; }
  check('lost_successful_reply_is_reported_unavailable', lost && await used(a) === 2);
  check('retry_recovers_without_second_debit', (await authorize(storeA, 'lost-card')).allowed && await used(a) === 2);
  const remounted = actualStore(a);
  check('fresh_store_recovers_same_receipt', (await authorize(remounted, 'lost-card')).allowed && await used(a) === 2);
  check('third_distinct_card_approved', (await authorize(remounted, 'third-card')).allowed && await used(a) === 3);
  check('receipt_replays_after_daily_exhaustion', (await authorize(remounted, 'shared-card')).allowed && await used(a) === 3);
  const denied = await Promise.all(Array.from({ length: 20 }, (_, i) => authorize(remounted, `denied-${i}`)));
  check('twenty_new_cards_denied_after_three', denied.every(x => !x.allowed) && await used(a) === 3);
  const storeB = actualStore(b);
  check('same_card_key_is_not_shared_across_accounts', (await authorize(storeB, 'shared-card')).used === 1 && await used(b) === 1);
  const distinct = await Promise.all(Array.from({ length: 20 }, (_, i) => authorize(storeB, `distinct-${i}`)));
  check('parallel_distinct_keys_share_atomic_counter', distinct.filter(x => x.allowed).length === 2 && await used(b) === 3);
  const rows = await admin.from('home_swipe_receipts').select('user_id,card_key,charged_used').in('user_id', [a.owner, b.owner]);
  check('exactly_three_committed_receipts_per_owner', !rows.error && [a,b].every(e => rows.data.filter(r => r.user_id === e.owner && r.charged_used > 0).length === 3) && rows.data.length === 6);
  const today = new Date().toISOString().slice(0,10);
  const invalid = await a.client.rpc('authorize_home_swipe', { p_card_key: 'shared-card', p_day: '2000-01-01' });
  check('old_day_rejected', invalid.error?.code === '22023');
  const forged = await a.client.rpc('authorize_home_swipe', { p_card_key: 'shared-card', p_day: today, p_user: b.owner });
  check('owner_override_rejected', !!forged.error);
  const unsigned = createClient(base, anon, options);
  const anonymous = await unsigned.rpc('authorize_home_swipe', { p_card_key: 'shared-card', p_day: today });
  check('anonymous_rpc_denied', !!anonymous.error);
  const direct = await a.client.from('home_swipe_receipts').select('*').limit(1);
  check('direct_receipt_table_access_denied', !!direct.error);
  report.requestCounts = { sameCardConcurrent: 20, exhaustedConcurrent: 20, distinctConcurrent: 20 };
} catch (error) { report.failure = error.message; process.exitCode = 1; }
finally {
  report.cleanup = [];
  for (const entry of fixtures) {
    const cleanup = { owner: entry.owner, sessionsRevoked: false, ownerDeleted: false };
    try { cleanup.sessionsRevoked = !(await entry.client.auth.signOut({ scope: 'global' })).error; } catch {}
    if (cleanup.sessionsRevoked) try { cleanup.ownerDeleted = !(await admin.auth.admin.deleteUser(entry.owner)).error; } catch {}
    report.cleanup.push(cleanup);
  }
  report.passed = !report.failure && report.checks.length === 16 && report.checks.every(x => x.passed)
    && report.cleanup.length === 2 && report.cleanup.every(x => x.sessionsRevoked && x.ownerDeleted);
  await writeFile('docs/reports/readiness-evidence/oct25-home-swipe-receipts-live.json', JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report));
  if (!report.passed) process.exitCode = 1;
}
