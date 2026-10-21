/** Bounded production HTTP regression on two disposable accounts; no AI calls. */
import 'dotenv/config';
import { readFile, writeFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
const base = process.env.EXPO_PUBLIC_SUPABASE_URL;
const anon = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
const service = process.env.SUPABASE_SERVICE_KEY || (process.env.SUPABASE_SERVICE_KEY_FILE
  ? (await readFile(process.env.SUPABASE_SERVICE_KEY_FILE, 'utf8')).trim() : '');
if (!base || !anon || !service) throw new Error('Supabase settings and temporary-account setup credential required');
const admin = { apikey: service, Authorization: `Bearer ${service}`, 'Content-Type': 'application/json' };
const report = { measuredAt: new Date().toISOString(), scope: 'Two disposable accounts; sequential correctness checks, not capacity testing', checks: [], cleanup: [] };
const users = [];
async function request(path, headers, method = 'GET', body) {
  const response = await fetch(new URL(path, base), { method, headers, redirect: 'error',
    signal: AbortSignal.timeout(20000), ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
  let data = null;
  try { data = await response.json(); } catch { /* DELETE may have no response body. */ }
  return { ok: response.ok, status: response.status, data };
}
function check(name, response, condition) {
  report.checks.push({ name, status: response.status, passed: Boolean(condition) });
  if (!condition) throw new Error(name);
}
try {
  for (let index = 0; index < 2; index++) {
    const tag = randomUUID(), email = `readiness-${tag}@aiwardrobe.club`, password = `${randomUUID()}Aa!9`;
    const created = await request('/auth/v1/admin/users', admin, 'POST', {
      email, password, email_confirm: true, user_metadata: { username: `readiness_${tag}`, readiness_test: true },
    });
    if (!created.ok || !created.data?.id) throw new Error('fixture_create');
    const user = { id: created.data.id }; users.push(user);
    const login = await request('/auth/v1/token?grant_type=password', { apikey: anon, 'Content-Type': 'application/json' }, 'POST', { email, password });
    if (!login.ok || !login.data?.access_token) throw new Error('fixture_login');
    user.headers = { apikey: anon, Authorization: `Bearer ${login.data.access_token}`, 'Content-Type': 'application/json' };
    user.itemId = randomUUID();
    const inserted = await request('/rest/v1/clothing_items', user.headers, 'POST', { id: user.itemId, user_id: user.id, name: 'Readiness wear fixture', category: 'top' });
    if (!inserted.ok) throw new Error('fixture_garment');
  }
  const [owner, other] = users, logId = randomUUID();
  const log = { id: logId, user_id: owner.id, item_ids: [owner.itemId, owner.itemId, other.itemId], date: '2026-10-16', created_at: '2026-10-16T12:00:00Z' };
  const writeHeaders = { ...owner.headers, Prefer: 'resolution=merge-duplicates,return=representation' };
  const first = await request('/rest/v1/wear_logs?on_conflict=id', writeHeaders, 'POST', log);
  check('insert_log_confirmed', first, first.ok && first.data?.[0]?.id === logId);
  const retry = await request('/rest/v1/wear_logs?on_conflict=id', writeHeaders, 'POST', log);
  check('retry_log_confirmed', retry, retry.ok && retry.data?.[0]?.id === logId);
  const stats = async (user, count, time) => {
    const response = await request(`/rest/v1/clothing_items?id=eq.${user.itemId}&select=id,wear_count,last_worn_at`, user.headers);
    return { response, passed: response.ok && response.data?.length === 1 && response.data[0].wear_count === count
      && (time === null ? response.data[0].last_worn_at === null : Date.parse(response.data[0].last_worn_at) === Date.parse(time)) };
  };
  let state = await stats(owner, 1, log.created_at);
  check('retry_and_duplicates_count_once', state.response, state.passed);
  state = await stats(other, 0, null);
  check('cross_owner_item_unchanged', state.response, state.passed);
  const overwrite = await request(`/rest/v1/clothing_items?id=eq.${owner.itemId}`, owner.headers, 'PATCH', { wear_count: 999, last_worn_at: '2099-01-01T00:00:00Z' });
  check('cached_garment_update_allowed', overwrite, overwrite.ok);
  state = await stats(owner, 1, log.created_at);
  check('server_statistics_preserved', state.response, state.passed);
  const isolated = await request(`/rest/v1/wear_logs?id=eq.${logId}&select=id`, other.headers);
  check('other_cannot_read_log', isolated, isolated.ok && Array.isArray(isolated.data) && isolated.data.length === 0);
  const removed = await request(`/rest/v1/wear_logs?id=eq.${logId}`, owner.headers, 'DELETE');
  check('delete_log_success', removed, removed.ok);
  state = await stats(owner, 0, null);
  check('delete_log_clears_statistics', state.response, state.passed);
  const restored = await request('/rest/v1/wear_logs?on_conflict=id', writeHeaders, 'POST', log);
  check('wear_history_restored_for_auth_deletion', restored, restored.ok);
  const deletedAccount = await request(`/auth/v1/admin/users/${owner.id}`, admin, 'DELETE');
  check('auth_deletion_with_wear_history', deletedAccount, deletedAccount.ok);
  const remaining = await request(`/rest/v1/wear_logs?user_id=eq.${owner.id}&select=id`, admin);
  check('auth_deletion_cascades_history', remaining, remaining.ok && Array.isArray(remaining.data) && remaining.data.length === 0);
} catch (error) {
  // Error text may come from transport libraries; retain only our safe check name/type.
  report.failure = report.checks.at(-1)?.passed === false ? report.checks.at(-1).name : error.name;
  process.exitCode = 1;
} finally {
  for (const user of users) {
    if (user.headers) {
      try { await request('/auth/v1/logout?scope=global', user.headers, 'POST'); } catch { /* Delete still required. */ }
    }
    let response = { ok: false, status: 0 };
    for (let attempt = 0; attempt < 3; attempt++) {
      try { response = await request(`/auth/v1/admin/users/${user.id}`, admin, 'DELETE'); } catch { response = { ok: false, status: 0 }; }
      if (response.ok || response.status === 404) break;
    }
    const passed = response.ok || response.status === 404;
    report.cleanup.push({ temporaryAccountDeleted: passed, status: response.status });
    if (!passed) process.exitCode = 1;
  }
  await writeFile(new URL('../../docs/reports/readiness-evidence/production-wear-consistency.json', import.meta.url), JSON.stringify(report, null, 2)+'\n');
  process.stdout.write(JSON.stringify(report)+'\n');
}
