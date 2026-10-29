/** Fixed bounded production workload: 4,800 upserts, 20 disposable owners. */
import 'dotenv/config';
import { randomUUID } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
import { runWearWrites } from '../load/wear-writes.mjs';
const base = process.env.EXPO_PUBLIC_SUPABASE_URL, anon = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
const service = process.env.SUPABASE_SERVICE_KEY;
const durationSec = Number(process.env.WEAR_LOAD_DURATION_SEC || 120);
if (![30,120].includes(durationSec)) throw new Error('Only 30-second diagnostic or 120-second capacity runs supported');
const rps = Number(process.env.WEAR_LOAD_RPS || 40);
if (![2,40].includes(rps) || (rps === 2 && durationSec !== 120)) throw new Error('Unsupported bounded write rate');
const expectedLogs = rps === 2 ? 6 : durationSec;
const instrumented = process.env.WEAR_LOAD_INSTRUMENTED === '1';
if (!base || !anon || !service) throw new Error('Fixture setup credential required');
const admin = { apikey: service, Authorization: `Bearer ${service}`, 'Content-Type': 'application/json' };
const users = [], report = { measuredAt: new Date().toISOString(), scope: `20 disposable owners; ${rps} wear-log upserts/second for ${durationSec} seconds; paired retries; excludes AI/uploads/native`, verification: [], cleanup: [] };
const controller = new AbortController(); process.once('SIGINT', () => controller.abort());
const emit = value => process.stdout.write(JSON.stringify(value)+'\n');
async function call(path, headers, method = 'GET', body) {
  const response = await fetch(new URL(path, base), { method, headers, redirect: 'error', signal: AbortSignal.timeout(20000),
    ...(body === undefined ? {} : { body: JSON.stringify(body) }) });
  let data = null; try { data = await response.json(); } catch { /* Empty response allowed. */ }
  return { ok: response.ok, status: response.status, data };
}
try {
  for (let index = 0; index < 20; index++) {
    const tag = randomUUID(), email = `readiness-${tag}@aiwardrobe.club`, password = `${randomUUID()}Aa!9`;
    const created = await call('/auth/v1/admin/users', admin, 'POST', { email, password, email_confirm: true,
      user_metadata: { username: `readiness_${tag}`, readiness_test: true } });
    if (!created.ok || !created.data?.id) throw new Error('FixtureSetupError');
    const user = { id: created.data.id, itemId: randomUUID(), logIds: Array.from({ length: expectedLogs }, () => randomUUID()) }; users.push(user);
    const login = await call('/auth/v1/token?grant_type=password', { apikey: anon, 'Content-Type': 'application/json' }, 'POST', { email, password });
    if (!login.ok || !login.data?.access_token) throw new Error('FixtureLoginError');
    user.accessToken = login.data.access_token;
    user.headers = { apikey: anon, Authorization: `Bearer ${user.accessToken}`, 'Content-Type': 'application/json' };
    const inserted = await call('/rest/v1/clothing_items', user.headers, 'POST', {
      id: user.itemId, user_id: user.id, name: 'Readiness write capacity fixture', category: 'top',
    });
    if (!inserted.ok) throw new Error('FixtureInsertError');
    if ((index + 1) % 5 === 0) emit({ stage: 'fixture_setup', ready: index + 1 });
  }
  report.load = await runWearWrites({ baseUrl: base, apiKey: anon, users, rps, durationSec,
    signal: controller.signal, onProgress: value => emit({ stage: 'write_progress', ...value }) });
  if (!report.load.passed) process.exitCode = 1;
  for (const user of users) {
    const logs = await call(`/rest/v1/wear_logs?user_id=eq.${user.id}&select=id,user_id&limit=250`, user.headers);
    const stats = await call(`/rest/v1/clothing_items?id=eq.${user.itemId}&select=id,user_id,wear_count,last_worn_at`, user.headers);
    const expected = new Set(user.logIds);
    const passed = logs.ok && Array.isArray(logs.data) && logs.data.length === expected.size
      && new Set(logs.data.map(row => row.id)).size === expected.size
      && logs.data.every(row => row.user_id === user.id && expected.has(row.id))
      && stats.ok && stats.data?.length === 1 && stats.data[0].user_id === user.id
      && stats.data[0].wear_count === expected.size
      && Date.parse(stats.data[0].last_worn_at) === Date.parse('2026-10-16T12:00:00Z');
    report.verification.push({ expectedDistinctLogs: expected.size, actualLogs: Array.isArray(logs.data) ? logs.data.length : null,
      actualWearCount: stats.data?.[0]?.wear_count ?? null, passed });
    if (!passed) process.exitCode = 1;
  }
} catch (error) { report.failure = error.name; process.exitCode = 1; }
finally {
  for (const user of users) {
    if (user.headers) try { await call('/auth/v1/logout?scope=global', user.headers, 'POST'); } catch { /* Delete still required. */ }
    let response = { ok: false, status: 0 };
    for (let attempt = 0; attempt < 3; attempt++) {
      try { response = await call(`/auth/v1/admin/users/${user.id}`, admin, 'DELETE'); } catch { response = { ok: false, status: 0 }; }
      if (response.ok || response.status === 404) break;
    }
    const passed = response.ok || response.status === 404;
    report.cleanup.push({ deleted: passed, status: response.status });
    if (!passed) process.exitCode = 1;
  }
  report.passed = Boolean(report.load?.passed) && report.verification.length === 20 && report.verification.every(value => value.passed)
    && report.cleanup.length === 20 && report.cleanup.every(value => value.deleted) && !report.failure;
  if (!report.passed) process.exitCode = 1;
  const filename = rps === 2 ? 'oct28-wear-load-2rps.json' : durationSec === 30 ? 'production-wear-load-diagnostic.json'
    : instrumented ? 'production-wear-load-instrumented.json' : 'production-wear-load.json';
  await writeFile(new URL(`../../docs/reports/readiness-evidence/${filename}`, import.meta.url), JSON.stringify(report, null, 2)+'\n');
  emit(report);
}
