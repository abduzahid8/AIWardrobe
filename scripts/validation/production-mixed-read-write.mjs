/** Bounded hosted load with temporary accounts and nonempty ownership datasets. */
import 'dotenv/config';
import { randomUUID } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
import { runLoad, capacityPlan } from '../load/readiness.mjs';
import { runWearWrites } from '../load/wear-writes.mjs';
const baseUrl = process.env.EXPO_PUBLIC_SUPABASE_URL;
const apiKey = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
const service = process.env.SUPABASE_SERVICE_KEY;
if (!service) throw new Error('Temporary-account setup credential missing');
const admin = { apikey: service, Authorization: `Bearer ${service}`, 'Content-Type': 'application/json' };
const users = [];
const durationSec = Number(process.env.MIXED_LOAD_SECONDS || 120);
if (!Number.isInteger(durationSec) || durationSec < 120 || durationSec > 600) throw new Error('Duration must be 120–600 seconds');
const totalRps = Number(process.env.MIXED_LOAD_RPS || 40);
if (![4, 40].includes(totalRps) || (totalRps === 4 && durationSec !== 600)) throw new Error('Unsupported mixed workload');
const readRps = totalRps * 0.8, writeRps = totalRps * 0.2;
const reportName = process.env.MIXED_LOAD_REPORT || 'production-mixed-read-write.json';
if (!/^[a-z0-9-]+\.json$/.test(reportName)) throw new Error('Invalid report filename');
const workload = 'app';
const runId = randomUUID();
const controller = new AbortController();
process.once('SIGINT', () => controller.abort());
process.once('SIGTERM', () => controller.abort());
const emit = value => process.stdout.write(JSON.stringify(value) + '\n');
const report = { measuredAt: new Date().toISOString(), runId, assumptions: capacityPlan(), scope: `Production concurrent ${readRps} app read RPS + ${writeRps} wear-log upsert RPS for ${durationSec} seconds; 20 temporary free owners, 20 garments/5 outfits each; excludes AI, image delivery, uploads, subscriptions and native` , fixtureUsers: 20, wardrobeItemsPerUser: 20, savedOutfitsPerUser: 5, cleanup: [] };
const output = new URL(`../../docs/reports/readiness-evidence/${reportName}`, import.meta.url);
const save = () => writeFile(output, JSON.stringify(report, null, 2));
const stopOnFailure = reason => { if (!controller.signal.aborted) { report.stopReason = reason; controller.abort(); } };
async function call(path, headers, method = 'GET', body) {
  const response = await fetch(new URL(path, baseUrl), { method, headers, ...(body === undefined ? {} : { body: JSON.stringify(body) }), signal: AbortSignal.timeout(20000), redirect: 'error' });
  let data; try { data = await response.json(); } catch { data = null; }
  return { ok: response.ok, status: response.status, data };
}
try {
  for (let i = 0; i < report.fixtureUsers; i++) {
    if (controller.signal.aborted) throw new Error('Cancelled');
    const tag = randomUUID(), email = `readiness-${tag}@aiwardrobe.club`, password = `${randomUUID()}Aa!9`;
    const created = await call('/auth/v1/admin/users', admin, 'POST', { email, password, email_confirm: true, app_metadata: {readiness_audit:runId}, user_metadata: { username: `readiness_${tag}`, readiness_test: true, readiness_run_id: runId } });
    if (!created.ok || !created.data?.id) throw new Error(`fixture_create_http_${created.status}`);
    const user = { id: created.data.id }; users.push(user);
    report.fixtureUserIds = users.map(owner => owner.id);
    await save();
    const session = await call('/auth/v1/token?grant_type=password', { apikey: apiKey, 'Content-Type': 'application/json' }, 'POST', { email, password });
    if (!session.ok || !session.data?.access_token) throw new Error(`fixture_signin_http_${session.status}`);
    user.accessToken = session.data.access_token;
    user.itemId = null; user.logIds = Array.from({ length: Math.ceil(writeRps * durationSec / (report.fixtureUsers * 2)) }, () => randomUUID());
    user.expectedWardrobeItems = report.wardrobeItemsPerUser;
    user.expectedSavedOutfits = report.savedOutfitsPerUser;
    const headers = { apikey: apiKey, Authorization: `Bearer ${user.accessToken}`, 'Content-Type': 'application/json' };
    const items = Array.from({ length: report.wardrobeItemsPerUser }, (_, n) => ({ id: randomUUID(), user_id: user.id, category: ['top', 'bottom', 'shoes'][n % 3], name: 'Readiness capacity fixture', image_url: 'https://example.invalid/readiness-fixture.jpg' }));
    user.itemId = items[0].id;
    user.headers = headers;
    const wardrobe = await call('/rest/v1/clothing_items', headers, 'POST', items);
    if (!wardrobe.ok) throw new Error(`fixture_wardrobe_http_${wardrobe.status}`);
    const outfits = await call('/rest/v1/saved_outfits', headers, 'POST', Array.from({ length: report.savedOutfitsPerUser }, () => ({ id: randomUUID(), user_id: user.id, items: items.slice(0, 3).map(item => ({ id: item.id, category: item.category })), name: 'Readiness capacity fixture' })));
    if (!outfits.ok) throw new Error(`fixture_outfit_http_${outfits.status}`);
    if ((i + 1) % 5 === 0) emit({ stage: 'fixture_setup', usersReady: i + 1 });
  }
  emit({ stage: 'populated_fixtures_ready', users: users.length, wardrobeItems: users.length * report.wardrobeItemsPerUser, savedOutfits: users.length * report.savedOutfitsPerUser });
  if (workload === 'app') {
    const preflight = await runLoad({ baseUrl, apiKey, users, rps: 4, durationSec: 1, workload });
    report.preflight = { completed: preflight.completed, failedRequests: preflight.failedRequests, statuses: preflight.statuses, failures: preflight.failures, passed: preflight.completed === 4 && preflight.failedRequests === 0 && preflight.dropped === 0 };
    if (!report.preflight.passed) throw new Error('App read preflight failed');
  }
  // Keep Node alive while a transport promise is unresolved so timeout and
  // abort callbacks can settle it and the finally block can remove fixtures.
  const keepAlive = setInterval(() => {}, 1000);
  let measured;
  try {
    measured = await Promise.allSettled([
      runLoad({ baseUrl, apiKey, users, rps: readRps, durationSec, workload, signal: controller.signal,
        onFailure: stopOnFailure, onProgress: progress => emit({ stage: 'mixed_read_progress', ...progress }) }),
      runWearWrites({ baseUrl, apiKey, users, rps: writeRps, durationSec, signal: controller.signal,
        onFailure: stopOnFailure, onProgress: progress => emit({ stage: 'mixed_write_progress', ...progress }) }),
    ]);
  } finally { clearInterval(keepAlive); }
  report.reads = measured[0].status === 'fulfilled' ? measured[0].value : { passed: false, harnessError: 'read_runner_rejected' };
  report.writes = measured[1].status === 'fulfilled' ? measured[1].value : { passed: false, harnessError: 'write_runner_rejected' };
  await save();
  report.verification = [];
  for (const user of controller.signal.aborted ? [] : users) {
    const logs = await call(`/rest/v1/wear_logs?user_id=eq.${user.id}&select=id,user_id,item_ids&limit=250`, user.headers);
    const stats = await call(`/rest/v1/clothing_items?id=eq.${user.itemId}&select=id,user_id,wear_count,last_worn_at`, user.headers);
    const expected = new Set(user.logIds);
    const passed = logs.ok && Array.isArray(logs.data) && logs.data.length === expected.size
      && new Set(logs.data.map(row => row.id)).size === expected.size
      && logs.data.every(row => row.user_id === user.id && expected.has(row.id) && row.item_ids.every(id => id === user.itemId))
      && stats.ok && stats.data?.length === 1 && stats.data[0].user_id === user.id
      && stats.data[0].wear_count === expected.size
      && Date.parse(stats.data[0].last_worn_at) === Date.parse('2026-10-16T12:00:00Z');
    report.verification.push({ expectedDistinctLogs: expected.size, actualLogs: Array.isArray(logs.data) ? logs.data.length : null,
      actualWearCount: stats.data?.[0]?.wear_count ?? null, passed });
  }
  if (!report.reads.passed || !report.writes.passed || report.verification.some(v => !v.passed)) process.exitCode = 1;

} catch (error) {
  report.failure = /^[a-z_]+_http_\d+$/.test(error.message) ? error.message : error.name;
  process.exitCode = 1;
} finally {
  for (const user of users) {
    if (user.accessToken) {
      try { await call('/auth/v1/logout?scope=global', { apikey: apiKey, Authorization: `Bearer ${user.accessToken}` }, 'POST'); } catch { /* Account deletion must still be attempted. */ }
    }
    let removed;
    for (let attempt = 0; attempt < 3; attempt++) {
      try { removed = await call(`/auth/v1/admin/users/${user.id}`, admin, 'DELETE'); } catch { removed = { ok: false, status: 0 }; }
      if (removed.ok || removed.status === 404) break;
    }
    const deleted = removed.ok || removed.status === 404;
    report.cleanup.push({ temporaryUserDeleted: deleted, status: removed.status });
    await save();
    if (!deleted) process.exitCode = 1;
  }
  report.passed = Boolean(report.reads?.passed && report.writes?.passed) && report.verification?.length === 20
    && report.verification.every(v => v.passed) && report.cleanup.length === 20
    && report.cleanup.every(v => v.temporaryUserDeleted) && !report.failure;
  if (!report.passed) process.exitCode = 1;
  await save();
  emit(report);
}
