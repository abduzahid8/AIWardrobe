/** Bounded hosted load with temporary accounts and nonempty ownership datasets. */
import 'dotenv/config';
import { randomUUID } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
import { runLoad } from '../load/readiness.mjs';
const baseUrl = process.env.EXPO_PUBLIC_SUPABASE_URL;
const apiKey = process.env.EXPO_PUBLIC_SUPABASE_ANON_KEY;
const service = process.env.SUPABASE_SERVICE_KEY;
if (!service) throw new Error('Temporary-account setup credential missing');
const admin = { apikey: service, Authorization: `Bearer ${service}`, 'Content-Type': 'application/json' };
const users = [];
const durationSec = Number(process.env.LOAD_DURATION_SEC || 120);
const rps = Number(process.env.LOAD_RPS || 40);
const reportName = process.env.LOAD_REPORT_NAME || (process.env.LOAD_WORKLOAD === 'app' ? 'production-app-read-soak.json' : 'production-populated-load.json');
if (![0.2, 2, 4, 40].includes(rps) || !/^[a-z0-9-]+\.json$/.test(reportName)) throw new Error('Unsupported bounded load configuration');
const workload = process.env.LOAD_WORKLOAD || 'lightweight';
if (![120, 600].includes(durationSec) || !['lightweight', 'app'].includes(workload)) throw new Error('Unsupported bounded load configuration');
const controller = new AbortController();
process.once('SIGINT', () => controller.abort());
const emit = value => process.stdout.write(JSON.stringify(value) + '\n');
const report = { measuredAt: new Date().toISOString(), scope: 'Production: 20 temporary free users, 20 wardrobe items and 5 saved outfits each; read-only traffic after fixture setup; excludes AI, image delivery and uploads', fixtureUsers: 20, wardrobeItemsPerUser: 20, savedOutfitsPerUser: 5, cleanup: [] };
async function call(path, headers, method = 'GET', body) {
  const response = await fetch(new URL(path, baseUrl), { method, headers, ...(body === undefined ? {} : { body: JSON.stringify(body) }), signal: AbortSignal.timeout(20000), redirect: 'error' });
  let data; try { data = await response.json(); } catch { data = null; }
  return { ok: response.ok, status: response.status, data };
}
try {
  for (let i = 0; i < report.fixtureUsers; i++) {
    const tag = randomUUID(), email = `readiness-${tag}@aiwardrobe.club`, password = `${randomUUID()}Aa!9`;
    const created = await call('/auth/v1/admin/users', admin, 'POST', { email, password, email_confirm: true, user_metadata: { username: `readiness_${tag}`, readiness_test: true } });
    if (!created.ok || !created.data?.id) throw new Error(`fixture_create_http_${created.status}`);
    const user = { id: created.data.id }; users.push(user);
    const session = await call('/auth/v1/token?grant_type=password', { apikey: apiKey, 'Content-Type': 'application/json' }, 'POST', { email, password });
    if (!session.ok || !session.data?.access_token) throw new Error(`fixture_signin_http_${session.status}`);
    user.accessToken = session.data.access_token;
    user.expectedWardrobeItems = report.wardrobeItemsPerUser;
    user.expectedSavedOutfits = report.savedOutfitsPerUser;
    const headers = { apikey: apiKey, Authorization: `Bearer ${user.accessToken}`, 'Content-Type': 'application/json' };
    const items = Array.from({ length: report.wardrobeItemsPerUser }, (_, n) => ({ id: randomUUID(), user_id: user.id, category: ['top', 'bottom', 'shoes'][n % 3], name: 'Readiness capacity fixture', image_url: 'https://example.invalid/readiness-fixture.jpg' }));
    const wardrobe = await call('/rest/v1/clothing_items', headers, 'POST', items);
    if (!wardrobe.ok) {
      report.fixtureErrorCode = typeof wardrobe.data?.code === 'string' && /^[A-Z0-9]{5,10}$/.test(wardrobe.data.code) ? wardrobe.data.code : null;
      throw new Error(`fixture_wardrobe_http_${wardrobe.status}`);
    }
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
  report.load = await runLoad({ baseUrl, apiKey, users, rps, durationSec, workload, signal: controller.signal, onProgress: progress => emit({ stage: 'load_progress', ...progress }) });
  if (!report.load.passed) process.exitCode = 1;
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
    if (!deleted) process.exitCode = 1;
  }
  await writeFile(new URL(`../../docs/reports/readiness-evidence/${reportName}`, import.meta.url), JSON.stringify(report, null, 2));
  emit(report);
}
