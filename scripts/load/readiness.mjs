/** Read-only, paced HTTP load verification. No provider work or account writes.
 * Run --plan without credentials; see docs/reports/READINESS_10000_DAU.md.
 */
import { responseDiagnostics, timingSummary } from './http-diagnostics.mjs';
import { readFile, writeFile } from 'node:fs/promises';
import { performance } from 'node:perf_hooks';
import { pathToFileURL } from 'node:url';

export function capacityPlan({ dau = 10000, sessionsPerDay = 2, requestsPerSession = 12, peakHourFraction = 0.2, headroom = 3 } = {}) {
  for (const value of [dau, sessionsPerDay, requestsPerSession, headroom]) {
    if (!Number.isFinite(value) || value <= 0) throw new Error('Capacity inputs must be positive');
  }
  if (!(peakHourFraction > 0 && peakHourFraction <= 1)) throw new Error('Invalid peak hour fraction');
  const dailyRequests = dau * sessionsPerDay * requestsPerSession;
  const peakRps = dailyRequests * peakHourFraction / 3600;
  return { dau, sessionsPerDay, requestsPerSession, peakHourFraction, headroom, dailyRequests, peakRps, targetRps: Math.ceil(peakRps * headroom) };
}

const routes = [
  { name: 'catalog', path: '/rest/v1/shop_catalog?select=id,name,garment_type&limit=24', private: false },
  { name: 'wardrobe', path: '/rest/v1/clothing_items?select=id,user_id&limit=30', private: true },
  { name: 'outfits', path: '/rest/v1/saved_outfits?select=id,user_id&limit=20', private: true },
  { name: 'profile', path: '/rest/v1/profiles?select=id&limit=1', private: true, ownerKey: 'id' },
];

const appRoutes = [
  { name: 'catalog', path: '/rest/v1/shop_catalog?select=id,name,garment_type,price,image_url,currency,brand&limit=24', private: false },
  { name: 'wardrobe', path: '/rest/v1/clothing_items?select=id,user_id,image_url,thumbnail_url,category,sub_category,primary_color,color_hex,pattern,material,brand,name,seasons,occasions,wear_count,last_worn_at,is_favorite,created_at,updated_at,detection_confidence&order=created_at.desc,id.desc&limit=250', private: true, expectedKey: 'expectedWardrobeItems' },
  { name: 'outfits', path: '/rest/v1/saved_outfits?select=id,user_id,date,occasion,items,created_at&order=created_at.desc,id.desc&limit=250', private: true, expectedKey: 'expectedSavedOutfits' },
  { name: 'profile', path: '/rest/v1/profiles?select=id,username,profile_image&limit=1', private: true, ownerKey: 'id' },
];

export async function runLoad({ baseUrl, apiKey, users, rps, durationSec, maxInflight = 200, timeoutMs = 10000, fetchImpl = fetch, workload = 'lightweight', onProgress, signal }) {
  if (!/^https?:\/\//.test(baseUrl) || !apiKey || !users?.length || users.some(u => !u.id || !u.accessToken)) throw new Error('URL, anon API key, and authenticated test users are required');
  for (const n of [rps, durationSec, maxInflight, timeoutMs]) if (!Number.isFinite(n) || n <= 0) throw new Error('Invalid load settings');
  if (!['lightweight', 'app'].includes(workload)) throw new Error('Invalid workload');
  const activeRoutes = workload === 'app' ? appRoutes : routes;
  const total = Math.ceil(rps * durationSec);
  const startedAt = new Date().toISOString();
  const started = performance.now();
  const headerTimes = [], bodyTimes = [];
  const latencies = [];
  const statuses = {};
  const failures = {};
  const failureDetails = [];
  const failedRequests = new Set();
  const pending = new Set();
  const exercisedUsers = new Set();
  let dropped = 0, lastProgress = 0;
  const recordFailure = reason => { failures[reason] = (failures[reason] || 0) + 1; };
  const execute = async index => {
    const fail = reason => { failedRequests.add(index); recordFailure(reason); };
    const route = activeRoutes[index % activeRoutes.length];
    const user = users[Math.floor(index / activeRoutes.length) % users.length];
    let stage = 'fetch';
    const before = performance.now();
    const requestStartedAt = new Date().toISOString();
    let headersAt, responseInfo = {};
    if (route.private) exercisedUsers.add(user.id);
    try {
      const url = new URL(route.path, baseUrl);
      if (workload === 'app' && route.private) url.searchParams.set(route.ownerKey || 'user_id', `eq.${user.id}`);
      const response = await fetchImpl(url, {
        headers: { apikey: apiKey, Authorization: `Bearer ${route.private ? user.accessToken : apiKey}` },
        signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(timeoutMs)]) : AbortSignal.timeout(timeoutMs),
        redirect: 'error',
      });
      headersAt = performance.now();
      headerTimes.push(headersAt - before);
      responseInfo = {status:response.status,...responseDiagnostics(response)};
      stage = 'read_body';
      statuses[response.status] = (statuses[response.status] || 0) + 1;
      if (response.status !== 200) {
        if (failureDetails.length < 10) failureDetails.push({route:route.name,stage:'http_response',requestIndex:index,startedAt:requestStartedAt,status:response.status,...responseInfo});
        await response.arrayBuffer(); fail(`http_${response.status}`); return;
      }
      stage = 'decode_json';
      const rows = await response.json();
      stage = 'validate_rows';
      if (!Array.isArray(rows)) { fail(`${route.name}_invalid_shape`); return; }
      if (route.expectedKey && Number.isInteger(user[route.expectedKey]) && rows.length !== Math.min(user[route.expectedKey], 250)) fail(`${route.name}_row_count`);
      if (route.name === 'catalog' && rows.length === 0) fail('catalog_empty');
      if (route.name === 'profile' && rows.length !== 1) fail('profile_missing');
      if (route.private && rows.some(row => row[route.ownerKey || 'user_id'] !== user.id)) fail(`${route.name}_ownership_violation`);
      if (rows.some(row => !row.id)) fail(`${route.name}_missing_id`);
    } catch (error) {
      fail('network_or_invalid_json');
      // Preserve diagnostic codes without recording URLs, headers, or tokens.
      if (failureDetails.length < 10) failureDetails.push({
        route: route.name, stage,
        requestIndex: index, startedAt: requestStartedAt, ...responseInfo,
        errorType: /^[A-Za-z]+Error$/.test(error?.name || '') ? error.name : 'UnknownError',
        ...(typeof error?.cause?.code === 'string' && /^[A-Z0-9_]+$/.test(error.cause.code) ? { causeCode: error.cause.code } : {}),
      });
    }
    finally { if (headersAt !== undefined) bodyTimes.push(performance.now() - headersAt); latencies.push(performance.now() - before); }
  };
  for (let i = 0; i < total; i++) {
    if (signal?.aborted) break;
    const delay = started + i * 1000 / rps - performance.now();
    if (delay > 0) await new Promise(resolve => setTimeout(resolve, delay));
    if (pending.size >= maxInflight) { dropped++; continue; }
    const task = execute(i).finally(() => pending.delete(task));
    pending.add(task);
    const elapsed = performance.now() - started;
    if (onProgress && elapsed - lastProgress >= 30000) {
      lastProgress = elapsed;
      onProgress({ elapsedSec: Math.round(elapsed / 1000), completed: latencies.length, inflight: pending.size, failedRequests: failedRequests.size, dropped });
    }
  }
  await Promise.all(pending);
  latencies.sort((a, b) => a - b);
  const percentile = p => latencies.length ? latencies[Math.min(latencies.length - 1, Math.ceil(latencies.length * p) - 1)] : null;
  const failed = failedRequests.size;
  const elapsedSec = (performance.now() - started) / 1000;
  return {
    startedAt, completedAt: new Date().toISOString(), timingMs: {headers:timingSummary(headerTimes),body:timingSummary(bodyTimes)},
    scope: 'read-only HTTP paths; excludes paid AI, uploads, writes, and native rendering', workload, aborted: signal?.aborted ?? false,
    requestedRps: rps, durationSec, elapsedSec, scheduled: total, completed: latencies.length, dropped,
    achievedRps: latencies.length / elapsedSec, distinctAuthenticatedUsers: exercisedUsers.size,
    statuses, failures, failureDetails, failedRequests: failed, errorRate: (failed + dropped) / total,
    latencyMs: { p50: percentile(0.5), p95: percentile(0.95), p99: percentile(0.99) },
    passed: !signal?.aborted && latencies.length / elapsedSec >= rps * 0.95 && dropped === 0 && failed === 0 && percentile(0.95) !== null && percentile(0.95) < 1000 && percentile(0.99) < 2000,
  };
}

async function main() {
  const plan = capacityPlan();
  if (process.argv.includes('--plan')) { process.stdout.write(JSON.stringify({ assumptions: plan, measurement: 'No requests performed; assumptions need production telemetry.' }, null, 2) + '\n'); return; }
  const users = JSON.parse(await readFile(process.env.LOAD_USERS_FILE, 'utf8'));
  const result = await runLoad({ baseUrl: process.env.LOAD_BASE_URL, apiKey: process.env.LOAD_ANON_KEY, users, rps: Number(process.env.LOAD_RPS || plan.targetRps), durationSec: Number(process.env.LOAD_DURATION_SEC || 600), workload: process.env.LOAD_WORKLOAD || 'lightweight' });
  const report = { measuredAt: new Date().toISOString(), assumptions: plan, ...result };
  if (process.env.LOAD_REPORT_FILE) await writeFile(process.env.LOAD_REPORT_FILE, JSON.stringify(report, null, 2));
  process.stdout.write(JSON.stringify(report, null, 2) + '\n');
  if (!result.passed) process.exitCode = 1;
}
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main().catch(() => { console.error('Load test could not start. Check URL, key, user-file path and configuration.'); process.exitCode = 1; });
