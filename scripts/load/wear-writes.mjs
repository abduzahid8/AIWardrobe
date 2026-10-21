import { responseDiagnostics, timingSummary } from './http-diagnostics.mjs';
import { performance, monitorEventLoopDelay } from 'node:perf_hooks';
export async function runWearWrites({ baseUrl, apiKey, users, rps = 40, durationSec = 120, maxInflight = 100, fetchImpl = fetch, onProgress, signal }) {
  if (!/^https?:\/\//.test(baseUrl) || !apiKey || !users?.length
    || users.some(user => !user.id || !user.accessToken || !user.itemId || !user.logIds?.length)) throw new Error('Incomplete write fixtures');
  for (const value of [rps, durationSec, maxInflight]) if (!Number.isFinite(value) || value <= 0) throw new Error('Invalid load settings');
  const startedAt = new Date().toISOString();
  const headerTimes = [], bodyTimes = [];
  const planned = Math.ceil(rps * durationSec), started = performance.now(), wallStarted = Date.now();
  const eventLoop = monitorEventLoopDelay({ resolution: 20 }); eventLoop.enable();
  const upstreamTimes = [], errorTypes = {}, perWindow = {};
  let maxScheduleLagMs = 0;
  const pending = new Set(), latencies = [], statuses = {}, failures = {}, failureDetails = [];
  let dropped = 0, completed = 0, failed = 0, progressAt = 0;
  const execute = async index => {
    const user = users[index % users.length];
    const logId = user.logIds[Math.floor(Math.floor(index / users.length) / 2)];
    const before = performance.now();
    const requestStartedAt = new Date().toISOString();
    let headersAt, responseInfo = {};
    const window = Math.floor((before - started) / 10000);
    const bucket = perWindow[window] ??= { attempted: 0, failed: 0 }; bucket.attempted++;
    let stage = 'fetch';
    try {
      if (!logId) throw new Error('FixtureRangeError');
      const response = await fetchImpl(new URL('/rest/v1/wear_logs?on_conflict=id', baseUrl), {
        method: 'POST', redirect: 'error', headers: { apikey: apiKey, Authorization: `Bearer ${user.accessToken}`,
          'Content-Type': 'application/json', Prefer: 'resolution=merge-duplicates,return=representation' },
        body: JSON.stringify({ id: logId, user_id: user.id, item_ids: [user.itemId, user.itemId],
          date: '2026-10-16', created_at: '2026-10-16T12:00:00Z' }),
        signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(10000)]) : AbortSignal.timeout(10000),
      });
      headersAt = performance.now();
      headerTimes.push(headersAt - before);
      responseInfo = {status:response.status,...responseDiagnostics(response)};
      statuses[response.status] = (statuses[response.status] || 0) + 1;
      const upstreamHeader = response.headers.get('x-envoy-upstream-service-time');
      if (upstreamHeader !== null && /^\d+(\.\d+)?$/.test(upstreamHeader)) upstreamTimes.push(Number(upstreamHeader));
      if (![200, 201].includes(response.status)) { await response.arrayBuffer(); throw new Error(`http_${response.status}`); }
      stage = 'decode_json';
      const rows = await response.json();
      stage = 'validate_confirmation';
      if (!Array.isArray(rows) || rows.length !== 1 || rows[0].id !== logId || rows[0].user_id !== user.id
          || !Array.isArray(rows[0].item_ids) || rows[0].item_ids.some(id => id !== user.itemId)) throw new Error('invalid_write_confirmation');
    } catch (error) {
      failed++;
      bucket.failed++;
      const errorType = /^[A-Za-z]+Error$/.test(error.name || '') ? error.name : 'UnknownError';
      errorTypes[errorType] = (errorTypes[errorType] || 0) + 1;
      const reason = /^(http_\d+|invalid_write_confirmation|FixtureRangeError)$/.test(error.message) ? error.message : 'transport_or_decode';
      failures[reason] = (failures[reason] || 0) + 1;
      if (failureDetails.length < 10) failureDetails.push({ requestIndex: index, stage, startedAt: requestStartedAt, ...responseInfo,
        errorType,
        ...(typeof error.cause?.code === 'string' && /^[A-Z0-9_]+$/.test(error.cause.code) ? { causeCode: error.cause.code } : {}),
      });
    } finally { if (headersAt !== undefined) bodyTimes.push(performance.now() - headersAt); completed++; latencies.push(performance.now() - before); }
  };
  for (let index = 0; index < planned; index++) {
    if (signal?.aborted) break;
    const delay = started + index * 1000 / rps - performance.now();
    if (delay > 0) await new Promise(resolve => setTimeout(resolve, delay));
    if (signal?.aborted) break;
    maxScheduleLagMs = Math.max(maxScheduleLagMs, performance.now() - started - index * 1000 / rps);
    if (pending.size >= maxInflight) { dropped++; continue; }
    const task = execute(index).finally(() => pending.delete(task)); pending.add(task);
    if (onProgress && performance.now() - progressAt - started >= 30000) {
      progressAt = performance.now() - started;
      onProgress({ scheduled: index + 1, completed, failed, dropped });
    }
  }
  await Promise.all(pending);
  eventLoop.disable();
  const elapsedSec = (performance.now() - started) / 1000;
  latencies.sort((a, b) => a - b);
  const percentile = value => latencies.length ? latencies[Math.min(latencies.length - 1, Math.ceil(value * latencies.length) - 1)] : null;
  const result = { startedAt, completedAt: new Date().toISOString(), timingMs: {headers:timingSummary(headerTimes),body:timingSummary(bodyTimes)}, planned, completed, failed, dropped, elapsedSec, achievedRps: completed / elapsedSec, statuses, failures, failureDetails,
    latencyMs: { p50: percentile(.5), p95: percentile(.95), p99: percentile(.99) }, aborted: Boolean(signal?.aborted),
    diagnostics: { errorTypes, tenSecondWindows: perWindow, maxScheduleLagMs, wallElapsedMs: Date.now() - wallStarted,
      eventLoopDelayMs: { mean: Number.isFinite(eventLoop.mean) ? eventLoop.mean / 1e6 : null, max: eventLoop.max / 1e6 },
      upstreamHeaderSamples: upstreamTimes.length, upstreamHeaderMaxMs: upstreamTimes.length ? Math.max(...upstreamTimes) : null,
      runtime: { node: process.version, platform: process.platform, arch: process.arch } },
  };
  result.passed = !result.aborted && completed === planned && failed === 0 && dropped === 0
    && result.achievedRps >= rps * .95 && result.latencyMs.p95 < 1000 && result.latencyMs.p99 < 2000;
  return result;
}
