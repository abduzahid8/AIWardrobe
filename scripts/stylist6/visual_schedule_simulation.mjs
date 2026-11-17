// Deterministic counterfactual using supplied stage durations, not provider I/O.
import { pathToFileURL } from 'node:url';
import { VISUAL_CONCURRENCY, VISUAL_PROVIDER_TIMEOUT_MS, VISUAL_TIMEOUT_MS } from '../../backend/stylist6/visual_critic_hosted.mjs';

export function simulateSchedule(durations, { concurrency, batchMs, providerMs = Infinity }) {
  const available = Array(concurrency).fill(0);
  const events = durations.map(({ media, provider }, index) => {
    const worker = available.indexOf(Math.min(...available));
    const queued = available[worker], dispatch = queued + media;
    if (queued >= batchMs) return { index, queued, dispatched: false, reason: 'batch_deadline_queued' };
    if (dispatch >= batchMs) {
      available[worker] = batchMs;
      return { index, queued, dispatched: false, reason: 'batch_deadline_media' };
    }
    const end = Math.min(dispatch + provider, batchMs, dispatch + providerMs);
    available[worker] = end;
    const reason = dispatch + provider <= end ? null
      : end === batchMs ? 'batch_deadline_in_flight' : 'provider_deadline';
    return { index, queued, dispatch, end, dispatched: true, reason };
  });
  return {
    wallMs: Math.max(...available),
    calls: events.filter(({ dispatched }) => dispatched).length,
    complete: events.filter(({ reason }) => reason === null).length,
    events,
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const policies = {
    sequential12s: { concurrency: 1, batchMs: 12_000 },
    twoWorker12s: { concurrency: 2, batchMs: 12_000 },
    separateDeadlines: { concurrency: VISUAL_CONCURRENCY, batchMs: VISUAL_TIMEOUT_MS, providerMs: VISUAL_PROVIDER_TIMEOUT_MS },
  };
  const scenarios = {
    // Rounded representative seven-second calls observed in prior receipts.
    twoWaves: [7_000, 7_000, 7_000, 7_000],
    // Distinguishes an inherently slow attempt from a worker that waited.
    oneSlow: [12_000, 6_000, 6_000, 6_000],
  };
  console.log(JSON.stringify(Object.fromEntries(Object.entries(scenarios).map(([name, times]) => [
    name, Object.fromEntries(Object.entries(policies).map(([id, policy]) => [
      id, simulateSchedule(times.map((provider) => ({ media: 20, provider })), policy),
    ])),
  ])), null, 2));
}
