import test from 'node:test';
import assert from 'node:assert/strict';
import { simulateSchedule } from './visual_schedule_simulation.mjs';

test('deadline-aware two-worker scheduling completes two measured-length waves', () => {
  const cases = Array.from({ length: 4 }, () => ({ media: 20, provider: 7000 }));
  assert.equal(simulateSchedule(cases, { concurrency: 1, batchMs: 12000 }).complete, 1);
  const old = simulateSchedule(cases, { concurrency: 2, batchMs: 12000 });
  assert.equal(old.complete, 2);
  assert.deepEqual(old.events.slice(2).map(({ reason }) => reason), ['batch_deadline_in_flight', 'batch_deadline_in_flight']);
  const revised = simulateSchedule(cases, { concurrency: 2, batchMs: 18000, providerMs: 8000 });
  assert.equal(revised.complete, 4);
  assert.equal(revised.calls, 4);
  assert.equal(revised.wallMs, 14040);
});

test('an inherent provider overrun is isolated from the other three calls', () => {
  const cases = [12000, 6000, 6000, 6000].map((provider) => ({ media: 20, provider }));
  const revised = simulateSchedule(cases, { concurrency: 2, batchMs: 18000, providerMs: 8000 });
  assert.equal(revised.complete, 3);
  assert.equal(revised.calls, 4);
  assert.equal(revised.events[0].reason, 'provider_deadline');
  assert.equal(revised.wallMs, 14040);
});
