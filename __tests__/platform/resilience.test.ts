/**
 * resilience.test.ts — provider health, circuit breaker, bounded retries,
 * backpressure, degraded mode, error taxonomy + UX.
 */
import {
  recordProviderOutcome, providerHealth, clearProviderHealth,
  shouldAllow, maxAttempts, shouldRetry,
  tryAcquire, release, resetGates,
  capabilityMode, degradedLabel, classifyError, userCopy,
} from '../../src/services/platform/resilience';

beforeEach(() => { clearProviderHealth(); resetGates(); });

test('provider health degrades then recovers through half-open', () => {
  recordProviderOutcome('vision', 'ok', 100);
  expect(providerHealth('vision').state).toBe('healthy');
  recordProviderOutcome('vision', 'timeout', 5000);
  recordProviderOutcome('vision', 'timeout', 5000);
  expect(providerHealth('vision').state).toBe('degraded');
  expect(providerHealth('vision').timeoutRate).toBeGreaterThan(0);
  // circuit opens at threshold
  const opts = { failureThreshold: 2, cooldownMs: 1000 };
  expect(shouldAllow('vision', opts)).toBe(false); // opens
  expect(shouldAllow('vision', opts)).toBe(false); // still open
  expect(shouldAllow('vision', opts, Date.now() + 2000)).toBe(true); // half-open recovery
});

test('retries are bounded; invalid input and validation never retry', () => {
  expect(maxAttempts('network-timeout')).toBe(2);
  expect(shouldRetry('network-timeout', 1)).toBe(true);
  expect(shouldRetry('network-timeout', 2)).toBe(false);
  expect(shouldRetry('invalid-input', 0)).toBe(false);
  expect(shouldRetry('validation-failure', 0)).toBe(false);
  expect(shouldRetry('malformed-output', 1)).toBe(true);
});

test('backpressure dedupes then rejects; core wardrobe never blocked', () => {
  const g = { maxConcurrent: 1, queueLimit: 1 };
  expect(tryAcquire('gen', g)).toBe('run');
  expect(tryAcquire('gen', g)).toBe('dedupe');
  expect(tryAcquire('gen', g)).toBe('reject');
  release('gen');
  expect(capabilityMode({ network: 'down', aiProvider: 'down', priceService: 'down' }, 'wardrobe-core')).toBe('full');
  expect(capabilityMode({ network: 'ok', aiProvider: 'down', priceService: 'ok' }, 'visual-ai')).toBe('degraded');
  expect(capabilityMode({ network: 'ok', aiProvider: 'down', priceService: 'ok' }, 'vton')).toBe('unavailable');
  expect(degradedLabel('full', 'x')).toBeNull();
  expect(degradedLabel('degraded', 'Visual analysis')!).toMatch(/saved local data/i);
});

test('error taxonomy is stable and user copy is actionable', () => {
  expect(classifyError(new Error('request timed out'))).toBe('PROVIDER_TIMEOUT');
  expect(classifyError(new Error('429 too many'))).toBe('PROVIDER_RATE_LIMIT');
  expect(classifyError(new Error('offline'))).toBe('NETWORK_UNAVAILABLE');
  expect(classifyError('weird string')).toBe('VALIDATION_FAILED');
  const copy = userCopy('PROVIDER_TIMEOUT');
  expect(copy.actions).toContain('fallback');
  expect(copy.title).not.toMatch(/something went wrong/i);
});
