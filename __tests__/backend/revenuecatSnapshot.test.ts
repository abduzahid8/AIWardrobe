import { normalizeRevenueCatSnapshot } from '../../backend/supabase/functions/_shared/revenuecatSnapshot';
const now = Date.parse('2026-10-15T10:00:00Z');
const snapshot = (changes: Record<string, unknown> = {}) => ({
  request_date_ms: now,
  subscriber: {
    entitlements: { premium: { product_identifier: 'com.aiwardrobe.premium.monthly', expires_date: '2026-11-01T00:00:00Z', purchase_date: '2026-10-01T00:00:00Z' } },
    subscriptions: { 'com.aiwardrobe.premium.monthly': { is_sandbox: false, period_type: 'normal', store: 'app_store', ...changes } },
  },
});
test('preserves a cancelled but unexpired paid period', () => {
  const result = normalizeRevenueCatSnapshot('owner', snapshot({ unsubscribe_detected_at: '2026-10-14T00:00:00Z' }), 'PRODUCTION');
  expect(result.entitlements).toEqual([expect.objectContaining({ tier: 'premium', status: 'cancelled', auto_renew: false, end_date: '2026-11-01T00:00:00.000Z' })]);
});
test('does not mix sandbox receipts into production access', () => {
  expect(normalizeRevenueCatSnapshot('owner', snapshot({ is_sandbox: true }), 'PRODUCTION').entitlements).toEqual([]);
});
test('preserves explicit billing grace beyond nominal expiration', () => {
  const input = snapshot({ grace_period_expires_date: '2026-10-18T00:00:00Z' });
  input.subscriber.entitlements.premium.expires_date = '2026-10-14T00:00:00Z';
  expect(normalizeRevenueCatSnapshot('owner', input, 'PRODUCTION').entitlements[0].end_date).toBe('2026-10-18T00:00:00.000Z');
});
test('expired entitlements grant no access', () => {
  const input = snapshot();
  input.subscriber.entitlements.premium.expires_date = '2026-10-14T00:00:00Z';
  expect(normalizeRevenueCatSnapshot('owner', input, 'PRODUCTION').entitlements).toEqual([]);
});
test('malformed customer info cannot silently clear or grant access', () => {
  for (const input of [{}, { request_date_ms: now, subscriber: { entitlements: [] } }, snapshot({ is_sandbox: undefined })]) {
    expect(() => normalizeRevenueCatSnapshot('owner', input, 'PRODUCTION')).toThrow();
  }
});
test('unknown active entitlement needs mapping rather than silently downgrading', () => {
  const input: any = snapshot();
  input.subscriber.entitlements = { custom: { product_identifier: 'new-plan', expires_date: null, purchase_date: '2026-10-01T00:00:00Z' } };
  input.subscriber.subscriptions['new-plan'] = { is_sandbox: false };
  expect(() => normalizeRevenueCatSnapshot('owner', input, 'PRODUCTION')).toThrow('Unmapped');
});
