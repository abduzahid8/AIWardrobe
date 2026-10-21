// Normalize verified RevenueCat Customer Info; never infer access from event order.
export type PaidTier = 'lite' | 'premium' | 'vip';
const PRODUCTS: Record<string, PaidTier> = {
  '2.99': 'lite', 'com.aiwardrobe.lite.monthly': 'lite',
  'com.aiwardrobe.premium.monthly': 'premium', 'com.aiwardrobe.premium.yearly': 'vip',
};
const ENTITLEMENTS: Record<string, PaidTier> = {
  lite: 'lite', aiwardrobe_lite: 'lite', premium: 'premium', pro: 'premium',
  'aiwardrobe pro': 'premium', vip: 'vip', max: 'vip', aiwardrobe_pro_yearly: 'vip',
};
export function normalizeRevenueCatSnapshot(userId: string, payload: any, environment: 'PRODUCTION' | 'SANDBOX') {
  if (!Number.isSafeInteger(payload?.request_date_ms) || payload.request_date_ms <= 0 ||
      !payload.subscriber || typeof payload.subscriber.entitlements !== 'object' || !payload.subscriber.entitlements || Array.isArray(payload.subscriber.entitlements)) {
    throw new Error('Invalid RevenueCat customer response');
  }
  const subscriber = payload.subscriber;
  const entries = new Map<string, any>();
  for (const [key, entitlement] of Object.entries(subscriber.entitlements) as Array<[string, any]>) {
    if (!entitlement || typeof entitlement.product_identifier !== 'string' || !Object.prototype.hasOwnProperty.call(entitlement, 'expires_date')) throw new Error('Invalid RevenueCat entitlement');
    const product = entitlement.product_identifier;
    const metadata = subscriber.subscriptions?.[product] ||
      [...(subscriber.non_subscriptions?.[product] || [])].sort((a: any, b: any) => Date.parse(b.purchase_date) - Date.parse(a.purchase_date))[0];
    if (!metadata || typeof metadata.is_sandbox !== 'boolean') throw new Error('Missing RevenueCat purchase metadata');
    if (metadata.is_sandbox !== (environment === 'SANDBOX')) continue;
    const date = (value: unknown): number | null => {
      if (value == null) return null;
      if (typeof value !== 'string' || !Number.isFinite(Date.parse(value))) throw new Error('Invalid RevenueCat purchase date');
      return Date.parse(value);
    };
    const expiry = date(entitlement.expires_date);
    const grace = date(entitlement.grace_period_expires_date || metadata.grace_period_expires_date);
    const effectiveExpiry = expiry === null ? null : Math.max(expiry, grace || 0);
    if (effectiveExpiry !== null && effectiveExpiry <= payload.request_date_ms) continue;
    const tier = PRODUCTS[product.toLowerCase()] || ENTITLEMENTS[key.toLowerCase()] || PRODUCTS[key.toLowerCase()];
    if (!tier) throw new Error('Unmapped active RevenueCat entitlement');
    const purchased = date(entitlement.purchase_date || metadata.purchase_date);
    if (purchased === null) throw new Error('Missing RevenueCat purchase date');
    const row = {
      tier, product_id: product,
      end_date: effectiveExpiry === null ? null : new Date(effectiveExpiry).toISOString(),
      start_date: new Date(purchased).toISOString(),
      status: metadata.unsubscribe_detected_at ? 'cancelled' : metadata.period_type === 'trial' ? 'trial' : 'active',
      auto_renew: !metadata.unsubscribe_detected_at && !subscriber.non_subscriptions?.[product],
      platform: metadata.store === 'app_store' ? 'apple' : metadata.store === 'play_store' ? 'google' : metadata.store === 'stripe' ? 'stripe' : 'revenuecat',
    };
    const previous = entries.get(product);
    if (!previous || ['lite', 'premium', 'vip'].indexOf(tier) > ['lite', 'premium', 'vip'].indexOf(previous.tier)) entries.set(product, row);
  }
  return { user_id: userId, request_date_ms: payload.request_date_ms, entitlements: [...entries.values()] };
}
