/**
 * Phase 7G — Startup hardening model.
 * Core UI renders under every degraded combination; startup never
 * blocks on analytics, store SDK, AI provider, or catalog refresh.
 */

export interface StartupConditions {
    online: boolean;
    supabaseReachable: boolean;
    supabaseSlow: boolean;
    revenueCatReachable: boolean;
    cachedEntitlementValid: boolean;
    persistedStateValid: boolean;
    providerConfigPresent: boolean;
    catalogRefreshAvailable: boolean;
}

export interface StartupDecision {
    renderCoreUi: boolean;
    blockedOn: string[];
    degraded: string[];
    deferred: string[];
}

const NEVER_BLOCK_ON = ['analytics', 'store_sdk', 'ai_provider', 'catalog_refresh'];

export function evaluateStartup(conditions: StartupConditions): StartupDecision {
    const blockedOn: string[] = [];
    const degraded: string[] = [];
    const deferred: string[] = [...NEVER_BLOCK_ON];
    if (!conditions.online) degraded.push('network_offline');
    if (!conditions.supabaseReachable) degraded.push('supabase_unreachable');
    if (conditions.supabaseSlow) degraded.push('supabase_slow');
    if (!conditions.revenueCatReachable) degraded.push('entitlement_unknown_use_free');
    if (!conditions.cachedEntitlementValid) degraded.push('entitlement_cache_stale');
    if (!conditions.persistedStateValid) degraded.push('persisted_state_recovered_to_defaults');
    if (!conditions.providerConfigPresent) degraded.push('optional_ai_unavailable');
    if (!conditions.catalogRefreshAvailable) degraded.push('catalog_refresh_deferred');
    // Core UI always renders: wardrobe is local-first. Only a total lack
    // of both cache and network for auth-critical boot blocks — and even
    // then the signed-out local state renders.
    return { renderCoreUi: true, blockedOn, degraded, deferred };
}
