/**
 * Phase 7AW–7AZ, 7BA–7BB, 7BC, 7BL–7BN, 7CW — Operations: incident
 * severity, runbooks registry, security/cost runbooks, backup model,
 * corruption safety, support boundary, observability/alert specs,
 * post-launch windows, operating cadence.
 */

import type { DiagnosticPackage, IncidentSeverity } from '../../types/launch';

export const INCIDENT_RESPONSE: Record<IncidentSeverity, { owner: string; immediateAction: string; userCommunication: boolean; rollbackFirst: boolean }> = {
    SEV0: { owner: 'on-call + security', immediateAction: 'Stop risky feature, preserve logs', userCommunication: true, rollbackFirst: true },
    SEV1: { owner: 'on-call', immediateAction: 'Kill switch / rollback to last good', userCommunication: true, rollbackFirst: true },
    SEV2: { owner: 'feature owner', immediateAction: 'Disable degraded feature, fallback on', userCommunication: false, rollbackFirst: true },
    SEV3: { owner: 'feature owner', immediateAction: 'File, schedule fix', userCommunication: false, rollbackFirst: false },
};

export type RunbookId =
    | 'supabase_outage' | 'auth_outage' | 'catalog_outage' | 'ai_provider_outage'
    | 'vton_outage' | 'revenuecat_outage' | 'bad_migration' | 'data_leak_suspicion'
    | 'runaway_cost' | 'broken_paywall' | 'price_corruption';

export const RUNBOOKS: Record<RunbookId, { symptoms: string; firstAction: string; rollback: string }> = {
    supabase_outage: { symptoms: 'Sync/signon failures, RPC errors', firstAction: 'Confirm status; app already offline-degraded', rollback: 'No deploy; wait + retry with backoff' },
    auth_outage: { symptoms: 'Refresh failures, unauthorized reads', firstAction: 'Signed-out local state; never show other-user data', rollback: 'No deploy' },
    catalog_outage: { symptoms: 'Empty search, discovery errors', firstAction: 'Cached catalog + unavailable-with-reason', rollback: 'Flag new_retrieval off' },
    ai_provider_outage: { symptoms: 'Timeouts, malformed outputs', firstAction: 'Circuit breaker opens; deterministic fallback', rollback: 'Kill remote_vision_provider' },
    vton_outage: { symptoms: 'VTON jobs failing', firstAction: 'Best-effort queue pause; history intact', rollback: 'Disable VTON entry' },
    revenuecat_outage: { symptoms: 'Entitlement sync failures', firstAction: 'Unknown → free features; wardrobe intact', rollback: 'No deploy; no paywall change' },
    bad_migration: { symptoms: 'Migration replay errors', firstAction: 'Stop cutover; do not apply further', rollback: 'Roll back additive migration' },
    data_leak_suspicion: { symptoms: 'User B data visible to user A', firstAction: 'SEV0: stop risky feature, preserve logs', rollback: 'Revert to last-known-good policies' },
    runaway_cost: { symptoms: 'Provider spend spike', firstAction: 'Kill switch on high-cost feature; lower limits', rollback: 'Flag off; cache/fallback on' },
    broken_paywall: { symptoms: 'Purchase/restore failing', firstAction: 'Sandbox repro; restore path first', rollback: 'Revert offering change in dashboard' },
    price_corruption: { symptoms: 'Impossible prices, currency mismatch', firstAction: 'Hide price history; label stale', rollback: 'Revert ingestion config' },
};

// ── Backup / recovery (7BA) + corruption safety (7BB) ───────────

export const BACKUP_CHECKLIST = [
    'database_point_in_time_recovery_verified',
    'storage_versioning_or_backup_verified',
    'migration_rollback_rehearsed',
    'local_cache_rebuild_documented',
] as const;

export const CORRUPTION_SIGNALS = [
    'orphan_wardrobe_images',
    'broken_product_family_links',
    'missing_variant',
    'invalid_price',
    'impossible_fit_evidence',
    'malformed_style_aggregate',
    'duplicate_lifecycle_events',
] as const;

/** Prefer repair over destructive cleanup; never delete user data blindly. */
export function corruptionResponse(signal: string): 'repair_tool' | 'quarantine' | 'unknown' {
    if ((CORRUPTION_SIGNALS as readonly string[]).includes(signal)) return 'repair_tool';
    return 'unknown';
}

// ── Observability + alerts (7BL–7BM) ─────────────────────────────

export const OBSERVABILITY_METRICS = [
    'crash_free_sessions', 'startup_success', 'auth_failure_rate', 'catalog_success_rate',
    'outfit_validity_rate', 'provider_error_rate', 'buycheck_latency', 'vton_success_rate',
    'subscription_sync_rate', 'migration_errors', 'cost_estimates',
] as const;

export const ALERT_TO_RUNBOOK: Record<string, RunbookId> = {
    crash_spike: 'supabase_outage',
    cross_user_anomaly: 'data_leak_suspicion',
    provider_failure: 'ai_provider_outage',
    rpc_failure: 'supabase_outage',
    latency_spike: 'catalog_outage',
    billing_sync_failure: 'revenuecat_outage',
    cost_spike: 'runaway_cost',
    migration_error: 'bad_migration',
};

// ── Post-launch windows (7BN) + cadence (7CW) ────────────────────

export const POST_LAUNCH_WINDOWS = [
    { window: 'first_hour', watch: 'crashes, startup, auth, fatal provider errors' },
    { window: 'first_24h', watch: 'activation funnel, billing sync, cost rate' },
    { window: 'first_7d', watch: 'retention, recommendation outcomes, guardrails' },
    { window: 'first_30d', watch: 'pricing signals, roadmap inputs, architecture debt' },
] as const;

export const OPERATING_CADENCE = {
    daily: 'crashes, auth, provider failure, billing, hard-rule violations',
    weekly: 'activation, retention, recommendation outcomes, cost, quality debt',
    monthly: 'pricing, roadmap, provider/model comparison, architecture debt, privacy/security review',
} as const;

// ── Support boundary (7BC) ───────────────────────────────────────

export function sanitizeDiagnostics(pkg: DiagnosticPackage): DiagnosticPackage {
    // Allowlist shape: the type itself excludes photos/body data/secrets.
    return { ...pkg, requestIds: pkg.requestIds.slice(0, 20) };
}
