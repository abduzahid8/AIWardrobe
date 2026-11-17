/**
 * Phase 7 — Launch readiness domain types.
 * Stabilization phase: no new product domains, only safety,
 * traceability, and the production operating model.
 */

export type ReadinessState = 'PASS' | 'CONDITIONAL' | 'BLOCKED' | 'MANUAL_REVIEW';

export type BlockerSeverity = 'BLOCKER' | 'HIGH' | 'MEDIUM' | 'LOW';

export interface LaunchBlocker {
    id: string;
    severity: BlockerSeverity;
    area: string;
    description: string;
    resolvesTo: string;
}

/** Data classification (7T) — drives logs, storage, retention, export. */
export type DataClass =
    | 'public'
    | 'account'
    | 'wardrobe_private'
    | 'body_fit_private'
    | 'purchase_private'
    | 'image_private'
    | 'operational'
    | 'analytics';

export type IncidentSeverity = 'SEV0' | 'SEV1' | 'SEV2' | 'SEV3';

export type FeatureReadiness =
    | 'local_complete'
    | 'needs_remote_activation'
    | 'beta'
    | 'production_candidate'
    | 'blocked'
    | 'disabled';

export type AppEnvironment = 'development' | 'preview' | 'production';

export type LocalStateVersion = number;

/** User-safe diagnostic export (7BD) — never photos, body data, secrets. */
export interface DiagnosticPackage {
    appVersion: string;
    environment: AppEnvironment;
    requestIds: string[];
    featureFlags: Record<string, string>;
    sanitizedErrors: string[];
    subscriptionCategory: 'free' | 'premium' | 'unknown';
    syncStatus: 'ok' | 'degraded' | 'offline';
}
