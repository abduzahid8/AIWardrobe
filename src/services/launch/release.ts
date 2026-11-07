/**
 * Phase 7AO–7AS, 7AT–7AV, 7CI, 7CM–7CO, 7CN, 7CP — Release engineering:
 * environment separation, versioning, compatibility matrix, cutover plan,
 * dependency graph, canary rollout, launch-safe flag defaults, release
 * manifest, release-candidate check, readiness scorecard, activation packet.
 */

import type { AppEnvironment, FeatureReadiness, LaunchBlocker, ReadinessState } from '../../types/launch';

// ── Environment separation (7AO) ─────────────────────────────────

export const ENV_CONFIG: Record<AppEnvironment, { apiEndpoint: 'production' | 'staging' | 'local'; logVerbosity: 'debug' | 'info' | 'warn'; mockProviders: boolean }> = {
    development: { apiEndpoint: 'local', logVerbosity: 'debug', mockProviders: true },
    preview: { apiEndpoint: 'staging', logVerbosity: 'info', mockProviders: false },
    production: { apiEndpoint: 'production', logVerbosity: 'warn', mockProviders: false },
};

export function productionPointsAtDev(env: AppEnvironment, apiUrl: string): boolean {
    if (env !== 'production') return false;
    return /localhost|staging|dev|onrender.*preview/i.test(apiUrl);
}

// ── Versioning (7AQ) + compatibility (7AS) ───────────────────────

export interface ReleaseVersion {
    appVersion: string; // semver
    buildNumber: number; // Android versionCode / iOS build
    minStorageVersion: number;
    backendCompatibilityWindow: string;
}

export const MIN_SUPPORTED_CLIENT = '3.0.0';

export function clientSupported(clientVersion: string): 'supported' | 'recommended_update' | 'hard_update' {
    const cmp = compareSemver(clientVersion, MIN_SUPPORTED_CLIENT);
    if (cmp < 0) return 'hard_update';
    if (cmp === 0) return 'recommended_update';
    return 'supported';
}

function compareSemver(a: string, b: string): number {
    const pa = a.split('.').map(Number);
    const pb = b.split('.').map(Number);
    for (let i = 0; i < 3; i++) {
        if ((pa[i] ?? 0) !== (pb[i] ?? 0)) return (pa[i] ?? 0) < (pb[i] ?? 0) ? -1 : 1;
    }
    return 0;
}

export type CompatDirection = 'old_client_new_backend' | 'new_client_old_backend';

export function compatibilityExpectation(input: {
    direction: CompatDirection;
    rpcPresent: boolean;
    migrationApplied: boolean;
}): string {
    if (input.rpcPresent && input.migrationApplied) return 'full';
    // Additive migrations + flag-guarded RPC callers ⇒ degraded, never crash.
    return 'degraded_with_fallback';
}

// ── Cutover plan (7AT) + dependency graph (7AU) ──────────────────

export interface CutoverStep {
    order: number;
    action: string;
    verify: string;
    rollback: string;
}

export function productionCutoverPlan(): CutoverStep[] {
    return [
        { order: 1, action: 'Apply additive DB migrations to hosted project', verify: 'Schema + RLS replay clean', rollback: 'Roll back migration; clients unaffected (additive only)' },
        { order: 2, action: 'Validate schema/RLS on hosted project', verify: 'RLS isolation probes pass', rollback: 'Fix policy, re-validate' },
        { order: 3, action: 'Deploy compatible Edge Functions', verify: 'Version endpoint reports expected set', rollback: 'Redeploy previous version' },
        { order: 4, action: 'Enable read paths behind flags', verify: 'Read-path probes pass', rollback: 'Flag off' },
        { order: 5, action: 'Release client with flags off', verify: 'Store rollout staged pause', rollback: 'Halt staged rollout' },
        { order: 6, action: 'Enable internal cohort', verify: 'Guardrails green 24h', rollback: 'Flag off for cohort' },
        { order: 7, action: 'Observe (crash, auth, provider, billing)', verify: 'Alerting spec thresholds hold', rollback: 'Cohort rollback' },
        { order: 8, action: 'Gradual feature activation per canary plan', verify: 'Stage gates pass', rollback: 'Per-stage flag off' },
    ];
}

export const DEPLOYMENT_DEPENDENCIES: Record<string, string[]> = {
    buycheck_v2: ['migration:phase2', 'rpc:buycheck', 'flag:buycheck_v2'],
    visual_parser: ['provider:vision', 'flag:visual_parser', 'kill:remote_vision_provider'],
    price_watch: ['migration:phase6', 'job:price_checks', 'entitlement:price_watch'],
    recreate_look: ['provider:vision', 'flag:recreate_look'],
};

// ── Canary (7AV) ─────────────────────────────────────────────────

export type CanaryStage = 'internal' | 'beta' | '1pct' | '5pct' | '25pct' | '50pct' | '100pct';

export interface CanaryGate {
    stage: CanaryStage;
    entryCriteria: string;
    abortThreshold: string;
    rollbackAction: string;
}

export function canaryPlan(): CanaryGate[] {
    const mk = (stage: CanaryStage, entry: string): CanaryGate => ({
        stage, entryCriteria: entry,
        abortThreshold: 'Crash spike, cross-user anomaly, provider failure rate, or guardrail breach',
        rollbackAction: 'Flag off for stage; previous client remains compatible (additive backend)',
    });
    return [
        mk('internal', 'Team accounts, 24h green'),
        mk('beta', 'Internal green 24h'),
        mk('1pct', 'Beta green 48h'),
        mk('5pct', '1% green 48h'),
        mk('25pct', '5% green 72h'),
        mk('50pct', '25% green 72h'),
        mk('100pct', '50% green 7d'),
    ];
}

// ── Launch-safe flag defaults (7CI) ──────────────────────────────

export interface FlagDefault {
    flag: string;
    default: 'on' | 'off' | 'limited';
    owner: string;
    killSwitch: string;
}

export function launchFlagDefaults(): FlagDefault[] {
    return [
        { flag: 'buycheck_v2', default: 'off', owner: 'purchase', killSwitch: 'flag:buycheck_v2' },
        { flag: 'fit_dna_v2', default: 'off', owner: 'fit', killSwitch: 'flag:fit_dna_v2' },
        { flag: 'visual_parser', default: 'off', owner: 'vision', killSwitch: 'kill:remote_vision_provider' },
        { flag: 'recreate_look', default: 'off', owner: 'vision', killSwitch: 'kill:remote_vision_provider' },
        { flag: 'daily_stylist', default: 'off', owner: 'growth', killSwitch: 'flag:daily_stylist' },
        { flag: 'new_retrieval', default: 'off', owner: 'shopping', killSwitch: 'flag:new_retrieval' },
        { flag: 'new_ranking', default: 'off', owner: 'stylist', killSwitch: 'flag:new_ranking' },
        { flag: 'wardrobe_core', default: 'on', owner: 'wardrobe', killSwitch: 'none (core is never killed)' },
    ];
}

// ── Release manifest (7CM) + RC check (7CN) ──────────────────────

export interface ReleaseManifest {
    appVersion: string;
    buildNumber: number;
    gitCommit: string;
    migrations: string[];
    edgeFunctionVersions: Record<string, string>;
    featureFlagDefaults: string;
    promptSchemaVersions: Record<string, number>;
    ruleVersions: Record<string, string>;
    knownLimitations: string[];
}

export type RcCheckItem = { check: string; result: 'PASS' | 'FAIL' | 'MANUAL'; detail: string };

export function releaseCandidateSummary(input: {
    typecheck: boolean;
    lint: boolean;
    criticalTests: boolean;
    fullRegression: boolean;
    sqlTests: boolean;
    migrationOrder: boolean;
    secretsScan: boolean;
    storeConfig: boolean;
}): RcCheckItem[] {
    const row = (check: string, ok: boolean | null, detail: string): RcCheckItem => ({
        check, result: ok === null ? 'MANUAL' : ok ? 'PASS' : 'FAIL', detail,
    });
    return [
        row('typecheck', input.typecheck, 'tsc --noEmit'),
        row('lint', input.lint, 'npm run lint'),
        row('critical_tests', input.criticalTests, 'Phase 1–6 focused suites'),
        row('full_regression', input.fullRegression, 'Full jest run; pre-existing failures listed'),
        row('sql_tests', input.sqlTests, 'Migration replay on CI postgres'),
        row('migration_order', input.migrationOrder, 'Additive, ordered, rollback documented'),
        row('secrets_scan', input.secretsScan, 'No keys in client/history/fixtures'),
        row('store_config', input.storeConfig, 'Checklist reviewed, no claim of approval'),
        row('manual_qa', null, 'Device/upgrade/offline/billing scripts pending hands-on'),
        row('remote_activation', null, 'Packet prepared, nothing applied'),
    ];
}

// ── Readiness scorecard (7CO) — never averaged ───────────────────

export const SCORECARD_DIMENSIONS = [
    'security', 'privacy', 'reliability', 'performance', 'data_integrity',
    'billing', 'ai_quality', 'ux', 'accessibility', 'store_readiness',
    'observability', 'rollback',
] as const;

export type ScorecardDimension = (typeof SCORECARD_DIMENSIONS)[number];

export function buildScorecard(ratings: Partial<Record<ScorecardDimension, ReadinessState>>): Record<ScorecardDimension, ReadinessState> {
    const out = {} as Record<ScorecardDimension, ReadinessState>;
    for (const d of SCORECARD_DIMENSIONS) out[d] = ratings[d] ?? 'MANUAL_REVIEW';
    return out;
}

/** A single BLOCKED dimension blocks launch; no averaging. */
export function launchBlockedBy(scorecard: Record<ScorecardDimension, ReadinessState>): ScorecardDimension[] {
    return SCORECARD_DIMENSIONS.filter((d) => scorecard[d] === 'BLOCKED');
}

// ── Feature readiness matrix (7BR) + blocker matrix (7BS) ────────

export function featureReadinessMatrix(): { feature: string; readiness: FeatureReadiness; dependency: string }[] {
    return [
        { feature: 'wardrobe_core', readiness: 'production_candidate', dependency: 'local persistence' },
        { feature: 'outfit_generation', readiness: 'production_candidate', dependency: 'ranking v1' },
        { feature: 'buycheck', readiness: 'production_candidate', dependency: 'phase2 rpc' },
        { feature: 'visual_parser', readiness: 'needs_remote_activation', dependency: 'vision provider' },
        { feature: 'vton', readiness: 'needs_remote_activation', dependency: 'vton provider (best-effort)' },
        { feature: 'price_watch', readiness: 'needs_remote_activation', dependency: 'phase6 migration + jobs' },
        { feature: 'subscriptions', readiness: 'needs_remote_activation', dependency: 'RevenueCat offerings' },
        { feature: 'notifications', readiness: 'needs_remote_activation', dependency: 'sender service' },
        { feature: 'recreate_look', readiness: 'beta', dependency: 'vision provider' },
    ];
}

export function launchBlockers(): LaunchBlocker[] {
    return [
        { id: 'B-cross-user', severity: 'BLOCKER', area: 'security', description: 'Any cross-user data exposure', resolvesTo: 'RLS probes green' },
        { id: 'B-data-loss', severity: 'BLOCKER', area: 'data', description: 'Wardrobe/history loss on upgrade or delete', resolvesTo: 'Upgrade tests green' },
        { id: 'B-restore', severity: 'BLOCKER', area: 'billing', description: 'Broken restore purchases', resolvesTo: 'Sandbox restore script passes' },
        { id: 'B-startup-crash', severity: 'BLOCKER', area: 'reliability', description: 'Crash on startup (cold/offline)', resolvesTo: 'Startup matrix green' },
        { id: 'B-deletion', severity: 'BLOCKER', area: 'privacy', description: 'Account deletion unavailable where required', resolvesTo: 'Deletion plan implemented' },
        { id: 'B-secret', severity: 'BLOCKER', area: 'security', description: 'Secret in client bundle or history', resolvesTo: 'Secret scan clean' },
        { id: 'B-migration', severity: 'BLOCKER', area: 'data', description: 'Destructive migration without rollback', resolvesTo: 'Additive + rollback doc' },
        { id: 'B-entitlement-forge', severity: 'BLOCKER', area: 'billing', description: 'Client-forged entitlement accepted server-side', resolvesTo: 'Server verifier authoritative' },
        { id: 'B-unbounded-ai', severity: 'BLOCKER', area: 'cost', description: 'Unbounded expensive API call path', resolvesTo: 'Budgets + kill switches' },
    ];
}
