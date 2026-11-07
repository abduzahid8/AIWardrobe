import { CRITICAL_JOURNEYS, journeysForArea } from '../../src/services/launch/journeys';
import { migrateStoredState, wrapWithVersion } from '../../src/services/launch/storage';
import { evaluateStartup } from '../../src/services/launch/startup';
import { validateImageUpload, sanitizeFilename, validateIngestUrl } from '../../src/services/launch/urlSafety';
import { deletionPlan, buildExportManifest, maySendToExternalAi, classifyField, defaultPrivacyControls, buildDiagnosticPackage } from '../../src/services/launch/privacy';
import {
    ENV_CONFIG, productionPointsAtDev, clientSupported, compatibilityExpectation,
    productionCutoverPlan, canaryPlan, launchFlagDefaults, releaseCandidateSummary,
    buildScorecard, launchBlockedBy, featureReadinessMatrix, launchBlockers,
} from '../../src/services/launch/release';
import { INCIDENT_RESPONSE, RUNBOOKS, corruptionResponse, ALERT_TO_RUNBOOK, sanitizeDiagnostics } from '../../src/services/launch/operations';
import { claimSafe, externalPurchaseDisclosure } from '../../src/services/launch/storeReadiness';
import { validateDeepLink, validateShareInput, staleNotificationCutoffHours, permissionMatrix, scanSourceForDebugLeakage, logLevelFor, sanitizeLogPayload } from '../../src/services/launch/platform';
import { triagePriority, loadModel, webhookIdempotencyKey, KNOWN_LIMITATIONS } from '../../src/services/launch/quality';

describe('critical journeys', () => {
    test('inventory covers all 8 areas; every journey has fallback + entitlement', () => {
        const areas = new Set(CRITICAL_JOURNEYS.map((j) => j.area));
        expect(areas.size).toBe(8);
        for (const j of CRITICAL_JOURNEYS) {
            expect(j.fallback).toBeTruthy();
            expect(j.entitlement).toBeTruthy();
            expect(j.entries.length).toBeGreaterThan(0);
        }
        expect(journeysForArea('subscription').length).toBeGreaterThan(0);
    });
});

describe('local storage migrations', () => {
    test('current passes through; old migrates; corrupt resets without throwing', () => {
        const cur = wrapWithVersion('growth', { a: 1 });
        expect(migrateStoredState({ slice: 'growth', stored: cur, defaults: { a: 0 } }).status).toBe('current');
        const migrated = migrateStoredState({
            slice: 'growth', stored: { schemaVersion: 0, data: { a: 1 } }, defaults: { a: 0, v: 0 },
            migrate: (v, d) => ({ a: (d as { a: number }).a, v }),
        });
        expect(migrated.status).toBe('migrated');
        expect(migrateStoredState({ slice: 'wardrobe', stored: null, defaults: [] }).status).toBe('reset_to_defaults');
        expect(migrateStoredState({ slice: 'wardrobe', stored: { nope: true }, defaults: [] }).status).toBe('reset_to_defaults');
        const newer = migrateStoredState({ slice: 'growth', stored: { schemaVersion: 99, data: {} }, defaults: { a: 0 } });
        expect(newer.status).toBe('reset_to_defaults');
        if (newer.status === 'reset_to_defaults') expect(newer.reason).toBe('newer_than_app');
    });
});

describe('startup hardening', () => {
    test('core UI renders under total degradation', () => {
        const d = evaluateStartup({
            online: false, supabaseReachable: false, supabaseSlow: false, revenueCatReachable: false,
            cachedEntitlementValid: false, persistedStateValid: false, providerConfigPresent: false, catalogRefreshAvailable: false,
        });
        expect(d.renderCoreUi).toBe(true);
        expect(d.blockedOn).toHaveLength(0);
        expect(d.degraded).toContain('entitlement_unknown_use_free');
        expect(d.deferred).toContain('analytics');
    });
});

describe('upload + URL safety', () => {
    test('rejects bad MIME, oversize, traversal, executables; SSRF ranges blocked', () => {
        expect(validateImageUpload({ mime: 'application/pdf', bytes: 10, width: 10, height: 10, filename: 'a.pdf' }).ok).toBe(false);
        expect(validateImageUpload({ mime: 'image/jpeg', bytes: 100 * 1024 * 1024, width: 10, height: 10, filename: 'a.jpg' }).reason).toBe('invalid_size');
        expect(validateImageUpload({ mime: 'image/png', bytes: 10, width: 10, height: 10, filename: '../x.png' }).reason).toBe('unsafe_filename');
        expect(validateImageUpload({ mime: 'image/jpeg', bytes: 100, width: 100, height: 100, filename: 'photo.jpg' }).ok).toBe(true);
        expect(sanitizeFilename('../../etc/passwd')).not.toContain('/');
        for (const u of ['http://example.com/x', 'https://localhost/x', 'https://127.0.0.1/', 'https://10.0.0.5/', 'https://192.168.1.1/', 'https://169.254.169.254/', 'https://[::1]/', 'https://user:pw@example.com/', 'not a url', 'ftp://example.com/f']) {
            expect(validateIngestUrl(u).ok).toBe(false);
        }
        expect(validateIngestUrl('https://merchant.com/product/123').ok).toBe(true);
    });
});

describe('privacy', () => {
    test('deletion covers all scopes; export excludes secrets; classification gates external AI', () => {
        expect(deletionPlan().length).toBe(10);
        const m = buildExportManifest(['wardrobe_metadata', 'secrets', 'goals']);
        expect(m.included).not.toContain('secrets');
        expect(m.excluded).toContain('secrets');
        expect(maySendToExternalAi('body_fit_private')).toBe(false);
        expect(maySendToExternalAi(classifyField('catalog_listing'))).toBe(true);
        expect(classifyField('fit_measurement')).toBe('body_fit_private');
        expect(defaultPrivacyControls().external_ai_processing).toBe(false);
        const pkg = buildDiagnosticPackage({ appVersion: '3.0.2', environment: 'production', requestIds: ['r1'], featureFlags: {}, errors: [], subscriptionCategory: 'free', syncStatus: 'ok' });
        expect(sanitizeDiagnostics(pkg).appVersion).toBe('3.0.2');
    });
});

describe('release engineering', () => {
    test('env separation, versioning, compat, cutover, canary, flags, RC, scorecard', () => {
        expect(ENV_CONFIG.production.mockProviders).toBe(false);
        expect(productionPointsAtDev('production', 'http://localhost:3000')).toBe(true);
        expect(productionPointsAtDev('development', 'http://localhost:3000')).toBe(false);
        expect(clientSupported('2.9.0')).toBe('hard_update');
        expect(clientSupported('3.0.2')).toBe('supported');
        expect(compatibilityExpectation({ direction: 'new_client_old_backend', rpcPresent: false, migrationApplied: false })).toBe('degraded_with_fallback');
        expect(productionCutoverPlan()).toHaveLength(8);
        expect(canaryPlan()).toHaveLength(7);
        expect(launchFlagDefaults().find((f) => f.flag === 'wardrobe_core')?.default).toBe('on');
        const rc = releaseCandidateSummary({ typecheck: true, lint: true, criticalTests: true, fullRegression: false, sqlTests: true, migrationOrder: true, secretsScan: true, storeConfig: false });
        expect(rc.find((r) => r.check === 'manual_qa')?.result).toBe('MANUAL');
        expect(rc.find((r) => r.check === 'full_regression')?.result).toBe('FAIL');
        const score = buildScorecard({ security: 'PASS', billing: 'BLOCKED' });
        expect(launchBlockedBy(score)).toEqual(['billing']);
        expect(score.privacy).toBe('MANUAL_REVIEW');
        expect(featureReadinessMatrix().length).toBeGreaterThan(5);
        expect(launchBlockers().filter((b) => b.severity === 'BLOCKER').length).toBeGreaterThan(5);
        expect(INCIDENT_RESPONSE.SEV0.rollbackFirst).toBe(true);
        expect(RUNBOOKS.runaway_cost.firstAction).toMatch(/Kill switch/);
        expect(corruptionResponse('orphan_wardrobe_images')).toBe('repair_tool');
        expect(ALERT_TO_RUNBOOK.cost_spike).toBe('runaway_cost');
    });
});

describe('store + quality', () => {
    test('claims gated; purchase disclosure; triage; scale; webhooks; limitations', () => {
        expect(claimSafe('Perfect fit guaranteed!').safe).toBe(false);
        expect(claimSafe('Helps compare fit evidence.').safe).toBe(true);
        expect(externalPurchaseDisclosure('Zara')).toMatch(/not AIWardrobe/);
        expect(triagePriority({ category: 'security_data', reproducible: true, affectedVersion: '3.0.2', workaround: null, severity: 'BLOCKER', owner: 'sec' })).toBeLessThan(
            triagePriority({ category: 'ui', reproducible: false, affectedVersion: '3.0.2', workaround: 'restart', severity: 'LOW', owner: 'ui' }),
        );
        const [est] = loadModel({ dau: 1000, outfitsPerUserPerDay: null });
        expect(est.buycheckRequestsPerDay).toBeNull(); // unknown baseline, not invented
        expect(webhookIdempotencyKey('e1', 'purchase')).toBe('purchase:e1');
        expect(KNOWN_LIMITATIONS.length).toBeGreaterThan(3);
    });
});

describe('platform readiness', () => {
    test('deep links: known routes pass, unknown fail safe, redirects blocked', () => {
        expect(validateDeepLink('com.aiwardrobe://buycheck?id=1')).toMatchObject({ ok: true, route: 'buycheck' });
        expect(validateDeepLink('com.aiwardrobe://nosuchthing')).toMatchObject({ ok: true, route: null });
        expect(validateDeepLink('com.aiwardrobe://buycheck?redirect=https://evil.com').ok).toBe(false);
        expect(validateDeepLink('https://merchant.com/p/1').ok).toBe(false);
        expect(validateDeepLink('notaurl').ok).toBe(false);
        expect(validateShareInput({ kind: 'url', sizeBytes: 100, authenticated: false }).reason).toBe('auth_required');
        expect(validateShareInput({ kind: 'image', sizeBytes: 100, authenticated: true }).ok).toBe(true);
        expect(staleNotificationCutoffHours('price_target_reached')).toBe(12);
        expect(permissionMatrix().length).toBe(4);
        expect(scanSourceForDebugLeakage('const x = 1;')).toHaveLength(0);
        expect(scanSourceForDebugLeakage('service_role key here')).toHaveLength(1);
        expect(logLevelFor('production')).toBe('warn');
        expect(sanitizeLogPayload({ a: 1, wardrobe_contents: [1] }, 'info')).toEqual({ a: 1 });
    });
});
