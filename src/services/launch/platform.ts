/**
 * Phase 7W–7Z, 7AA–7AC, 7CJ–7CK — Platform readiness: deep-link
 * validation, share-target finalization, notification config,
 * permission matrix (grounded in app.json), debug-leakage gate,
 * logging levels.
 */

export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

// ── Deep linking (7AA) ───────────────────────────────────────────

export const APP_SCHEME = 'com.aiwardrobe';
export const KNOWN_DEEP_LINK_HOSTS = ['product', 'buycheck', 'outfit', 'subscription', 'referral', 'auth'] as const;

export function validateDeepLink(raw: string): { ok: boolean; route: string | null; reason: string } {
    let url: URL;
    try {
        url = new URL(raw);
    } catch {
        return { ok: false, route: null, reason: 'malformed_url' };
    }
    const scheme = url.protocol.replace(/:$/, '');
    if (scheme !== APP_SCHEME && scheme !== 'https') return { ok: false, route: null, reason: 'unknown_scheme' };
    // Never allow open redirects: only in-app routes, no external passthrough.
    if (url.searchParams.has('redirect') || url.searchParams.has('url')) {
        return { ok: false, route: null, reason: 'open_redirect_blocked' };
    }
    const host = url.hostname.toLowerCase();
    if (scheme === 'https') {
        // Universal links: only our own domain (placeholder until associated domains land).
        return { ok: false, route: null, reason: 'universal_links_not_configured' };
    }
    if (!(KNOWN_DEEP_LINK_HOSTS as readonly string[]).includes(host)) {
        return { ok: true, route: null, reason: 'unknown_route_ignored' }; // fail safely
    }
    return { ok: true, route: host, reason: 'ok' };
}

// ── Share targets (7AB) ──────────────────────────────────────────

export type ShareInputKind = 'url' | 'text' | 'image';

export function validateShareInput(input: { kind: ShareInputKind; sizeBytes: number; authenticated: boolean }): {
    ok: boolean;
    reason: string;
} {
    if (!input.authenticated) return { ok: false, reason: 'auth_required' };
    if (input.sizeBytes <= 0 || input.sizeBytes > 15 * 1024 * 1024) return { ok: false, reason: 'invalid_size' };
    return { ok: true, reason: 'ok' };
}

// ── Notification config (7AC) — local only, no sending ───────────

export const NOTIFICATION_CHANNELS = ['outfit_reminders', 'event_reminders', 'price_alerts', 'weekly_review', 'subscription_lifecycle'] as const;

export function staleNotificationCutoffHours(type: string): number {
    // Time-sensitive notifications expire; weekly digests last longer.
    if (type === 'outfit_weather_conflict' || type === 'price_target_reached') return 12;
    return 72;
}

// ── Permission matrix (7W–7X), grounded in app.json ──────────────

export interface PermissionEntry {
    permission: string;
    requestedWhen: string;
    denialFallback: string;
    copy: string;
}

export function permissionMatrix(): PermissionEntry[] {
    return [
        { permission: 'CAMERA', requestedWhen: 'Closet scan / add-from-photo only', denialFallback: 'Manual add + URL import', copy: 'Photograph clothing items you choose to add.' },
        { permission: 'READ_MEDIA_IMAGES', requestedWhen: 'Photo-library import only', denialFallback: 'Camera or manual add', copy: 'Import only the clothing images you select.' },
        { permission: 'NOTIFICATIONS', requestedWhen: 'User enables a reminder category', denialFallback: 'In-app cards only', copy: 'Only the reminder categories you enable.' },
        { permission: 'LOCATION (coarse/fine)', requestedWhen: 'Weather-aware outfit request', denialFallback: 'Manual city entry', copy: 'Used once per session for weather; not stored.' },
    ];
}

// ── Debug leakage gate (7CJ) + logging levels (7CK) ─────────────

export const PRODUCTION_FORBIDDEN_PATTERNS = [
    'debug panel',
    '__tests__',
    'console.log(',
    'provider_raw_response',
    'service_role',
    'localhost',
] as const;

export function scanSourceForDebugLeakage(source: string): string[] {
    return PRODUCTION_FORBIDDEN_PATTERNS.filter((p) => source.includes(p));
}

export function logLevelFor(environment: 'development' | 'preview' | 'production'): LogLevel {
    return environment === 'development' ? 'debug' : environment === 'preview' ? 'info' : 'warn';
}

/** Wardrobe contents must never be logged above debug level. */
export function sanitizeLogPayload(payload: Record<string, unknown>, level: LogLevel): Record<string, unknown> {
    if (level === 'debug') return payload;
    const { wardrobe_contents: _drop, raw_images: _drop2, body_data: _drop3, ...rest } = payload;
    return rest;
}
