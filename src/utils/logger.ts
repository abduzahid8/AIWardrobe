/**
 * Logger — scoped, tagged, and production-safe.
 *
 * Usage:
 *   import { createLogger } from '@/utils/logger';
 *   const log = createLogger('AuthStore');
 *   log.info('User signed in', { userId });
 *   log.warn('Slow network', { ms });
 *   log.error('Failed to fetch profile', err, { userId });
 *
 * Rules:
 *   - debug / info logs are NO-OPS in production.
 *   - warn / error always log and (if crashReporting is loaded) are
 *     sent to Sentry as breadcrumbs / exceptions.
 *   - No secrets, tokens, or bearer strings should be passed to logs.
 *     A guard strips obvious token-shaped values before printing.
 */

/* eslint-disable no-console */

type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface Logger {
    debug: (message: string, data?: unknown) => void;
    info: (message: string, data?: unknown) => void;
    warn: (message: string, data?: unknown) => void;
    error: (message: string, error?: unknown, data?: unknown) => void;
}

const TOKEN_PATTERNS = [
    /Bearer\s+[A-Za-z0-9._-]{12,}/gi,
    /sk-[A-Za-z0-9]{16,}/g,
    /nvapi-[A-Za-z0-9_-]{16,}/g,
    /r8_[A-Za-z0-9]{16,}/g,
    /hf_[A-Za-z0-9]{16,}/g,
    /eyJ[A-Za-z0-9._-]{40,}/g,
];

function redact(value: unknown): unknown {
    if (typeof value === 'string') {
        let out = value;
        for (const re of TOKEN_PATTERNS) out = out.replace(re, '[redacted]');
        return out;
    }
    if (value && typeof value === 'object') {
        try {
            const str = JSON.stringify(value);
            let redacted = str;
            for (const re of TOKEN_PATTERNS) redacted = redacted.replace(re, '[redacted]');
            return redacted === str ? value : JSON.parse(redacted);
        } catch {
            return value;
        }
    }
    return value;
}

function send(level: LogLevel, scope: string, message: string, data?: unknown, error?: unknown) {
    const tag = `[${scope}]`;
    const payload = data !== undefined ? redact(data) : undefined;
    const isDev = typeof __DEV__ !== 'undefined' && __DEV__;

    if ((level === 'debug' || level === 'info') && !isDev) return;

    if (level === 'error') {
        console.error(tag, message, error ?? '', payload ?? '');
        void reportToCrashService(scope, message, error, payload);
        return;
    }
    if (level === 'warn') {
        console.warn(tag, message, payload ?? '');
        void breadcrumb(scope, `WARN: ${message}`);
        return;
    }
    if (level === 'info') {
        console.log(tag, message, payload ?? '');
        return;
    }
    console.log(tag, message, payload ?? '');
}

async function reportToCrashService(
    scope: string,
    message: string,
    error: unknown,
    payload: unknown,
): Promise<void> {
    try {
        const mod = await import('../services/crashReporting');
        const svc = mod.default ?? mod.crashReporting;
        if (!svc || typeof svc.reportCrash !== 'function') return;
        const err = error instanceof Error ? error : new Error(String(error ?? message));
        svc.reportCrash(err, { scope, message, payload });
    } catch {
        // crashReporting may not be initialized yet during boot — that's fine.
    }
}

async function breadcrumb(scope: string, message: string): Promise<void> {
    try {
        const mod = await import('../services/crashReporting');
        const svc = mod.default ?? mod.crashReporting;
        if (svc?.logBreadcrumb) svc.logBreadcrumb(`[${scope}] ${message}`);
    } catch {
        // No-op
    }
}

export function createLogger(scope: string): Logger {
    return {
        debug: (message, data) => send('debug', scope, message, data),
        info: (message, data) => send('info', scope, message, data),
        warn: (message, data) => send('warn', scope, message, data),
        error: (message, error, data) => send('error', scope, message, data, error),
    };
}

/**
 * Default global logger — prefer createLogger(scope) in new code.
 * Kept for backwards compatibility with existing imports.
 */
const logger: Logger = createLogger('app');
export default logger;
