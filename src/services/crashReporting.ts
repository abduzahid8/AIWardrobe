/**
 * Crash Reporting Service — Sentry Integration
 *
 * Production: Reports crashes to Sentry for real-time monitoring.
 * Development: Logs to console + stores locally in AsyncStorage.
 * Fallback: If Sentry DSN is not configured, falls back to local-only storage.
 *
 * Usage:
 *   crashReporting.initialize();
 *   crashReporting.reportCrash(error, { screen: 'HomeScreen' });
 *   crashReporting.setUser('user-id-123');
 */
import AsyncStorage from '@react-native-async-storage/async-storage';
import Config from '../config/env';

// Sentry lazy import — only loaded when DSN is configured
// @ts-ignore — Sentry SDK is optional; loaded at runtime via require()
let Sentry: any = null;

const CRASH_LOG_KEY = 'crash_reports';
const MAX_STORED_CRASHES = 50;

interface CrashContext {
    screen?: string;
    action?: string;
    componentStack?: string;
    [key: string]: unknown;
}

interface CrashReport {
    id: string;
    timestamp: string;
    message: string;
    stack?: string;
    context: CrashContext;
    userId?: string;
}

interface Breadcrumb {
    timestamp: string;
    message: string;
    category?: string;
}

class CrashReportingService {
    private userId: string | null = null;
    private breadcrumbs: Breadcrumb[] = [];
    private isInitialized = false;
    private sentryAvailable = false;

    /**
     * Initialize the crash reporting service.
     * Call once in App.tsx on startup.
     */
    async initialize(): Promise<void> {
        if (this.isInitialized) return;
        this.isInitialized = true;

        // Try to initialize Sentry if DSN is configured
        const dsn = Config.sentry.dsn;
        if (dsn && dsn !== 'your-sentry-dsn') {
            try {
                Sentry = require('@sentry/react-native');
                Sentry.init({
                    dsn,
                    debug: __DEV__,
                    environment: __DEV__ ? 'development' : 'production',
                    tracesSampleRate: __DEV__ ? 1.0 : 0.2,
                    // Don't send events in development unless explicitly enabled
                    enabled: !__DEV__ || process.env.EXPO_PUBLIC_SENTRY_DEBUG === 'true',
                    beforeSend(event: any) {
                        // Scrub sensitive data before sending to Sentry
                        if (event.request?.headers) {
                            delete event.request.headers['Authorization'];
                            delete event.request.headers['Cookie'];
                        }
                        return event;
                    },
                });
                this.sentryAvailable = true;
            } catch (err) {
                // Sentry SDK not installed — fall back to local-only
                console.warn('[CrashReporting] Sentry SDK not available, using local-only mode');
                this.sentryAvailable = false;
            }
        }

        // Set up global error handlers (always, regardless of Sentry)
        const originalHandler = ErrorUtils.getGlobalHandler();
        ErrorUtils.setGlobalHandler((error: Error, isFatal?: boolean) => {
            this.reportCrash(error, { isFatal, source: 'globalHandler' });
            originalHandler(error, isFatal);
        });

        // Handle unhandled promise rejections
        const originalRejectionHandler = (globalThis as any).onunhandledrejection;
        (globalThis as any).onunhandledrejection = (event: any) => {
            this.reportCrash(
                event?.reason instanceof Error
                    ? event.reason
                    : new Error(String(event?.reason || 'Unhandled promise rejection')),
                { source: 'unhandledRejection' }
            );
            originalRejectionHandler?.(event);
        };
    }

    /**
     * Set the current user for crash context.
     */
    setUser(userId: string): void {
        this.userId = userId;
        if (this.sentryAvailable && Sentry) {
            Sentry.setUser({ id: userId });
        }
    }

    /**
     * Clear the current user (on logout).
     */
    clearUser(): void {
        this.userId = null;
        if (this.sentryAvailable && Sentry) {
            Sentry.setUser(null);
        }
    }

    /**
     * Add a breadcrumb for debugging context.
     * Breadcrumbs are attached to the next crash report.
     */
    logBreadcrumb(message: string, category?: string): void {
        this.breadcrumbs.push({
            timestamp: new Date().toISOString(),
            message,
            category,
        });
        // Keep last 20 breadcrumbs locally
        if (this.breadcrumbs.length > 20) {
            this.breadcrumbs = this.breadcrumbs.slice(-20);
        }

        // Also send to Sentry for richer crash context
        if (this.sentryAvailable && Sentry) {
            Sentry.addBreadcrumb({
                message,
                category: category || 'app',
                level: 'info',
                timestamp: Date.now() / 1000,
            });
        }
    }

    /**
     * Report a crash or error.
     * Sends to Sentry in production, stores locally as fallback.
     */
    async reportCrash(error: Error, context: CrashContext = {}): Promise<void> {
        // Send to Sentry if available
        if (this.sentryAvailable && Sentry) {
            Sentry.captureException(error, {
                extra: {
                    ...context,
                    localBreadcrumbs: [...this.breadcrumbs],
                },
                tags: {
                    screen: context.screen || 'unknown',
                    source: (context.source as string) || 'manual',
                },
            });
        }

        const report: CrashReport = {
            id: `crash_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            timestamp: new Date().toISOString(),
            message: error.message,
            stack: error.stack,
            context: {
                ...context,
                breadcrumbs: [...this.breadcrumbs],
            },
            userId: this.userId || undefined,
        };

        // Log to console in dev
        if (__DEV__) {
            console.error('[CrashReporting]', report.message, report.context);
        }

        // Store locally (fallback and for debugging)
        try {
            const existing = await AsyncStorage.getItem(CRASH_LOG_KEY);
            const reports: CrashReport[] = existing ? JSON.parse(existing) : [];
            reports.push(report);
            const trimmed = reports.slice(-MAX_STORED_CRASHES);
            await AsyncStorage.setItem(CRASH_LOG_KEY, JSON.stringify(trimmed));
        } catch (storageError) {
            if (__DEV__) {
                console.error('[CrashReporting] Failed to store report:', storageError);
            }
        }
    }

    /**
     * Capture a non-fatal message in Sentry.
     */
    captureMessage(message: string, level: 'info' | 'warning' | 'error' = 'info'): void {
        if (this.sentryAvailable && Sentry) {
            Sentry.captureMessage(message, level);
        }
        if (__DEV__) {
            console.log(`[CrashReporting:${level}]`, message);
        }
    }

    /**
     * Get stored crash reports (for debugging screen or upload).
     */
    async getStoredReports(): Promise<CrashReport[]> {
        try {
            const data = await AsyncStorage.getItem(CRASH_LOG_KEY);
            return data ? JSON.parse(data) : [];
        } catch {
            return [];
        }
    }

    /**
     * Clear all stored crash reports.
     */
    async clearReports(): Promise<void> {
        await AsyncStorage.removeItem(CRASH_LOG_KEY);
    }

    /**
     * Wrap a React component tree with Sentry error boundary.
     * Use for the top-level App component.
     */
    get ErrorBoundary() {
        if (this.sentryAvailable && Sentry) {
            return Sentry.wrap;
        }
        // Return passthrough if Sentry is not available
        return (component: any) => component;
    }
}

export const crashReporting = new CrashReportingService();
export default crashReporting;
