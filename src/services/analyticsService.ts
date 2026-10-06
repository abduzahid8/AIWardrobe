/**
 * Analytics Service — client-side event tracking with server flush.
 *
 * Stores events locally in AsyncStorage, flushes to backend periodically.
 * Backend can forward to PostHog/Mixpanel/Amplitude when configured.
 *
 * Usage:
 *   analyticsService.trackEvent('signup', { method: 'email' });
 *   analyticsService.setUserId('user-123');
 */
import AsyncStorage from '@react-native-async-storage/async-storage';
import { AppState, AppStateStatus } from 'react-native';
import { createLogger } from '../utils/logger';

const logger = createLogger('Analytics');

const ANALYTICS_QUEUE_KEY = 'analytics_event_queue';
const ANALYTICS_ENABLED_KEY = 'analytics_enabled';
const MAX_QUEUE_SIZE = 200;
const FLUSH_THRESHOLD = 50; // Auto-flush when queue reaches this size
const FLUSH_INTERVAL_MS = 5 * 60 * 1000; // Auto-flush every 5 minutes

interface AnalyticsEvent {
    name: string;
    properties?: Record<string, unknown>;
    timestamp: string;
    userId?: string;
    sessionId?: string;
}

class AnalyticsService {
    private userId: string | null = null;
    private sessionId: string;
    private flushTimer: ReturnType<typeof setInterval> | null = null;
    private isFlushing = false;
    private enabled = true;
    // In-memory queue — avoids AsyncStorage getItem/setItem + JSON
    // parse/stringify of up to 200 events on EVERY trackEvent call
    // (that was running on the JS thread per screen_view/tap).
    private memoryQueue: AnalyticsEvent[] = [];
    private persistTimer: ReturnType<typeof setTimeout> | null = null;
    private appStateSub: { remove: () => void } | null = null;
    private queueLoaded = false;

    constructor() {
        this.sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
    }

    /**
     * Initialize analytics — starts auto-flush timer and app state listener.
     * Call once in App.tsx.
     */
    initialize(): void {
        // Guard against double-init (RootNavigator remounts would previously
        // leak an extra setInterval + AppState listener per mount).
        if (this.flushTimer || this.appStateSub) {
            return;
        }
        // Best-effort: load persisted preference. Defaults to enabled.
        void this.loadEnabledPreference();
        void this.loadPersistedQueue();

        // Auto-flush on interval
        this.flushTimer = setInterval(() => {
            this.flush();
        }, FLUSH_INTERVAL_MS);

        // Flush when app goes to background
        this.appStateSub = AppState.addEventListener('change', (state: AppStateStatus) => {
            if (state === 'background') {
                this.flush();
            }
        });
    }

    /**
     * Enable/disable analytics collection and flush.
     * When disabled, no events are queued and flush() becomes a no-op.
     */
    async setEnabled(value: boolean): Promise<void> {
        this.enabled = Boolean(value);
        try {
            await AsyncStorage.setItem(ANALYTICS_ENABLED_KEY, JSON.stringify(this.enabled));
        } catch {
            // ignore persistence errors
        }
        if (!this.enabled) {
            // Optional: clear queued events when user opts out.
            try {
                this.memoryQueue = [];
                await AsyncStorage.removeItem(ANALYTICS_QUEUE_KEY);
            } catch {
                // ignore
            }
        }
    }

    /**
     * Returns current analytics enabled state (default: true).
     * Note: initialize() also loads persisted preference in the background.
     */
    getEnabled(): boolean {
        return this.enabled;
    }

    private async loadEnabledPreference(): Promise<void> {
        try {
            const raw = await AsyncStorage.getItem(ANALYTICS_ENABLED_KEY);
            if (raw === null) return;
            const parsed = JSON.parse(raw);
            if (typeof parsed === 'boolean') {
                this.enabled = parsed;
            }
        } catch {
            // ignore
        }
    }

    /**
     * Set the current user ID for event attribution.
     */
    setUserId(userId: string): void {
        this.userId = userId;
    }

    /**
     * Clear the user (on logout).
     */
    clearUserId(): void {
        this.userId = null;
    }

    /**
     * Track an event — fire-and-forget, never blocks the caller.
     * Appends to the in-memory queue synchronously and persists
     * debounced (500ms) so rapid screen_view/tap storms don't
     * serialize JSON on the JS thread per event.
     */
    trackEvent(name: string, properties?: Record<string, unknown>): void {
        if (!this.enabled) return;

        const event: AnalyticsEvent = {
            name,
            properties,
            timestamp: new Date().toISOString(),
            userId: this.userId || undefined,
            sessionId: this.sessionId,
        };

        if (__DEV__) {
            logger.debug(name, properties);
        }

        try {
            this.memoryQueue.push(event);
            // Trim queue to max size (O(1) amortized, no JSON serde)
            if (this.memoryQueue.length > MAX_QUEUE_SIZE) {
                this.memoryQueue.splice(0, this.memoryQueue.length - MAX_QUEUE_SIZE);
            }
            this.schedulePersist();

            // Auto-flush when threshold is reached
            if (this.memoryQueue.length >= FLUSH_THRESHOLD) {
                void this.flush();
            }
        } catch {
            // Don't let analytics errors affect the app
        }
    }

    private schedulePersist(): void {
        if (this.persistTimer) return;
        this.persistTimer = setTimeout(() => {
            this.persistTimer = null;
            // Fire-and-forget background write — never awaited by UI code.
            AsyncStorage.setItem(ANALYTICS_QUEUE_KEY, JSON.stringify(this.memoryQueue)).catch(() => {});
        }, 500);
    }

    private async loadPersistedQueue(): Promise<void> {
        if (this.queueLoaded) return;
        this.queueLoaded = true;
        try {
            const raw = await AsyncStorage.getItem(ANALYTICS_QUEUE_KEY);
            if (raw) {
                const parsed = JSON.parse(raw);
                if (Array.isArray(parsed)) {
                    this.memoryQueue = parsed.slice(-MAX_QUEUE_SIZE);
                }
            }
        } catch {
            // ignore — start with empty queue
        }
    }

    /**
     * Get all queued events (for sending to a backend or debugging).
     */
    async getQueue(): Promise<AnalyticsEvent[]> {
        // Serve from memory when available — avoids a disk read per flush.
        if (this.queueLoaded || this.memoryQueue.length > 0) {
            return [...this.memoryQueue];
        }
        try {
            const data = await AsyncStorage.getItem(ANALYTICS_QUEUE_KEY);
            return data ? JSON.parse(data) : [];
        } catch {
            return [];
        }
    }

    /**
     * Flush the event queue to the backend.
     * Sends batched events to /api/analytics/events.
     * Falls back to clearing local queue if backend is unavailable.
     */
    async flush(): Promise<void> {
        if (!this.enabled) return;
        if (this.isFlushing) return;
        // Nothing to send — skip the AsyncStorage read entirely.
        if (this.queueLoaded && this.memoryQueue.length === 0) return;
        this.isFlushing = true;

        try {
            const queue = await this.getQueue();
            if (queue.length === 0) {
                this.isFlushing = false;
                return;
            }

            // Try to send to backend (Edge `analytics` function replaces
            // Express POST /api/analytics/events; same body contract).
            try {
                const { supabase } = require('../../lib/supabase') as typeof import('../../lib/supabase');
                await supabase.functions.invoke('analytics', {
                    body: {
                        events: queue,
                        sessionId: this.sessionId,
                    },
                });
            } catch {
                // Backend unavailable — events are kept locally for next flush.
                // Only clear if queue is getting too large to prevent unbounded growth.
                if (queue.length < MAX_QUEUE_SIZE) {
                    this.isFlushing = false;
                    return;
                }
            }

            // Clear the queue after successful send (or if too large)
            this.memoryQueue = [];
            await AsyncStorage.removeItem(ANALYTICS_QUEUE_KEY);
        } catch {
            // Fail silently
        } finally {
            this.isFlushing = false;
        }
    }

    /**
     * Cleanup timers (call on app unmount)
     */
    destroy(): void {
        if (this.flushTimer) {
            clearInterval(this.flushTimer);
            this.flushTimer = null;
        }
        if (this.persistTimer) {
            clearTimeout(this.persistTimer);
            this.persistTimer = null;
        }
        if (this.appStateSub) {
            try {
                this.appStateSub.remove();
            } catch {
                // ignore
            }
            this.appStateSub = null;
        }
    }

    // ── Convenience methods for common events ──

    trackSignup(method: string): void {
        this.trackEvent('signup', { method });
    }

    trackLogin(method: string): void {
        this.trackEvent('login', { method });
    }

    trackItemAdded(category: string): void {
        this.trackEvent('item_added', { category });
    }

    trackOutfitGenerated(source: string): void {
        this.trackEvent('outfit_generated', { source });
    }

    trackSubscriptionPurchased(tier: string, price: number): void {
        this.trackEvent('subscription_purchased', { tier, price });
    }

    trackScreenView(screenName: string): void {
        this.trackEvent('screen_view', { screen: screenName });
    }

    trackError(errorType: string, message: string): void {
        this.trackEvent('error', { errorType, message });
    }
}

export const analyticsService = new AnalyticsService();
export default analyticsService;
