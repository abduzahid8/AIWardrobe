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

const ANALYTICS_QUEUE_KEY = 'analytics_event_queue';
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
    private pendingCount = 0;

    constructor() {
        this.sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;
    }

    /**
     * Initialize analytics — starts auto-flush timer and app state listener.
     * Call once in App.tsx.
     */
    initialize(): void {
        // Auto-flush on interval
        this.flushTimer = setInterval(() => {
            this.flush();
        }, FLUSH_INTERVAL_MS);

        // Flush when app goes to background
        AppState.addEventListener('change', (state: AppStateStatus) => {
            if (state === 'background') {
                this.flush();
            }
        });
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
     * Track an event.
     */
    async trackEvent(name: string, properties?: Record<string, unknown>): Promise<void> {
        const event: AnalyticsEvent = {
            name,
            properties,
            timestamp: new Date().toISOString(),
            userId: this.userId || undefined,
            sessionId: this.sessionId,
        };

        if (__DEV__) {
            console.log('[Analytics]', name, properties || '');
        }

        try {
            const existing = await AsyncStorage.getItem(ANALYTICS_QUEUE_KEY);
            const queue: AnalyticsEvent[] = existing ? JSON.parse(existing) : [];
            queue.push(event);

            // Trim queue to max size
            const trimmed = queue.slice(-MAX_QUEUE_SIZE);
            await AsyncStorage.setItem(ANALYTICS_QUEUE_KEY, JSON.stringify(trimmed));

            this.pendingCount = trimmed.length;

            // Auto-flush when threshold is reached
            if (this.pendingCount >= FLUSH_THRESHOLD) {
                this.flush();
            }
        } catch {
            // Don't let analytics errors affect the app
        }
    }

    /**
     * Get all queued events (for sending to a backend or debugging).
     */
    async getQueue(): Promise<AnalyticsEvent[]> {
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
        if (this.isFlushing) return;
        this.isFlushing = true;

        try {
            const queue = await this.getQueue();
            if (queue.length === 0) {
                this.isFlushing = false;
                return;
            }

            // Try to send to backend
            try {
                // Dynamic import to avoid circular dependency with apiClient
                const { default: apiClient } = await import('./apiClient');
                await apiClient.post('/api/analytics/events', {
                    events: queue,
                    sessionId: this.sessionId,
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
            await AsyncStorage.removeItem(ANALYTICS_QUEUE_KEY);
            this.pendingCount = 0;
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
