/**
 * useTrackingPermission
 *
 * Requests App Tracking Transparency (ATT) permission on iOS 14+ as required
 * by Apple Guideline 5.1.2(i) when the app collects data used to track users.
 *
 * Rules:
 *  - The system dialog is shown at most ONCE per install (iOS enforces this
 *    natively; we also track it ourselves for clean state management).
 *  - Must be called at the ROOT of the app (before analytics fire).
 *  - On iOS < 14 and Android the hook is a safe no-op that immediately
 *    resolves to `granted` (no dialog required on those platforms).
 *  - The ATT prompt is deliberately shown after the app's first meaningful
 *    render so the user has context about what the app does before deciding.
 */

import { useEffect, useRef, useState } from 'react';
import { Platform } from 'react-native';

// Lazy import — only available on iOS builds with the native module installed.
// Using a try/require so the app doesn't crash on Android or web.
let TrackingTransparency: {
    requestTrackingPermissionsAsync: () => Promise<{ status: string }>;
    getTrackingPermissionsAsync: () => Promise<{ status: string }>;
} | null = null;

try {
    TrackingTransparency = require('expo-tracking-transparency');
} catch {
    // SDK not linked (Android, web, old SDK) — no-op.
}

export type TrackingStatus =
    | 'unavailable'   // Android / web / iOS < 14
    | 'not-determined'
    | 'restricted'
    | 'denied'
    | 'authorized'
    | 'granted';      // alias for authorized (RevenueCat naming)

interface TrackingPermissionResult {
    status: TrackingStatus;
    isGranted: boolean;
    isReady: boolean;
}

/**
 * Requests ATT permission once at app startup and returns the resulting
 * status. Safe to call on all platforms.
 *
 * @param delayMs  Milliseconds to wait before showing the system prompt.
 *                 Defaults to 1200 ms — gives the first screen time to render
 *                 so the user understands what app they're in before deciding.
 */
export function useTrackingPermission(delayMs = 1200): TrackingPermissionResult {
    const [status, setStatus] = useState<TrackingStatus>('not-determined');
    const [isReady, setIsReady] = useState(false);
    const hasRequested = useRef(false);

    useEffect(() => {
        // Only iOS 14+ needs ATT
        if (Platform.OS !== 'ios' || !TrackingTransparency) {
            setStatus('unavailable');
            setIsReady(true);
            return;
        }

        if (hasRequested.current) return;
        hasRequested.current = true;

        const run = async () => {
            try {
                // Check if already determined (reinstall edge case, etc.)
                const existing = await TrackingTransparency!.getTrackingPermissionsAsync();
                if (existing.status !== 'not-determined') {
                    setStatus(existing.status as TrackingStatus);
                    setIsReady(true);
                    return;
                }

                // Brief delay so the app UI is visible before the system dialog appears
                await new Promise<void>((resolve) => setTimeout(resolve, delayMs));

                const result = await TrackingTransparency!.requestTrackingPermissionsAsync();
                setStatus(result.status as TrackingStatus);
            } catch (err) {
                // Non-critical — fall through with 'denied' so the app keeps working
                console.warn('[ATT] requestTrackingPermissionsAsync error:', err);
                setStatus('denied');
            } finally {
                setIsReady(true);
            }
        };

        void run();
    }, [delayMs]);

    return {
        status,
        isGranted: status === 'authorized' || status === 'granted' || status === 'unavailable',
        isReady,
    };
}

export default useTrackingPermission;
