/**
 * useNetworkStatus — React hook for online/offline detection
 *
 * Uses a lightweight fetch-based approach to detect connectivity.
 * Assumes online by default and only marks offline after consecutive failures.
 */

import { useState, useEffect, useCallback, useRef } from 'react';

interface NetworkStatus {
    isOnline: boolean;
}

/**
 * Hook to monitor network connectivity.
 * Polls every `pollIntervalMs` milliseconds. Requires 2 consecutive failures before marking offline.
 */
export function useNetworkStatus(pollIntervalMs: number = 15000): NetworkStatus {
    const [isOnline, setIsOnline] = useState(true);
    const failures = useRef(0);

    const checkNetwork = useCallback(async () => {
        try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 8000);
            await fetch('https://clients3.google.com/generate_204', {
                method: 'HEAD',
                signal: controller.signal,
            });
            clearTimeout(timeout);
            failures.current = 0;
            setIsOnline(true);
        } catch {
            failures.current += 1;
            if (failures.current >= 2) {
                setIsOnline(false);
            }
        }
    }, []);

    useEffect(() => {
        checkNetwork();
        const interval = setInterval(checkNetwork, pollIntervalMs);
        return () => clearInterval(interval);
    }, [checkNetwork, pollIntervalMs]);

    return { isOnline };
}

export default useNetworkStatus;
