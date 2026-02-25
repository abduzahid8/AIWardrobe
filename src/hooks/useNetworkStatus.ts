/**
 * useNetworkStatus — React hook for online/offline detection
 *
 * Uses expo-network to detect connectivity state.
 * Falls back to assuming online if expo-network is unavailable.
 */

import { useState, useEffect, useCallback } from 'react';
import * as Network from 'expo-network';

interface NetworkStatus {
    isOnline: boolean;
    isWifi: boolean;
    type: string;
}

/**
 * Hook to monitor network connectivity.
 * Polls every 10 seconds (expo-network doesn't have listeners on all platforms).
 */
export function useNetworkStatus(pollIntervalMs: number = 10000): NetworkStatus {
    const [status, setStatus] = useState<NetworkStatus>({
        isOnline: true,
        isWifi: false,
        type: 'unknown',
    });

    const checkNetwork = useCallback(async () => {
        try {
            const state = await Network.getNetworkStateAsync();
            setStatus({
                isOnline: state.isConnected ?? true,
                isWifi: state.type === Network.NetworkStateType.WIFI,
                type: state.type ?? 'unknown',
            });
        } catch {
            // If we can't check, assume online
            setStatus((prev) => ({ ...prev, isOnline: true }));
        }
    }, []);

    useEffect(() => {
        // Check immediately
        checkNetwork();

        // Then poll
        const interval = setInterval(checkNetwork, pollIntervalMs);
        return () => clearInterval(interval);
    }, [checkNetwork, pollIntervalMs]);

    return status;
}

export default useNetworkStatus;
