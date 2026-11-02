import { useEffect, useRef } from 'react';
import NetInfo from '@react-native-community/netinfo';

/** Retry only after an observed disconnection has recovered, never on the initial event. */
export function useNetworkRecovery(onRecovery: () => void): void {
    const callback = useRef(onRecovery);
    callback.current = onRecovery;
    useEffect(() => {
        let wasOffline = false;
        return NetInfo.addEventListener(state => {
            if (state.isConnected === false || state.isInternetReachable === false) {
                wasOffline = true;
            } else if (state.isConnected === true && state.isInternetReachable === true && wasOffline) {
                wasOffline = false;
                callback.current();
            }
        });
    }, []);
}
