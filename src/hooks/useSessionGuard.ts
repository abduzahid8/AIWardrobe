/**
 * useSessionGuard — detects expired sessions on app foreground.
 *
 * Checks Supabase token validity when the app returns from background.
 * If the session has expired, shows an alert and triggers logout.
 *
 * Usage: Call once in RootNavigator or App.tsx:
 *   useSessionGuard();
 */
import { useEffect, useRef } from 'react';
import { AppState, AppStateStatus, Alert } from 'react-native';
import { supabase } from '../../lib/supabase';
import useAuthStore from '../../store/auth';
import { useTranslation } from 'react-i18next';

export function useSessionGuard(): void {
    const appState = useRef(AppState.currentState);

    useEffect(() => {
        const handleAppStateChange = async (nextAppState: AppStateStatus) => {
            // Only check when coming back to foreground
            if (
                appState.current.match(/inactive|background/) &&
                nextAppState === 'active'
            ) {
                const { isAuthenticated } = useAuthStore.getState();
                if (!isAuthenticated) return;

                try {
                    const { data: { session }, error } = await supabase.auth.getSession();

                    if (error || !session) {
                        // Session expired
                        const { logout } = useAuthStore.getState();
                        const { t } = useTranslation();
                        await logout();
                        Alert.alert(
                            t('sessionGuard.sessionExpired'),
                            t('sessionGuard.sessionExpiredMessage'),
                        );
                    } else {
                        // Refresh session if it's about to expire (within 5 minutes)
                        const expiresAt = session.expires_at;
                        if (expiresAt) {
                            const expiresInMs = expiresAt * 1000 - Date.now();
                            if (expiresInMs < 5 * 60 * 1000) {
                                await supabase.auth.refreshSession();
                            }
                        }
                    }
                } catch {
                    // Don't crash on session check failure
                }
            }

            appState.current = nextAppState;
        };

        const subscription = AppState.addEventListener('change', handleAppStateChange);
        return () => subscription.remove();
    }, []);
}

export default useSessionGuard;
