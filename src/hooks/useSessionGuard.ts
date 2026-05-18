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
    const { t } = useTranslation();
    // Capture translated strings in a ref so the async AppState handler can
    // read them without calling useTranslation() inside the callback (which
    // would violate the Rules of Hooks).
    const translationsRef = useRef({
        sessionExpired: t('sessionGuard.sessionExpired'),
        sessionExpiredMessage: t('sessionGuard.sessionExpiredMessage'),
    });

    // Keep the ref up-to-date whenever the language changes.
    useEffect(() => {
        translationsRef.current = {
            sessionExpired: t('sessionGuard.sessionExpired'),
            sessionExpiredMessage: t('sessionGuard.sessionExpiredMessage'),
        };
    }, [t]);

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
                        // Session expired — read pre-resolved strings from the ref
                        // instead of calling useTranslation() here (hooks cannot be
                        // called inside async callbacks).
                        const { logout } = useAuthStore.getState();
                        await logout();
                        Alert.alert(
                            translationsRef.current.sessionExpired,
                            translationsRef.current.sessionExpiredMessage,
                        );
                    } else {
                        // Refresh session if it's about to expire (within 5 minutes)
                        const expiresAt = session.expires_at;
                        if (expiresAt) {
                            const expiresInMs = expiresAt * 1000 - Date.now();
                            if (expiresInMs < 5 * 60 * 1000) {
                                const { error: refreshError } = await supabase.auth.refreshSession();
                                if (refreshError) {
                                    const refreshMsg = (refreshError as any)?.message ?? '';
                                    if (
                                        refreshMsg.includes('Invalid Refresh Token') ||
                                        refreshMsg.includes('Refresh Token Not Found')
                                    ) {
                                        // Stale token — log out silently
                                        const { logout } = useAuthStore.getState();
                                        await logout();
                                        Alert.alert(
                                            translationsRef.current.sessionExpired,
                                            translationsRef.current.sessionExpiredMessage,
                                        );
                                    }
                                }
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
