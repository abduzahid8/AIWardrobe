import { create } from 'zustand';
import { Platform } from 'react-native';
import { Session } from '@supabase/supabase-js';
import type { Subscription } from '@supabase/supabase-js';
import * as WebBrowser from 'expo-web-browser';
import * as Linking from 'expo-linking';

import { supabase } from '../lib/supabase';
import { analyticsService } from '../src/services/analyticsService';
import { crashReporting } from '../src/services/crashReporting';
import { iapService } from '../src/services/iapService';
import useSubscriptionStore from './subscriptionStore';
import { createLogger } from '../src/utils/logger';
import { clearAllPersistedUserData } from '../src/lib/persistence';
import { authEvents } from './authEvents';
import { uploadQueue } from '../src/services/uploadQueue';

const log = createLogger('AuthStore');

// ============================================
// TYPES
// ============================================

export type AuthMethod = 'email' | 'apple' | 'google';

export interface AuthUser {
    id: string;
    email: string;
    username: string;
    gender?: 'male' | 'female' | 'other' | 'prefer_not_to_say';
    profile_image?: string;

    subscription_tier?: 'free' | 'premium';
    subscription_expires_at?: string;

    created_at?: string;
    updated_at?: string;
}

export interface AuthState {
    user: AuthUser | null;
    session: Session | null;
    loading: boolean;
    error: string | null;
    isAuthenticated: boolean;
    isInitialized: boolean;
}

export interface AuthActions {
    initializeAuth: () => Promise<void>;
    register: (
        email: string,
        password: string,
        username: string,
        gender?: string,
        profileImage?: string,
    ) => Promise<void>;
    login: (email: string, password: string) => Promise<void>;
    signInWithApple: () => Promise<void>;
    signInWithGoogle: () => Promise<void>;
    logout: () => Promise<void>;
    deleteAccount: () => Promise<void>;
    fetchUser: () => Promise<void>;
    handleDeepLink: (url: string) => Promise<void>;
    clearError: () => void;
}

export type AuthStore = AuthState & AuthActions;

// ============================================
// SHARED HELPERS
// ============================================

let _authSubscription: Subscription | null = null;

/**
 * Unsubscribes the Supabase onAuthStateChange listener registered by
 * `initializeAuth`. Called from the `RootNavigator` useEffect cleanup so
 * the listener does not persist across component remounts (Defect 5.5).
 */
export function cleanupAuthSubscription(): void {
  if (_authSubscription) {
    _authSubscription.unsubscribe();
    _authSubscription = null;
  }
}

/**
 * Parses the access_token and refresh_token from a Supabase auth URL hash.
 */
function parseSupabaseUrl(url: string) {
    const hash = url.split('#')[1];
    if (!hash) return null;

    const params = new URLSearchParams(hash);
    const accessToken = params.get('access_token');
    const refreshToken = params.get('refresh_token');

    if (!accessToken || !refreshToken) return null;
    return { accessToken, refreshToken };
}

/**
 * Central side-effect pipeline for every successful authentication.
 * Exactly one place where we tag analytics, crash reporting, IAP,
 * and emit the in-app login event.
 */
async function onAuthSuccess(
    set: (partial: Partial<AuthStore>) => void,
    get: () => AuthStore,
    session: Session,
    method: AuthMethod,
): Promise<void> {
    set({ session, isAuthenticated: true, loading: false, error: null });

    await get().fetchUser();

    if (method === 'email') analyticsService.trackLogin('email');
    else if (method === 'apple') analyticsService.trackLogin('apple');
    else if (method === 'google') analyticsService.trackLogin('google');

    analyticsService.setUserId(session.user.id);
    crashReporting.setUser(session.user.id);
    iapService.identify(session.user.id).catch(() => {});
    authEvents.emitLogin(session.user.id);
    
    // Ensure subscription and trial status are resolved immediately
    await useSubscriptionStore.getState().verifySubscriptionFromServer().catch(() => {});

    // Automatically initialize 7-day free trial on signup/login
    await useSubscriptionStore.getState().initializeTrial(session.user.id).catch(() => {});

    log.info('Authentication succeeded', { method, userId: session.user.id });
}

function setAuthError(
    set: (partial: Partial<AuthStore>) => void,
    error: unknown,
    fallback: string,
): never {
    // Accept Error instances, Supabase { message } objects, or bare strings.
    let message: string = fallback;
    if (error instanceof Error) {
        message = error.message || fallback;
    } else if (typeof error === 'string') {
        message = error;
    } else if (error && typeof error === 'object' && 'message' in error) {
        const m = (error as { message?: unknown }).message;
        if (typeof m === 'string' && m) message = m;
    }

    log.error(fallback, error);
    set({ error: message, loading: false });
    throw error instanceof Error ? error : new Error(message);
}

// ============================================
// STORE
// ============================================

const useAuthStore = create<AuthStore>((set, get) => ({
    user: null,
    session: null,
    loading: false,
    error: null,
    isAuthenticated: false,
    isInitialized: false,

    initializeAuth: async () => {
        log.debug('initializeAuth called');
        set({ loading: true });
        try {
            const { data: { session }, error } = await supabase.auth.getSession();
            // An "Invalid Refresh Token" error means the stored session is stale
            // (e.g. the user was signed out server-side or the token expired).
            // Treat it as "no session" rather than a hard error so the app
            // lands on the sign-in screen instead of crashing.
            if (error) {
                const msg = (error as any)?.message ?? '';
                if (
                    msg.includes('Invalid Refresh Token') ||
                    msg.includes('Refresh Token Not Found') ||
                    msg.includes('refresh_token_not_found')
                ) {
                    log.warn('Stale refresh token detected — clearing session', msg);
                    await supabase.auth.signOut().catch(() => undefined);
                    // Fall through with no session — user will see sign-in screen
                } else {
                    throw error;
                }
            } else if (session) {
                // Fast-path: set basic user from session metadata immediately so
                // the UI can render without waiting for the profiles round-trip.
                // fetchUser() enriches with the full profile in the background.
                const fastUser: AuthUser = {
                    id: session.user.id,
                    email: session.user.email || '',
                    username: session.user.user_metadata?.username || '',
                    gender: session.user.user_metadata?.gender,
                    profile_image: session.user.user_metadata?.profile_image,
                };
                set({ session, isAuthenticated: true, user: fastUser });

                // Fire profile enrichment in the background — do NOT block
                // initializeAuth() on this Supabase round-trip (fixes slow
                // cold-start where fetchUser took 1-2s on marginal networks).
                get().fetchUser().catch((err) => {
                    log.warn('Background fetchUser failed during session restore', err);
                });

                iapService.identify(session.user.id).catch(() => {});
                authEvents.emitLogin(session.user.id);
                log.info('Restored existing session', { userId: session.user.id });
            }

            const { data: { subscription } } = supabase.auth.onAuthStateChange((event, nextSession) => {
                log.debug('Auth state changed', { event, hasSession: !!nextSession });
                if (nextSession) {
                    set({ session: nextSession, isAuthenticated: true });
                } else {
                    set({ session: null, user: null, isAuthenticated: false });
                }
            });

            _authSubscription = subscription;
        } catch (err) {
            log.error('Auth initialization failed', err);
        } finally {
            set({ loading: false, isInitialized: true });
        }
    },

    handleDeepLink: async (url: string) => {
        log.debug('handleDeepLink called', { url });
        const tokens = parseSupabaseUrl(url);
        if (!tokens) return;

        // Check if this is a password-recovery deep link (type=recovery in the hash)
        const hash = url.split('#')[1];
        const params = new URLSearchParams(hash || '');
        const isRecovery = params.get('type') === 'recovery';

        set({ loading: true, error: null });
        try {
            const { data, error } = await supabase.auth.setSession({
                access_token: tokens.accessToken,
                refresh_token: tokens.refreshToken,
            });
            if (error) throw error;
            if (data.session) {
                if (isRecovery) {
                    // Password-recovery flow: just set session + auth state without
                    // triggering login analytics, IAP identity, or trial initialisation.
                    set({ session: data.session, isAuthenticated: true, loading: false, error: null });
                } else {
                    await onAuthSuccess(set, get, data.session, 'email');
                }
                log.info('Deep link auth successful', { userId: data.session.user.id, isRecovery });
            }
        } catch (err) {
            log.error('Deep link auth failed', err);
            set({ error: 'Session expired or invalid link. Please try again.' });
        } finally {
            set({ loading: false });
        }
    },

    register: async (email, password, username, gender, profileImage) => {
        log.debug('Register attempt', { email, username });
        set({ loading: true, error: null });
        try {
            const { data, error } = await supabase.auth.signUp({
                email,
                password,
                options: {
                    data: { username, gender, profile_image: profileImage },
                },
            });
            if (error) throw error;

            if (data.session) {
                await onAuthSuccess(set, get, data.session, 'email');
                analyticsService.trackSignup('email');
            } else {
                set({
                    loading: false,
                    error: 'Please check your email for a confirmation link.',
                });
            }
        } catch (err) {
            setAuthError(set, err, 'Registration failed');
        }
    },

    login: async (email, password) => {
        log.debug('Login attempt', { email });
        set({ loading: true, error: null });
        try {
            const { data, error } = await supabase.auth.signInWithPassword({ email, password });
            if (error) throw error;
            if (data.session) await onAuthSuccess(set, get, data.session, 'email');
        } catch (err) {
            setAuthError(set, err, 'Login failed');
        }
    },

    signInWithApple: async () => {
        // Apple Sign-In is iOS-only. On Android, fail fast with a clear message.
        if (Platform.OS !== 'ios') {
            throw new Error('Sign in with Apple is only available on iOS.');
        }

        set({ loading: true, error: null });
        try {
            // Dynamic import so Android and Expo Go don't crash on missing module.
            const AppleAuthentication = await import('expo-apple-authentication').catch(() => null);
            if (!AppleAuthentication) {
                throw new Error(
                    'Sign in with Apple requires a native development build with expo-apple-authentication installed.',
                );
            }

            const available = await AppleAuthentication.isAvailableAsync();
            if (!available) {
                throw new Error('Sign in with Apple is not available on this device.');
            }

            const credential = await AppleAuthentication.signInAsync({
                requestedScopes: [
                    AppleAuthentication.AppleAuthenticationScope.FULL_NAME,
                    AppleAuthentication.AppleAuthenticationScope.EMAIL,
                ],
            });

            if (!credential.identityToken) {
                throw new Error('No identity token returned by Apple.');
            }

            const { data, error } = await supabase.auth.signInWithIdToken({
                provider: 'apple',
                token: credential.identityToken,
            });
            if (error) throw error;
            if (!data.session) throw new Error('Supabase did not return a session for Apple sign-in.');

            // Apple only returns the user's name on FIRST sign-in. Capture it
            // into the profiles row if present, so future logins still show it.
            const fullName = credential.fullName;
            if (fullName && (fullName.givenName || fullName.familyName)) {
                const username = [fullName.givenName, fullName.familyName]
                    .filter(Boolean)
                    .join(' ')
                    .trim();
                if (username) {
                    await supabase
                        .from('profiles')
                        .update({ username })
                        .eq('id', data.session.user.id)
                        .then(({ error: updateErr }) => {
                            if (updateErr) log.warn('Failed to persist Apple display name', updateErr);
                        });
                }
            }

            await onAuthSuccess(set, get, data.session, 'apple');
        } catch (err) {
            const anyErr = err as { code?: string; message?: string };
            // User cancelling the Apple sheet is not an error the UI should show.
            if (anyErr?.code === 'ERR_REQUEST_CANCELED') {
                set({ loading: false, error: null });
                return;
            }
            // "Unacceptable audience in id_token" happens in Expo Go / simulator
            // because the Apple identity token is issued for the Expo client app ID,
            // not the production bundle ID. This is an expected dev-environment
            // limitation — downgrade to a warning instead of a crash report.
            const msg = anyErr?.message ?? '';
            if (msg.includes('Unacceptable audience') || msg.includes('id_token')) {
                log.warn('Apple Sign-In failed (expected in Expo Go / simulator):', msg);
                set({
                    loading: false,
                    error: 'Apple Sign-In is not available in Expo Go. Please use a development build.',
                });
                return;
            }
            setAuthError(set, err, 'Apple sign-in failed');
        }
    },

    signInWithGoogle: async () => {
      set({ loading: true, error: null });
      try {
        const redirectUrl = Linking.createURL('/auth/callback');

        const { data, error } = await supabase.auth.signInWithOAuth({
          provider: 'google',
          options: {
            redirectTo: redirectUrl,
            skipBrowserRedirect: true,
          },
        });

        if (error) throw error;
        if (!data?.url) throw new Error('Failed to get Google auth URL');

        const result = await WebBrowser.openAuthSessionAsync(data.url, redirectUrl);

        if (result.type === 'success') {
          const tokens = parseSupabaseUrl(result.url);
          if (!tokens) throw new Error('Failed to parse auth response');

          const { data: sessionData, error: sessionError } = await supabase.auth.setSession({
            access_token: tokens.accessToken,
            refresh_token: tokens.refreshToken,
          });
          if (sessionError) throw sessionError;
          if (sessionData.session) {
            await onAuthSuccess(set, get, sessionData.session, 'google');
          }
        } else {
          set({ loading: false });
        }
      } catch (err) {
        setAuthError(set, err, 'Google sign-in failed');
      }
    },

    logout: async () => {
        if (_authSubscription) {
            _authSubscription.unsubscribe();
            _authSubscription = null;
        }
        // Remove the AppState listener registered by uploadQueue.init() to prevent
        // listener accumulation across login/logout cycles (Defect 5.2).
        uploadQueue.destroy();
        authEvents.emitLogout();

        try {
            await supabase.auth.signOut();
        } catch (err) {
            log.warn('supabase.auth.signOut failed — continuing local cleanup', err);
        }

        analyticsService.trackEvent('logout');
        analyticsService.clearUserId();
        crashReporting.clearUser();
        await iapService.logout().catch(() => undefined);

        // Wipe per-user local state so subsequent logins on shared
        // devices never inherit the previous user's data.
        await clearAllPersistedUserData();

        set({ user: null, session: null, isAuthenticated: false });
        log.info('User logged out');
    },

    deleteAccount: async () => {
        set({ loading: true, error: null });
        try {
            const { data: { session } } = await supabase.auth.getSession();
            if (!session?.access_token) throw new Error('Not authenticated');

            const { data, error: fnError } = await supabase.functions.invoke('delete-account', {
                method: 'POST',
            });
            if (fnError) throw new Error(fnError.message || 'Account deletion failed');
            if (data && !data.success) throw new Error(data.error || 'Account deletion failed');

            analyticsService.trackEvent('account_deleted');

            if (_authSubscription) {
                _authSubscription.unsubscribe();
                _authSubscription = null;
            }
            authEvents.emitLogout();

            // The auth user may already be deleted server-side; ignore failures.
            await supabase.auth.signOut().catch(() => undefined);

            analyticsService.clearUserId();
            crashReporting.clearUser();
            await iapService.logout().catch(() => undefined);

            await clearAllPersistedUserData();

            set({ user: null, session: null, isAuthenticated: false, loading: false });
            log.info('Account deleted and local data cleared');
        } catch (err) {
            setAuthError(set, err, 'Account deletion failed');
        }
    },

    fetchUser: async () => {
        log.debug('fetchUser called');
        try {
            // Read the session from Zustand state — initializeAuth() already
            // stored it there, so we don't need another getSession() round-trip
            // (fixes Defect 1.2).
            const existingSession = get().session;
            const user = existingSession?.user;
            if (!user) {
                set({ user: null });
                return;
            }

            const { data: profile, error } = await supabase
                .from('profiles')
                .select('*')
                .eq('id', user.id)
                .maybeSingle();

            if (error) log.warn('Error fetching profile', error);

            if (profile) {
                set({ user: profile as AuthUser, error: null });
                return;
            }

            const fallbackUser: AuthUser = {
                id: user.id,
                email: user.email || '',
                username: user.user_metadata?.username || '',
                gender: user.user_metadata?.gender,
                profile_image: user.user_metadata?.profile_image,
            };
            set({ user: fallbackUser, error: null });
        } catch (err) {
            log.error('Failed to fetch user', err);
            set({
                user: null,
                error: err instanceof Error ? err.message : 'Failed to fetch user',
            });
        }
    },

    clearError: () => set({ error: null }),
}));

export default useAuthStore;
