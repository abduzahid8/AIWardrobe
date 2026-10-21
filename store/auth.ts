import { AccountNameError, assertHandleAvailable, normalizeHandle } from '../src/services/accountNames';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { create } from 'zustand';
import { Platform } from 'react-native';
import { Session } from '@supabase/supabase-js';
import type { Subscription } from '@supabase/supabase-js';
import * as WebBrowser from 'expo-web-browser';
import * as Linking from 'expo-linking';

import { supabase } from '../lib/supabase';
import { withRequestDeadline } from '../src/lib/requestDeadline';
import { analyticsService } from '../src/services/analyticsService';
import { crashReporting } from '../src/services/crashReporting';
import { iapService } from '../src/services/iapService';
import useSubscriptionStore from './subscriptionStore';
import { createLogger } from '../src/utils/logger';
import { clearAllPersistedUserData, resetInMemoryUserState, ensurePreviousAccountCleanup } from '../src/lib/persistence';
import { authEvents } from './authEvents';
import { uploadQueue } from '../src/services/uploadQueue';
import { createAuthOperationQueue } from '../src/lib/authOperationQueue';
import { beginSignOutIntent, hasPendingSignOutIntent, completeSignOutIntent } from '../src/lib/signOutIntent';

const log = createLogger('AuthStore');
const authOperationQueue = createAuthOperationQueue();
let retainAuthWork: (<T>(work: Promise<T>) => Promise<T>) | null = null;

// ============================================
// TYPES
// ============================================

export type AuthMethod = 'email' | 'apple' | 'google';

export interface AuthUser {
    id: string;
    email: string;
    username: string;
    handle?: string;
    display_name?: string;
    gender?: 'male' | 'female' | 'other' | 'prefer_not_to_say';
    profile_image?: string;

    subscription_tier?: 'free' | 'premium';
    subscription_expires_at?: string;

    created_at?: string;
    updated_at?: string;
}

export interface AuthState {
    passwordRecovery: { ownerId: string; revision: number } | null;
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
    logout: (expectedOwnerId?: string) => Promise<void>;
    updatePassword: (password: string, expectedOwnerId: string) => Promise<void>;
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
let profileReadVersion = 0;
let authInitializationVersion = 0;
let authLifecycleRevision = 0;

/**
 * Unsubscribes the Supabase onAuthStateChange listener registered by
 * `initializeAuth`. Called from the `RootNavigator` useEffect cleanup so
 * the listener does not persist across component remounts (Defect 5.5).
 * Also invalidates startup reads and callbacks queued before unsubscribe.
 */
export function cleanupAuthSubscription(): void {
  authLifecycleRevision++;
  disposeAuthSubscription();
}

function disposeAuthSubscription(): void {
  authInitializationVersion++;
  if (_authSubscription) {
    _authSubscription.unsubscribe();
    _authSubscription = null;
  }
}

/**
 * Parses a Supabase auth redirect URL.
 *
 * Supports BOTH flows Supabase may emit:
 *  - Implicit flow (legacy): tokens in the URL hash
 *    `...#access_token=...&refresh_token=...&type=recovery`
 *  - PKCE flow (current default): auth code in the query string
 *    `...?code=...&type=recovery`
 *
 * Returns a discriminated union so callers can either `setSession`
 * (implicit) or `exchangeCodeForSession` (PKCE).
 */
export type ParsedSupabaseAuthUrl =
    | { kind: 'tokens'; accessToken: string; refreshToken: string; type: string | null }
    | { kind: 'code'; code: string; type: string | null };

// The auth browser and native URL listener can deliver the same callback.
// Retain one completed credential in memory only, scoped to its exact session.
let completedGoogleRedirect: {
    credential: string; ownerId: string; accessToken: string; revision: number;
} | null = null;

function rememberGoogleRedirect(parsed: ParsedSupabaseAuthUrl, session: Session, state: AuthStore) {
    if (state.isAuthenticated && state.session?.user.id === session.user.id
        && state.session.access_token === session.access_token) {
        completedGoogleRedirect = { credential: JSON.stringify(parsed), ownerId: session.user.id,
            accessToken: session.access_token, revision: authEvents.getRevision() };
    }
}

function getQueryParam(url: string, name: string): string | null {
    try {
        const qIndex = url.indexOf('?');
        if (qIndex === -1) return null;
        const hashIndex = url.indexOf('#');
        const query = hashIndex === -1 ? url.slice(qIndex + 1) : url.slice(qIndex + 1, hashIndex);
        return new URLSearchParams(query).get(name);
    } catch {
        return null;
    }
}

export function parseSupabaseUrl(url: string): ParsedSupabaseAuthUrl | null {
    // 1. Implicit flow: #access_token=...&refresh_token=...
    const hash = url.split('#')[1];
    if (hash) {
        const params = new URLSearchParams(hash);
        const accessToken = params.get('access_token');
        const refreshToken = params.get('refresh_token');
        if (accessToken && refreshToken) {
            return { kind: 'tokens', accessToken, refreshToken, type: params.get('type') };
        }
    }

    // 2. PKCE flow: ?code=...
    const code = getQueryParam(url, 'code');
    if (code) {
        return { kind: 'code', code, type: getQueryParam(url, 'type') };
    }

    return null;
}

/**
 * @deprecated Use the PKCE-aware `parseSupabaseUrl` above. Kept for
 * backwards-compat with existing unit tests that mock the old shape.
 * Prefer asserting on `ParsedSupabaseAuthUrl`.
 */
export function parseSupabaseTokensOnly(
    url: string,
): { accessToken: string; refreshToken: string } | null {
    const parsed = parseSupabaseUrl(url);
    if (parsed?.kind === 'tokens') {
        return { accessToken: parsed.accessToken, refreshToken: parsed.refreshToken };
    }
    return null;
}

let publicationVersion = 0;
let pendingPublication: { owner: string; token: string; revision: number; work: Promise<boolean> } | null = null;
let callbackResetBarrier: Promise<unknown> = Promise.resolve();

/** Keep the next account anonymous until previous-account reset writes finish. */
function publishAuthenticatedSession(
    set: (partial: Partial<AuthStore>) => void,
    get: () => AuthStore,
    session: Session,
    restoredUser?: AuthUser,
    allowCachedRestore = false,
): Promise<boolean> {
    if (pendingPublication?.owner === session.user.id && pendingPublication.token === session.access_token
        && pendingPublication.revision === authEvents.getRevision()) return pendingPublication.work;
    const previousOwner = get().session?.user.id ?? get().user?.id ?? null;
    // Only startup may reuse disk caches, and only after their explicit owner
    // stamp and the pending logout/cleanup barriers have been checked.
    const coldRestore = allowCachedRestore && !previousOwner && !authEvents.getUserId() && !pendingPublication;
    const changedAccount = previousOwner !== session.user.id || authEvents.getUserId() !== session.user.id;
    const publish = () => {
        authEvents.emitLogin(session.user.id);
        set({ session, user: restoredUser?.id === session.user.id ? restoredUser
            : get().user?.id === session.user.id ? get().user : null,
            isAuthenticated: true, loading: false, error: null });
    };
    if (!changedAccount) { publish(); return Promise.resolve(true); }

    const version = ++publicationVersion;
    const previousWork = pendingPublication?.work;
    authEvents.emitLogout();
    const revision = authEvents.getRevision();
    set({ session: null, user: null, isAuthenticated: false, loading: true });
    let resetWork: Promise<void>;
    try { resetWork = coldRestore ? Promise.resolve() : Promise.resolve(resetInMemoryUserState()); }
    catch (error) { resetWork = Promise.reject(error); }
    const work = (async () => {
        // A superseded reset may still be deleting shared keys. Wait for it too.
        try {
            const outcomes = await Promise.allSettled([previousWork, ensurePreviousAccountCleanup(), resetWork, hasPendingSignOutIntent()]);
            const failure = outcomes.find((outcome): outcome is PromiseRejectedResult => outcome.status === 'rejected');
            if (failure) throw failure.reason;
            const intent = outcomes[3];
            if (intent.status === 'fulfilled' && intent.value) throw new Error('Sign out must complete before restoring an account');
            if (version !== publicationVersion || revision !== authEvents.getRevision()) return false;
            if (coldRestore && await AsyncStorage.getItem('private_cache_owner_v1') !== session.user.id) {
                await resetInMemoryUserState();
            }
            if (version !== publicationVersion || revision !== authEvents.getRevision()) return false;
            await AsyncStorage.setItem('private_cache_owner_v1', session.user.id);
        }
        catch (error) {
            if (version === publicationVersion && revision === authEvents.getRevision()) {
                set({ loading: false, error: 'Unable to clear previous account data. Please try again.' });
            }
            throw error;
        }
        if (version !== publicationVersion || revision !== authEvents.getRevision()) return false;
        publish();
        return true;
    })();
    const publication = { owner: session.user.id, token: session.access_token, revision, work };
    pendingPublication = publication;
    const clear = () => { if (pendingPublication === publication) pendingPublication = null; };
    void work.then(clear, clear);
    return work;
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
    if (!await publishAuthenticatedSession(set, get, session)) return;
    const revision = authEvents.getRevision();
    const stillCurrent = () => get().isAuthenticated && get().session?.user.id === session.user.id
        && authEvents.getRevision() === revision;

    await get().fetchUser();
    if (!stillCurrent()) return;

    if (method === 'email') analyticsService.trackLogin('email');
    else if (method === 'apple') analyticsService.trackLogin('apple');
    else if (method === 'google') analyticsService.trackLogin('google');

    analyticsService.setUserId(session.user.id);
    crashReporting.setUser(session.user.id);
    await withLogoutTimeout(iapService.identify(session.user.id).catch(() => undefined), 8000, 'iapService.identify');
    if (!stillCurrent()) return;
    
    // Ensure subscription and trial status are resolved immediately
    await useSubscriptionStore.getState().verifySubscriptionFromServer().catch(() => {});
    if (!stillCurrent()) return;

    // Automatically initialize 7-day free trial on signup/login
    await useSubscriptionStore.getState().initializeTrial(session.user.id).catch(() => {});
    if (!stillCurrent()) return;

    // Kick off the new user's wardrobe fetch in the background — never block
    // login on it (it takes 1-2s on marginal networks). force=true bypasses
    // the TTL guard, which is user-scoped but the in-memory reset above
    // cleared lastSyncedAt anyway. Lazy-require avoids an import cycle
    // (wardrobeSyncSlice imports this auth store).
    try {
        const useWardrobeStore = require('./wardrobeStore').default;
        void useWardrobeStore
            .getState()
            .rehydrateFromCloud(true)
            .catch(() => undefined);
    } catch {}

    log.info('Authentication succeeded', { method, userId: session.user.id });
}

/**
 * Race a cleanup promise against a timeout so logout/delete can NEVER hang
 * forever on a stalled network call (signOut / IAP / storage). Local state
 * clearing is what actually protects shared devices — a hung remote call
 * must not block it. Resolves undefined on timeout; the underlying promise
 * keeps running in the background.
 */
function withLogoutTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T | undefined> {
    retainAuthWork?.(promise);
    let timer: ReturnType<typeof setTimeout> | undefined;
    const timeout = new Promise<undefined>((resolve) => {
        timer = setTimeout(() => {
            timer = undefined;
            log.warn(`${label} timed out after ${ms}ms — continuing local cleanup`);
            resolve(undefined);
        }, ms);
    });
    // Clear the timer as soon as either branch wins so a fast success never
    // fires a spurious "timed out" warning seconds later (and leaves no
    // stray timer behind in tests).
    return Promise.race([promise, timeout]).finally(() => {
        if (timer !== undefined) clearTimeout(timer);
    });
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
    passwordRecovery: null,
    user: null,
    session: null,
    loading: false,
    error: null,
    isAuthenticated: false,
    isInitialized: false,

    initializeAuth: async () => {
        log.debug('initializeAuth called');
        disposeAuthSubscription();
        const initializationVersion = authInitializationVersion;
        const startingRevision = authEvents.getRevision();
        // A same-account login can replace the session without changing the auth revision.
        const startingSession = get().session;
        set({ loading: true });
        try {
            const { data: { session }, error } = await supabase.auth.getSession();
            if (initializationVersion !== authInitializationVersion) return;
            // A newer login/logout owns the identity even if its startup read finishes later.
            if (startingRevision === authEvents.getRevision() && get().session === startingSession) {
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
                        handle: session.user.user_metadata?.handle,
                        display_name: session.user.user_metadata?.display_name || session.user.user_metadata?.full_name || session.user.user_metadata?.name,
                        gender: session.user.user_metadata?.gender,
                        profile_image: session.user.user_metadata?.profile_image,
                    };
                    if (!await publishAuthenticatedSession(set, get, session, fastUser, true)) return;

                    // Fire profile enrichment in the background — do NOT block
                    // initializeAuth() on this Supabase round-trip (fixes slow
                    // cold-start where fetchUser took 1-2s on marginal networks).
                    get().fetchUser().catch((err) => {
                        log.warn('Background fetchUser failed during session restore', err);
                    });

                    await withLogoutTimeout(iapService.identify(session.user.id).catch(() => undefined), 8000, 'iapService.identify (restore)');
                    log.info('Restored existing session', { userId: session.user.id });
                }
            }
            if (initializationVersion !== authInitializationVersion) return;
            const { data: { subscription } } = supabase.auth.onAuthStateChange((event, nextSession) => {
                if (initializationVersion !== authInitializationVersion) return;
                log.debug('Auth state changed', { event, hasSession: !!nextSession });
                const previousId = get().session?.user.id ?? get().user?.id;
                if (nextSession) {
                    const changedAccount = previousId !== nextSession.user.id;
                    if (changedAccount) {
                        // Keep SDK callbacks synchronous: the Auth client holds a lock.
                        const publication = publishAuthenticatedSession(set, get, nextSession);
                        retainAuthWork?.(publication);
                        // Never await inside an SDK callback. Register a queue barrier
                        // so subsequent user actions also wait for these reset writes.
                        void authOperationQueue.run(async () => { await publication; }).catch(() => {
                            log.warn('Account-switch cleanup failed');
                        });
                    } else {
                        set({ session: nextSession, isAuthenticated: true });
                    }
                } else {
                    authEvents.emitLogout();
                    try { uploadQueue.destroy(); } catch {}
                    try {
                        const reset = resetInMemoryUserState();
                        const resetWork = Promise.all([callbackResetBarrier, reset]);
                        // Keep ordering after failure without permanently poisoning
                        // retries. New publication still runs the cleanup gate/reset.
                        callbackResetBarrier = resetWork.catch(() => undefined);
                        retainAuthWork?.(resetWork);
                        void authOperationQueue.run(async () => { await resetWork; }).catch(() => undefined);
                    } catch {}
                    set({ session: null, user: null, isAuthenticated: false, loading: false });
                }
            });

            _authSubscription = subscription;
        } catch (err) {
            if (initializationVersion === authInitializationVersion) log.error('Auth initialization failed', err);
        } finally {
            if (initializationVersion === authInitializationVersion) set({ loading: false, isInitialized: true });
        }
    },

    handleDeepLink: async (url: string) => {
        // Each incoming link must establish its own recovery authority.
        set({ passwordRecovery: null });
        // Auth URLs may contain access/refresh tokens or a one-use PKCE code.
        log.debug('handleDeepLink called');

        // Surface OAuth provider errors (e.g. user denied consent) instead of
        // silently ignoring the deep link.
        const fragment = new URLSearchParams(url.split('#')[1] || '');
        const oauthError = getQueryParam(url, 'error') ?? getQueryParam(url, 'error_code')
            ?? fragment.get('error') ?? fragment.get('error_code');
        if (oauthError) {
            completedGoogleRedirect = null;
            const description =
                getQueryParam(url, 'error_description') ?? fragment.get('error_description')
                ?? 'Sign-in was not completed. Please try again.';
            log.warn('OAuth deep link returned an error', { oauthError, description });
            // URLSearchParams already decoded this value. Decoding again can
            // corrupt literal '%' text or throw before the error reaches UI.
            set({ error: description, loading: false });
            return;
        }

        const parsed = parseSupabaseUrl(url);
        if (!parsed) return;
        if (completedGoogleRedirect && completedGoogleRedirect.credential === JSON.stringify(parsed)
            && completedGoogleRedirect.revision === authEvents.getRevision()
            && get().isAuthenticated && get().session?.user.id === completedGoogleRedirect.ownerId
            && get().session?.access_token === completedGoogleRedirect.accessToken) return;
        completedGoogleRedirect = null;

        // Password-recovery links carry type=recovery in either the hash
        // (implicit) or the query string (PKCE) — never treat them as logins.
        // PKCE callbacks can omit type; the registered reset route still
        // identifies the recovery flow after a successful code exchange.
        const route = url.split(/[?#]/)[0].replace(/\/$/, '');
        const recoveryRoute = Linking.createURL('reset-password').split(/[?#]/)[0].replace(/\/$/, '');
        const isRecovery = parsed.type === 'recovery' || (parsed.kind === 'code' && route === recoveryRoute);

        set({ loading: true, error: null });
        try {
            if (parsed.kind === 'code') {
                // PKCE flow (current Supabase default): exchange ?code= for a session.
                const { data, error } = await supabase.auth.exchangeCodeForSession(parsed.code);
                if (error) throw error;
                if (data.session) {
                    if (isRecovery) {
                        if (!await publishAuthenticatedSession(set, get, data.session)) return;
                        set({ passwordRecovery: { ownerId: data.session.user.id, revision: authEvents.getRevision() } });
                    } else {
                        await onAuthSuccess(set, get, data.session, 'email');
                    }
                    log.info('Deep link auth successful (PKCE)', { userId: data.session.user.id, isRecovery });
                } else {
                    throw new Error('Sign-in link did not return a session. Please try again.');
                }
            } else {
                const { data, error } = await supabase.auth.setSession({
                    access_token: parsed.accessToken,
                    refresh_token: parsed.refreshToken,
                });
                if (error) throw error;
                if (data.session) {
                    if (isRecovery) {
                        // Password-recovery flow: just set session + auth state without
                        // triggering login analytics, IAP identity, or trial initialisation.
                        if (!await publishAuthenticatedSession(set, get, data.session)) return;
                        set({ passwordRecovery: { ownerId: data.session.user.id, revision: authEvents.getRevision() } });
                    } else {
                        await onAuthSuccess(set, get, data.session, 'email');
                    }
                    log.info('Deep link auth successful', { userId: data.session.user.id, isRecovery });
                }
            }
        } catch (err) {
            log.error('Deep link auth failed', err);
            set({ error: 'Session expired or invalid link. Please try again.' });
        } finally {
            set({ loading: false });
        }
    },

    register: async (email, password, username, gender, profileImage) => {
        const normalizedEmail = email.trim().toLowerCase();
        const normalizedUsername = normalizeHandle(username);
        log.debug('Register attempt', { email: normalizedEmail, username: normalizedUsername });
        set({ loading: true, error: null });
        try {
            await assertHandleAvailable(normalizedUsername);
            const { data, error } = await supabase.auth.signUp({
                email: normalizedEmail,
                password,
                options: {
                    data: { username: normalizedUsername, handle: normalizedUsername, display_name: username.trim(), gender, profile_image: profileImage },
                },
            });
            if (error) {
                // Availability is advisory; the unique constraint arbitrates races.
                if (/database|username|handle|duplicate/i.test(error.message || '')) {
                    try { await assertHandleAvailable(normalizedUsername); }
                    catch (availabilityError) {
                        if (availabilityError instanceof AccountNameError && availabilityError.code === 'username_exists') throw availabilityError;
                    }
                }
                throw error;
            }

            if (data.session) {
                await onAuthSuccess(set, get, data.session, 'email');
                analyticsService.trackSignup('email');
            } else {
                // Email confirmation required (Supabase "Confirm email" ON).
                // Throw a typed sentinel so the UI can show a check-email
                // success state instead of silently doing nothing. The store
                // must NOT stash this informational message in `error`.
                set({ loading: false, error: null });
                throw new Error(
                    'EMAIL_CONFIRMATION_REQUIRED: Please check your email for a confirmation link.',
                );
            }
        } catch (err) {
            // Let the confirmation sentinel pass through with loading=false
            // and no error state — the SignUp screen renders its own UI.
            if (
                err instanceof Error &&
                err.message.startsWith('EMAIL_CONFIRMATION_REQUIRED')
            ) {
                throw err;
            }
            setAuthError(set, err, 'Registration failed');
        }
    },

    login: async (email, password) => {
        const normalizedEmail = email.trim().toLowerCase();
        log.debug('Login attempt', { email: normalizedEmail });
        set({ loading: true, error: null });
        try {
            const { data, error } = await supabase.auth.signInWithPassword({
                email: normalizedEmail,
                password,
            });
            if (error) throw error;
            if (data.session) await onAuthSuccess(set, get, data.session, 'email');
            else throw new Error('Login failed: no session returned. Please try again.');
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
            // Use require so Android/iOS/Expo Go don't crash at startup or run dynamic import.
            let AppleAuthentication: any = null;
            try {
                AppleAuthentication = require('expo-apple-authentication');
            } catch (e) {}
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
                        .update({ display_name: username })
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
        const redirectUrl = Linking.createURL('auth/callback');

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
          const fragment = new URLSearchParams(result.url.split('#')[1] || '');
          const providerError = getQueryParam(result.url, 'error') ?? getQueryParam(result.url, 'error_code')
            ?? fragment.get('error') ?? fragment.get('error_code');
          const providerDescription = getQueryParam(result.url, 'error_description')
            ?? fragment.get('error_description');
          // URLSearchParams has already decoded these values. A second
          // decode corrupts literal percent signs and can hide the cause.
          if (providerError) throw new Error(providerDescription || providerError);
          const parsed = parseSupabaseUrl(result.url);
          if (!parsed) {
            throw new Error(providerDescription || 'Failed to parse auth response');
          }

          if (parsed.kind === 'code') {
            // PKCE flow (current Supabase default): ?code=... must be exchanged.
            const { data: sessionData, error: sessionError } =
              await supabase.auth.exchangeCodeForSession(parsed.code);
            if (sessionError) throw sessionError;
            if (!sessionData.session) throw new Error('Google sign-in did not return a session.');
            await onAuthSuccess(set, get, sessionData.session, 'google');
            rememberGoogleRedirect(parsed, sessionData.session, get());
          } else {
            const { data: sessionData, error: sessionError } = await supabase.auth.setSession({
              access_token: parsed.accessToken,
              refresh_token: parsed.refreshToken,
            });
            if (sessionError) throw sessionError;
            if (sessionData.session) {
              await onAuthSuccess(set, get, sessionData.session, 'google');
              rememberGoogleRedirect(parsed, sessionData.session, get());
            } else {
              throw new Error('Google sign-in did not return a session.');
            }
          }
        } else if (result.type === 'cancel' || result.type === 'dismiss') {
          // User backed out of the browser — not an error, just reset loading.
          set({ loading: false, error: null });
        } else {
          set({ loading: false });
        }
      } catch (err) {
        setAuthError(set, err, 'Google sign-in failed');
      }
    },

    updatePassword: async (password, expectedOwnerId) => {
        const revision = authEvents.getRevision();
        const ownsSession = () => get().isAuthenticated
            && get().session?.user.id === expectedOwnerId
            && get().passwordRecovery?.ownerId === expectedOwnerId
            && get().passwordRecovery?.revision === revision
            && authEvents.getRevision() === revision;
        if (!expectedOwnerId || !ownsSession()) throw new Error('Password reset session changed');
        const { data: { session }, error } = await supabase.auth.getSession();
        if (error) throw error;
        if (!ownsSession() || session?.user.id !== expectedOwnerId) throw new Error('Password reset session changed');
        const result = await supabase.auth.updateUser({ password });
        if (!ownsSession()) throw new Error('Password reset session changed');
        if (result.error) throw result.error;
        if (result.data.user?.id !== expectedOwnerId) throw new Error('Password update could not be confirmed');
    },

    logout: async () => {
        disposeAuthSubscription();
        // Remove the AppState listener registered by uploadQueue.init() to prevent
        // listener accumulation across login/logout cycles (Defect 5.2).
        uploadQueue.destroy();
        authEvents.emitLogout();
        const logoutRevision = authEvents.getRevision();
        const stillCurrent = () => authEvents.getRevision() === logoutRevision;
        let intentError: unknown;
        try { await beginSignOutIntent(); } catch (error) { intentError = error; }

        try {
            await withLogoutTimeout(supabase.auth.signOut(), 8000, 'supabase.auth.signOut');
        } catch (err) {
            log.warn('supabase.auth.signOut failed — continuing local cleanup', err);
        }

        if (!stillCurrent()) return;
        analyticsService.trackEvent('logout');
        analyticsService.clearUserId();
        crashReporting.clearUser();
        await withLogoutTimeout(
            iapService.logout().catch(() => undefined),
            8000,
            'iapService.logout',
        );

        if (!stillCurrent()) return;
        // Wipe per-user local state so subsequent logins on shared
        // devices never inherit the previous user's data.
        let cleanupConfirmed: boolean | undefined;
        try {
            cleanupConfirmed = await withLogoutTimeout(clearAllPersistedUserData(logoutRevision).then(() => true), 10000, 'clearAllPersistedUserData');
        } finally {
            if (stillCurrent()) set({ user: null, session: null, isAuthenticated: false, loading: false });
        }
        if (!stillCurrent()) return;
        if (cleanupConfirmed !== true) throw new Error('Local account cleanup is still pending');
        await confirmLocalSignOut();
        if (intentError) throw new Error('Sign-out intent could not be recorded');
        log.info('User logged out');
    },

    deleteAccount: async () => {
        let deletionConfirmed = false;
        let intentError: unknown;
        let operationRevision = authEvents.getRevision();
        const currentOwner = () => get().session?.user?.id ?? get().user?.id ?? null;
        const owner = currentOwner();
        const stillCurrent = () => authEvents.getRevision() === operationRevision && currentOwner() === owner;
        set({ loading: true, error: null });
        try {
            if (!owner || !get().isAuthenticated) throw new Error('Not authenticated');
            const { data: { session } } = await supabase.auth.getSession();
            if (!stillCurrent()) return;
            if (!session?.access_token || session.user.id !== owner) throw new Error('Session does not match the selected account');

            const { data, error: fnError } = await supabase.functions.invoke('delete-account', {
                method: 'POST',
                headers: { Authorization: `Bearer ${session.access_token}` },
            });
            if (!stillCurrent()) return;
            if (fnError) throw new Error(fnError.message || 'Account deletion failed');
            if (data?.success !== true) throw new Error(data?.error || 'Account deletion failed');
            deletionConfirmed = true;
            try { await beginSignOutIntent(); } catch (error) { intentError = error; }

            analyticsService.trackEvent('account_deleted');

            disposeAuthSubscription();
            // Parity with logout: remove the AppState listener registered by
            // uploadQueue.init() (clearAllPersistedUserData also does this via
            // resetInMemoryUserState, but keep it explicit so the two paths
            // can never drift again).
            try {
                uploadQueue.destroy();
            } catch {}
            authEvents.emitLogout();
            operationRevision = authEvents.getRevision();

            // The auth user may already be deleted server-side; ignore failures.
            await withLogoutTimeout(
                supabase.auth.signOut().catch(() => undefined),
                8000,
                'supabase.auth.signOut (delete)',
            );

            if (!stillCurrent()) return;
            analyticsService.clearUserId();
            crashReporting.clearUser();
            await withLogoutTimeout(
                iapService.logout().catch(() => undefined),
                8000,
                'iapService.logout (delete)',
            );

            if (!stillCurrent()) return;
            const cleanupConfirmed = await withLogoutTimeout(clearAllPersistedUserData(operationRevision).then(() => true), 10000, 'clearAllPersistedUserData (delete)');
            if (cleanupConfirmed !== true) throw new Error('Local account cleanup is still pending');
            await confirmLocalSignOut();
            if (intentError) throw new Error('Sign-out intent could not be recorded');

            if (!stillCurrent()) return;
            set({ user: null, session: null, isAuthenticated: false, loading: false });
            log.info('Account deleted and local data cleared');
        } catch (err) {
            if (!stillCurrent()) return;
            if (deletionConfirmed) {
                set({ user: null, session: null, isAuthenticated: false, loading: false, error: 'Local account cleanup failed' });
                throw err;
            }
            setAuthError(set, err, 'Account deletion failed');
        }
    },

    fetchUser: async () => {
        const readVersion = ++profileReadVersion;
        const sessionUser = get().session?.user;
        const revision = authEvents.getRevision();
        if (!sessionUser) { set({ user: null }); return; }
        const stillCurrent = () => readVersion === profileReadVersion && get().session?.user.id === sessionUser.id && authEvents.getRevision() === revision;
        const basicUser = (): AuthUser => ({
            id: sessionUser.id, email: sessionUser.email || '', username: sessionUser.user_metadata?.username || '',
            handle: sessionUser.user_metadata?.handle,
            display_name: sessionUser.user_metadata?.display_name || sessionUser.user_metadata?.full_name || sessionUser.user_metadata?.name,
            gender: sessionUser.user_metadata?.gender, profile_image: sessionUser.user_metadata?.profile_image,
        });
        try {
            const { data: profile, error } = await withRequestDeadline(signal =>
                supabase.from('profiles').select('*').eq('id', sessionUser.id).abortSignal(signal).maybeSingle());
            if (!stillCurrent()) return;
            if (error) {
                log.warn('Error fetching profile', error);
                // Do not replace a known profile with incomplete session metadata.
                if (get().user?.id === sessionUser.id) return;
            }
            if (profile) {
                if (profile.id !== sessionUser.id) throw new Error('Profile identity mismatch');
                set({ user: profile as AuthUser, error: null });
            } else {
                set({ user: basicUser(), error: null });
            }
        } catch (err) {
            if (!stillCurrent()) return;
            log.error('Failed to fetch user', err);
            set({
                user: get().user?.id === sessionUser.id ? get().user : basicUser(),
                error: err instanceof Error ? err.message : 'Failed to fetch user',
            });
        }
    },

    clearError: () => set({ error: null }),
}));

/** Confirm the local SDK state; an HTTP logout result alone is insufficient. */
async function confirmLocalSignOut(): Promise<void> {
    const { data: { session }, error } = await supabase.auth.getSession();
    if (error || session) throw new Error('Local session removal could not be confirmed');
    await completeSignOutIntent();
}

async function recoverPendingSignOut(): Promise<void> {
    if (!await hasPendingSignOutIntent()) return;
    disposeAuthSubscription();
    authEvents.emitLogout();
    useAuthStore.setState({ user: null, session: null, isAuthenticated: false, loading: true });
    let failed = false;
    try { const result = await supabase.auth.signOut(); if (result?.error) failed = true; }
    catch { failed = true; }
    try { await iapService.logout(); } catch { failed = true; }
    try { await clearAllPersistedUserData(authEvents.getRevision()); } catch { failed = true; }
    await callbackResetBarrier;
    if (failed) throw new Error('Pending sign out could not be completed');
    await confirmLocalSignOut();
}

// Serialize the whole action, not only SDK calls: profile publication, IAP
// identity and local cleanup must complete before the next identity starts.
const coordinatedActions = ['initializeAuth', 'handleDeepLink', 'register', 'login',
    'signInWithApple', 'signInWithGoogle', 'logout', 'deleteAccount', 'updatePassword'] as const;
for (const name of coordinatedActions) {
    const action = useAuthStore.getState()[name] as (...args: never[]) => Promise<void>;
    const coordinated = (...args: never[]) => {
        const requestedOwner = useAuthStore.getState().session?.user?.id ?? useAuthStore.getState().user?.id;
        const requestedLifecycle = authLifecycleRevision;
        const requestedIdentityRevision = authEvents.getRevision();
        return authOperationQueue.run(async retain => {
            await callbackResetBarrier;
            await pendingPublication?.work;
            if (name !== 'logout' && name !== 'deleteAccount') {
                try { await recoverPendingSignOut(); }
                catch (error) {
                    disposeAuthSubscription();
                    authEvents.emitLogout();
                    useAuthStore.setState({ user: null, session: null, isAuthenticated: false, loading: false,
                        error: 'Unable to complete sign out. Please try again.',
                        ...(name === 'initializeAuth' ? { isInitialized: true } : {}) });
                    throw error;
                }
            }
            if (name === 'initializeAuth' && requestedLifecycle !== authLifecycleRevision) return;
            const currentOwner = useAuthStore.getState().session?.user?.id ?? useAuthStore.getState().user?.id;
            const expectedOwner = name === 'updatePassword' ? args[1] : name === 'logout' ? args[0] : undefined;
            if (expectedOwner !== undefined && expectedOwner !== currentOwner) {
                throw new Error('Account changed before the operation started');
            }
            if ((name === 'updatePassword' || (name === 'logout' && expectedOwner !== undefined))
                && requestedIdentityRevision !== authEvents.getRevision()) {
                throw new Error('Account changed before the operation started');
            }
            // Deletion is authorized for the account visible when requested,
            // never for an account established by a previously queued login.
            if (name === 'deleteAccount' && requestedOwner !==
                (useAuthStore.getState().session?.user?.id ?? useAuthStore.getState().user?.id)) {
                throw new Error('Account changed before deletion started');
            }
            if (name !== 'initializeAuth' && name !== 'handleDeepLink' && name !== 'updatePassword') {
                completedGoogleRedirect = null;
                useAuthStore.setState({ passwordRecovery: null });
            }
            retainAuthWork = retain;
            try { await action(...args); }
            finally { retainAuthWork = null; }
        });
    };
    useAuthStore.setState({ [name]: coordinated });
}

export default useAuthStore;
