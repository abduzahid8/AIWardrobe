/**
 * Tests for the auth Zustand store
 */

// Mock supabase
const mockGetSession = jest.fn();
const mockSignUp = jest.fn();
const mockSignInWithPassword = jest.fn();
const mockSignOut = jest.fn();
const mockUpdateUser = jest.fn();
const mockSetSession = jest.fn();
const mockExchangeCodeForSession = jest.fn();
const mockSignInWithOAuth = jest.fn();
const mockSignInWithIdToken = jest.fn();
const mockOnAuthStateChange = jest.fn((_callback?: (event: string, session: any) => void) => ({ data: { subscription: { unsubscribe: jest.fn() } } }));
const mockFrom = jest.fn();
const mockNameAvailability = jest.fn();
const mockAppleNameCredential = jest.fn();
jest.mock('expo-apple-authentication', () => ({ isAvailableAsync: jest.fn().mockResolvedValue(true), signInAsync: (...args: any[]) => mockAppleNameCredential(...args), AppleAuthenticationScope: { FULL_NAME: 0, EMAIL: 1 } }));

jest.mock('../../lib/supabase', () => ({
    supabase: {
        auth: {
            getSession: () => mockGetSession(),
            signUp: (...args: any[]) => mockSignUp(...args),
            signInWithPassword: (...args: any[]) => mockSignInWithPassword(...args),
            signOut: () => mockSignOut(),
            updateUser: (...args: any[]) => mockUpdateUser(...args),
            setSession: (...args: any[]) => mockSetSession(...args),
            exchangeCodeForSession: (...args: any[]) => mockExchangeCodeForSession(...args),
            signInWithOAuth: (...args: any[]) => mockSignInWithOAuth(...args),
            signInWithIdToken: (...args: any[]) => mockSignInWithIdToken(...args),
            onAuthStateChange: (callback: any) => mockOnAuthStateChange(callback),
        },
        from: (table: string) => mockFrom(table),
        rpc: (...args: any[]) => mockNameAvailability(...args),
        functions: { invoke: jest.fn().mockResolvedValue({ data: { success: true }, error: null }) },
    },
}));

jest.mock('../../src/services/iapService', () => ({
    iapService: {
        identify: jest.fn().mockResolvedValue(undefined),
        logout: jest.fn().mockResolvedValue(undefined),
        initialize: jest.fn().mockResolvedValue(undefined),
    },
}));

jest.mock('../../src/services/analyticsService', () => ({
    analyticsService: {
        trackLogin: jest.fn(),
        trackSignup: jest.fn(),
        trackEvent: jest.fn(),
        setUserId: jest.fn(),
        clearUserId: jest.fn(),
        initialize: jest.fn(),
    },
}));

jest.mock('../../src/services/crashReporting', () => ({
    crashReporting: { setUser: jest.fn(), clearUser: jest.fn(), initialize: jest.fn() },
}));

jest.mock('../../store/subscriptionStore', () => ({
    __esModule: true,
    default: { getState: () => ({ verifySubscriptionFromServer: jest.fn().mockResolvedValue(undefined), initializeTrial: jest.fn().mockResolvedValue(undefined) }) },
}));

jest.mock('../../src/lib/persistence', () => ({
    clearAllPersistedUserData: jest.fn().mockResolvedValue(undefined),
    resetInMemoryUserState: jest.fn(),
    ensurePreviousAccountCleanup: jest.fn().mockResolvedValue(undefined),
}));

jest.mock('../../src/services/uploadQueue', () => ({
    uploadQueue: { destroy: jest.fn(), init: jest.fn() },
}));

jest.mock('expo-web-browser', () => ({
    WebBrowserResultType: { CANCEL: 'cancel', DISMISS: 'dismiss' },
    openAuthSessionAsync: jest.fn(),
}));

jest.mock('expo-linking', () => ({
    createURL: jest.fn((p: string) => `com.aiwardrobe://${p}`),
}));

// Import after mocks
import { waitFor } from '@testing-library/react-native';
import useAuthStore, { cleanupAuthSubscription, parseSupabaseUrl } from '../../store/auth';
import { authEvents } from '../../store/authEvents';
import { completeSignOutIntent, beginSignOutIntent, SIGN_OUT_INTENT_KEY } from '../../src/lib/signOutIntent';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { resetInMemoryUserState } from '../../src/lib/persistence';
import { iapService } from '../../src/services/iapService';
import { analyticsService } from '../../src/services/analyticsService';
import * as WebBrowser from 'expo-web-browser';

beforeEach(async () => {
    await completeSignOutIntent();
    jest.clearAllMocks();
    jest.mocked(WebBrowser.openAuthSessionAsync).mockReset();
    jest.mocked(resetInMemoryUserState).mockReset().mockResolvedValue(undefined);
    for (const mock of [mockGetSession, mockSignUp, mockSignInWithPassword, mockSignOut,
        mockSetSession, mockExchangeCodeForSession, mockSignInWithOAuth, mockSignInWithIdToken, mockUpdateUser, mockFrom]) mock.mockReset();
    mockGetSession.mockResolvedValue({ data: { session: null }, error: null });
    mockSignOut.mockResolvedValue({});
    jest.mocked(iapService.identify).mockReset().mockResolvedValue(undefined);
    jest.mocked(iapService.logout).mockReset().mockResolvedValue(undefined);
    require('../../src/lib/persistence').clearAllPersistedUserData.mockReset().mockResolvedValue(undefined);
    require('../../lib/supabase').supabase.functions.invoke.mockReset().mockResolvedValue({ data: { success: true }, error: null });
    // Reset store state
    useAuthStore.setState({
        user: null,
        session: null,
        loading: false,
        error: null,
        isAuthenticated: false,
    });
    mockNameAvailability.mockResolvedValue({ data: true, error: null });
    mockFrom.mockReturnValue({
        select: () => ({
            eq: () => ({
                abortSignal() { return this; }, maybeSingle: () => Promise.resolve({ data: null, error: null }),
            }),
        }),
        update: () => ({
            eq: () => Promise.resolve({ error: null }),
        }),
    });
});

describe('auth store', () => {
    describe('initial state', () => {
        it('should have correct defaults', () => {
            const state = useAuthStore.getState();
            expect(state.user).toBeNull();
            expect(state.session).toBeNull();
            expect(state.loading).toBe(false);
            expect(state.error).toBeNull();
            expect(state.isAuthenticated).toBe(false);
        });
    });

    describe('parseSupabaseUrl (implicit + PKCE)', () => {
        it('parses implicit #access_token flow', () => {
            const parsed = parseSupabaseUrl('com.aiwardrobe://auth/callback#access_token=a&refresh_token=r&type=signup');
            expect(parsed).toMatchObject({ kind: 'tokens', accessToken: 'a', refreshToken: 'r' });
        });

        it('parses PKCE ?code flow', () => {
            const parsed = parseSupabaseUrl('com.aiwardrobe://auth/callback?code=pkce-123&type=recovery');
            expect(parsed).toMatchObject({ kind: 'code', code: 'pkce-123', type: 'recovery' });
        });

        it('returns null for unrelated URLs', () => {
            expect(parseSupabaseUrl('com.aiwardrobe://reset-password')).toBeNull();
        });
    });

    describe('clearError', () => {
        it('should set error to null', () => {
            useAuthStore.setState({ error: 'some error' });
            useAuthStore.getState().clearError();
            expect(useAuthStore.getState().error).toBeNull();
        });
    });

    describe('logout', () => {
        it('should clear user state and call signOut', async () => {
            mockSignOut.mockResolvedValue({});
            useAuthStore.setState({
                user: { id: '1', email: 'test@example.com', username: 'test' },
                session: {} as any,
                isAuthenticated: true,
            });

            await useAuthStore.getState().logout();

            const state = useAuthStore.getState();
            expect(state.user).toBeNull();
            expect(state.session).toBeNull();
            expect(state.isAuthenticated).toBe(false);
            expect(mockSignOut).toHaveBeenCalledTimes(1);
        });
    });

    describe('login', () => {
        it('should set authenticated state on successful login', async () => {
            const mockSession = { access_token: 'abc', user: { id: '1', email: 'test@example.com' } };
            mockSignInWithPassword.mockResolvedValue({
                data: { session: mockSession },
                error: null,
            });
            // Mock fetchUser calls
            mockGetSession.mockResolvedValue({ data: { session: { user: { id: '1', email: 'test@example.com' } } } });
            mockFrom.mockReturnValue({
                select: () => ({
                    eq: () => ({
                        abortSignal() { return this; }, maybeSingle: () => Promise.resolve({ data: { id: '1', email: 'test@example.com', username: 'test' }, error: null }),
                    }),
                }),
            });

            await useAuthStore.getState().login('test@example.com', 'password');

            const state = useAuthStore.getState();
            expect(state.isAuthenticated).toBe(true);
            expect(state.loading).toBe(false);
        });

        it('normalizes email (trim + lowercase) before Supabase', async () => {
            const mockSession = { access_token: 'abc', user: { id: '1', email: 'test@example.com' } };
            mockSignInWithPassword.mockResolvedValue({ data: { session: mockSession }, error: null });

            await useAuthStore.getState().login('  Test@Example.COM  ', 'password');

            expect(mockSignInWithPassword).toHaveBeenCalledWith({
                email: 'test@example.com',
                password: 'password',
            });
        });

        it('should set error on login failure', async () => {
            mockSignInWithPassword.mockResolvedValue({
                data: {},
                error: { message: 'Invalid credentials' },
            });

            try {
                await useAuthStore.getState().login('test@example.com', 'wrongpassword');
            } catch {
                // Store may or may not re-throw
            }

            const state = useAuthStore.getState();
            expect(state.error).toBe('Invalid credentials');
            expect(state.loading).toBe(false);
        });
    });

    describe('register', () => {
        it('should set authenticated state on successful registration with session', async () => {
            const mockSession = { access_token: 'abc', user: { id: '1', email: 'new@example.com' } };
            mockSignUp.mockResolvedValue({
                data: { session: mockSession },
                error: null,
            });
            mockGetSession.mockResolvedValue({ data: { session: { user: { id: '1', email: 'new@example.com', user_metadata: { username: 'newuser' } } } } });
            mockFrom.mockReturnValue({
                select: () => ({
                    eq: () => ({
                        abortSignal() { return this; }, maybeSingle: () => Promise.resolve({ data: null, error: null }),
                    }),
                }),
            });

            await useAuthStore.getState().register('new@example.com', 'password', 'newuser');

            const state = useAuthStore.getState();
            expect(state.isAuthenticated).toBe(true);
        });

        it('throws EMAIL_CONFIRMATION_REQUIRED without setting store error when no session', async () => {
            mockSignUp.mockResolvedValue({ data: { session: null, user: { id: '1' } }, error: null });

            await expect(
                useAuthStore.getState().register('new@example.com', 'password', 'newuser'),
            ).rejects.toThrow('EMAIL_CONFIRMATION_REQUIRED');

            // Informational state must not pollute the error field.
            expect(useAuthStore.getState().error).toBeNull();
            expect(useAuthStore.getState().loading).toBe(false);
            expect(useAuthStore.getState().isAuthenticated).toBe(false);
        });
    });

    describe('handleDeepLink', () => {
        it('surfaces OAuth errors containing literal percent signs without throwing', async () => {
            await expect(useAuthStore.getState().handleDeepLink(
                'com.aiwardrobe://auth/callback?error=access_denied&error_description=Consent%20was%20100%25%20denied',
            )).resolves.toBeUndefined();
            expect(useAuthStore.getState().error).toBe('Consent was 100% denied');
            expect(mockExchangeCodeForSession).not.toHaveBeenCalled();
        });

        it('surfaces implicit-flow errors in the URL fragment', async () => {
            await useAuthStore.getState().handleDeepLink(
                'com.aiwardrobe://auth/callback#error=access_denied&error_description=Consent+denied',
            );
            expect(useAuthStore.getState().error).toBe('Consent denied');
            expect(mockSetSession).not.toHaveBeenCalled();
        });

        it('exchanges PKCE ?code for a session', async () => {
            const mockSession = { access_token: 'abc', user: { id: '1', email: 'a@b.c' } };
            mockExchangeCodeForSession.mockResolvedValue({ data: { session: mockSession }, error: null });

            await useAuthStore.getState().handleDeepLink('com.aiwardrobe://auth/callback?code=pkce-123');

            expect(mockExchangeCodeForSession).toHaveBeenCalledWith('pkce-123');
            expect(useAuthStore.getState().isAuthenticated).toBe(true);
        });

        it('sets session directly for implicit #tokens links', async () => {
            const mockSession = { access_token: 'abc', user: { id: '1', email: 'a@b.c' } };
            mockSetSession.mockResolvedValue({ data: { session: mockSession }, error: null });

            await useAuthStore
                .getState()
                .handleDeepLink('com.aiwardrobe://auth/callback#access_token=a&refresh_token=r');

            expect(mockSetSession).toHaveBeenCalledWith({ access_token: 'a', refresh_token: 'r' });
            expect(useAuthStore.getState().isAuthenticated).toBe(true);
        });

        it('does not trigger login pipeline for recovery links', async () => {
            const mockSession = { access_token: 'abc', user: { id: '1', email: 'a@b.c' } };
            mockExchangeCodeForSession.mockResolvedValue({ data: { session: mockSession }, error: null });

            await useAuthStore
                .getState()
                .handleDeepLink('com.aiwardrobe://reset-password?code=rec-123&type=recovery');

            expect(useAuthStore.getState().isAuthenticated).toBe(true);
            expect(useAuthStore.getState().session).toEqual(mockSession);
        });

        it('ignores non-auth URLs', async () => {
            await useAuthStore.getState().handleDeepLink('com.aiwardrobe://reset-password');
            expect(mockSetSession).not.toHaveBeenCalled();
            expect(mockExchangeCodeForSession).not.toHaveBeenCalled();
        });

        it.each(['pkce', 'implicit'])('clears prior recovery authority after a rejected %s link without targeting the signed-in account', async flow => {
            const session = { access_token: 'synthetic-current', user: { id: 'current-owner' } };
            authEvents.emitLogin('current-owner');
            useAuthStore.setState({ session: session as any, isAuthenticated: true,
                passwordRecovery: { ownerId: 'current-owner', revision: authEvents.getRevision() } });
            const failure = { data: { session: null }, error: { message: 'synthetic expired link' } };
            mockExchangeCodeForSession.mockResolvedValueOnce(failure);
            mockSetSession.mockResolvedValueOnce(failure);
            await useAuthStore.getState().handleDeepLink(flow === 'pkce'
                ? 'com.aiwardrobe://reset-password?code=expired&type=recovery'
                : 'com.aiwardrobe://reset-password#access_token=expired&refresh_token=expired&type=recovery');
            expect(useAuthStore.getState().session).toEqual(session);
            expect(useAuthStore.getState().passwordRecovery).toBeNull();
            expect(mockUpdateUser).not.toHaveBeenCalled();
        });

        it.each(['pkce', 'implicit', 'pkce-without-type'])('grants recovery only after a successful %s exchange', async flow => {
            const session = { access_token: 'synthetic-recovery', user: { id: 'recovered-owner' } };
            const response = { data: { session }, error: null };
            mockExchangeCodeForSession.mockResolvedValueOnce(response);
            mockSetSession.mockResolvedValueOnce(response);
            await useAuthStore.getState().handleDeepLink(flow === 'implicit'
                ? 'com.aiwardrobe://reset-password#access_token=synthetic&refresh_token=synthetic&type=recovery'
                : `com.aiwardrobe://reset-password?code=synthetic${flow === 'pkce' ? '&type=recovery' : ''}`);
            expect(useAuthStore.getState().passwordRecovery).toEqual({ ownerId: 'recovered-owner', revision: authEvents.getRevision() });
            expect(analyticsService.trackLogin).not.toHaveBeenCalled();
        });
    });

    describe('deleteAccount', () => {
        it('does not clear a signed-in account when the server omits deletion confirmation', async () => {
            const session = { access_token: 'test-session', user: { id: 'owner' } };
            mockGetSession.mockResolvedValue({ data: { session }, error: null });
            useAuthStore.setState({ session: session as any, isAuthenticated: true });
            const { supabase } = require('../../lib/supabase');
            supabase.functions.invoke.mockResolvedValueOnce({ data: null, error: null });
            await expect(useAuthStore.getState().deleteAccount()).rejects.toThrow('Account deletion failed');
            expect(useAuthStore.getState().isAuthenticated).toBe(true);
            expect(useAuthStore.getState().session).toBe(session);
            expect(require('../../src/lib/persistence').clearAllPersistedUserData).not.toHaveBeenCalled();
        });
    });

    describe('logout hang resistance', () => {
        afterEach(() => {
            jest.useRealTimers();
        });

        it('completes local logout while signOut is stalled', async () => {
            jest.useFakeTimers();
            let release!: () => void;
            mockSignOut.mockReturnValueOnce(new Promise<void>(resolve => { release = resolve; }));
            useAuthStore.setState({
                user: { id: '1', email: 'test@example.com', username: 'test' },
                session: {} as any,
                isAuthenticated: true,
            });

            const done = jest.fn();
            const p = useAuthStore.getState().logout().then(done);
            // Flush past all logout timeouts (8s signOut + 8s IAP + 10s wipe).
            await jest.advanceTimersByTimeAsync(40000);
            await p;

            expect(done).toHaveBeenCalled();
            const state = useAuthStore.getState();
            expect(state.user).toBeNull();
            expect(state.session).toBeNull();
            expect(state.isAuthenticated).toBe(false);
            release();
            await Promise.resolve();
        });
    });
});

describe('profile response account isolation', () => {
    const profileQuery = (result: Promise<unknown>) => ({ select: () => ({ eq: () => ({ abortSignal() { return this; }, maybeSingle: () => result }) }) });
    const session = (id: string) => ({ access_token: `test-${id}`, user: { id, email: `${id}@example.invalid`, user_metadata: {} } });
    const profile = (id: string) => ({ id, email: `${id}@example.invalid`, username: `profile-${id}` });

    it('discards a late previous-account profile result', async () => {
        const events = require('../../store/authEvents').authEvents;
        events.emitLogin('a');
        useAuthStore.setState({ session: session('a') as any, user: profile('a'), isAuthenticated: true });
        let finish!: (value: unknown) => void;
        mockFrom.mockReturnValueOnce(profileQuery(new Promise(resolve => { finish = resolve; })));
        const pending = useAuthStore.getState().fetchUser();
        events.emitLogout(); events.emitLogin('b');
        useAuthStore.setState({ session: session('b') as any, user: profile('b') });
        finish({ data: profile('a'), error: null }); await pending;
        expect(useAuthStore.getState().user).toEqual(profile('b'));
    });

    it('discards a late previous-account fetch exception without clearing the new profile', async () => {
        const events = require('../../store/authEvents').authEvents;
        events.emitLogin('a');
        useAuthStore.setState({ session: session('a') as any, user: profile('a'), isAuthenticated: true });
        let fail!: (error: Error) => void;
        mockFrom.mockReturnValueOnce(profileQuery(new Promise((_, reject) => { fail = reject; })));
        const pending = useAuthStore.getState().fetchUser();
        events.emitLogout(); events.emitLogin('b');
        useAuthStore.setState({ session: session('b') as any, user: profile('b') });
        fail(new Error('offline')); await pending;
        expect(useAuthStore.getState().user).toEqual(profile('b'));
        expect(useAuthStore.getState().error).toBeNull();
    });

    it('preserves a known profile when the refresh returns an error', async () => {
        const events = require('../../store/authEvents').authEvents; events.emitLogin('a');
        useAuthStore.setState({ session: session('a') as any, user: profile('a'), isAuthenticated: true });
        mockFrom.mockReturnValueOnce(profileQuery(Promise.resolve({ data: null, error: { message: 'temporarily unavailable' } })));
        await useAuthStore.getState().fetchUser();
        expect(useAuthStore.getState().user).toEqual(profile('a'));
        expect(useAuthStore.getState().isAuthenticated).toBe(true);
    });

    it('does not identify the previous user after a newer login completes', async () => {
        let finish!: (value: unknown) => void;
        mockSignInWithPassword.mockResolvedValueOnce({ data: { session: session('a') }, error: null })
            .mockResolvedValueOnce({ data: { session: session('b') }, error: null });
        mockFrom.mockReturnValueOnce(profileQuery(new Promise(resolve => { finish = resolve; })))
            .mockReturnValueOnce(profileQuery(Promise.resolve({ data: profile('b'), error: null })));
        const first = useAuthStore.getState().login('a@example.invalid', 'test-password');
        await Promise.resolve();
        const second = useAuthStore.getState().login('b@example.invalid', 'test-password');
        await waitFor(() => expect(typeof finish).toBe('function'));
        finish({ data: profile('a'), error: null });
        await Promise.all([first, second]);
        expect(useAuthStore.getState().user).toEqual(profile('b'));
        const analytics = require('../../src/services/analyticsService').analyticsService;
        expect(analytics.setUserId).toHaveBeenCalledWith('b');
        expect(analytics.setUserId.mock.calls.map((call: string[]) => call[0])).toEqual(['a', 'b']);
        expect(require('../../store/authEvents').authEvents.canApplyToUser('b')).toBe(true);
    });
});

test('restored-session profile enrichment survives its own login event', async () => {
    mockGetSession.mockResolvedValueOnce({ data: { session: { access_token: 'restored', user: { id: 'restore-user', email: 'restore@example.invalid', user_metadata: {} } } }, error: null });
    mockFrom.mockReturnValueOnce({ select: () => ({ eq: () => ({ abortSignal() { return this; }, maybeSingle: async () => ({ data: { id: 'restore-user', email: 'restore@example.invalid', username: 'server-profile' }, error: null }) }) }) });
    await useAuthStore.getState().initializeAuth();
    await waitFor(() => expect(useAuthStore.getState().user?.username).toBe('server-profile'));
});

test('an older same-account profile read cannot overwrite a newer refresh', async () => {
    const events = require('../../store/authEvents').authEvents; events.emitLogin('same-user');
    useAuthStore.setState({ session: { user: { id: 'same-user', user_metadata: {} } } as any, user: { id: 'same-user', email: '', username: 'before' }, isAuthenticated: true });
    let finish!: (value: unknown) => void;
    mockFrom.mockReturnValueOnce({ select: () => ({ eq: () => ({ abortSignal() { return this; }, maybeSingle: () => new Promise(resolve => { finish = resolve; }) }) }) })
        .mockReturnValueOnce({ select: () => ({ eq: () => ({ abortSignal() { return this; }, maybeSingle: async () => ({ data: { id: 'same-user', username: 'newest' }, error: null }) }) }) });
    const first = useAuthStore.getState().fetchUser();
    await useAuthStore.getState().fetchUser();
    finish({ data: { id: 'same-user', username: 'stale' }, error: null }); await first;
    expect(useAuthStore.getState().user?.username).toBe('newest');
});

test('a stalled profile query times out, preserves the known profile and ignores a late response', async () => {
    jest.useFakeTimers();
    const errorLog = jest.spyOn(console, 'error').mockImplementation(() => undefined);
    try {
        const events = require('../../store/authEvents').authEvents; events.emitLogin('timeout-owner');
        const known = { id: 'timeout-owner', email: '', username: 'known-profile' };
        useAuthStore.setState({ session: { user: { id: 'timeout-owner', user_metadata: {} } } as any, user: known, isAuthenticated: true });
        let signal!: AbortSignal, finish!: (value: unknown) => void;
        mockFrom.mockReturnValueOnce({ select: () => ({ eq: () => ({ abortSignal: (value: AbortSignal) => {
            signal = value;
            return { maybeSingle: () => new Promise(resolve => { finish = resolve; }) };
        } }) }) });
        const pending = useAuthStore.getState().fetchUser();
        await jest.advanceTimersByTimeAsync(15_000); await pending;
        expect(signal.aborted).toBe(true);
        expect(useAuthStore.getState().user).toEqual(known);
        expect(useAuthStore.getState().error).toContain('timed out');
        finish({ data: { ...known, username: 'late-result' }, error: null }); await Promise.resolve();
        expect(useAuthStore.getState().user).toEqual(known);
    } finally { errorLog.mockRestore(); jest.useRealTimers(); }
});


describe('SDK authentication event isolation', () => {
    const session = (id: string) => ({ access_token: `test-${id}`, user: { id, user_metadata: {} } });
    async function subscribe() {
        mockGetSession.mockResolvedValueOnce({ data: { session: null }, error: null });
        await useAuthStore.getState().initializeAuth();
        return mockOnAuthStateChange.mock.calls.at(-1)![0]!;
    }
    it('invalidates old-account work and clears memory when the SDK signs out', async () => {
        const callback = await subscribe();
        const events = require('../../store/authEvents').authEvents;
        events.emitLogin('sdk-a');
        useAuthStore.setState({ session: session('sdk-a') as any, user: { id: 'sdk-a', email: '', username: 'old' }, isAuthenticated: true });
        const revision = events.getRevision();
        callback('SIGNED_OUT', null);
        expect(useAuthStore.getState()).toMatchObject({ user: null, session: null, isAuthenticated: false });
        expect(events.getRevision()).toBeGreaterThan(revision);
        expect(events.canApplyToUser('sdk-a')).toBe(false);
        expect(require('../../src/lib/persistence').resetInMemoryUserState).toHaveBeenCalledTimes(1);
    });
    it('clears the previous profile and switches ownership on an SDK account change', async () => {
        const callback = await subscribe();
        const events = require('../../store/authEvents').authEvents; events.emitLogin('sdk-a');
        useAuthStore.setState({ session: session('sdk-a') as any, user: { id: 'sdk-a', email: '', username: 'old' }, isAuthenticated: true });
        callback('SIGNED_IN', session('sdk-b'));
        await waitFor(() => expect(useAuthStore.getState().session?.user.id).toBe('sdk-b'));
        expect(useAuthStore.getState()).toMatchObject({ user: null, session: session('sdk-b'), isAuthenticated: true });
        expect(events.canApplyToUser('sdk-a')).toBe(false);
        expect(events.canApplyToUser('sdk-b')).toBe(true);
        expect(require('../../src/lib/persistence').resetInMemoryUserState).toHaveBeenCalledTimes(1);
    });
    it('preserves profile, memory and revision during a same-account token refresh', async () => {
        const callback = await subscribe();
        const events = require('../../store/authEvents').authEvents; events.emitLogin('sdk-a');
        const user = { id: 'sdk-a', email: '', username: 'known' };
        useAuthStore.setState({ session: session('sdk-a') as any, user, isAuthenticated: true });
        const revision = events.getRevision();
        callback('TOKEN_REFRESHED', { ...session('sdk-a'), access_token: 'refreshed' });
        expect(useAuthStore.getState().user).toEqual(user);
        expect(events.getRevision()).toBe(revision);
        expect(require('../../src/lib/persistence').resetInMemoryUserState).not.toHaveBeenCalled();
    });
});


describe('startup session read isolation', () => {
    const session = (id: string) => ({ access_token: `test-${id}`, user: { id, email: `${id}@example.invalid`, user_metadata: {} } });
    async function loginNewAccount() {
        mockSignInWithPassword.mockResolvedValueOnce({ data: { session: session('new-login') }, error: null });
        await useAuthStore.getState().login('new-login@example.invalid', 'test-password');
    }
    it('cannot restore an old session over a newer completed login', async () => {
        let finish!: (value: unknown) => void;
        mockGetSession.mockReturnValueOnce(new Promise(resolve => { finish = resolve; }));
        const pending = useAuthStore.getState().initializeAuth();
        const login = loginNewAccount();
        await waitFor(() => expect(typeof finish).toBe('function'));
        finish({ data: { session: session('old-restore') }, error: null });
        await Promise.all([pending, login]);
        expect(useAuthStore.getState().session?.user.id).toBe('new-login');
        expect(require('../../src/services/iapService').iapService.identify.mock.calls.map((call: string[]) => call[0])).toEqual(['old-restore', 'new-login']);
        // A newer login still needs the SDK observer even if its startup result was stale.
        expect(mockOnAuthStateChange).toHaveBeenCalledTimes(1);
    });
    it('cannot sign out a newer same-account login because an old startup read returned an expired-token error', async () => {
        require('../../store/authEvents').authEvents.emitLogin('new-login');
        useAuthStore.setState({ session: session('new-login') as any, user: { id: 'new-login', email: '', username: 'before' }, isAuthenticated: true });
        let finish!: (value: unknown) => void;
        mockGetSession.mockReturnValueOnce(new Promise(resolve => { finish = resolve; }));
        const pending = useAuthStore.getState().initializeAuth();
        const login = loginNewAccount();
        await waitFor(() => expect(typeof finish).toBe('function'));
        finish({ data: { session: null }, error: { message: 'Invalid Refresh Token' } });
        await Promise.all([pending, login]);
        expect(mockSignOut).toHaveBeenCalledTimes(1);
        expect(useAuthStore.getState().session?.user.id).toBe('new-login');
    });
    it('keeps the latest initialization and only one registered observer', async () => {
        let finish!: (value: unknown) => void;
        mockGetSession.mockReturnValueOnce(new Promise(resolve => { finish = resolve; }))
            .mockResolvedValueOnce({ data: { session: session('latest-startup') }, error: null });
        const first = useAuthStore.getState().initializeAuth();
        const second = useAuthStore.getState().initializeAuth();
        await waitFor(() => expect(typeof finish).toBe('function'));
        finish({ data: { session: session('obsolete-startup') }, error: null });
        await Promise.all([first, second]);
        expect(useAuthStore.getState().session?.user.id).toBe('latest-startup');
        expect(mockOnAuthStateChange).toHaveBeenCalledTimes(2);
    });
    it('does not register a late observer after navigator cleanup', async () => {
        let finish!: (value: unknown) => void;
        mockGetSession.mockReturnValueOnce(new Promise(resolve => { finish = resolve; }));
        const pending = useAuthStore.getState().initializeAuth();
        await waitFor(() => expect(typeof finish).toBe('function'));
        cleanupAuthSubscription();
        finish({ data: { session: session('disposed') }, error: null }); await pending;
        expect(mockOnAuthStateChange).not.toHaveBeenCalled();
        expect(useAuthStore.getState().session).toBeNull();
    });
    it('ignores a queued callback from an unsubscribed observer', async () => {
        mockGetSession.mockResolvedValueOnce({ data: { session: null }, error: null });
        await useAuthStore.getState().initializeAuth();
        const callback = mockOnAuthStateChange.mock.calls.at(-1)![0]!;
        cleanupAuthSubscription();
        await loginNewAccount();
        callback('SIGNED_OUT', null);
        expect(useAuthStore.getState().session?.user.id).toBe('new-login');
    });
});

describe('logout completion isolation', () => {
  it('does not clean or clear a newer login after a delayed SDK sign-out returns', async () => {
    const {authEvents}=require('../../store/authEvents');
    let finish!:(value:object)=>void;
    mockSignOut.mockImplementationOnce(()=>new Promise(resolve=>{finish=resolve;}));
    const logout=useAuthStore.getState().logout();
    await waitFor(() => expect(typeof finish).toBe('function'));
    const session={access_token:'new-test-token',user:{id:'new-owner'}};
    authEvents.emitLogin('new-owner');
    useAuthStore.setState({session:session as any,user:{id:'new-owner',email:'new@example.com',username:'new'},isAuthenticated:true});
    finish({});await logout;
    expect(require('../../src/lib/persistence').clearAllPersistedUserData).not.toHaveBeenCalled();
    expect(useAuthStore.getState().session).toBe(session);
    expect(useAuthStore.getState().isAuthenticated).toBe(true);
  });

  it('does not clear a newer login after previously started local cleanup returns', async () => {
    const {authEvents}=require('../../store/authEvents');
    const cleanup=require('../../src/lib/persistence').clearAllPersistedUserData as jest.Mock;
    let finish!:()=>void, started!:()=>void;
    const active=new Promise<void>(resolve=>{started=resolve;});
    cleanup.mockImplementationOnce(()=>new Promise<void>(resolve=>{finish=resolve;started();}));
    mockSignOut.mockResolvedValueOnce({});
    const logout=useAuthStore.getState().logout();await active;
    const session={access_token:'new-test-token',user:{id:'new-owner'}};
    authEvents.emitLogin('new-owner');
    useAuthStore.setState({session:session as any,user:{id:'new-owner',email:'new@example.com',username:'new'},isAuthenticated:true});
    finish();await logout;
    expect(useAuthStore.getState().session).toBe(session);
    expect(useAuthStore.getState().isAuthenticated).toBe(true);
  });
});

describe('account deletion identity isolation',()=>{
 const setup=(id='delete-owner')=>{
  const events=require('../../store/authEvents').authEvents;
  events.emitLogin(id);
  const session={access_token:`token-${id}`,user:{id}};
  useAuthStore.setState({session:session as any,user:{id,email:`${id}@example.com`,username:id},isAuthenticated:true,loading:false,error:null});
  return session;
 };
 const invoke=()=>require('../../lib/supabase').supabase.functions.invoke as jest.Mock;

 it('binds the confirmed delete request to the initiating session token',async()=>{
  const session=setup();mockGetSession.mockResolvedValueOnce({data:{session},error:null});
  invoke().mockResolvedValueOnce({data:{success:true},error:null});mockSignOut.mockResolvedValueOnce({});
  await useAuthStore.getState().deleteAccount();
  expect(invoke()).toHaveBeenCalledWith('delete-account',{method:'POST',headers:{Authorization:'Bearer token-delete-owner'}});
  expect(useAuthStore.getState()).toMatchObject({isAuthenticated:false,loading:false});
 });

 it('does not invoke deletion when SDK session ownership differs from the selected account',async()=>{
  setup();mockGetSession.mockResolvedValueOnce({data:{session:{access_token:'other-token',user:{id:'other-owner'}}},error:null});
  await expect(useAuthStore.getState().deleteAccount()).rejects.toThrow('Session does not match');
  expect(invoke()).not.toHaveBeenCalled();
  expect(useAuthStore.getState().session?.user.id).toBe('delete-owner');
 });

 it('cancels before invocation when account changes during session lookup',async()=>{
  const old=setup();let finish!:(value:unknown)=>void;
  mockGetSession.mockImplementationOnce(()=>new Promise(resolve=>{finish=resolve;}));
  const pending=useAuthStore.getState().deleteAccount();
  await waitFor(() => expect(typeof finish).toBe('function'));
  const fresh=setup('new-owner');
  finish({data:{session:old},error:null});await pending;
  expect(invoke()).not.toHaveBeenCalled();expect(useAuthStore.getState().session).toBe(fresh);
 });

 it('ignores successful completion for an old account after a newer login',async()=>{
  const session=setup();mockGetSession.mockResolvedValueOnce({data:{session},error:null});
  let finish!:(value:unknown)=>void,started!:()=>void;
  const active=new Promise<void>(resolve=>{started=resolve;});
  invoke().mockImplementationOnce(()=>new Promise(resolve=>{finish=resolve;started();}));
  const pending=useAuthStore.getState().deleteAccount();await active;const fresh=setup('new-owner');
  finish({data:{success:true},error:null});await pending;
  expect(mockSignOut).not.toHaveBeenCalled();
  expect(require('../../src/lib/persistence').clearAllPersistedUserData).not.toHaveBeenCalled();
  expect(useAuthStore.getState().session).toBe(fresh);
 });

 it('does not publish an old deletion failure into the new account',async()=>{
  const session=setup();mockGetSession.mockResolvedValueOnce({data:{session},error:null});
  let fail!:(error:Error)=>void,started!:()=>void;
  const active=new Promise<void>(resolve=>{started=resolve;});
  invoke().mockImplementationOnce(()=>new Promise((_resolve,reject)=>{fail=reject;started();}));
  const pending=useAuthStore.getState().deleteAccount();await active;const fresh=setup('new-owner');
  fail(new Error('Old deletion failed'));await pending;
  expect(useAuthStore.getState()).toMatchObject({session:fresh,error:null,loading:false,isAuthenticated:true});
 });

 it('does not clear a newer account after already-started deletion cleanup finishes',async()=>{
  const session=setup();mockGetSession.mockResolvedValueOnce({data:{session},error:null});
  invoke().mockResolvedValueOnce({data:{success:true},error:null});mockSignOut.mockResolvedValueOnce({});
  let finish!:()=>void,started!:()=>void;
  const active=new Promise<void>(resolve=>{started=resolve;});
  const cleanup=require('../../src/lib/persistence').clearAllPersistedUserData as jest.Mock;
  cleanup.mockImplementationOnce(()=>new Promise<void>(resolve=>{finish=resolve;started();}));
  const pending=useAuthStore.getState().deleteAccount();await active;const fresh=setup('new-owner');
  finish();await pending;
  expect(useAuthStore.getState()).toMatchObject({session:fresh,isAuthenticated:true,loading:false});
 });
});

describe('deletion provider cleanup isolation',()=>{
 it.each(['signOut','iapLogout'])('does not continue cleanup after a newer login during %s',async stage=>{
  const events=require('../../store/authEvents').authEvents;
  events.emitLogin('old-owner');
  const old={access_token:'old-owner-token',user:{id:'old-owner'}};
  useAuthStore.setState({session:old as any,user:{id:'old-owner',email:'old@example.com',username:'old'},isAuthenticated:true});
  mockGetSession.mockResolvedValueOnce({data:{session:old},error:null});
  require('../../lib/supabase').supabase.functions.invoke.mockResolvedValueOnce({data:{success:true},error:null});
  let finish!:()=>void,started!:()=>void;
  const active=new Promise<void>(resolve=>{started=resolve;});
  const gate=stage==='signOut'?mockSignOut:require('../../src/services/iapService').iapService.logout;
  if(stage==='iapLogout')mockSignOut.mockResolvedValueOnce({});
  gate.mockImplementationOnce(()=>new Promise<void>(resolve=>{finish=resolve;started();}));
  const pending=useAuthStore.getState().deleteAccount();await active;
  events.emitLogin('new-owner');
  const fresh={access_token:'new-owner-token',user:{id:'new-owner'}};
  useAuthStore.setState({session:fresh as any,user:{id:'new-owner',email:'new@example.com',username:'new'},isAuthenticated:true,loading:false});
  finish();await pending;
  expect(require('../../src/lib/persistence').clearAllPersistedUserData).not.toHaveBeenCalled();
  expect(useAuthStore.getState()).toMatchObject({session:fresh,isAuthenticated:true,loading:false});
 });
});


describe('password recovery account isolation', () => {
    it.each(['pkce', 'implicit'])('invalidates the previous account before publishing a %s recovery session', async (flow) => {
        authEvents.emitLogin('recovery-old-owner');
        useAuthStore.setState({
            session: { user: { id: 'recovery-old-owner' } } as any,
            user: { id: 'recovery-old-owner', email: 'old@example.com', username: 'old-private-profile' },
            isAuthenticated: true,
        });
        const session = { access_token: 'synthetic-recovery-token', user: { id: 'recovery-new-owner' } };
        const exchange = flow === 'pkce' ? mockExchangeCodeForSession : mockSetSession;
        exchange.mockResolvedValueOnce({ data: { session }, error: null });
        const observations: unknown[] = [];
        const stop = useAuthStore.subscribe(state => {
            if (state.session === session) observations.push({
                owner: authEvents.getUserId(), profile: state.user,
                resets: jest.mocked(resetInMemoryUserState).mock.calls.length,
            });
        });
        try {
            await useAuthStore.getState().handleDeepLink(flow === 'pkce'
                ? 'com.aiwardrobe://reset-password?code=synthetic-code&type=recovery'
                : 'com.aiwardrobe://reset-password#access_token=synthetic-access&refresh_token=synthetic-refresh&type=recovery');
        } finally { stop(); }
        expect(observations.length).toBeGreaterThan(0);
        for (const observation of observations) expect(observation).toEqual({
            owner: 'recovery-new-owner', profile: null, resets: 1,
        });
        expect(iapService.identify).not.toHaveBeenCalled();
        expect(analyticsService.trackLogin).not.toHaveBeenCalled();
        expect(mockFrom).not.toHaveBeenCalled();
    });

    it('preserves the current profile and caches for recovery of the same account', async () => {
        authEvents.emitLogin('recovery-same-owner');
        const user = { id: 'recovery-same-owner', email: 'same@example.com', username: 'known' };
        useAuthStore.setState({ session: { user: { id: user.id } } as any, user, isAuthenticated: true });
        const revision = authEvents.getRevision();
        mockExchangeCodeForSession.mockResolvedValueOnce({ data: { session: { user: { id: user.id } } }, error: null });
        await useAuthStore.getState().handleDeepLink('com.aiwardrobe://reset-password?code=synthetic-code&type=recovery');
        expect(useAuthStore.getState().user).toBe(user);
        expect(authEvents.getRevision()).toBe(revision);
        expect(resetInMemoryUserState).not.toHaveBeenCalled();
    });
});

describe('authenticated session publication', () => {
    it.each(['login', 'register', 'restore', 'observer'])('publishes %s only after cache ownership changes', async (entry) => {
        authEvents.emitLogin('publication-old');
        useAuthStore.setState({ session: { user: { id: 'publication-old' } } as any,
            user: { id: 'publication-old', email: '', username: 'old' }, isAuthenticated: true });
        const session = { access_token: 'synthetic-publication-token', user: { id: 'publication-new', email: 'new@example.invalid' } };
        const response = { data: { session }, error: null };
        if (entry === 'login') mockSignInWithPassword.mockResolvedValueOnce(response);
        if (entry === 'register') mockSignUp.mockResolvedValueOnce(response);
        if (entry === 'restore') mockGetSession.mockResolvedValueOnce(response);
        if (entry === 'observer') {
            mockGetSession.mockResolvedValueOnce({ data: { session: null }, error: null });
            await useAuthStore.getState().initializeAuth();
        }
        const eventResets: number[] = [];
        const stopEvent = authEvents.onLogin(owner => {
            if (owner === 'publication-new') eventResets.push(jest.mocked(resetInMemoryUserState).mock.calls.length);
        });
        const observations: Array<{ owner: string | null; resets: number; profileId: string | undefined }> = [];
        const stop = useAuthStore.subscribe(state => {
            if (state.session === session) observations.push({ owner: authEvents.getUserId(),
                resets: jest.mocked(resetInMemoryUserState).mock.calls.length, profileId: state.user?.id });
        });
        try {
            if (entry === 'login') await useAuthStore.getState().login('new@example.invalid', 'synthetic-password');
            if (entry === 'register') await useAuthStore.getState().register('new@example.invalid', 'synthetic-password', 'new');
            if (entry === 'restore') await useAuthStore.getState().initializeAuth();
            if (entry === 'observer') {
                mockOnAuthStateChange.mock.calls.at(-1)![0]!('SIGNED_IN', session);
                await waitFor(() => expect(useAuthStore.getState().session).toBe(session));
            }
        } finally { stop(); stopEvent(); cleanupAuthSubscription(); }
        expect(eventResets).toEqual([1]);
        expect(observations.length).toBeGreaterThan(0);
        for (const state of observations) {
            expect(state.owner).toBe('publication-new');
            expect(state.resets).toBe(1);
            expect(state.profileId).not.toBe('publication-old');
        }
    });
});

describe('coordinated authentication actions', () => {
    const session = (id: string) => ({ access_token: `synthetic-${id}`, user: { id } });
    it('orders two pending sign-ins followed by logout without overlapping SDK mutations', async () => {
        let finish!: (value: unknown) => void;
        mockSignInWithPassword.mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }))
            .mockResolvedValueOnce({ data: { session: session('queued-b') }, error: null });
        const first = useAuthStore.getState().login('a@example.invalid', 'synthetic');
        await waitFor(() => expect(typeof finish).toBe('function'));
        const second = useAuthStore.getState().login('b@example.invalid', 'synthetic');
        const logout = useAuthStore.getState().logout();
        await Promise.resolve();
        expect(mockSignInWithPassword).toHaveBeenCalledTimes(1);
        expect(mockSignOut).not.toHaveBeenCalled();
        finish({ data: { session: session('queued-a') }, error: null });
        await Promise.all([first, second, logout]);
        expect(mockSignInWithPassword).toHaveBeenCalledTimes(2);
        expect(mockSignOut).toHaveBeenCalledTimes(1);
        expect(useAuthStore.getState()).toMatchObject({ session: null, user: null, isAuthenticated: false });
    });

    it('holds new login behind cleanup that outlives its UI deadline', async () => {
        jest.useFakeTimers();
        let finish!: () => void, started!: () => void;
        const active = new Promise<void>(resolve => { started = resolve; });
        jest.mocked(require('../../src/lib/persistence').clearAllPersistedUserData)
            .mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; started(); }));
        mockSignInWithPassword.mockResolvedValueOnce({ data: { session: session('after-cleanup') }, error: null });
        try {
            const logout = useAuthStore.getState().logout();
            const pendingLogout = expect(logout).rejects.toThrow('still pending');
            await active;
            await jest.advanceTimersByTimeAsync(10001);
            await pendingLogout;
            const login = useAuthStore.getState().login('new@example.invalid', 'synthetic');
            await Promise.resolve();
            expect(mockSignInWithPassword).not.toHaveBeenCalled();
            finish(); await login;
            expect(useAuthStore.getState().session?.user.id).toBe('after-cleanup');
        } finally { finish?.(); jest.useRealTimers(); }
    });

    it('never redirects queued deletion to an account established while waiting', async () => {
        authEvents.emitLogin('delete-request-owner');
        useAuthStore.setState({ session: session('delete-request-owner') as any, isAuthenticated: true });
        let finish!: (value: unknown) => void;
        mockSignInWithPassword.mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }));
        const login = useAuthStore.getState().login('new@example.invalid', 'synthetic');
        await waitFor(() => expect(typeof finish).toBe('function'));
        const deletion = useAuthStore.getState().deleteAccount();
        const rejected = expect(deletion).rejects.toThrow('Account changed');
        finish({ data: { session: session('replacement-owner') }, error: null });
        await login; await rejected;
        expect(require('../../lib/supabase').supabase.functions.invoke).not.toHaveBeenCalled();
        expect(useAuthStore.getState().session?.user.id).toBe('replacement-owner');
    });
});

it('does not attach an initialization queued before navigator disposal', async () => {
    let finish!: (value: unknown) => void;
    mockSignInWithPassword.mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }));
    const login = useAuthStore.getState().login('queued@example.invalid', 'synthetic');
    await waitFor(() => expect(typeof finish).toBe('function'));
    const initialize = useAuthStore.getState().initializeAuth();
    cleanupAuthSubscription();
    finish({ data: { session: { access_token: 'synthetic', user: { id: 'queued-initialization-owner' } } }, error: null });
    await Promise.all([login, initialize]);
    expect(mockOnAuthStateChange).not.toHaveBeenCalled();
});

describe('coordinated password recovery update', () => {
    function setup(id = 'password-owner') {
        const session = { access_token: 'synthetic-password-token', user: { id } };
        authEvents.emitLogin(id);
        useAuthStore.setState({ session: session as any, isAuthenticated: true,
            passwordRecovery: { ownerId: id, revision: authEvents.getRevision() } });
        mockGetSession.mockResolvedValueOnce({ data: { session }, error: null });
        return session;
    }
    it.each(['missing', 'different-owner', 'stale-revision'])('rejects a %s recovery grant before contacting the SDK', async kind => {
        setup();
        useAuthStore.setState({ passwordRecovery: kind === 'missing' ? null : {
            ownerId: kind === 'different-owner' ? 'another-owner' : 'password-owner',
            revision: authEvents.getRevision() - (kind === 'stale-revision' ? 1 : 0),
        } });
        await expect(useAuthStore.getState().updatePassword('SyntheticPass9', 'password-owner')).rejects.toThrow('session changed');
        expect(mockGetSession).not.toHaveBeenCalled();
        expect(mockUpdateUser).not.toHaveBeenCalled();
    });
    it('rejects a recovery grant revoked while the SDK session check is pending', async () => {
        setup();
        let finish!: (value: unknown) => void;
        mockGetSession.mockReset().mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }));
        const update = useAuthStore.getState().updatePassword('SyntheticPass9', 'password-owner');
        const rejected = expect(update).rejects.toThrow('session changed');
        await waitFor(() => expect(typeof finish).toBe('function'));
        useAuthStore.setState({ passwordRecovery: null });
        finish({ data: { session: { user: { id: 'password-owner' } } }, error: null });
        await rejected;
        expect(mockUpdateUser).not.toHaveBeenCalled();
    });
    it('confirms the SDK owner before updating and confirms the returned user', async () => {
        setup(); mockUpdateUser.mockResolvedValueOnce({ data: { user: { id: 'password-owner' } }, error: null });
        await useAuthStore.getState().updatePassword('SyntheticPass9', 'password-owner');
        expect(mockUpdateUser).toHaveBeenCalledWith({ password: 'SyntheticPass9' });
    });
    it('rejects mismatched SDK ownership before sending a password mutation', async () => {
        setup(); mockGetSession.mockReset().mockResolvedValueOnce({ data: { session: { user: { id: 'different-owner' } } }, error: null });
        await expect(useAuthStore.getState().updatePassword('SyntheticPass9', 'password-owner')).rejects.toThrow('session changed');
        expect(mockUpdateUser).not.toHaveBeenCalled();
    });
    it('keeps a queued sign-in behind an unfinished password mutation', async () => {
        setup(); let finish!: (value: unknown) => void;
        mockUpdateUser.mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }));
        const update = useAuthStore.getState().updatePassword('SyntheticPass9', 'password-owner');
        await waitFor(() => expect(typeof finish).toBe('function'));
        mockSignInWithPassword.mockResolvedValueOnce({ data: { session: { user: { id: 'new-password-account' } } }, error: null });
        const login = useAuthStore.getState().login('new@example.invalid', 'synthetic');
        await Promise.resolve(); expect(mockSignInWithPassword).not.toHaveBeenCalled();
        finish({ data: { user: { id: 'password-owner' } }, error: null });
        await Promise.all([update, login]);
        expect(useAuthStore.getState().session?.user.id).toBe('new-password-account');
        await expect(useAuthStore.getState().logout('password-owner')).rejects.toThrow('Account changed');
        expect(mockSignOut).not.toHaveBeenCalled();
    });
    it('preserves the SDK failure for localized screen feedback', async () => {
        setup(); const error = { message: 'same_password', code: 'same_password' };
        mockUpdateUser.mockResolvedValueOnce({ data: { user: null }, error });
        await expect(useAuthStore.getState().updatePassword('SyntheticPass9', 'password-owner')).rejects.toBe(error);
    });
    it('does not confirm success for a different response owner', async () => {
        setup(); mockUpdateUser.mockResolvedValueOnce({ data: { user: { id: 'different-owner' } }, error: null });
        await expect(useAuthStore.getState().updatePassword('SyntheticPass9', 'password-owner')).rejects.toThrow('could not be confirmed');
    });
});

it('rejects a password update queued before logout and re-login to the same owner finishes', async () => {
    const session = { access_token: 'synthetic-original', user: { id: 'reused-owner' } };
    authEvents.emitLogin('reused-owner');
    useAuthStore.setState({ session: session as any, isAuthenticated: true });
    let finish!: () => void;
    mockSignOut.mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; }));
    const logout = useAuthStore.getState().logout();
    await waitFor(() => expect(typeof finish).toBe('function'));
    mockSignInWithPassword.mockResolvedValueOnce({ data: { session: { ...session, access_token: 'synthetic-new' } }, error: null });
    const login = useAuthStore.getState().login('same@example.invalid', 'synthetic');
    const update = useAuthStore.getState().updatePassword('SyntheticPass9', 'reused-owner');
    const rejected = expect(update).rejects.toThrow('Account changed');
    finish(); await Promise.all([logout, login]); await rejected;
    expect(mockUpdateUser).not.toHaveBeenCalled();
    expect(useAuthStore.getState().session?.access_token).toBe('synthetic-new');
});

describe('account publication waits for reset persistence', () => {
    it.each(['login', 'recovery', 'restore', 'callback'])('keeps %s anonymous until reset writes finish', async entry => {
        if (entry === 'callback') {
            mockGetSession.mockResolvedValueOnce({ data: { session: null }, error: null });
            await useAuthStore.getState().initializeAuth();
        }
        authEvents.emitLogin('before-reset-owner');
        useAuthStore.setState({ session: { user: { id: 'before-reset-owner' } } as any,
            user: { id: 'before-reset-owner', email: '', username: 'private-old' }, isAuthenticated: true });
        let finish!: () => void, started!: () => void;
        const active = new Promise<void>(resolve => { started = resolve; });
        jest.mocked(resetInMemoryUserState).mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; started(); }));
        const session = { access_token: 'synthetic-pending-reset', user: { id: 'after-reset-owner' } };
        const response = { data: { session }, error: null };
        let action = Promise.resolve();
        if (entry === 'login') { mockSignInWithPassword.mockResolvedValueOnce(response); action = useAuthStore.getState().login('new@example.invalid', 'synthetic'); }
        if (entry === 'recovery') { mockExchangeCodeForSession.mockResolvedValueOnce(response); action = useAuthStore.getState().handleDeepLink('com.aiwardrobe://reset-password?code=synthetic&type=recovery'); }
        if (entry === 'restore') { mockGetSession.mockResolvedValueOnce(response); action = useAuthStore.getState().initializeAuth(); }
        if (entry === 'callback') mockOnAuthStateChange.mock.calls.at(-1)![0]!('SIGNED_IN', session);
        await active;
        try {
            expect(useAuthStore.getState()).toMatchObject({ session: null, user: null, isAuthenticated: false });
            expect(authEvents.getUserId()).toBeNull();
        } finally { finish(); await action; }
        await waitFor(() => expect(useAuthStore.getState().session).toBe(session));
        expect(authEvents.getUserId()).toBe('after-reset-owner');
        cleanupAuthSubscription();
    });

    it('does not revive a pending SDK account switch after an SDK sign-out', async () => {
        mockGetSession.mockResolvedValueOnce({ data: { session: null }, error: null });
        await useAuthStore.getState().initializeAuth();
        const callback = mockOnAuthStateChange.mock.calls.at(-1)![0]!;
        let finish!: () => void;
        jest.mocked(resetInMemoryUserState).mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; }));
        callback('SIGNED_IN', { access_token: 'synthetic-stale', user: { id: 'stale-switch' } });
        callback('SIGNED_OUT', null);
        finish();
        await new Promise<void>(resolve => setImmediate(resolve));
        expect(useAuthStore.getState()).toMatchObject({ session: null, isAuthenticated: false });
        expect(authEvents.getUserId()).toBeNull();
        cleanupAuthSubscription();
    });
});

it('shares one pending reset between the SDK callback and the sign-in completion', async () => {
    mockGetSession.mockResolvedValueOnce({ data: { session: null }, error: null });
    await useAuthStore.getState().initializeAuth();
    const callback = mockOnAuthStateChange.mock.calls.at(-1)![0]!;
    let finish!: () => void, started!: () => void;
    const active = new Promise<void>(resolve => { started = resolve; });
    jest.mocked(resetInMemoryUserState).mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; started(); }));
    const session = { access_token: 'synthetic-shared-publication', user: { id: 'shared-publication-owner' } };
    mockSignInWithPassword.mockImplementationOnce(async () => { callback('SIGNED_IN', session); return { data: { session }, error: null }; });
    const login = useAuthStore.getState().login('new@example.invalid', 'synthetic');
    await active;
    expect(useAuthStore.getState().isAuthenticated).toBe(false);
    finish(); await login;
    expect(resetInMemoryUserState).toHaveBeenCalledTimes(1);
    expect(useAuthStore.getState().session).toBe(session);
    cleanupAuthSubscription();
});

it('does not expose a new account when its reset barrier rejects', async () => {
    jest.mocked(resetInMemoryUserState).mockRejectedValueOnce(new Error('synthetic reset failure'));
    mockSignInWithPassword.mockResolvedValueOnce({ data: { session: { user: { id: 'blocked-publication' } } }, error: null });
    await expect(useAuthStore.getState().login('new@example.invalid', 'synthetic')).rejects.toThrow('synthetic reset failure');
    expect(useAuthStore.getState()).toMatchObject({ session: null, user: null, isAuthenticated: false, loading: false });
    expect(authEvents.getUserId()).toBeNull();
});

it('clears the displayed session while reporting failed logout cleanup', async () => {
    authEvents.emitLogin('failed-cleanup-owner');
    useAuthStore.setState({ session: { user: { id: 'failed-cleanup-owner' } } as any, isAuthenticated: true });
    require('../../src/lib/persistence').clearAllPersistedUserData.mockRejectedValueOnce(new Error('Local account cleanup failed'));
    await expect(useAuthStore.getState().logout()).rejects.toThrow('Local account cleanup failed');
    expect(useAuthStore.getState()).toMatchObject({ session: null, user: null, isAuthenticated: false, loading: false });
});

it('does not retain the deleted account session after confirmed server deletion and local cleanup failure', async () => {
    const session = { access_token: 'synthetic-deleted', user: { id: 'deleted-cleanup-owner' } };
    authEvents.emitLogin(session.user.id);
    useAuthStore.setState({ session: session as any, isAuthenticated: true });
    mockGetSession.mockResolvedValueOnce({ data: { session }, error: null });
    require('../../src/lib/persistence').clearAllPersistedUserData.mockRejectedValueOnce(new Error('Local account cleanup failed'));
    await expect(useAuthStore.getState().deleteAccount()).rejects.toThrow('Local account cleanup failed');
    expect(useAuthStore.getState()).toMatchObject({ session: null, user: null, isAuthenticated: false, loading: false });
});


it('does not publish an account when checking the incomplete-cleanup marker fails', async () => {
    require('../../src/lib/persistence').ensurePreviousAccountCleanup.mockRejectedValueOnce(new Error('synthetic marker read failure'));
    mockSignInWithPassword.mockResolvedValueOnce({ data: { session: { user: { id: 'marker-blocked-owner' } } }, error: null });
    await expect(useAuthStore.getState().login('new@example.invalid', 'synthetic')).rejects.toThrow('synthetic marker read failure');
    expect(useAuthStore.getState()).toMatchObject({ session: null, isAuthenticated: false });
    expect(authEvents.getUserId()).toBeNull();
});

it('can retry sign-in after a failed SDK sign-out reset has settled', async () => {
    mockGetSession.mockResolvedValueOnce({ data: { session: null }, error: null });
    await useAuthStore.getState().initializeAuth();
    const callback = mockOnAuthStateChange.mock.calls.at(-1)![0]!;
    jest.mocked(resetInMemoryUserState).mockRejectedValueOnce(new Error('synthetic callback reset failure'));
    callback('SIGNED_OUT', null);
    await new Promise<void>(resolve => setImmediate(resolve));
    mockSignInWithPassword.mockResolvedValueOnce({ data: { session: { user: { id: 'recovered-reset-owner' } } }, error: null });
    await useAuthStore.getState().login('new@example.invalid', 'synthetic');
    expect(useAuthStore.getState().session?.user.id).toBe('recovered-reset-owner');
    cleanupAuthSubscription();
});

describe('durable local sign-out intent', () => {
    it('writes intent before invoking SDK sign-out and rejects an unremoved SDK session', async () => {
        const session = { user: { id: 'residual-sdk-owner' }, access_token: 'synthetic-residual' };
        authEvents.emitLogin(session.user.id);
        useAuthStore.setState({ session: session as any, isAuthenticated: true });
        let markerAtSdkCall: string | null = null;
        mockSignOut.mockImplementationOnce(async () => { markerAtSdkCall = await AsyncStorage.getItem(SIGN_OUT_INTENT_KEY); return {}; });
        mockGetSession.mockResolvedValueOnce({ data: { session }, error: null });
        await expect(useAuthStore.getState().logout()).rejects.toThrow('session removal could not be confirmed');
        expect(markerAtSdkCall).toBe('1');
        expect(await AsyncStorage.getItem(SIGN_OUT_INTENT_KEY)).toBe('1');
        expect(useAuthStore.getState().isAuthenticated).toBe(false);
    });

    it('finishes a saved sign-out before a new sign-in reaches the SDK', async () => {
        await beginSignOutIntent();
        const order: string[] = [];
        mockSignOut.mockImplementationOnce(async () => { order.push('old sign out'); return {}; });
        mockSignInWithPassword.mockImplementationOnce(async () => { order.push('new sign in'); return { data: { session: { user: { id: 'intent-new-owner' } } }, error: null }; });
        await useAuthStore.getState().login('new@example.invalid', 'synthetic');
        expect(order).toEqual(['old sign out', 'new sign in']);
        expect(await AsyncStorage.getItem(SIGN_OUT_INTENT_KEY)).toBeNull();
        expect(useAuthStore.getState().session?.user.id).toBe('intent-new-owner');
    });

    it('keeps startup signed out and initialized when recovery fails', async () => {
        await beginSignOutIntent();
        mockSignOut.mockRejectedValueOnce(new Error('synthetic SDK deletion failure'));
        await expect(useAuthStore.getState().initializeAuth()).rejects.toThrow('Pending sign out');
        expect(useAuthStore.getState()).toMatchObject({ session: null, isAuthenticated: false, isInitialized: true, loading: false });
        expect(authEvents.getUserId()).toBeNull();
        expect(await AsyncStorage.getItem(SIGN_OUT_INTENT_KEY)).toBe('1');
        expect(mockOnAuthStateChange).not.toHaveBeenCalled();
    });

    it('does not start new sign-in when pending local cleanup fails', async () => {
        await beginSignOutIntent();
        require('../../src/lib/persistence').clearAllPersistedUserData.mockRejectedValueOnce(new Error('synthetic cleanup failure'));
        await expect(useAuthStore.getState().login('new@example.invalid', 'synthetic')).rejects.toThrow('Pending sign out');
        expect(mockSignInWithPassword).not.toHaveBeenCalled();
        expect(await AsyncStorage.getItem(SIGN_OUT_INTENT_KEY)).toBe('1');
    });
});

it('retains sign-out intent after confirmed deletion if SDK session removal fails', async () => {
    const session = { access_token: 'synthetic-deletion-residual', user: { id: 'deleted-residual-owner' } };
    authEvents.emitLogin(session.user.id);
    useAuthStore.setState({ session: session as any, isAuthenticated: true });
    mockGetSession.mockResolvedValueOnce({ data: { session }, error: null }).mockResolvedValueOnce({ data: { session }, error: null });
    mockSignOut.mockRejectedValueOnce(new Error('synthetic SDK removal failure'));
    await expect(useAuthStore.getState().deleteAccount()).rejects.toThrow('session removal could not be confirmed');
    expect(useAuthStore.getState()).toMatchObject({ session: null, isAuthenticated: false });
    expect(await AsyncStorage.getItem(SIGN_OUT_INTENT_KEY)).toBe('1');
});

describe('Google OAuth callback handling', () => {
    beforeEach(() => {
        mockSignInWithOAuth.mockResolvedValue({ data: { url: 'https://accounts.google.com/synthetic' }, error: null });
    });
    it.each([
        ['query', 'com.aiwardrobe://auth/callback?error=access_denied&error_description=Consent%20was%20100%25%20denied'],
        ['fragment', 'com.aiwardrobe://auth/callback#error=access_denied&error_description=Consent%20was%20100%25%20denied'],
    ])('preserves the Google denial message from the %s', async (_kind, url) => {
        jest.mocked(WebBrowser.openAuthSessionAsync).mockResolvedValueOnce({ type: 'success', url });
        await expect(useAuthStore.getState().signInWithGoogle()).rejects.toThrow('Consent was 100% denied');
        expect(useAuthStore.getState().error).toBe('Consent was 100% denied');
        expect(mockExchangeCodeForSession).not.toHaveBeenCalled();
        expect(mockSetSession).not.toHaveBeenCalled();
    });
    it.each(['pkce', 'implicit'])('publishes the Google account from a successful %s callback', async flow => {
        const session = { access_token: 'synthetic-google', user: { id: 'google-account', email: 'google@example.invalid' } };
        const response = { data: { session }, error: null };
        mockExchangeCodeForSession.mockResolvedValueOnce(response);
        mockSetSession.mockResolvedValueOnce(response);
        jest.mocked(WebBrowser.openAuthSessionAsync).mockResolvedValueOnce({ type: 'success', url: flow === 'pkce'
            ? 'com.aiwardrobe://auth/callback?code=synthetic-code'
            : 'com.aiwardrobe://auth/callback#access_token=synthetic-access&refresh_token=synthetic-refresh' });
        await useAuthStore.getState().signInWithGoogle();
        expect(mockSignInWithOAuth).toHaveBeenCalledWith({ provider: 'google', options: {
            redirectTo: 'com.aiwardrobe://auth/callback', skipBrowserRedirect: true,
        } });
        expect(useAuthStore.getState().session).toEqual(session);
        expect(useAuthStore.getState().error).toBeNull();
        expect(analyticsService.trackLogin).toHaveBeenCalledWith('google');
    });
    it.each([WebBrowser.WebBrowserResultType.CANCEL, WebBrowser.WebBrowserResultType.DISMISS])('returns to the form after browser %s without a session mutation', async type => {
        jest.mocked(WebBrowser.openAuthSessionAsync).mockResolvedValueOnce({ type });
        await useAuthStore.getState().signInWithGoogle();
        expect(useAuthStore.getState().loading).toBe(false);
        expect(useAuthStore.getState().error).toBeNull();
        expect(mockSetSession).not.toHaveBeenCalled();
        expect(mockExchangeCodeForSession).not.toHaveBeenCalled();
    });
});

describe('Google callback delivered through browser result and URL listener', () => {
    it.each(['pkce', 'implicit'])('processes one %s credential once across both delivery paths', async flow => {
        const url = flow === 'pkce' ? 'com.aiwardrobe://auth/callback?code=dual-synthetic'
            : 'com.aiwardrobe://auth/callback#access_token=dual-access&refresh_token=dual-refresh';
        const session = { access_token: 'dual-published', user: { id: `dual-${flow}`, email: 'dual@example.invalid' } };
        const exchange = flow === 'pkce' ? mockExchangeCodeForSession : mockSetSession;
        exchange.mockResolvedValueOnce({ data: { session }, error: null })
            .mockResolvedValueOnce({ data: { session: null }, error: { message: 'one-use credential consumed' } });
        mockSignInWithOAuth.mockResolvedValueOnce({ data: { url: 'https://accounts.google.com/synthetic' }, error: null });
        let delivered!: Promise<void>;
        jest.mocked(WebBrowser.openAuthSessionAsync).mockImplementationOnce(async () => {
            delivered = useAuthStore.getState().handleDeepLink(url);
            return { type: 'success', url };
        });
        await useAuthStore.getState().signInWithGoogle();
        await delivered;
        expect(exchange).toHaveBeenCalledTimes(1);
        expect(useAuthStore.getState().session).toEqual(session);
        expect(useAuthStore.getState().error).toBeNull();
        expect(analyticsService.trackLogin).toHaveBeenCalledTimes(1);
    });
});

it.each(['owner', 'token', 'revision', 'credential'])('does not suppress a Google callback after its %s changes', async changed => {
    const url = 'com.aiwardrobe://auth/callback?code=identity-scoped-code';
    const session = { access_token: 'identity-scoped-token', user: { id: 'identity-scoped-owner', email: 'scoped@example.invalid' } };
    mockExchangeCodeForSession.mockResolvedValue({ data: { session }, error: null });
    mockSignInWithOAuth.mockResolvedValueOnce({ data: { url: 'https://accounts.google.com/synthetic' }, error: null });
    jest.mocked(WebBrowser.openAuthSessionAsync).mockResolvedValueOnce({ type: 'success', url });
    await useAuthStore.getState().signInWithGoogle();
    if (changed === 'owner') useAuthStore.setState({ session: { ...session, user: { ...session.user, id: 'another-owner' } } as any });
    if (changed === 'token') useAuthStore.setState({ session: { ...session, access_token: 'replaced-token' } as any });
    if (changed === 'revision') { authEvents.emitLogout(); authEvents.emitLogin(session.user.id); }
    await useAuthStore.getState().handleDeepLink(changed === 'credential'
        ? 'com.aiwardrobe://auth/callback?code=different-code' : url);
    expect(mockExchangeCodeForSession).toHaveBeenCalledTimes(2);
});

describe('cold-start private cache ownership', () => {
    it.each(['cold-owner', 'other-owner', null])('preserves completed onboarding only for a stamped matching owner (%s)', async storedOwner => {
        const matching = storedOwner === 'cold-owner';
        authEvents.emitLogout();
        const { useStylePreferenceStore } = require('../../store/stylePreferenceStore');
        await useStylePreferenceStore.persist.rehydrate();
        useStylePreferenceStore.getState().completeOnboarding();
        if (storedOwner) await AsyncStorage.setItem('private_cache_owner_v1', storedOwner);
        else await AsyncStorage.removeItem('private_cache_owner_v1');
        jest.mocked(resetInMemoryUserState).mockImplementation(async () => {
            useStylePreferenceStore.getState().clearAll();
        });
        mockGetSession.mockResolvedValueOnce({ data: { session: {
            access_token: 'synthetic-cold-token', user: { id: 'cold-owner', email: 'cold@example.invalid' },
        } }, error: null });
        await useAuthStore.getState().initializeAuth();
        expect(useAuthStore.getState().isAuthenticated).toBe(true);
        expect(useStylePreferenceStore.getState().hasCompletedOnboarding).toBe(matching);
        expect(resetInMemoryUserState).toHaveBeenCalledTimes(matching ? 0 : 1);
        expect(await AsyncStorage.getItem('private_cache_owner_v1')).toBe('cold-owner');
        cleanupAuthSubscription();
    });

    it('fails closed if the owner stamp cannot be written', async () => {
        authEvents.emitLogout();
        const original = AsyncStorage.setItem;
        jest.spyOn(AsyncStorage, 'setItem').mockImplementation((key, value) =>
            key === 'private_cache_owner_v1' ? Promise.reject(new Error('disk unavailable')) : original(key, value));
        mockGetSession.mockResolvedValueOnce({ data: { session: {
            access_token: 'synthetic-cold-token', user: { id: 'cold-failed' },
        } }, error: null });
        try {
            await useAuthStore.getState().initializeAuth();
            expect(useAuthStore.getState().isAuthenticated).toBe(false);
            expect(useAuthStore.getState().session).toBeNull();
        } finally { jest.mocked(AsyncStorage.setItem).mockRestore(); cleanupAuthSubscription(); }
    });
});

describe('unique signup handles', () => {
    it('does not create an account for an occupied handle', async () => {
        mockNameAvailability.mockResolvedValueOnce({data:false,error:null});
        await expect(useAuthStore.getState().register('new@example.invalid','password','existing')).rejects.toThrow('username_exists');
        expect(mockSignUp).not.toHaveBeenCalled();
    });
    it('passes the canonical handle separately from the initial display name', async () => {
        mockSignUp.mockResolvedValueOnce({data:{session:null},error:null});
        await expect(useAuthStore.getState().register('new@example.invalid','password',' Mixed_Case ')).rejects.toThrow('EMAIL_CONFIRMATION_REQUIRED');
        expect(mockSignUp).toHaveBeenCalledWith(expect.objectContaining({options:{data:expect.objectContaining({handle:'mixed_case',display_name:'Mixed_Case'})}}));
    });
    it('reports a handle claimed between availability and account creation', async () => {
        mockNameAvailability.mockResolvedValueOnce({data:true,error:null}).mockResolvedValueOnce({data:false,error:null});
        mockSignUp.mockResolvedValueOnce({data:null,error:{message:'Database error saving new user'}});
        await expect(useAuthStore.getState().register('new@example.invalid','password','race_handle')).rejects.toThrow('username_exists');
    });
});


it('Apple first-sign-in persists a display name without changing the unique handle', async () => {
    const { Platform } = require('react-native');
    const oldOS = Platform.OS;
    Object.defineProperty(Platform, 'OS', { value: 'ios', configurable: true });
    const update = jest.fn((_updates: Record<string, unknown>) => ({ eq: () => Promise.resolve({ error: null }) }));
    mockFrom.mockReturnValue({ update, select: () => ({ eq: () => ({ abortSignal() { return this; }, maybeSingle: async () => ({ data: { id: 'apple-name-owner', username: 'Shared Full Name', display_name: 'Shared Full Name', handle: 'server_handle' }, error: null }) }) }) });
    mockAppleNameCredential.mockResolvedValue({ identityToken: 'synthetic-apple-token', fullName: { givenName: 'Shared', familyName: 'Full Name' } });
    mockSignInWithIdToken.mockResolvedValue({ data: { session: { access_token: 'synthetic-access', refresh_token: 'synthetic-refresh', user: { id: 'apple-name-owner', email: 'apple@example.invalid', user_metadata: {} } } }, error: null });
    try {
        await useAuthStore.getState().signInWithApple();
        expect(update).toHaveBeenCalledWith({ display_name: 'Shared Full Name' });
        expect(update.mock.calls.every(([value]) => !('handle' in value) && !('username' in value))).toBe(true);
    } finally { Object.defineProperty(Platform, 'OS', { value: oldOS, configurable: true }); }
});
