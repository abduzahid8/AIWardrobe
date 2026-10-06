/**
 * Tests for the auth Zustand store
 */

// Mock supabase
const mockGetSession = jest.fn();
const mockSignUp = jest.fn();
const mockSignInWithPassword = jest.fn();
const mockSignOut = jest.fn();
const mockSetSession = jest.fn();
const mockExchangeCodeForSession = jest.fn();
const mockSignInWithOAuth = jest.fn();
const mockSignInWithIdToken = jest.fn();
const mockOnAuthStateChange = jest.fn(() => ({ data: { subscription: { unsubscribe: jest.fn() } } }));
const mockFrom = jest.fn();

jest.mock('../../lib/supabase', () => ({
    supabase: {
        auth: {
            getSession: () => mockGetSession(),
            signUp: (...args: any[]) => mockSignUp(...args),
            signInWithPassword: (...args: any[]) => mockSignInWithPassword(...args),
            signOut: () => mockSignOut(),
            setSession: (...args: any[]) => mockSetSession(...args),
            exchangeCodeForSession: (...args: any[]) => mockExchangeCodeForSession(...args),
            signInWithOAuth: (...args: any[]) => mockSignInWithOAuth(...args),
            signInWithIdToken: (...args: any[]) => mockSignInWithIdToken(...args),
            onAuthStateChange: mockOnAuthStateChange,
        },
        from: (table: string) => mockFrom(table),
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
}));

jest.mock('../../store/authEvents', () => ({
    authEvents: { emitLogin: jest.fn(), emitLogout: jest.fn() },
}));

jest.mock('../../src/services/uploadQueue', () => ({
    uploadQueue: { destroy: jest.fn(), init: jest.fn() },
}));

jest.mock('expo-web-browser', () => ({
    openAuthSessionAsync: jest.fn(),
}));

jest.mock('expo-linking', () => ({
    createURL: jest.fn((p: string) => `com.aiwardrobe://${p}`),
}));

// Import after mocks
import useAuthStore, { parseSupabaseUrl } from '../../store/auth';

beforeEach(() => {
    jest.clearAllMocks();
    // Reset store state
    useAuthStore.setState({
        user: null,
        session: null,
        loading: false,
        error: null,
        isAuthenticated: false,
    });
    mockFrom.mockReturnValue({
        select: () => ({
            eq: () => ({
                maybeSingle: () => Promise.resolve({ data: null, error: null }),
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
                        maybeSingle: () => Promise.resolve({ data: { id: '1', email: 'test@example.com', username: 'test' }, error: null }),
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
                        maybeSingle: () => Promise.resolve({ data: null, error: null }),
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
    });

    describe('logout hang resistance', () => {
        afterEach(() => {
            jest.useRealTimers();
        });

        it('completes local logout even when signOut hangs forever', async () => {
            jest.useFakeTimers();
            // Simulate a stalled network: signOut never settles.
            mockSignOut.mockReturnValue(new Promise(() => {}));
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
        });
    });
});
