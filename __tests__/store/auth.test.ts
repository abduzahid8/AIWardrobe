/**
 * Tests for the auth Zustand store
 */

import { create } from 'zustand';

// Mock supabase
const mockGetSession = jest.fn();
const mockSignUp = jest.fn();
const mockSignInWithPassword = jest.fn();
const mockSignOut = jest.fn();
const mockOnAuthStateChange = jest.fn(() => ({ data: { subscription: { unsubscribe: jest.fn() } } }));
const mockFrom = jest.fn();

jest.mock('../../lib/supabase', () => ({
    supabase: {
        auth: {
            getSession: () => mockGetSession(),
            signUp: (args: any) => mockSignUp(args),
            signInWithPassword: (args: any) => mockSignInWithPassword(args),
            signOut: () => mockSignOut(),
            onAuthStateChange: mockOnAuthStateChange,
        },
        from: (table: string) => mockFrom(table),
    },
}));

jest.mock('../../src/services/iapService', () => ({
    iapService: {
        identify: jest.fn().mockResolvedValue(undefined),
        logout: jest.fn().mockResolvedValue(undefined),
    },
}));

// Import after mocks
import useAuthStore from '../../store/auth';

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
    });
});
