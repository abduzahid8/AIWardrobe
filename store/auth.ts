import { create } from 'zustand';
import { supabase } from '../lib/supabase';
import { Session } from '@supabase/supabase-js';
import { analyticsService } from '../src/services/analyticsService';
import { crashReporting } from '../src/services/crashReporting';
import { iapService } from '../src/services/iapService';

// ============================================
// AUTH TYPES
// ============================================

export interface AuthUser {
    id: string; // Changed from _id to id to match Supabase
    email: string;
    username: string;
    gender?: 'male' | 'female' | 'other' | 'prefer_not_to_say';
    profile_image?: string; // Changed from profileImage to snake_case to match DB, or we map it

    // Subscription
    subscription_tier?: 'free' | 'premium' | 'vip';
    subscription_expires_at?: string;

    // Legacy/Metadata
    created_at?: string;
    updated_at?: string;
}

export interface AuthState {
    user: AuthUser | null;
    session: Session | null;
    loading: boolean;
    error: string | null;
    isAuthenticated: boolean;
    isTrialMode: boolean;
}

export interface AuthActions {
    initializeAuth: () => Promise<void>;
    register: (
        email: string,
        password: string,
        username: string,
        gender?: string,
        profileImage?: string
    ) => Promise<void>;
    login: (email: string, password: string) => Promise<void>;
    logout: () => Promise<void>;
    deleteAccount: () => Promise<void>;
    startTrial: () => void;
    endTrial: () => void;
    fetchUser: () => Promise<void>;
    clearError: () => void;
}

export type AuthStore = AuthState & AuthActions;

// ============================================
// AUTH STORE
// ============================================

const useAuthStore = create<AuthStore>((set, get) => ({
    // Initial state
    user: null,
    session: null,
    loading: false,
    error: null,
    isAuthenticated: false,
    isTrialMode: false,

    initializeAuth: async (): Promise<void> => {
        set({ loading: true });
        try {
            // Check for existing session
            const { data: { session }, error } = await supabase.auth.getSession();

            if (error) throw error;

            if (session) {
                set({ session, isAuthenticated: true, isTrialMode: false });
                await get().fetchUser();
            }

            // Listen for auth changes (store subscription for cleanup)
            const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
                if (session) {
                    set({ session, isAuthenticated: true, isTrialMode: false });
                } else {
                    set({ session: null, user: null, isAuthenticated: false });
                }
            });

            // Store unsubscribe for cleanup (e.g. on logout)
            (useAuthStore as any)._authSubscription = subscription;

        } catch (err) {
            console.error('Auth initialization failed', err);
        } finally {
            set({ loading: false });
        }
    },

    register: async (
        email: string,
        password: string,
        username: string,
        gender?: string,
        profileImage?: string
    ): Promise<void> => {
        set({ loading: true, error: null });
        try {
            const { data, error } = await supabase.auth.signUp({
                email,
                password,
                options: {
                    data: {
                        username,
                        gender,
                        profile_image: profileImage,
                    },
                },
            });

            if (error) throw error;

            if (data.session) {
                set({ session: data.session, isAuthenticated: true, isTrialMode: false, loading: false });
                await get().fetchUser();
                analyticsService.trackSignup('email');
                analyticsService.setUserId(data.session.user.id);
                crashReporting.setUser(data.session.user.id);
                iapService.identify(data.session.user.id);
            } else {
                // If email confirmation is enabled, session might be null
                set({ loading: false, error: "Please checks your email for confirmation link." });
            }

        } catch (error: any) {

            set({
                error: error.message || 'Registration failed',
                loading: false,
            });
            throw error;
        }
    },

    login: async (email: string, password: string): Promise<void> => {
        set({ loading: true, error: null });
        try {
            const { data, error } = await supabase.auth.signInWithPassword({
                email,
                password,
            });

            if (error) throw error;

            if (data.session) {
                set({ session: data.session, isAuthenticated: true, isTrialMode: false, loading: false });
                await get().fetchUser();
                analyticsService.trackLogin('email');
                analyticsService.setUserId(data.session.user.id);
                crashReporting.setUser(data.session.user.id);
                iapService.identify(data.session.user.id);
            }

        } catch (err: any) {

            set({
                error: err.message || 'Login failed',
                loading: false,
            });
            throw err;
        }
    },

    logout: async (): Promise<void> => {
        // Clean up auth listener to prevent memory leak
        const sub = (useAuthStore as any)._authSubscription;
        if (sub) {
            sub.unsubscribe();
            (useAuthStore as any)._authSubscription = null;
        }
        await supabase.auth.signOut();
        analyticsService.trackEvent('logout');
        analyticsService.clearUserId();
        crashReporting.clearUser();
        set({ user: null, session: null, isAuthenticated: false, isTrialMode: false });
    },

    deleteAccount: async (): Promise<void> => {
        set({ loading: true, error: null });
        try {
            const { data: { session } } = await supabase.auth.getSession();
            if (!session?.access_token) {
                throw new Error('Not authenticated');
            }

            // Call the account deletion API
            const Config = (await import('../src/config/env')).default;
            const response = await fetch(`${Config.api.url}/api/account`, {
                method: 'DELETE',
                headers: {
                    'Authorization': `Bearer ${session.access_token}`,
                    'Content-Type': 'application/json',
                },
            });

            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.error || 'Account deletion failed');
            }

            analyticsService.trackEvent('account_deleted');

            // Clean up local state
            const sub = (useAuthStore as any)._authSubscription;
            if (sub) {
                sub.unsubscribe();
                (useAuthStore as any)._authSubscription = null;
            }
            await supabase.auth.signOut();
            analyticsService.clearUserId();
            crashReporting.clearUser();
            set({ user: null, session: null, isAuthenticated: false, isTrialMode: false, loading: false });
        } catch (err: any) {
            set({
                error: err.message || 'Account deletion failed',
                loading: false,
            });
            throw err;
        }
    },

    startTrial: (): void => {
        set({ isTrialMode: true, isAuthenticated: false });
    },

    endTrial: (): void => {
        set({ isTrialMode: false });
    },

    fetchUser: async (): Promise<void> => {
        try {
            const { data: sessionData } = await supabase.auth.getSession();
            const user = sessionData.session?.user;

            if (!user) {
                set({ user: null });
                return;
            }

            // Fetch profile data from 'profiles' table
            const { data: profile, error } = await supabase
                .from('profiles')
                .select('*')
                .eq('id', user.id)
                .maybeSingle();

            if (error) {
                console.error('Error fetching profile:', error);
            }

            if (profile) {
                set({ user: profile as AuthUser, error: null });
            } else {
                // Fallback to auth metadata if no profile row exists yet
                const fallbackUser: AuthUser = {
                    id: user.id,
                    email: user.email || '',
                    username: user.user_metadata?.username || '',
                    gender: user.user_metadata?.gender,
                    profile_image: user.user_metadata?.profile_image,
                };
                set({ user: fallbackUser, error: null });
            }
        } catch (err: any) {
            console.error('Failed to fetch user:', err);
            set({
                user: null,
                error: err.message || 'Failed to fetch user',
            });
        }
    },

    clearError: (): void => {
        set({ error: null });
    },
}));

export default useAuthStore;

