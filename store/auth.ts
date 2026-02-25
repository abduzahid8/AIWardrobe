import { create } from 'zustand';
import { supabase } from '../lib/supabase';
import { Session } from '@supabase/supabase-js';

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

            // Listen for auth changes
            supabase.auth.onAuthStateChange((_event, session) => {
                if (session) {
                    set({ session, isAuthenticated: true, isTrialMode: false });
                    // We could fetch user here, but verify logic to avoid loops
                } else {
                    set({ session: null, user: null, isAuthenticated: false });
                }
            });

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
            } else {
                // If email confirmation is enabled, session might be null
                set({ loading: false, error: "Please checks your email for confirmation link." });
            }

        } catch (error: any) {
            console.log('Registration error details:', error);
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
            }

        } catch (err: any) {
            console.log('Login error details:', err);
            set({
                error: err.message || 'Login failed',
                loading: false,
            });
            throw err;
        }
    },

    logout: async (): Promise<void> => {
        await supabase.auth.signOut();
        set({ user: null, session: null, isAuthenticated: false, isTrialMode: false });
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
                .from('user_profiles')
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

