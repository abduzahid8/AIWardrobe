import { create } from 'zustand';
import { supabase } from '../lib/supabase';
import { Session } from '@supabase/supabase-js';
import { analyticsService } from '../src/services/analyticsService';
import { crashReporting } from '../src/services/crashReporting';
import { iapService } from '../src/services/iapService';

// Lazy import to avoid circular dependency
const getWardrobeStore = () => require('./wardrobeStore').default;

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
        console.log('[AuthStore] initializeAuth called');
        set({ loading: true });
        try {
            // Check for existing session
            const { data: { session }, error } = await supabase.auth.getSession();

            if (error) throw error;
            console.log('[AuthStore] Session check result:', session ? 'Session found' : 'No session');

            if (session) {
                set({ session, isAuthenticated: true, isTrialMode: false });
                await get().fetchUser();
                console.log('[AuthStore] User authenticated successfully');

                // Rehydrate wardrobe from cloud and start realtime sync
                try {
                    const wardrobeStore = getWardrobeStore();
                    await wardrobeStore.getState().rehydrateFromCloud();
                    wardrobeStore.getState().subscribeToRealtime();
                    console.log('[AuthStore] Wardrobe rehydration and realtime sync completed');
                } catch (e) {
                    console.error('[Auth] Wardrobe rehydration failed:', e);
                }
            }

            // Listen for auth changes (store subscription for cleanup)
            const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
                console.log('[AuthStore] Auth state changed:', _event, session ? 'Session exists' : 'No session');
                if (session) {
                    set({ session, isAuthenticated: true, isTrialMode: false });
                } else {
                    set({ session: null, user: null, isAuthenticated: false });
                }
            });

            // Store unsubscribe for cleanup (e.g. on logout)
            (useAuthStore as any)._authSubscription = subscription;

        } catch (err) {
            console.error('[AuthStore] Auth initialization failed:', err);
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
        console.log('[AuthStore] Registration attempt - email:', email, 'username:', username);
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
            console.log('[AuthStore] Registration successful - session:', data.session ? 'Created' : 'Email confirmation required');

            if (data.session) {
                set({ session: data.session, isAuthenticated: true, isTrialMode: false, loading: false });
                await get().fetchUser();
                analyticsService.trackSignup('email');
                analyticsService.setUserId(data.session.user.id);
                crashReporting.setUser(data.session.user.id);
                iapService.identify(data.session.user.id);
                console.log('[AuthStore] User fully registered and tracked');

                // Migrate trial/guest data to the new account
                try {
                    const wardrobeStore = getWardrobeStore();
                    const state = wardrobeStore.getState();

                    // Update any local items that have an empty/missing userId
                    const userId = data.session.user.id;
                    const hasOrphanedItems = state.items.some((i: any) => !i.userId || i.userId === '');
                    if (hasOrphanedItems) {
                        console.log('[AuthStore] Migrating orphaned items to user:', userId);
                        state.items.forEach((item: any) => {
                            if (!item.userId || item.userId === '') {
                                state.updateItem(item.id, { userId });
                            }
                        });
                    }

                    await state.rehydrateFromCloud();
                    state.subscribeToRealtime();
                } catch (e) {
                    console.error('[Auth] Wardrobe migration/rehydration failed:', e);
                }
            } else {
                // If email confirmation is enabled, session might be null
                console.log('[AuthStore] Registration successful but email confirmation required');
                set({ loading: false, error: "Please checks your email for confirmation link." });
            }

        } catch (error: any) {
            console.error('[AuthStore] Registration failed:', error.message);
            set({
                error: error.message || 'Registration failed',
                loading: false,
            });
            throw error;
        }
    },

    login: async (email: string, password: string): Promise<void> => {
        console.log('[AuthStore] Login attempt - email:', email);
        set({ loading: true, error: null });
        try {
            const { data, error } = await supabase.auth.signInWithPassword({
                email,
                password,
            });

            if (error) throw error;
            console.log('[AuthStore] Login successful - session created');

            if (data.session) {
                set({ session: data.session, isAuthenticated: true, isTrialMode: false, loading: false });
                await get().fetchUser();
                analyticsService.trackLogin('email');
                analyticsService.setUserId(data.session.user.id);
                crashReporting.setUser(data.session.user.id);
                iapService.identify(data.session.user.id);
                console.log('[AuthStore] User logged in and tracked');

                // Rehydrate wardrobe from cloud — prevents data loss on re-login
                // Migrate trial/guest data if applicable
                try {
                    const wardrobeStore = getWardrobeStore();
                    const state = wardrobeStore.getState();

                    const userId = data.session.user.id;
                    const hasOrphanedItems = state.items.some((i: any) => !i.userId || i.userId === '');
                    if (hasOrphanedItems) {
                        console.log('[AuthStore] Migrating orphaned items to user:', userId);
                        state.items.forEach((item: any) => {
                            if (!item.userId || item.userId === '') {
                                state.updateItem(item.id, { userId });
                            }
                        });
                    }

                    await state.rehydrateFromCloud();
                    state.subscribeToRealtime();
                } catch (e) {
                    console.error('[Auth] Wardrobe rehydration failed:', e);
                }
            }

        } catch (err: any) {
            console.error('[AuthStore] Login failed:', err.message);
            set({
                error: err.message || 'Login failed',
                loading: false,
            });
            throw err;
        }
    },

    logout: async (): Promise<void> => {
        console.log('[AuthStore] Logout called');
        // Clean up auth listener to prevent memory leak
        const sub = (useAuthStore as any)._authSubscription;
        if (sub) {
            sub.unsubscribe();
            (useAuthStore as any)._authSubscription = null;
            console.log('[AuthStore] Auth subscription cleaned up');
        }

        // Unsubscribe from wardrobe realtime
        try {
            const wardrobeStore = getWardrobeStore();
            wardrobeStore.getState().unsubscribeRealtime();
            console.log('[AuthStore] Wardrobe realtime subscription cleaned up');
        } catch (e) {
            // Store may not be initialized
            console.log('[AuthStore] Wardrobe store not initialized, skipping cleanup');
        }

        await supabase.auth.signOut();
        analyticsService.trackEvent('logout');
        analyticsService.clearUserId();
        crashReporting.clearUser();
        set({ user: null, session: null, isAuthenticated: false, isTrialMode: false });
        console.log('[AuthStore] Logout completed');
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
        console.log('[AuthStore] Trial mode started');
        set({ isTrialMode: true, isAuthenticated: false });
    },

    endTrial: (): void => {
        console.log('[AuthStore] Trial mode ended');
        set({ isTrialMode: false });
    },

    fetchUser: async (): Promise<void> => {
        console.log('[AuthStore] Fetching user profile');
        try {
            const { data: sessionData } = await supabase.auth.getSession();
            const user = sessionData.session?.user;

            if (!user) {
                console.log('[AuthStore] No user session found');
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
                console.error('[AuthStore] Error fetching profile:', error);
            }

            if (profile) {
                console.log('[AuthStore] User profile loaded:', profile.username);
                set({ user: profile as AuthUser, error: null });
            } else {
                // Fallback to auth metadata if no profile row exists yet
                console.log('[AuthStore] No profile found, using auth metadata fallback');
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
            console.error('[AuthStore] Failed to fetch user:', err);
            set({
                user: null,
                error: err.message || 'Failed to fetch user',
            });
        }
    },

    clearError: (): void => {
        console.log('[AuthStore] Clearing error state');
        set({ error: null });
    },
}));

export default useAuthStore;

