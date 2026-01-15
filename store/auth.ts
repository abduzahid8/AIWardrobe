import { create } from 'zustand';
import axios, { AxiosError } from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { API_URL } from '../api/config';

const BASE_URL = API_URL;

// ============================================
// AUTH TYPES
// ============================================

export interface AuthUser {
    _id: string;
    email: string;
    username: string;
    gender?: 'male' | 'female' | 'other' | 'prefer_not_to_say';
    profileImage?: string;
    outfits?: string[];
    followers?: string[];
    following?: string[];
    createdAt?: string;
    updatedAt?: string;
}

export interface AuthState {
    user: AuthUser | null;
    token: string | null;
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
    token: null,
    loading: false,
    error: null,
    isAuthenticated: false,
    isTrialMode: false,

    initializeAuth: async (): Promise<void> => {
        try {
            const token = await AsyncStorage.getItem('userToken');
            if (token) {
                set({ token, isAuthenticated: true, isTrialMode: false });
                await get().fetchUser();
            }
        } catch (err) {
            console.error('Auth initialization failed', err);
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
            const response = await axios.post<{ token: string }>(`${BASE_URL}/register`, {
                email,
                password,
                username,
                gender,
                profileImage,
            });
            console.log('data', response.data);
            const { token } = response.data;
            await AsyncStorage.setItem('userToken', token);
            set({ token, loading: false, isAuthenticated: true, isTrialMode: false });
            await get().fetchUser();
        } catch (error) {
            const axiosError = error as AxiosError<{ error: string }>;
            set({
                error: axiosError.response?.data?.error || 'Registration failed',
                loading: false,
            });
        }
    },

    login: async (email: string, password: string): Promise<void> => {
        set({ loading: true, error: null });
        try {
            const response = await axios.post<{ token: string }>(`${BASE_URL}/login`, {
                email,
                password,
            });
            const { token } = response.data;
            console.log('token', response.data);
            await AsyncStorage.setItem('userToken', token);
            set({ token, loading: false, isAuthenticated: true, isTrialMode: false });
            await get().fetchUser();
        } catch (err) {
            const axiosError = err as AxiosError<{ error: string }>;
            set({
                error: axiosError.response?.data?.error || 'Login failed',
                loading: false,
            });
        }
    },

    logout: async (): Promise<void> => {
        await AsyncStorage.removeItem('userToken');
        set({ user: null, token: null, isAuthenticated: false, isTrialMode: false });
    },

    startTrial: (): void => {
        set({ isTrialMode: true, isAuthenticated: false });
    },

    endTrial: (): void => {
        set({ isTrialMode: false });
    },

    fetchUser: async (): Promise<void> => {
        try {
            const token = await AsyncStorage.getItem('userToken');
            console.log('da', token);
            if (!token) {
                set({ user: null, error: 'No token available' });
                return;
            }
            const response = await axios.get<AuthUser>(`${BASE_URL}/me`, {
                headers: { Authorization: `Bearer ${token}` },
            });
            set({ user: response.data, error: null });
        } catch (err) {
            console.error('Failed to fetch user:', err);
            const axiosError = err as AxiosError<{ error: string }>;
            set({
                user: null,
                error: axiosError.response?.data?.error || 'Failed to fetch user',
            });
        }
    },

    clearError: (): void => {
        set({ error: null });
    },
}));

export default useAuthStore;
