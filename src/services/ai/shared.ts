/**
 * Shared utilities for AI domain services.
 *
 * Provides retry logic, error handling, and auth header generation
 * used across all AI service modules.
 */

import axios, { AxiosError } from 'axios';
import AsyncStorage from '@react-native-async-storage/async-storage';

const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 2000;

/**
 * Retry wrapper with exponential backoff.
 */
export async function withRetry<T>(
    fn: () => Promise<T>,
    retries: number = MAX_RETRIES,
    delay: number = RETRY_DELAY_MS
): Promise<T> {
    try {
        return await fn();
    } catch (error) {
        if (retries > 0) {
            await new Promise(resolve => setTimeout(resolve, delay));
            return withRetry(fn, retries - 1, delay * 1.5);
        }
        throw error;
    }
}

/**
 * Throw a user-friendly error based on Axios error shape.
 */
export function handleAPIError(error: unknown, context: string): never {
    if (axios.isAxiosError(error)) {
        const axiosError = error as AxiosError;

        if (axiosError.response?.status === 500) {
            throw new Error('Server is temporarily unavailable. Please try again in a moment.');
        }
        if (axiosError.response?.status === 404) {
            throw new Error('This feature is currently being updated. Please try again later.');
        }
        if (axiosError.code === 'ECONNABORTED') {
            throw new Error('Request timed out. Please check your connection and try again.');
        }
        if (!axiosError.response) {
            throw new Error('Unable to connect to server. Please check your internet connection.');
        }
    }

    throw new Error(`${context} failed. Please try again.`);
}

/**
 * Build auth headers from the live Supabase session (preferred) with a
 * fallback to the legacy `userToken` key for older installs.
 *
 * The legacy key alone caused every AI call to 401 once Supabase became
 * the sole auth source — the session JWT is what the backend validates.
 */
export async function getAuthHeaders(): Promise<Record<string, string>> {
    const headers: Record<string, string> = {
        'Content-Type': 'application/json',
    };
    try {
        // Lazy require avoids a hard import cycle (supabase client is tiny,
        // but shared.ts is imported very early in the boot sequence).
        // eslint-disable-next-line @typescript-eslint/no-var-requires
        const { supabase } = require('../../../lib/supabase') as typeof import('../../../lib/supabase');
        const { data } = await supabase.auth.getSession();
        const sessionToken = data?.session?.access_token;
        if (sessionToken) {
            headers['Authorization'] = `Bearer ${sessionToken}`;
            return headers;
        }
    } catch {
        // Fall through to legacy token below.
    }
    const token = await AsyncStorage.getItem('userToken');
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    return headers;
}
