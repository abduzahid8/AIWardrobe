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
 * Build auth headers from stored token.
 */
export async function getAuthHeaders(): Promise<Record<string, string>> {
    const headers: Record<string, string> = {
        'Content-Type': 'application/json',
    };
    const token = await AsyncStorage.getItem('userToken');
    if (token) {
        headers['Authorization'] = `Bearer ${token}`;
    }
    return headers;
}
