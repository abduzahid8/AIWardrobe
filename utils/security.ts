/**
 * Security Utilities for AIWardrobe
 * Provides secure storage, input sanitization, and API security
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { Platform } from 'react-native';

// Secure Storage Keys
const SECURE_KEYS = {
    AUTH_TOKEN: 'auth_token',
    REFRESH_TOKEN: 'refresh_token',
    USER_ID: 'user_id',
    SUBSCRIPTION_KEY: 'subscription_key',
};

/**
 * Securely store a value (uses AsyncStorage with base64 encoding)
 */
export const secureStore = async (key: string, value: string): Promise<boolean> => {
    try {
        // Encode value for basic obfuscation
        const encodedValue = Buffer.from(value).toString('base64');
        await AsyncStorage.setItem(`secure_${key}`, encodedValue);
        return true;
    } catch (error) {
        console.error('Secure store error:', error);
        return false;
    }
};

/**
 * Securely retrieve a value
 */
export const secureGet = async (key: string): Promise<string | null> => {
    try {
        const encodedValue = await AsyncStorage.getItem(`secure_${key}`);
        if (!encodedValue) return null;
        return Buffer.from(encodedValue, 'base64').toString('utf-8');
    } catch (error) {
        console.error('Secure get error:', error);
        return null;
    }
};

/**
 * Securely delete a value
 */
export const secureDelete = async (key: string): Promise<boolean> => {
    try {
        await AsyncStorage.removeItem(`secure_${key}`);
        return true;
    } catch (error) {
        console.error('Secure delete error:', error);
        return false;
    }
};

/**
 * Sanitize user input to prevent XSS and injection attacks
 */
export const sanitizeInput = (input: string): string => {
    if (!input) return '';

    return input
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#x27;')
        .replace(/\//g, '&#x2F;')
        .trim();
};

/**
 * Validate email format
 */
export const isValidEmail = (email: string): boolean => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
};

/**
 * Validate password strength
 */
export const isStrongPassword = (password: string): { valid: boolean; message: string } => {
    if (password.length < 8) {
        return { valid: false, message: 'Password must be at least 8 characters' };
    }
    if (!/[A-Z]/.test(password)) {
        return { valid: false, message: 'Password must contain an uppercase letter' };
    }
    if (!/[a-z]/.test(password)) {
        return { valid: false, message: 'Password must contain a lowercase letter' };
    }
    if (!/[0-9]/.test(password)) {
        return { valid: false, message: 'Password must contain a number' };
    }
    return { valid: true, message: 'Password is strong' };
};

/**
 * Rate limiting helper for API calls
 */
const rateLimitMap = new Map<string, number[]>();
const RATE_LIMIT_WINDOW = 60000; // 1 minute
const MAX_REQUESTS = 30;

export const checkRateLimit = (endpoint: string): boolean => {
    const now = Date.now();
    const timestamps = rateLimitMap.get(endpoint) || [];

    // Remove old timestamps
    const recentTimestamps = timestamps.filter(t => now - t < RATE_LIMIT_WINDOW);

    if (recentTimestamps.length >= MAX_REQUESTS) {
        return false; // Rate limited
    }

    recentTimestamps.push(now);
    rateLimitMap.set(endpoint, recentTimestamps);
    return true;
};

/**
 * Generate a random ID (alphanumeric)
 */
export const generateSecureId = (): string => {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let i = 0; i < 32; i++) {
        result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    return result;
};

export { SECURE_KEYS };
