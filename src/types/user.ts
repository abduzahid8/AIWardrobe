/**
 * User Types
 * Types for user authentication and profile management
 */

/**
 * User profile from database
 */
export interface User {
    _id: string;
    email: string;
    username: string;
    gender: Gender;
    profileImage?: string;
    outfits: string[];
    createdAt?: Date;
    updatedAt?: Date;
}

/**
 * User gender options
 */
export type Gender = 'male' | 'female' | 'other' | 'prefer_not_to_say';

/**
 * User registration input
 */
export interface RegisterInput {
    email: string;
    password: string;
    username: string;
    gender?: Gender;
    profileImage?: string;
}

/**
 * User login input
 */
export interface LoginInput {
    email: string;
    password: string;
}

/**
 * Authentication response
 */
export interface AuthResponse {
    token: string;
}

/**
 * User profile for display (without sensitive data)
 */
export interface UserProfile {
    _id: string;
    email: string;
    username: string;
    gender: Gender;
    profileImage?: string;
    outfitCount: number;
    itemCount: number;
}

/**
 * User settings
 */
export interface UserSettings {
    notifications: boolean;
    darkMode: boolean;
    language: SupportedLanguage;
    measurementUnit: 'metric' | 'imperial';
}

/**
 * Supported languages
 */
export type SupportedLanguage = 'en' | 'ru' | 'uz';

/**
 * JWT token payload
 */
export interface TokenPayload {
    id: string;
    iat: number;
    exp?: number;
}

/**
 * Auth store state
 */
export interface AuthState {
    isAuthenticated: boolean;
    token: string | null;
    user: User | null;
    isLoading: boolean;
    error: string | null;
}
