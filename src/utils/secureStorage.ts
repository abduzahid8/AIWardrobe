/**
 * Secure Storage — wraps expo-secure-store for sensitive data.
 * Falls back to AsyncStorage on web (where SecureStore is unavailable).
 */
import { Platform } from 'react-native';
import * as SecureStore from 'expo-secure-store';
import AsyncStorage from '@react-native-async-storage/async-storage';

const isNative = Platform.OS === 'ios' || Platform.OS === 'android';

/**
 * Store a value securely.
 * Uses Keychain (iOS) / Keystore (Android), falls back to AsyncStorage on web.
 */
export async function setSecureItem(key: string, value: string): Promise<void> {
    if (isNative) {
        await SecureStore.setItemAsync(key, value);
    } else {
        await AsyncStorage.setItem(`__secure_${key}`, value);
    }
}

/**
 * Retrieve a securely stored value.
 */
export async function getSecureItem(key: string): Promise<string | null> {
    if (isNative) {
        return SecureStore.getItemAsync(key);
    }
    return AsyncStorage.getItem(`__secure_${key}`);
}

/**
 * Delete a securely stored value.
 */
export async function deleteSecureItem(key: string): Promise<void> {
    if (isNative) {
        await SecureStore.deleteItemAsync(key);
    } else {
        await AsyncStorage.removeItem(`__secure_${key}`);
    }
}

export default { setSecureItem, getSecureItem, deleteSecureItem };
