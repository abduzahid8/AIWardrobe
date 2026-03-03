import 'react-native-url-polyfill/auto';
import { createClient } from '@supabase/supabase-js';
import { setSecureItem, getSecureItem, deleteSecureItem } from '../src/utils/secureStorage';
import Config from '../src/config/env';

/**
 * Supabase-compatible storage adapter backed by SecureStore (Keychain/Keystore).
 * Falls back to AsyncStorage on web (handled inside secureStorage.ts).
 */
const SecureStorageAdapter = {
    getItem: (key: string) => getSecureItem(key),
    setItem: (key: string, value: string) => setSecureItem(key, value),
    removeItem: (key: string) => deleteSecureItem(key),
};

export const supabase = createClient(Config.supabase.url, Config.supabase.anonKey, {
    auth: {
        storage: SecureStorageAdapter,
        autoRefreshToken: true,
        persistSession: true,
        detectSessionInUrl: false,
    },
});
