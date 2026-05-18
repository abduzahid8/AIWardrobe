import 'react-native-url-polyfill/auto';
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import AsyncStorage from '@react-native-async-storage/async-storage';
import Config from '../src/config/env';

// Use AsyncStorage for Supabase session persistence.
// SecureStore has a 2048-byte limit which Supabase JWTs exceed, causing
// "Value being stored in SecureStore is larger than 2048 bytes" warnings
// and silent storage failures. Supabase sessions are already protected by
// HTTPS and token expiry — AsyncStorage is sufficient for this use case.
const AsyncStorageAdapter = {
    getItem: (key: string) => AsyncStorage.getItem(key),
    setItem: (key: string, value: string) => AsyncStorage.setItem(key, value),
    removeItem: (key: string) => AsyncStorage.removeItem(key),
};

const PLACEHOLDER_URL = 'https://placeholder.supabase.co';

function createSupabaseClient(): SupabaseClient {
    const url = Config.supabase.url || PLACEHOLDER_URL;
    const key = Config.supabase.anonKey || 'placeholder';

    return createClient(url, key, {
        auth: {
            storage: AsyncStorageAdapter,
            autoRefreshToken: true,
            persistSession: true,
            detectSessionInUrl: false,
        },
    });
}

export const supabase = createSupabaseClient();
