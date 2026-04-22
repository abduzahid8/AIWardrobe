import 'react-native-url-polyfill/auto';
import { createClient, SupabaseClient } from '@supabase/supabase-js';
import { setSecureItem, getSecureItem, deleteSecureItem } from '../src/utils/secureStorage';
import Config from '../src/config/env';

const SecureStorageAdapter = {
    getItem: (key: string) => getSecureItem(key),
    setItem: (key: string, value: string) => setSecureItem(key, value),
    removeItem: (key: string) => deleteSecureItem(key),
};

const PLACEHOLDER_URL = 'https://placeholder.supabase.co';

function createSupabaseClient(): SupabaseClient {
    const url = Config.supabase.url || PLACEHOLDER_URL;
    const key = Config.supabase.anonKey || 'placeholder';

    return createClient(url, key, {
        auth: {
            storage: SecureStorageAdapter,
            autoRefreshToken: true,
            persistSession: true,
            detectSessionInUrl: false,
        },
    });
}

export const supabase = createSupabaseClient();
