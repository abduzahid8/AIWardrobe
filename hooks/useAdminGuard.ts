import { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import useAuthStore from '../store/auth';
import { createLogger } from '../src/utils/logger';
import Config from '../src/config/env';

const logger = createLogger('useAdminGuard');

const ADMIN_EMAIL = Config.admin?.email || 'info@aiwardrobe.club';

// Stable selectors — defined at module scope so they never change reference.
// useAuthStore(s => s.user) returns an object (new ref every render) which
// causes infinite re-renders when used inline. Selecting only the primitives
// we actually need avoids that.
const selectUserId = (s: any) => s.user?.id as string | undefined;
const selectUserEmail = (s: any) => s.user?.email as string | undefined;
const selectIsAuthenticated = (s: any) => s.isAuthenticated as boolean;

export interface AdminGuardResult {
    isAdmin: boolean;
    loading: boolean;
    checkAdmin: () => Promise<boolean>;
}

// Admin configuration type
interface AdminConfig {
    email: string;
    allowedEmails?: string[];
}

// Module-level cache so multiple components (HomeScreen, TabNavigator, etc.)
// that call useAdminGuard() share a single result instead of each firing
// an independent Supabase query.
let cachedAdminUserId: string | null = null;
let cachedAdminResult: boolean = false;
let cacheResolved: boolean = false;

export function useAdminGuard(): AdminGuardResult {
    const userId = useAuthStore(selectUserId);
    const userEmail = useAuthStore(selectUserEmail);
    const isAuthenticated = useAuthStore(selectIsAuthenticated);

    // Initialize state synchronously from cache when possible so the component
    // never renders with a stale `loading=true` / `isAdmin=false` pair that
    // immediately triggers a re-render when the cache resolves (fixes Defects 1.4 + 1.5).
    const isCacheHit = cacheResolved && cachedAdminUserId === userId;
    const [isAdmin, setIsAdmin] = useState(isCacheHit ? cachedAdminResult : false);
    const [loading, setLoading] = useState(!isCacheHit);

    const checkAdmin = async (): Promise<boolean> => {
        if (!isAuthenticated || !userId) {
            setIsAdmin(false);
            setLoading(false);
            // Reset cache on logout so the next user gets a fresh check (Defect 3.6).
            cacheResolved = false;
            cachedAdminUserId = null;
            cachedAdminResult = false;
            return false;
        }

        // Return cached result if the same user was already checked this session.
        // Skip setState calls when the initial state was already set from cache
        // to avoid a redundant re-render (fixes Defects 1.4 + 1.5).
        if (cacheResolved && cachedAdminUserId === userId) {
            // Only update state if it differs from what was set at init time
            // (e.g. if userId changed between renders).
            if (isAdmin !== cachedAdminResult) setIsAdmin(cachedAdminResult);
            if (loading) setLoading(false);
            return cachedAdminResult;
        }

        // Fast path: email match
        if (userEmail?.toLowerCase() === ADMIN_EMAIL) {
            cachedAdminUserId = userId;
            cachedAdminResult = true;
            cacheResolved = true;
            setIsAdmin(true);
            setLoading(false);
            return true;
        }

        // DB check: is_admin flag on profile
        try {
            const { data, error } = await supabase
                .from('profiles')
                .select('is_admin')
                .eq('id', userId)
                .maybeSingle();

            if (error) {
                logger.warn('Failed to check admin status', error);
                setIsAdmin(false);
                return false;
            }

            const admin = data?.is_admin === true;
            cachedAdminUserId = userId;
            cachedAdminResult = admin;
            cacheResolved = true;
            setIsAdmin(admin);
            return admin;
        } catch (err) {
            logger.error('Admin check error', err);
            setIsAdmin(false);
            return false;
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        checkAdmin();
    }, [isAuthenticated, userId]);

    return { isAdmin, loading, checkAdmin };
}
