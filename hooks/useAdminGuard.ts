import { useState, useEffect, useCallback } from 'react';
import { supabase } from '../lib/supabase';
import useAuthStore, { type AuthStore } from '../store/auth';
import { authEvents } from '../store/authEvents';
import { withRequestDeadline } from '../src/lib/requestDeadline';
import { createLogger } from '../src/utils/logger';

const logger = createLogger('useAdminGuard');
const selectUserId = (state: AuthStore) => state.session?.user.id ?? state.user?.id;
const selectIsAuthenticated = (state: AuthStore) => state.isAuthenticated;
const CACHE_TTL_MS = 30_000;
let cached: { key: string; allowed: boolean; expires: number } | null = null;
const pending = new Map<string, Promise<boolean>>();

export interface AdminGuardResult {
    isAdmin: boolean;
    loading: boolean;
    checkAdmin: () => Promise<boolean>;
}

/** UI visibility only. Database/Edge authorization remains mandatory. */
export function useAdminGuard(): AdminGuardResult {
    const userId = useAuthStore(selectUserId);
    const isAuthenticated = useAuthStore(selectIsAuthenticated);
    const revision = authEvents.getRevision();
    const key = `${revision}:${userId ?? ''}`;
    const [result, setResult] = useState<{ key: string; allowed: boolean } | null>(null);
    const [loadingKey, setLoadingKey] = useState<string | null>(null);

    const readAdmin = useCallback(async (refresh: boolean): Promise<boolean> => {
        const stillCurrent = () => authEvents.getRevision() === revision
            && useAuthStore.getState().isAuthenticated
            && selectUserId(useAuthStore.getState()) === userId;
        if (!isAuthenticated || !userId || !stillCurrent()) return false;
        setLoadingKey(key);
        try {
            if (!refresh && cached?.key === key && cached.expires > Date.now()) {
                setResult({ key, allowed: cached.allowed });
                return cached.allowed;
            }
            let request = pending.get(key);
            if (!request) {
                request = withRequestDeadline(signal => supabase.from('profiles')
                    .select('is_admin').eq('id', userId).abortSignal(signal).maybeSingle())
                    .then(({ data, error }) => {
                        if (error) throw error;
                        return data?.is_admin === true;
                    });
                pending.set(key, request);
                void request.finally(() => { if (pending.get(key) === request) pending.delete(key); }).catch(() => undefined);
            }
            const allowed = await request;
            if (!stillCurrent()) return false;
            cached = { key, allowed, expires: Date.now() + CACHE_TTL_MS };
            setResult({ key, allowed });
            return allowed;
        } catch (error) {
            if (stillCurrent()) {
                cached = null;
                setResult({ key, allowed: false });
                logger.warn('Failed to check admin status', error);
            }
            return false;
        } finally {
            setLoadingKey(current => current === key ? null : current);
        }
    }, [isAuthenticated, userId, revision, key]);

    const checkAdmin = useCallback(() => readAdmin(true), [readAdmin]);
    useEffect(() => { void readAdmin(false); }, [readAdmin]);

    return {
        // Identity-tagged results cannot flash the old account's admin controls.
        isAdmin: isAuthenticated && result?.key === key && result.allowed,
        loading: isAuthenticated && !!userId && (loadingKey === key || result?.key !== key),
        checkAdmin,
    };
}
