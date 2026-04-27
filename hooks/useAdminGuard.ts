import { useState, useEffect } from 'react';
import { supabase } from '../lib/supabase';
import useAuthStore from '../store/auth';
import { createLogger } from '../src/utils/logger';

const logger = createLogger('useAdminGuard');

const ADMIN_EMAIL = 'info@aiwardrobe.club';

export interface AdminGuardResult {
    isAdmin: boolean;
    loading: boolean;
    checkAdmin: () => Promise<boolean>;
}

export function useAdminGuard(): AdminGuardResult {
    const { user, isAuthenticated } = useAuthStore();
    const [isAdmin, setIsAdmin] = useState(false);
    const [loading, setLoading] = useState(true);

    const checkAdmin = async (): Promise<boolean> => {
        if (!isAuthenticated || !user?.id) {
            setIsAdmin(false);
            setLoading(false);
            return false;
        }

        // Fast path: email match
        if (user.email?.toLowerCase() === ADMIN_EMAIL) {
            setIsAdmin(true);
            setLoading(false);
            return true;
        }

        // DB check: is_admin flag on profile
        try {
            const { data, error } = await supabase
                .from('profiles')
                .select('is_admin')
                .eq('id', user.id)
                .maybeSingle();

            if (error) {
                logger.warn('Failed to check admin status', error);
                setIsAdmin(false);
                return false;
            }

            const admin = data?.is_admin === true;
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
    }, [isAuthenticated, user?.id]);

    return { isAdmin, loading, checkAdmin };
}
