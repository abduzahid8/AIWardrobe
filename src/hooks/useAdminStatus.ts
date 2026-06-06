/**
 * useAdminStatus Hook
 * 
 * React hook for checking admin status and permissions.
 * Provides admin role, permissions, and helper functions.
 */

import type React from 'react';
import { useState, useEffect, useCallback } from 'react';
import { supabase } from '../../lib/supabase';
import useAuthStore from '../../store/auth';
import { createLogger } from '../utils/logger';

const logger = createLogger('useAdminStatus');

// ============================================
// TYPES
// ============================================

export type AdminRole = 'super_admin' | 'admin' | 'moderator' | null;

export interface AdminStatus {
    isAdmin: boolean;
    isSuperAdmin: boolean;
    isLoading: boolean;
    error: string | null;
    role: AdminRole;
    permissions: string[];
    canAssignAdmins: boolean;
    canManageUsers: boolean;
    canViewLogs: boolean;
    canManagePermissions: boolean;
}

// ============================================
// HOOK
// ============================================

/**
 * Hook to check admin status and permissions
 * 
 * Usage:
 * ```typescript
 * const { isAdmin, role, canAssignAdmins } = useAdminStatus();
 * 
 * if (!isAdmin) {
 *   return <Text>Admin access required</Text>;
 * }
 * ```
 */
export function useAdminStatus(): AdminStatus {
    const user = useAuthStore((s: any) => s.user);
    const isAuthenticated = useAuthStore((s: any) => s.isAuthenticated);

    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [role, setRole] = useState<AdminRole>(null);
    const [permissions, setPermissions] = useState<string[]>([]);

    // Fetch admin status
    const fetchAdminStatus = useCallback(async () => {
        if (!isAuthenticated || !user?.id) {
            setIsLoading(false);
            setRole(null);
            setPermissions([]);
            return;
        }

        try {
            setIsLoading(true);
            setError(null);

            // Get user profile with admin info
            const { data: profile, error: profileError } = await supabase
                .from('profiles')
                .select('is_admin, admin_role')
                .eq('id', user.id)
                .single();

            if (profileError) {
                throw profileError;
            }

            if (!profile?.is_admin) {
                setRole(null);
                setPermissions([]);
                setIsLoading(false);
                return;
            }

            setRole(profile.admin_role as AdminRole);

            // Get permissions if admin
            const { data: perms, error: permsError } = await supabase
                .from('admin_permissions')
                .select('permission')
                .eq('admin_id', user.id);

            if (permsError) {
                logger.warn('Failed to fetch permissions:', permsError);
            } else {
                setPermissions(perms?.map((p) => p.permission) || []);
            }
        } catch (err) {
            const message = err instanceof Error ? err.message : 'Failed to fetch admin status';
            logger.error('Error fetching admin status:', err);
            setError(message);
        } finally {
            setIsLoading(false);
        }
    }, [isAuthenticated, user?.id]);

    // Fetch on mount and when user changes
    useEffect(() => {
        fetchAdminStatus();
    }, [fetchAdminStatus]);

    // Compute derived values
    const isAdmin = role !== null;
    const isSuperAdmin = role === 'super_admin';
    const canAssignAdmins = isSuperAdmin;
    const canManageUsers = isAdmin;
    const canViewLogs = isAdmin;
    const canManagePermissions = isSuperAdmin;

    return {
        isAdmin,
        isSuperAdmin,
        isLoading,
        error,
        role,
        permissions,
        canAssignAdmins,
        canManageUsers,
        canViewLogs,
        canManagePermissions,
    };
}

// ============================================
// HELPER HOOKS
// ============================================

/**
 * Hook to check if user has a specific permission
 */
export function useHasPermission(permission: string): boolean {
    const { permissions } = useAdminStatus();
    return permissions.includes(permission);
}

/**
 * Hook to check if user is super admin
 */
export function useIsSuperAdmin(): boolean {
    const { isSuperAdmin } = useAdminStatus();
    return isSuperAdmin;
}

/**
 * Hook to check if user is any type of admin
 */
export function useIsAdmin(): boolean {
    const { isAdmin } = useAdminStatus();
    return isAdmin;
}

/**
 * Hook to get admin role
 */
export function useAdminRole(): AdminRole {
    const { role } = useAdminStatus();
    return role;
}

// ============================================
// GUARD COMPONENT
// ============================================

interface AdminGuardProps {
    children: React.ReactNode;
    requiredRole?: AdminRole;
    fallback?: React.ReactNode;
}

/**
 * Component to guard content behind admin check
 * 
 * Usage:
 * ```typescript
 * <AdminGuard requiredRole="super_admin">
 *   <AdminManagement />
 * </AdminGuard>
 * ```
 */
export function AdminGuard({
    children,
    requiredRole = 'admin',
    fallback = null,
}: AdminGuardProps): React.ReactNode {
    const { isAdmin, role, isLoading } = useAdminStatus();

    if (isLoading) {
        return null;
    }

    if (!isAdmin) {
        return fallback;
    }

    // Check role hierarchy
    const roleHierarchy: Record<NonNullable<AdminRole>, number> = {
        super_admin: 3,
        admin: 2,
        moderator: 1,
    };

    const userLevel = role ? roleHierarchy[role] : 0;
    const requiredLevel = requiredRole ? roleHierarchy[requiredRole] : 0;

    if (userLevel < requiredLevel) {
        return fallback;
    }

    return children;
}

// ============================================
// PERMISSION CHECKER
// ============================================

/**
 * Check if user has required permissions
 */
export function useCheckPermissions(requiredPermissions: string[]): boolean {
    const { permissions } = useAdminStatus();
    return requiredPermissions.every((perm) => permissions.includes(perm));
}

/**
 * Check if user has any of the required permissions
 */
export function useCheckAnyPermission(requiredPermissions: string[]): boolean {
    const { permissions } = useAdminStatus();
    return requiredPermissions.some((perm) => permissions.includes(perm));
}
