/**
 * Admin Service
 * 
 * Client-side service for admin operations.
 * Handles API calls for admin role management, user management, and audit logs.
 */

import { supabase } from '../lib/supabase';
import { createLogger } from '../utils/logger';

const logger = createLogger('AdminService');

// ============================================
// TYPES
// ============================================

export interface AdminUser {
    id: string;
    email: string;
    username: string;
    is_admin: boolean;
    admin_role: 'super_admin' | 'admin' | 'moderator' | null;
    admin_assigned_at: string | null;
    created_at: string;
}

export interface UserDetails extends AdminUser {
    gender?: string;
    profile_image?: string;
    subscription_tier: string;
    subscription_expires_at: string | null;
    is_active: boolean;
    is_email_verified: boolean;
    updated_at: string;
}

export interface AdminLog {
    id: string;
    admin_id: string;
    action: 'assign_admin' | 'revoke_admin' | 'update_role' | 'delete_user' | 'view_user_data';
    target_user_id: string | null;
    details: Record<string, any>;
    created_at: string;
}

export interface AdminPermission {
    id: string;
    admin_id: string;
    permission: string;
    granted_at: string;
}

export interface AdminStats {
    totalUsers: number;
    totalAdmins: number;
    actionCounts: Record<string, number>;
}

// ============================================
// API BASE URL
// ============================================

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:3000';

// ============================================
// HELPER FUNCTIONS
// ============================================

/**
 * Get the current user's auth token
 */
async function getAuthToken(): Promise<string> {
    const { data: { session }, error } = await supabase.auth.getSession();
    
    if (error || !session?.access_token) {
        throw new Error('No active session');
    }
    
    return session.access_token;
}

/**
 * Make authenticated API request
 */
async function apiRequest<T>(
    endpoint: string,
    options: RequestInit = {}
): Promise<T> {
    const token = await getAuthToken();
    
    const response = await fetch(`${API_BASE}/api/admin${endpoint}`, {
        ...options,
        headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`,
            ...options.headers,
        },
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ error: 'Unknown error' }));
        throw new Error(error.error || `API error: ${response.status}`);
    }

    return response.json();
}

// ============================================
// USER MANAGEMENT
// ============================================

/**
 * Get all users with admin status
 */
export async function getAllUsers(): Promise<AdminUser[]> {
    try {
        const result = await apiRequest<{ success: boolean; data: AdminUser[] }>('/users');
        return result.data;
    } catch (err) {
        logger.error('Failed to fetch users:', err);
        throw err;
    }
}

/**
 * Get detailed information about a specific user
 */
export async function getUserDetails(userId: string): Promise<UserDetails> {
    try {
        const result = await apiRequest<{ success: boolean; data: UserDetails }>(`/users/${userId}`);
        return result.data;
    } catch (err) {
        logger.error(`Failed to fetch user ${userId}:`, err);
        throw err;
    }
}

/**
 * Search users by email
 */
export async function searchUsersByEmail(email: string): Promise<AdminUser[]> {
    try {
        const users = await getAllUsers();
        return users.filter(u => u.email.toLowerCase().includes(email.toLowerCase()));
    } catch (err) {
        logger.error('Failed to search users:', err);
        throw err;
    }
}

// ============================================
// ADMIN ROLE MANAGEMENT
// ============================================

export type AdminRole = 'super_admin' | 'admin' | 'moderator';

/**
 * Assign admin role to a user by email
 */
export async function assignAdminRole(email: string, role: AdminRole): Promise<void> {
    try {
        await apiRequest('/assign-admin', {
            method: 'POST',
            body: JSON.stringify({ email, role }),
        });
        logger.info(`Admin role assigned: ${email} -> ${role}`);
    } catch (err) {
        logger.error(`Failed to assign admin role to ${email}:`, err);
        throw err;
    }
}

/**
 * Revoke admin role from a user by email
 */
export async function revokeAdminRole(email: string): Promise<void> {
    try {
        await apiRequest('/revoke-admin', {
            method: 'POST',
            body: JSON.stringify({ email }),
        });
        logger.info(`Admin role revoked: ${email}`);
    } catch (err) {
        logger.error(`Failed to revoke admin role from ${email}:`, err);
        throw err;
    }
}

// ============================================
// AUDIT LOGS
// ============================================

export interface LogsOptions {
    limit?: number;
    offset?: number;
    admin_id?: string;
    target_user_id?: string;
    action?: string;
}

/**
 * Get admin action logs
 */
export async function getAdminLogs(options: LogsOptions = {}): Promise<{
    logs: AdminLog[];
    pagination: {
        limit: number;
        offset: number;
        total: number;
    };
}> {
    try {
        const params = new URLSearchParams();
        if (options.limit) params.append('limit', options.limit.toString());
        if (options.offset) params.append('offset', options.offset.toString());
        if (options.admin_id) params.append('admin_id', options.admin_id);
        if (options.target_user_id) params.append('target_user_id', options.target_user_id);
        if (options.action) params.append('action', options.action);

        const result = await apiRequest<{
            success: boolean;
            data: AdminLog[];
            pagination: any;
        }>(`/logs?${params.toString()}`);

        return {
            logs: result.data,
            pagination: result.pagination,
        };
    } catch (err) {
        logger.error('Failed to fetch admin logs:', err);
        throw err;
    }
}

// ============================================
// PERMISSIONS
// ============================================

/**
 * Get permissions for an admin user
 */
export async function getAdminPermissions(userId: string): Promise<AdminPermission[]> {
    try {
        const result = await apiRequest<{ success: boolean; data: AdminPermission[] }>(
            `/permissions/${userId}`
        );
        return result.data;
    } catch (err) {
        logger.error(`Failed to fetch permissions for ${userId}:`, err);
        throw err;
    }
}

/**
 * Grant a permission to an admin
 */
export async function grantPermission(adminId: string, permission: string): Promise<void> {
    try {
        await apiRequest('/permissions', {
            method: 'POST',
            body: JSON.stringify({ admin_id: adminId, permission }),
        });
        logger.info(`Permission granted: ${adminId} -> ${permission}`);
    } catch (err) {
        logger.error(`Failed to grant permission:`, err);
        throw err;
    }
}

/**
 * Revoke a permission from an admin
 */
export async function revokePermission(permissionId: string): Promise<void> {
    try {
        await apiRequest(`/permissions/${permissionId}`, {
            method: 'DELETE',
        });
        logger.info(`Permission revoked: ${permissionId}`);
    } catch (err) {
        logger.error(`Failed to revoke permission:`, err);
        throw err;
    }
}

// ============================================
// STATISTICS
// ============================================

/**
 * Get admin dashboard statistics
 */
export async function getAdminStats(): Promise<AdminStats> {
    try {
        const result = await apiRequest<{ success: boolean; data: AdminStats }>('/stats');
        return result.data;
    } catch (err) {
        logger.error('Failed to fetch admin stats:', err);
        throw err;
    }
}

// ============================================
// BATCH OPERATIONS
// ============================================

/**
 * Assign admin role to multiple users
 */
export async function assignAdminRoleBatch(
    emails: string[],
    role: AdminRole
): Promise<{ success: string[]; failed: { email: string; error: string }[] }> {
    const results = {
        success: [] as string[],
        failed: [] as { email: string; error: string }[],
    };

    for (const email of emails) {
        try {
            await assignAdminRole(email, role);
            results.success.push(email);
        } catch (err) {
            results.failed.push({
                email,
                error: err instanceof Error ? err.message : 'Unknown error',
            });
        }
    }

    return results;
}

/**
 * Revoke admin role from multiple users
 */
export async function revokeAdminRoleBatch(emails: string[]): Promise<{
    success: string[];
    failed: { email: string; error: string }[];
}> {
    const results = {
        success: [] as string[],
        failed: [] as { email: string; error: string }[],
    };

    for (const email of emails) {
        try {
            await revokeAdminRole(email);
            results.success.push(email);
        } catch (err) {
            results.failed.push({
                email,
                error: err instanceof Error ? err.message : 'Unknown error',
            });
        }
    }

    return results;
}
