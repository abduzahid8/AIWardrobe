/**
 * Admin Management Routes
 * 
 * Endpoints for managing admin roles and permissions.
 * All endpoints require admin authentication.
 */

import express from 'express';
import { supabase } from '../lib/supabase.js';
import { authenticateToken, requireAdmin } from '../middleware/auth.js';
import logger from '../utils/logger.js';

const router = express.Router();

// ============================================
// MIDDLEWARE
// ============================================

/**
 * Verify super admin role (stricter than regular admin)
 */
const requireSuperAdmin = async (req, res, next) => {
    try {
        const { data: profile, error } = await supabase
            .from('profiles')
            .select('admin_role')
            .eq('id', req.user.id)
            .single();

        if (error || !profile || profile.admin_role !== 'super_admin') {
            return res.status(403).json({
                error: 'Super admin access required',
                code: 'INSUFFICIENT_PERMISSIONS'
            });
        }

        next();
    } catch (err) {
        logger.error('Super admin check failed:', err.message);
        res.status(500).json({ error: 'Permission check failed' });
    }
};

// ============================================
// ENDPOINTS
// ============================================

/**
 * GET /api/admin/users
 * List all users with admin status
 */
router.get('/users', authenticateToken, requireAdmin, async (req, res) => {
    try {
        const { data: users, error } = await supabase
            .from('profiles')
            .select('id, email, username, is_admin, admin_role, admin_assigned_at, created_at')
            .order('created_at', { ascending: false });

        if (error) throw error;

        res.json({
            success: true,
            data: users,
            count: users.length
        });
    } catch (err) {
        logger.error('Failed to fetch users:', err.message);
        res.status(500).json({ error: 'Failed to fetch users' });
    }
});

/**
 * GET /api/admin/users/:userId
 * Get detailed user information
 */
router.get('/users/:userId', authenticateToken, requireAdmin, async (req, res) => {
    try {
        const { userId } = req.params;

        const { data: user, error } = await supabase
            .from('profiles')
            .select(`
                id, email, username, gender, profile_image,
                is_admin, admin_role, admin_assigned_at, admin_assigned_by,
                subscription_tier, subscription_expires_at,
                is_active, is_email_verified,
                created_at, updated_at
            `)
            .eq('id', userId)
            .single();

        if (error || !user) {
            return res.status(404).json({ error: 'User not found' });
        }

        // Log the view action
        await supabase.rpc('log_admin_action', {
            p_admin_id: req.user.id,
            p_action: 'view_user_data',
            p_target_user_id: userId,
            p_ip_address: req.ip,
            p_user_agent: req.get('user-agent')
        });

        res.json({ success: true, data: user });
    } catch (err) {
        logger.error('Failed to fetch user:', err.message);
        res.status(500).json({ error: 'Failed to fetch user' });
    }
});

/**
 * POST /api/admin/assign-admin
 * Assign admin role to a user
 * 
 * Body:
 * {
 *   "email": "user@example.com",
 *   "role": "admin" | "moderator" | "super_admin"
 * }
 */
router.post('/assign-admin', authenticateToken, requireSuperAdmin, async (req, res) => {
    try {
        const { email, role } = req.body;

        if (!email || !role) {
            return res.status(400).json({
                error: 'Email and role are required',
                code: 'MISSING_FIELDS'
            });
        }

        if (!['admin', 'moderator', 'super_admin'].includes(role)) {
            return res.status(400).json({
                error: 'Invalid role. Must be admin, moderator, or super_admin',
                code: 'INVALID_ROLE'
            });
        }

        // Find user by email
        const { data: user, error: userError } = await supabase
            .from('profiles')
            .select('id, email, is_admin, admin_role')
            .eq('email', email)
            .single();

        if (userError || !user) {
            return res.status(404).json({
                error: 'User not found',
                code: 'USER_NOT_FOUND'
            });
        }

        // Update user with admin role
        const { error: updateError } = await supabase
            .from('profiles')
            .update({
                is_admin: true,
                admin_role: role,
                admin_assigned_at: new Date().toISOString(),
                admin_assigned_by: req.user.id
            })
            .eq('id', user.id);

        if (updateError) throw updateError;

        // Log the action
        await supabase.rpc('log_admin_action', {
            p_admin_id: req.user.id,
            p_action: 'assign_admin',
            p_target_user_id: user.id,
            p_details: { role, previous_role: user.admin_role },
            p_ip_address: req.ip,
            p_user_agent: req.get('user-agent')
        });

        logger.info(`Admin role assigned: ${email} -> ${role}`);

        res.json({
            success: true,
            message: `Admin role '${role}' assigned to ${email}`,
            data: {
                userId: user.id,
                email: user.email,
                role: role
            }
        });
    } catch (err) {
        logger.error('Failed to assign admin role:', err.message);
        res.status(500).json({ error: 'Failed to assign admin role' });
    }
});

/**
 * POST /api/admin/revoke-admin
 * Revoke admin role from a user
 * 
 * Body:
 * {
 *   "email": "user@example.com"
 * }
 */
router.post('/revoke-admin', authenticateToken, requireSuperAdmin, async (req, res) => {
    try {
        const { email } = req.body;

        if (!email) {
            return res.status(400).json({
                error: 'Email is required',
                code: 'MISSING_EMAIL'
            });
        }

        // Find user by email
        const { data: user, error: userError } = await supabase
            .from('profiles')
            .select('id, email, is_admin, admin_role')
            .eq('email', email)
            .single();

        if (userError || !user) {
            return res.status(404).json({
                error: 'User not found',
                code: 'USER_NOT_FOUND'
            });
        }

        if (!user.is_admin) {
            return res.status(400).json({
                error: 'User is not an admin',
                code: 'NOT_ADMIN'
            });
        }

        const previousRole = user.admin_role;

        // Revoke admin role
        const { error: updateError } = await supabase
            .from('profiles')
            .update({
                is_admin: false,
                admin_role: null,
                admin_assigned_at: null,
                admin_assigned_by: null
            })
            .eq('id', user.id);

        if (updateError) throw updateError;

        // Log the action
        await supabase.rpc('log_admin_action', {
            p_admin_id: req.user.id,
            p_action: 'revoke_admin',
            p_target_user_id: user.id,
            p_details: { previous_role: previousRole },
            p_ip_address: req.ip,
            p_user_agent: req.get('user-agent')
        });

        logger.info(`Admin role revoked: ${email}`);

        res.json({
            success: true,
            message: `Admin role revoked from ${email}`,
            data: {
                userId: user.id,
                email: user.email
            }
        });
    } catch (err) {
        logger.error('Failed to revoke admin role:', err.message);
        res.status(500).json({ error: 'Failed to revoke admin role' });
    }
});

/**
 * GET /api/admin/logs
 * Get admin action logs
 * 
 * Query params:
 * - limit: number (default 50)
 * - offset: number (default 0)
 * - admin_id: UUID (filter by admin)
 * - target_user_id: UUID (filter by target user)
 * - action: string (filter by action type)
 */
router.get('/logs', authenticateToken, requireAdmin, async (req, res) => {
    try {
        const { limit = 50, offset = 0, admin_id, target_user_id, action } = req.query;

        let query = supabase
            .from('admin_logs')
            .select('id, admin_id, action, target_user_id, details, created_at', { count: 'exact' })
            .order('created_at', { ascending: false })
            .range(parseInt(offset), parseInt(offset) + parseInt(limit) - 1);

        if (admin_id) query = query.eq('admin_id', admin_id);
        if (target_user_id) query = query.eq('target_user_id', target_user_id);
        if (action) query = query.eq('action', action);

        const { data: logs, error, count } = await query;

        if (error) throw error;

        res.json({
            success: true,
            data: logs,
            pagination: {
                limit: parseInt(limit),
                offset: parseInt(offset),
                total: count
            }
        });
    } catch (err) {
        logger.error('Failed to fetch admin logs:', err.message);
        res.status(500).json({ error: 'Failed to fetch admin logs' });
    }
});

/**
 * GET /api/admin/permissions/:userId
 * Get admin permissions for a user
 */
router.get('/permissions/:userId', authenticateToken, requireAdmin, async (req, res) => {
    try {
        const { userId } = req.params;

        const { data: permissions, error } = await supabase
            .from('admin_permissions')
            .select('id, permission, granted_at')
            .eq('admin_id', userId);

        if (error) throw error;

        res.json({
            success: true,
            data: permissions
        });
    } catch (err) {
        logger.error('Failed to fetch permissions:', err.message);
        res.status(500).json({ error: 'Failed to fetch permissions' });
    }
});

/**
 * POST /api/admin/permissions
 * Grant a permission to an admin
 * 
 * Body:
 * {
 *   "admin_id": "UUID",
 *   "permission": "string"
 * }
 */
router.post('/permissions', authenticateToken, requireSuperAdmin, async (req, res) => {
    try {
        const { admin_id, permission } = req.body;

        if (!admin_id || !permission) {
            return res.status(400).json({
                error: 'admin_id and permission are required',
                code: 'MISSING_FIELDS'
            });
        }

        // Verify target is an admin
        const { data: admin, error: adminError } = await supabase
            .from('profiles')
            .select('id, is_admin')
            .eq('id', admin_id)
            .single();

        if (adminError || !admin || !admin.is_admin) {
            return res.status(404).json({
                error: 'Admin user not found',
                code: 'ADMIN_NOT_FOUND'
            });
        }

        // Grant permission
        const { error: insertError } = await supabase
            .from('admin_permissions')
            .insert({
                admin_id,
                permission,
                granted_by: req.user.id
            });

        if (insertError) {
            if (insertError.code === '23505') {
                return res.status(400).json({
                    error: 'Permission already granted',
                    code: 'PERMISSION_EXISTS'
                });
            }
            throw insertError;
        }

        logger.info(`Permission granted: ${admin_id} -> ${permission}`);

        res.json({
            success: true,
            message: `Permission '${permission}' granted`,
            data: { admin_id, permission }
        });
    } catch (err) {
        logger.error('Failed to grant permission:', err.message);
        res.status(500).json({ error: 'Failed to grant permission' });
    }
});

/**
 * DELETE /api/admin/permissions/:permissionId
 * Revoke a permission from an admin
 */
router.delete('/permissions/:permissionId', authenticateToken, requireSuperAdmin, async (req, res) => {
    try {
        const { permissionId } = req.params;

        const { error } = await supabase
            .from('admin_permissions')
            .delete()
            .eq('id', permissionId);

        if (error) throw error;

        logger.info(`Permission revoked: ${permissionId}`);

        res.json({
            success: true,
            message: 'Permission revoked'
        });
    } catch (err) {
        logger.error('Failed to revoke permission:', err.message);
        res.status(500).json({ error: 'Failed to revoke permission' });
    }
});

/**
 * GET /api/admin/stats
 * Get admin dashboard statistics
 */
router.get('/stats', authenticateToken, requireAdmin, async (req, res) => {
    try {
        // Total users
        const { count: totalUsers } = await supabase
            .from('profiles')
            .select('id', { count: 'exact' });

        // Total admins
        const { count: totalAdmins } = await supabase
            .from('profiles')
            .select('id', { count: 'exact' })
            .eq('is_admin', true);

        // Recent logs
        const { data: recentLogs } = await supabase
            .from('admin_logs')
            .select('action')
            .order('created_at', { ascending: false })
            .limit(100);

        const actionCounts = recentLogs.reduce((acc, log) => {
            acc[log.action] = (acc[log.action] || 0) + 1;
            return acc;
        }, {});

        res.json({
            success: true,
            data: {
                totalUsers,
                totalAdmins,
                actionCounts
            }
        });
    } catch (err) {
        logger.error('Failed to fetch admin stats:', err.message);
        res.status(500).json({ error: 'Failed to fetch admin stats' });
    }
});

export default router;
