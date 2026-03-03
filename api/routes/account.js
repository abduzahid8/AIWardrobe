import express from "express";
import { createClient } from "@supabase/supabase-js";
import { authenticateToken } from "../middleware/auth.js";
import "dotenv/config";

import logger from '../utils/logger.js';
const router = express.Router();

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

let supabaseAdmin = null;
if (SUPABASE_URL && SUPABASE_SERVICE_ROLE_KEY) {
    supabaseAdmin = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
        auth: { autoRefreshToken: false, persistSession: false },
    });
}

/**
 * DELETE /api/account
 * Permanently delete user account and all associated data.
 * GDPR Article 17: Right to erasure.
 *
 * Supabase ON DELETE CASCADE handles:
 *   - profiles → clothing_items → wear_logs
 *   - profiles → saved_outfits
 *   - profiles → subscriptions → payments
 *
 * This endpoint also deletes the Supabase auth.users record.
 */
router.delete("/", authenticateToken, async (req, res) => {
    try {
        const userId = req.user.id;
        const userEmail = req.user.email;

        if (!supabaseAdmin) {
            return res.status(503).json({ error: "Service unavailable" });
        }

        logger.info(JSON.stringify({
            type: 'audit',
            action: 'ACCOUNT_DELETE_REQUESTED',
            userId,
            email: userEmail,
            timestamp: new Date().toISOString(),
        }));

        // 1. Delete user storage files (if any)
        try {
            const { data: files } = await supabaseAdmin.storage
                .from('user_uploads')
                .list(`${userId}/`);

            if (files?.length) {
                const filePaths = files.map(f => `${userId}/${f.name}`);
                await supabaseAdmin.storage
                    .from('user_uploads')
                    .remove(filePaths);
            }
        } catch (storageError) {
            // Non-blocking — continue with account deletion
            logger.error("Storage cleanup error:", storageError.message);
        }

        // 2. Delete profile (cascades to all related tables via FK constraints)
        const { error: profileError } = await supabaseAdmin
            .from('profiles')
            .delete()
            .eq('id', userId);

        if (profileError) {
            logger.error("Profile deletion error:", profileError);
            return res.status(500).json({ error: "Failed to delete account data" });
        }

        // 3. Delete the auth.users record (removes login credentials)
        const { error: authError } = await supabaseAdmin.auth.admin.deleteUser(userId);

        if (authError) {
            logger.error("Auth deletion error:", authError);
            // Profile data is already deleted — log for manual cleanup
            return res.status(500).json({
                error: "Account data deleted but auth cleanup failed. Contact support.",
                partial: true,
            });
        }

        logger.info(JSON.stringify({
            type: 'audit',
            action: 'ACCOUNT_DELETED',
            userId,
            email: userEmail,
            timestamp: new Date().toISOString(),
        }));

        res.json({
            success: true,
            message: "Your account and all associated data have been permanently deleted.",
        });
    } catch (error) {
        logger.error("Account deletion error:", error.message);
        res.status(500).json({ error: "Account deletion failed. Please contact support." });
    }
});

/**
 * GET /api/account/data-export
 * Export all user data (GDPR Article 20: Right to data portability).
 */
router.get("/data-export", authenticateToken, async (req, res) => {
    try {
        const userId = req.user.id;

        if (!supabaseAdmin) {
            return res.status(503).json({ error: "Service unavailable" });
        }

        const [profile, items, outfits, wearLogs, subscriptions, payments] = await Promise.all([
            supabaseAdmin.from('profiles').select('*').eq('id', userId).maybeSingle(),
            supabaseAdmin.from('clothing_items').select('*').eq('user_id', userId),
            supabaseAdmin.from('saved_outfits').select('*').eq('user_id', userId),
            supabaseAdmin.from('wear_logs').select('*').eq('user_id', userId),
            supabaseAdmin.from('subscriptions').select('*').eq('user_id', userId),
            supabaseAdmin.from('payments').select('*').eq('user_id', userId),
        ]);

        res.json({
            exportDate: new Date().toISOString(),
            profile: profile.data,
            clothingItems: items.data || [],
            savedOutfits: outfits.data || [],
            wearLogs: wearLogs.data || [],
            subscriptions: subscriptions.data || [],
            payments: payments.data || [],
        });
    } catch (error) {
        logger.error("Data export error:", error.message);
        res.status(500).json({ error: "Failed to export data" });
    }
});

export default router;
