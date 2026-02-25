import express from "express";
import { createClient } from "@supabase/supabase-js";
import { authenticateToken } from "../middleware/auth.js";
import { authLimiter } from "../middleware/rateLimit.js";
import "dotenv/config";

/**
 * Auth Routes
 *
 * Registration and login are handled entirely by Supabase on the mobile client.
 * This router only exposes server-side profile endpoints that require a valid
 * Supabase access token.
 *
 * Flow:
 *   1. Mobile calls supabase.auth.signUp() / signInWithPassword() directly
 *   2. Supabase returns an access_token
 *   3. Mobile sends that token in Authorization: Bearer <token> to this API
 *   4. authenticateToken middleware validates it via supabaseAdmin.auth.getUser()
 */

const router = express.Router();

const supabaseAdmin = createClient(
    process.env.SUPABASE_URL,
    process.env.SUPABASE_SERVICE_ROLE_KEY,
    { auth: { autoRefreshToken: false, persistSession: false } }
);

/**
 * GET /me
 * Returns the authenticated user's profile from Supabase.
 * Requires: Authorization: Bearer <supabase_access_token>
 */
router.get("/me", authLimiter, authenticateToken, async (req, res) => {
    try {
        const { data: profile, error } = await supabaseAdmin
            .from("profiles")
            .select("id, email, username, gender, profile_image, subscription_tier, subscription_expires_at, created_at")
            .eq("id", req.user.id)
            .single();

        if (error || !profile) {
            return res.status(404).json({
                error: "Profile not found",
                code: "USER_NOT_FOUND"
            });
        }

        res.json({ user: profile });
    } catch (err) {
        console.error("Get profile error:", err.message);
        res.status(500).json({
            error: "Failed to get user data",
            code: "FETCH_ERROR"
        });
    }
});

/**
 * GET /subscription-status
 * Returns the authenticated user's current subscription tier and expiry.
 * Used by the mobile app to verify subscription server-side.
 */
router.get("/subscription-status", authLimiter, authenticateToken, async (req, res) => {
    try {
        const { data: profile, error } = await supabaseAdmin
            .from("profiles")
            .select("subscription_tier, subscription_expires_at")
            .eq("id", req.user.id)
            .single();

        if (error || !profile) {
            return res.status(404).json({ error: "Profile not found", code: "USER_NOT_FOUND" });
        }

        const isExpired = profile.subscription_expires_at
            ? new Date(profile.subscription_expires_at) < new Date()
            : false;

        const effectiveTier = isExpired ? "free" : (profile.subscription_tier || "free");

        res.json({
            tier: effectiveTier,
            expiresAt: profile.subscription_expires_at,
            isActive: !isExpired && effectiveTier !== "free",
        });
    } catch (err) {
        console.error("Subscription status error:", err.message);
        res.status(500).json({ error: "Failed to get subscription status", code: "FETCH_ERROR" });
    }
});

export default router;

