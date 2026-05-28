import { createClient } from "@supabase/supabase-js";
import "dotenv/config";
import logger from "../utils/logger.js";

/**
 * Subscription Guard Middleware
 *
 * Server-side enforcement of subscription tiers on premium features.
 * Queries Supabase `subscriptions` table directly — this is the source of truth.
 *
 * Usage:
 *   router.post('/premium-feature', authenticateToken, requireTier('premium'), handler);
 *   router.post('/vip-feature', authenticateToken, requireTier('vip'), handler);
 */

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

// Service-role client for server-side queries
let supabaseAdmin = null;
if (SUPABASE_URL && SUPABASE_SERVICE_ROLE_KEY) {
    supabaseAdmin = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
        auth: { autoRefreshToken: false, persistSession: false },
    });
}

// In-memory tier cache to reduce DB lookups (5-minute TTL)
const tierCache = new Map();
const CACHE_TTL_MS = 5 * 60 * 1000;

/**
 * Tier hierarchy: free < premium < vip
 */
const TIER_LEVEL = { free: 0, premium: 1, vip: 2 };

/**
 * Look up user's active subscription tier.
 * Returns 'free' if no active subscription exists.
 */
async function getUserTier(userId) {
    // Check cache first
    const cached = tierCache.get(userId);
    if (cached && Date.now() - cached.timestamp < CACHE_TTL_MS) {
        return cached.tier;
    }

    if (!supabaseAdmin) {
        // If Supabase admin is not configured, allow access (graceful degradation)
        return "free";
    }

    try {
        const { data, error } = await supabaseAdmin
            .from("subscriptions")
            .select("tier, status, end_date")
            .eq("user_id", userId)
            .in("status", ["active", "trial"])
            .gte("end_date", new Date().toISOString())
            .order("end_date", { ascending: false })
            .limit(1)
            .maybeSingle();

        if (error) {
            logger.error("Subscription lookup error:", error);
            return "free";
        }

        const tier = data?.tier || "free";

        // Cache the result
        tierCache.set(userId, { tier, timestamp: Date.now() });

        return tier;
    } catch (error) {
        logger.error("Subscription guard error:", error);
        return "free";
    }
}

/**
 * Middleware factory: requires a minimum subscription tier.
 *
 * @param {string} minTier — 'premium' or 'vip'
 * @returns Express middleware
 */
export const requireTier = (minTier) => {
    return async (req, res, next) => {
        if (!req.user?.id) {
            return res.status(401).json({
                error: "Authentication required",
                code: "AUTH_REQUIRED",
            });
        }

        const userTier = await getUserTier(req.user.id);
        const userLevel = TIER_LEVEL[userTier] || 0;
        const requiredLevel = TIER_LEVEL[minTier] || 0;

        if (userLevel < requiredLevel) {
            return res.status(403).json({
                error: `This feature requires a ${minTier} subscription.`,
                code: "UPGRADE_REQUIRED",
                currentTier: userTier,
                requiredTier: minTier,
                upgradeUrl: "/paywall",
            });
        }

        // Attach tier info to request for downstream use
        req.subscriptionTier = userTier;
        next();
    };
};

/**
 * Middleware: attach subscription tier to request without blocking.
 * Useful for routes that behave differently based on tier.
 */
export const attachTier = async (req, res, next) => {
    if (req.user?.id) {
        req.subscriptionTier = await getUserTier(req.user.id);
    } else {
        req.subscriptionTier = "free";
    }
    next();
};

/**
 * Clear the tier cache for a specific user (call after subscription changes).
 */
export const clearTierCache = (userId) => {
    tierCache.delete(userId);
};

export default { requireTier, attachTier, clearTierCache };
