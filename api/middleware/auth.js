import { createClient } from "@supabase/supabase-js";
import "dotenv/config";
import logger from "../utils/logger.js";

/**
 * Unified Auth Middleware — Supabase JWT Validation
 *
 * The mobile app uses Supabase Auth exclusively.
 * This middleware validates the Supabase access token sent in the
 * Authorization header and attaches the Supabase user to req.user.
 *
 * Mobile sends: Authorization: Bearer <supabase_access_token>
 */

// Use global supabase client to validate tokens server-side
import { supabase } from "../lib/supabase.js";

const isLocalDevAuthBypassEnabled = () => {
    const bypassEnabled = String(process.env.DEV_AUTH_BYPASS || '').toLowerCase() === 'true';
    const nodeEnv = String(process.env.NODE_ENV || '').toLowerCase();
    const allowInDevelopment = nodeEnv === 'development';
    return bypassEnabled && allowInDevelopment;
};

const attachBypassUser = (req) => {
    req.user = {
        id: process.env.DEV_AUTH_BYPASS_USER_ID || 'dev-bypass-user',
        email: process.env.DEV_AUTH_BYPASS_EMAIL || 'dev-bypass@local.test',
        username: process.env.DEV_AUTH_BYPASS_USERNAME || 'dev_bypass',
    };
};

/**
 * Middleware to authenticate Supabase JWT tokens.
 * Attaches the Supabase user object to req.user on success.
 */
export const authenticateToken = async (req, res, next) => {
    const authHeader = req.headers["authorization"];
    const token = authHeader?.split(" ")[1];

    if (!token) {
        if (isLocalDevAuthBypassEnabled()) {
            attachBypassUser(req);
            return next();
        }
        return res.status(401).json({
            error: "Authentication required",
            code: "NO_TOKEN"
        });
    }

    try {
        const { data: { user }, error } = await supabase.auth.getUser(token);

        if (error || !user) {
            return res.status(401).json({
                error: "Invalid or expired token",
                code: "INVALID_TOKEN"
            });
        }

        // Attach normalized user object — matches Supabase profile shape
        req.user = {
            id: user.id,
            email: user.email,
            username: user.user_metadata?.username || '',
        };

        next();
    } catch (err) {
        logger.error("Auth middleware error:", err.message);
        return res.status(500).json({
            error: "Authentication check failed",
            code: "AUTH_ERROR"
        });
    }
};

/**
 * Optional authentication — attaches user if token is valid, continues either way.
 * Use for routes that work both authenticated and unauthenticated.
 */
/**
 * Admin-only guard. Must be used AFTER authenticateToken.
 * Checks `profiles.is_admin` flag (preferred) and falls back to a hard-coded
 * admin email allow-list to mirror the SQL is_admin() helper.
 */
const ADMIN_EMAILS = new Set(
    (process.env.ADMIN_EMAILS || 'info@aiwardrobe.club')
        .split(',').map(e => e.trim().toLowerCase()).filter(Boolean)
);

export const requireAdmin = async (req, res, next) => {
    try {
        if (!req.user?.id) {
            return res.status(401).json({ error: 'Authentication required', code: 'NO_TOKEN' });
        }
        if (req.user.email && ADMIN_EMAILS.has(req.user.email.toLowerCase())) {
            return next();
        }
        const { data, error } = await supabase
            .from('profiles')
            .select('is_admin')
            .eq('id', req.user.id)
            .maybeSingle();
        if (error) {
            logger.error('requireAdmin lookup error:', error.message);
            return res.status(500).json({ error: 'Admin check failed', code: 'ADMIN_CHECK_FAILED' });
        }
        if (!data?.is_admin) {
            return res.status(403).json({ error: 'Admin access required', code: 'NOT_ADMIN' });
        }
        next();
    } catch (err) {
        logger.error('requireAdmin error:', err.message);
        return res.status(500).json({ error: 'Admin check failed', code: 'ADMIN_CHECK_FAILED' });
    }
};

export const optionalAuth = async (req, res, next) => {
    const authHeader = req.headers["authorization"];
    const token = authHeader?.split(" ")[1];

    if (!token) {
        if (isLocalDevAuthBypassEnabled()) {
            attachBypassUser(req);
            return next();
        }
        req.user = null;
        return next();
    }

    try {
        const { data: { user } } = await supabase.auth.getUser(token);
        req.user = user
            ? { id: user.id, email: user.email, username: user.user_metadata?.username || '' }
            : null;
    } catch {
        req.user = null;
    }

    next();
};

