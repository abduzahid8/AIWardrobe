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

const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

if (!SUPABASE_URL || !SUPABASE_SERVICE_ROLE_KEY) {
    logger.error("❌ FATAL: SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set in .env");
    logger.error("   Get the service role key from: Supabase Dashboard → Settings → API");
    process.exit(1);
}

// Use service role client to validate tokens server-side
const supabaseAdmin = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
    auth: { autoRefreshToken: false, persistSession: false }
});

/**
 * Middleware to authenticate Supabase JWT tokens.
 * Attaches the Supabase user object to req.user on success.
 */
export const authenticateToken = async (req, res, next) => {
    const authHeader = req.headers["authorization"];
    const token = authHeader?.split(" ")[1];

    if (!token) {
        return res.status(401).json({
            error: "Authentication required",
            code: "NO_TOKEN"
        });
    }

    try {
        const { data: { user }, error } = await supabaseAdmin.auth.getUser(token);

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
export const optionalAuth = async (req, res, next) => {
    const authHeader = req.headers["authorization"];
    const token = authHeader?.split(" ")[1];

    if (!token) {
        req.user = null;
        return next();
    }

    try {
        const { data: { user } } = await supabaseAdmin.auth.getUser(token);
        req.user = user
            ? { id: user.id, email: user.email, username: user.user_metadata?.username || '' }
            : null;
    } catch {
        req.user = null;
    }

    next();
};

