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
export const optionalAuth = async (req, res, next) => {
    const authHeader = req.headers["authorization"];
    const token = authHeader?.split(" ")[1];

    if (!token) {
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

