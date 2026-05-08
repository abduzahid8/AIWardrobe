import { createClient } from "@supabase/supabase-js";
import "dotenv/config";
import logger from "../utils/logger.js";

/**
 * Security Middleware — Supabase-backed
 *
 * Provides account lockout, audit logging, and suspicious activity detection.
 * All queries go to Supabase `profiles` table (replaces broken MongoDB User model).
 */

// Configuration
const MAX_LOGIN_ATTEMPTS = 5;
const LOCKOUT_DURATION_MINUTES = 30;
const IP_BLOCK_THRESHOLD = 10;

// Supabase admin client
const SUPABASE_URL = process.env.SUPABASE_URL;
const SUPABASE_SERVICE_ROLE_KEY = process.env.SUPABASE_SERVICE_ROLE_KEY;

let supabaseAdmin = null;
if (SUPABASE_URL && SUPABASE_SERVICE_ROLE_KEY) {
    supabaseAdmin = createClient(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, {
        auth: { autoRefreshToken: false, persistSession: false },
    });
}

// In-memory IP tracking for rate limiting (resets on server restart)
const ipFailureMap = new Map();

/**
 * Extract client IP from request
 */
export const getClientIP = (req) => {
    // Express resolves req.ip correctly when `app.set('trust proxy', 1)` is on.
    // Keep header fallbacks for non-Express callers / tests.
    return req.ip ||
        req.headers['x-forwarded-for']?.split(',')[0]?.trim() ||
        req.headers['x-real-ip'] ||
        req.socket?.remoteAddress ||
        'unknown';
};

/**
 * Extract user agent from request
 */
export const getUserAgent = (req) => {
    return req.headers['user-agent'] || 'unknown';
};

/**
 * Audit logging middleware — structured log output.
 * Logs significant actions (mutations, auth events) as JSON for log aggregation.
 */
export const auditLogger = (req, res, next) => {
    req.startTime = Date.now();

    const originalEnd = res.end;
    res.end = function (...args) {
        const duration = Date.now() - req.startTime;

        // Only log significant actions (not every GET request)
        const shouldLog = req.method !== 'GET' ||
            req.path.includes('/login') ||
            req.path.includes('/register');

        if (shouldLog) {
            const logEntry = {
                type: 'audit',
                timestamp: new Date().toISOString(),
                userId: req.user?.id || null,
                action: determineAction(req, res),
                method: req.method,
                path: req.path,
                statusCode: res.statusCode,
                durationMs: duration,
                ip: getClientIP(req),
                userAgent: getUserAgent(req).substring(0, 200),
                success: res.statusCode < 400,
            };

            // Structured JSON log — parseable by log aggregation services
            logger.info(JSON.stringify(logEntry));
        }

        originalEnd.apply(res, args);
    };

    next();
};

/**
 * Determine action type from request
 */
function determineAction(req, res) {
    const path = req.path.toLowerCase();
    const method = req.method;
    const statusCode = res.statusCode;

    if (path.includes('/login')) return statusCode < 400 ? 'LOGIN_SUCCESS' : 'LOGIN_FAILED';
    if (path.includes('/register')) return 'REGISTER';
    if (path.includes('/subscription') && method === 'POST') return 'SUBSCRIPTION_CREATED';
    if (path.includes('/subscription') && path.includes('cancel')) return 'SUBSCRIPTION_CANCELLED';
    if (path.includes('/ai') || path.includes('/gemini') || path.includes('/outfit')) return 'AI_REQUEST';
    if (path.includes('/password')) return path.includes('reset') ? 'PASSWORD_RESET_REQUEST' : 'PASSWORD_CHANGE';
    if (path.includes('/account') && method === 'DELETE') return 'ACCOUNT_DELETED';
    return 'API_REQUEST';
}

/**
 * Account lockout middleware for login routes.
 * Checks Supabase `profiles` table for lockout status.
 *
 * Usage: router.post('/login', checkAccountLock, ...)
 */
export const checkAccountLock = async (req, res, next) => {
    try {
        const { email } = req.body;
        if (!email || !supabaseAdmin) return next();

        // Check if user account is locked
        const { data: profile, error } = await supabaseAdmin
            .from('profiles')
            .select('id, locked_until, failed_login_attempts')
            .eq('email', email.toLowerCase())
            .maybeSingle();

        if (error) {
            logger.error('Account lock check error:', error);
            return next(); // Don't block on DB errors
        }

        if (profile?.locked_until && new Date(profile.locked_until) > new Date()) {
            const remainingMinutes = Math.ceil(
                (new Date(profile.locked_until).getTime() - Date.now()) / 60000
            );

            logger.info(JSON.stringify({
                type: 'security',
                action: 'LOGIN_BLOCKED_LOCKED',
                email,
                ip: getClientIP(req),
                remainingMinutes,
            }));

            return res.status(423).json({
                error: `Account is locked. Try again in ${remainingMinutes} minutes.`,
                code: 'ACCOUNT_LOCKED',
                lockedUntil: profile.locked_until,
            });
        }

        // Check for IP-based blocking (in-memory)
        const ip = getClientIP(req);
        const ipRecord = ipFailureMap.get(ip);
        if (ipRecord && ipRecord.count >= IP_BLOCK_THRESHOLD &&
            Date.now() - ipRecord.firstFailure < 60 * 60 * 1000) {

            logger.info(JSON.stringify({
                type: 'security',
                action: 'LOGIN_BLOCKED_IP',
                ip,
                failureCount: ipRecord.count,
            }));

            return res.status(429).json({
                error: 'Too many failed login attempts from this IP. Please try again later.',
                code: 'IP_BLOCKED',
            });
        }

        next();
    } catch (error) {
        logger.error('Account lock check error:', error);
        next(); // Don't block on errors
    }
};

/**
 * Handle failed login — increment counter and potentially lock account.
 */
export const handleFailedLogin = async (email, req) => {
    if (!email || !supabaseAdmin) return;

    try {
        // Track IP failures in memory
        const ip = getClientIP(req);
        const ipRecord = ipFailureMap.get(ip) || { count: 0, firstFailure: Date.now() };
        ipRecord.count++;
        ipFailureMap.set(ip, ipRecord);

        // Look up user in Supabase
        const { data: profile } = await supabaseAdmin
            .from('profiles')
            .select('id, failed_login_attempts')
            .eq('email', email.toLowerCase())
            .maybeSingle();

        if (!profile) return;

        const failedAttempts = (profile.failed_login_attempts || 0) + 1;
        const updates = {
            failed_login_attempts: failedAttempts,
            last_failed_login: new Date().toISOString(),
        };

        // Lock account if too many failed attempts
        if (failedAttempts >= MAX_LOGIN_ATTEMPTS) {
            updates.locked_until = new Date(Date.now() + LOCKOUT_DURATION_MINUTES * 60000).toISOString();
            updates.failed_login_attempts = 0;

            logger.info(JSON.stringify({
                type: 'security',
                action: 'ACCOUNT_LOCKED',
                userId: profile.id,
                lockDurationMinutes: LOCKOUT_DURATION_MINUTES,
                ip: getClientIP(req),
            }));
        }

        await supabaseAdmin
            .from('profiles')
            .update(updates)
            .eq('id', profile.id);

        return failedAttempts;
    } catch (error) {
        logger.error('Failed to handle failed login:', error);
    }
};

/**
 * Handle successful login — reset counters.
 */
export const handleSuccessfulLogin = async (email, req) => {
    if (!email || !supabaseAdmin) return;

    try {
        // Clear IP failure tracking
        const ip = getClientIP(req);
        ipFailureMap.delete(ip);

        const { data: profile } = await supabaseAdmin
            .from('profiles')
            .select('id')
            .eq('email', email.toLowerCase())
            .maybeSingle();

        if (!profile) return;

        await supabaseAdmin
            .from('profiles')
            .update({
                failed_login_attempts: 0,
                locked_until: null,
                last_login_at: new Date().toISOString(),
                last_login_ip: getClientIP(req),
            })
            .eq('id', profile.id);
    } catch (error) {
        logger.error('Failed to handle successful login:', error);
    }
};

/**
 * Sanitize sensitive headers before logging
 */
export const sanitizeHeaders = (headers) => {
    const sanitized = { ...headers };
    const sensitiveHeaders = ['authorization', 'cookie', 'x-api-key'];
    sensitiveHeaders.forEach(header => {
        if (sanitized[header]) {
            sanitized[header] = '[REDACTED]';
        }
    });
    return sanitized;
};

// Clean up stale IP records every hour
setInterval(() => {
    const cutoff = Date.now() - 60 * 60 * 1000;
    for (const [ip, record] of ipFailureMap.entries()) {
        if (record.firstFailure < cutoff) {
            ipFailureMap.delete(ip);
        }
    }
}, 60 * 60 * 1000);

export default {
    auditLogger,
    checkAccountLock,
    handleFailedLogin,
    handleSuccessfulLogin,
    getClientIP,
    getUserAgent,
    sanitizeHeaders,
};
