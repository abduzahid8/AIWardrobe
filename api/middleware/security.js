import AuditLog from "../models/AuditLog.js";
import User from "../models/user.js";

/**
 * Security Middleware
 * Provides account lockout, audit logging, and suspicious activity detection
 */

// Configuration
const MAX_LOGIN_ATTEMPTS = 5;
const LOCKOUT_DURATION_MINUTES = 30;
const IP_BLOCK_THRESHOLD = 10;
const IP_BLOCK_WINDOW_MINUTES = 60;

/**
 * Extract client IP from request
 */
export const getClientIP = (req) => {
    return req.headers['x-forwarded-for']?.split(',')[0]?.trim() ||
        req.headers['x-real-ip'] ||
        req.connection?.remoteAddress ||
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
 * Audit logging middleware
 * Use: app.use(auditLogger)
 */
export const auditLogger = (req, res, next) => {
    // Store request start time
    req.startTime = Date.now();

    // Capture response
    const originalEnd = res.end;
    res.end = function (...args) {
        const duration = Date.now() - req.startTime;

        // Only log significant actions (not every GET request)
        const shouldLog = req.method !== 'GET' ||
            req.path.includes('/login') ||
            req.path.includes('/register');

        if (shouldLog) {
            const logData = {
                userId: req.user?.id,
                userEmail: req.user?.email,
                action: determineAction(req, res),
                resource: req.path,
                ipAddress: getClientIP(req),
                userAgent: getUserAgent(req),
                success: res.statusCode < 400,
                details: {
                    method: req.method,
                    statusCode: res.statusCode,
                    durationMs: duration
                }
            };

            // Don't await - fire and forget
            AuditLog.log(logData).catch(() => { });
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

    if (path.includes('/login')) {
        return statusCode < 400 ? 'LOGIN_SUCCESS' : 'LOGIN_FAILED';
    }
    if (path.includes('/register')) {
        return 'REGISTER';
    }
    if (path.includes('/subscription') && method === 'POST') {
        return 'SUBSCRIPTION_CREATED';
    }
    if (path.includes('/subscription') && path.includes('cancel')) {
        return 'SUBSCRIPTION_CANCELLED';
    }
    if (path.includes('/ai') || path.includes('/stylist') || path.includes('/outfit')) {
        return 'AI_REQUEST';
    }
    if (path.includes('/password')) {
        return path.includes('reset') ? 'PASSWORD_RESET_REQUEST' : 'PASSWORD_CHANGE';
    }

    return 'API_REQUEST'; // Generic
}

/**
 * Account lockout middleware for login routes
 * Use: router.post('/login', checkAccountLock, ...)
 */
export const checkAccountLock = async (req, res, next) => {
    try {
        const { email } = req.body;

        if (!email) {
            return next();
        }

        // Check if user account is locked
        const user = await User.findOne({ email: email.toLowerCase() });

        if (user?.lockedUntil && user.lockedUntil > new Date()) {
            const remainingMinutes = Math.ceil((user.lockedUntil - new Date()) / 60000);

            // Log the blocked attempt
            await AuditLog.log({
                userId: user._id,
                userEmail: email,
                action: 'LOGIN_FAILED',
                resource: '/login',
                ipAddress: getClientIP(req),
                userAgent: getUserAgent(req),
                success: false,
                errorMessage: 'Account locked',
                details: { remainingMinutes }
            });

            return res.status(423).json({
                error: `Account is locked. Try again in ${remainingMinutes} minutes.`,
                code: 'ACCOUNT_LOCKED',
                lockedUntil: user.lockedUntil
            });
        }

        // Check for IP-based blocking
        const shouldBlock = await AuditLog.shouldBlockIP(
            getClientIP(req),
            IP_BLOCK_THRESHOLD,
            IP_BLOCK_WINDOW_MINUTES
        );

        if (shouldBlock) {
            await AuditLog.log({
                action: 'SUSPICIOUS_ACTIVITY',
                resource: '/login',
                ipAddress: getClientIP(req),
                userAgent: getUserAgent(req),
                success: false,
                errorMessage: 'IP blocked due to too many failed attempts'
            });

            return res.status(429).json({
                error: 'Too many failed login attempts from this IP. Please try again later.',
                code: 'IP_BLOCKED'
            });
        }

        next();
    } catch (error) {
        console.error('Account lock check error:', error);
        next(); // Don't block on errors
    }
};

/**
 * Handle failed login - increment counter and potentially lock account
 */
export const handleFailedLogin = async (user, req) => {
    if (!user) return;

    try {
        const failedAttempts = (user.failedLoginAttempts || 0) + 1;

        const updateData = {
            failedLoginAttempts: failedAttempts,
            lastFailedLogin: new Date()
        };

        // Lock account if too many failed attempts
        if (failedAttempts >= MAX_LOGIN_ATTEMPTS) {
            updateData.lockedUntil = new Date(Date.now() + LOCKOUT_DURATION_MINUTES * 60000);
            updateData.failedLoginAttempts = 0; // Reset counter

            await AuditLog.log({
                userId: user._id,
                userEmail: user.email,
                action: 'ACCOUNT_LOCKED',
                resource: '/login',
                ipAddress: getClientIP(req),
                userAgent: getUserAgent(req),
                success: true,
                details: { lockDurationMinutes: LOCKOUT_DURATION_MINUTES }
            });
        }

        await User.findByIdAndUpdate(user._id, updateData);

        return failedAttempts;
    } catch (error) {
        console.error('Failed to handle failed login:', error);
    }
};

/**
 * Handle successful login - reset counters
 */
export const handleSuccessfulLogin = async (user, req) => {
    if (!user) return;

    try {
        await User.findByIdAndUpdate(user._id, {
            failedLoginAttempts: 0,
            lockedUntil: null,
            lastLoginAt: new Date(),
            lastLoginIP: getClientIP(req)
        });

        await AuditLog.log({
            userId: user._id,
            userEmail: user.email,
            action: 'LOGIN_SUCCESS',
            resource: '/login',
            ipAddress: getClientIP(req),
            userAgent: getUserAgent(req),
            success: true
        });
    } catch (error) {
        console.error('Failed to handle successful login:', error);
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

export default {
    auditLogger,
    checkAccountLock,
    handleFailedLogin,
    handleSuccessfulLogin,
    getClientIP,
    getUserAgent,
    sanitizeHeaders
};
