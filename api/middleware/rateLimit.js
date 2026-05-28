import rateLimit, { ipKeyGenerator } from 'express-rate-limit';

/**
 * Rate Limiting Middleware
 *
 * NOTE on scaling: the default in-memory store is per-process. When the API
 * runs on more than one instance, swap `store` for `rate-limit-redis` backed
 * by a shared Redis (Upstash). The key generator below already prefers
 * authenticated user id, so per-user fairness survives behind a NAT.
 */

/**
 * Prefer authenticated user id, fall back to IPv6-safe client IP.
 * Uses ipKeyGenerator to correctly handle IPv6 addresses and prevent bypass.
 * Requires `app.set('trust proxy', 1)` so req.ip is the real client IP behind Render.
 */
const userOrIp = (req) => req.user?.id ? `u:${req.user.id}` : ipKeyGenerator(req);

/**
 * General API rate limiter
 */
export const apiLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 500,
    validate: { xForwardedForHeader: false },
    keyGenerator: userOrIp,
    message: {
        error: 'Too many requests, please try again after 15 minutes.',
        retryAfter: 900
    },
    standardHeaders: true,
    legacyHeaders: false,
});

/**
 * Strict rate limiter for authentication routes
 * 5 attempts per hour per IP (prevents brute force)
 */
export const authLimiter = rateLimit({
    windowMs: 60 * 60 * 1000, // 1 hour
    max: 5,
    validate: { xForwardedForHeader: false },
    message: {
        error: 'Too many login attempts from this IP, please try again after an hour.',
        retryAfter: 3600
    },
    standardHeaders: true,
    legacyHeaders: false,
    skipSuccessfulRequests: true, // Don't count successful logins
});

/**
 * Rate limiter for registration
 * 3 registrations per hour per IP
 */
export const registrationLimiter = rateLimit({
    windowMs: 60 * 60 * 1000, // 1 hour
    max: 3,
    validate: { xForwardedForHeader: false },
    message: {
        error: 'Too many accounts created from this IP, please try again after an hour.',
        retryAfter: 3600
    },
    standardHeaders: true,
    legacyHeaders: false,
});

/**
 * Rate limiter for AI-powered routes (expensive operations)
 * 20 requests per minute per user/IP
 */
export const aiLimiter = rateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 20,
    validate: { xForwardedForHeader: false },
    keyGenerator: userOrIp,
    message: {
        error: 'AI rate limit exceeded. Please wait a moment before trying again.',
        retryAfter: 60
    },
    standardHeaders: true,
    legacyHeaders: false,
});

/**
 * Rate limiter for file upload routes
 * 5 uploads per minute per IP
 */
export const uploadLimiter = rateLimit({
    windowMs: 60 * 1000, // 1 minute
    max: 5,
    validate: { xForwardedForHeader: false },
    message: {
        error: 'Too many uploads. Please wait a moment before trying again.',
        retryAfter: 60
    },
    standardHeaders: true,
    legacyHeaders: false,
});
