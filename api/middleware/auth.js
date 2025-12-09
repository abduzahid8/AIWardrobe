import jwt from "jsonwebtoken";
import "dotenv/config";

/**
 * JWT Secret from environment variable
 * CRITICAL: Set JWT_SECRET in production for security!
 */
const JWT_SECRET = process.env.JWT_SECRET || 'aiwardrobe-dev-secret-change-in-production';

if (!process.env.JWT_SECRET) {
    console.warn("⚠️ WARNING: JWT_SECRET not set, using default. Set this in production!");
}

/**
 * JWT Token expiration (7 days)
 */
export const JWT_EXPIRES_IN = '7d';

/**
 * Middleware to authenticate JWT tokens
 * Extracts user info and attaches to req.user
 */
export const authenticateToken = (req, res, next) => {
    const authHeader = req.headers["authorization"];
    const token = authHeader?.split(" ")[1];

    if (!token) {
        return res.status(401).json({
            error: "Authentication required",
            code: "NO_TOKEN"
        });
    }

    jwt.verify(token, JWT_SECRET || 'fallback-dev-secret', (err, decoded) => {
        if (err) {
            if (err.name === 'TokenExpiredError') {
                return res.status(401).json({
                    error: "Token has expired",
                    code: "TOKEN_EXPIRED"
                });
            }
            return res.status(403).json({
                error: "Invalid token",
                code: "INVALID_TOKEN"
            });
        }
        req.user = decoded;
        next();
    });
};

/**
 * Optional authentication - doesn't fail if no token
 * Useful for routes that work both authenticated and not
 */
export const optionalAuth = (req, res, next) => {
    const authHeader = req.headers["authorization"];
    const token = authHeader?.split(" ")[1];

    if (!token) {
        req.user = null;
        return next();
    }

    jwt.verify(token, JWT_SECRET || 'fallback-dev-secret', (err, decoded) => {
        req.user = err ? null : decoded;
        next();
    });
};

export { JWT_SECRET };

