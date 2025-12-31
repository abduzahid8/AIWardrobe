import express from "express";
import jwt from "jsonwebtoken";
import User from "../models/user.js";
import { authenticateToken, JWT_SECRET, JWT_EXPIRES_IN } from "../middleware/auth.js";
import { validateRegistration, validateLogin } from "../middleware/validators.js";
import { authLimiter, registrationLimiter } from "../middleware/rateLimit.js";
import {
    checkAccountLock,
    handleFailedLogin,
    handleSuccessfulLogin,
    getClientIP
} from "../middleware/security.js";

const router = express.Router();

/**
 * POST /register
 * Register a new user
 * 
 * Rate limited: 3 attempts per hour
 * Validated: email, password strength, username format
 */
router.post("/register", registrationLimiter, validateRegistration, async (req, res) => {
    try {
        const { email, password, username, gender, profileImage } = req.body;
        console.log("📧 Registration attempt for:", email);

        // Check for existing email
        const existingUser = await User.findOne({ email: email.toLowerCase() });
        if (existingUser) {
            return res.status(400).json({
                error: "Email already exists",
                code: "EMAIL_EXISTS"
            });
        }

        // Check for existing username
        const existingUsername = await User.findOne({
            username: { $regex: new RegExp(`^${username}$`, 'i') }
        });
        if (existingUsername) {
            return res.status(400).json({
                error: "Username already exists",
                code: "USERNAME_EXISTS"
            });
        }

        // Create user (password hashing should be in User model pre-save hook)
        const user = new User({
            email: email.toLowerCase(),
            password,
            username,
            gender: gender || 'prefer_not_to_say',
            profileImage,
            outfits: [],
        });

        console.log("👤 Creating user:", user.username);

        await user.save();

        // Generate token with expiration
        const token = jwt.sign(
            { id: user._id, username: user.username },
            JWT_SECRET,
            { expiresIn: JWT_EXPIRES_IN }
        );

        res.status(201).json({
            token,
            user: {
                id: user._id,
                email: user.email,
                username: user.username
            }
        });
    } catch (error) {
        console.error("Registration error:", error.message);
        res.status(500).json({
            error: "Registration failed. Please try again.",
            code: "REGISTRATION_ERROR"
        });
    }
});

/**
 * POST /login
 * Authenticate user and return JWT token
 * 
 * Rate limited: 5 attempts per hour
 * Validated: email format, password required
 * Security: Account lockout after 5 failed attempts
 */
router.post("/login", authLimiter, checkAccountLock, validateLogin, async (req, res) => {
    try {
        const { email, password } = req.body;
        console.log("🔐 Login attempt for:", email);

        const user = await User.findOne({ email: email.toLowerCase() });

        // Check credentials
        if (!user || !(await user.comparePassword(password))) {
            // Track failed login
            if (user) {
                const attempts = await handleFailedLogin(user, req);
                console.log(`❌ Failed login for ${email}, attempt ${attempts}`);
            }

            // Use same error message for both cases to prevent user enumeration
            return res.status(401).json({
                error: "Invalid email or password",
                code: "INVALID_CREDENTIALS"
            });
        }

        // Check if account is locked
        if (user.isLocked()) {
            const remainingMinutes = Math.ceil((user.lockedUntil - new Date()) / 60000);
            return res.status(423).json({
                error: `Account is locked. Try again in ${remainingMinutes} minutes.`,
                code: "ACCOUNT_LOCKED"
            });
        }

        // Track successful login
        await handleSuccessfulLogin(user, req);
        console.log(`✅ Successful login for ${email}`);

        // Generate token with expiration
        const token = jwt.sign(
            { id: user._id, username: user.username },
            JWT_SECRET,
            { expiresIn: JWT_EXPIRES_IN }
        );

        res.json({
            token,
            user: {
                id: user._id,
                email: user.email,
                username: user.username,
                subscriptionTier: user.subscriptionTier || 'free'
            }
        });
    } catch (err) {
        console.error("Login error:", err.message);
        res.status(500).json({
            error: "Login failed. Please try again.",
            code: "LOGIN_ERROR"
        });
    }
});

/**
 * GET /me
 * Get current authenticated user
 */
router.get("/me", authenticateToken, async (req, res) => {
    try {
        const user = await User.findById(req.user.id).select("-password");
        if (!user) {
            return res.status(404).json({
                error: "User not found",
                code: "USER_NOT_FOUND"
            });
        }
        res.json(user);
    } catch (err) {
        console.error("Get user error:", err.message);
        res.status(500).json({
            error: "Failed to get user data",
            code: "FETCH_ERROR"
        });
    }
});

/**
 * POST /refresh-token
 * Refresh an expiring token
 */
router.post("/refresh-token", authenticateToken, async (req, res) => {
    try {
        // Generate new token
        const token = jwt.sign(
            { id: req.user.id, username: req.user.username },
            JWT_SECRET,
            { expiresIn: JWT_EXPIRES_IN }
        );

        res.json({ token });
    } catch (err) {
        console.error("Token refresh error:", err.message);
        res.status(500).json({
            error: "Failed to refresh token",
            code: "REFRESH_ERROR"
        });
    }
});

export default router;

