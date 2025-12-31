import mongoose from "mongoose";

/**
 * AuditLog Model
 * Tracks security-relevant actions for compliance and debugging
 */
const auditLogSchema = new mongoose.Schema({
    // User info
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "User",
        index: true
    },
    userEmail: String,

    // Action details
    action: {
        type: String,
        required: true,
        enum: [
            'LOGIN_SUCCESS',
            'LOGIN_FAILED',
            'LOGOUT',
            'REGISTER',
            'PASSWORD_CHANGE',
            'PASSWORD_RESET_REQUEST',
            'PASSWORD_RESET_COMPLETE',
            'ACCOUNT_LOCKED',
            'ACCOUNT_UNLOCKED',
            'SUBSCRIPTION_CREATED',
            'SUBSCRIPTION_CANCELLED',
            'PAYMENT_COMPLETED',
            'PAYMENT_FAILED',
            'AI_REQUEST',
            'DATA_EXPORT',
            'DATA_DELETE',
            'SETTINGS_CHANGE',
            'API_KEY_GENERATED',
            'API_KEY_REVOKED',
            'SUSPICIOUS_ACTIVITY'
        ],
        index: true
    },

    // Resource info
    resource: String, // e.g., "/login", "/api/subscription"
    resourceId: String, // ID of affected resource

    // Request info
    ipAddress: {
        type: String,
        index: true
    },
    userAgent: String,
    origin: String,

    // Result
    success: {
        type: Boolean,
        default: true
    },
    errorMessage: String,

    // Additional context
    details: mongoose.Schema.Types.Mixed,

    // Timestamp
    timestamp: {
        type: Date,
        default: Date.now,
        index: true
    }
}, {
    timestamps: false // We use our own timestamp field
});

// Compound indexes for efficient queries
auditLogSchema.index({ userId: 1, timestamp: -1 });
auditLogSchema.index({ action: 1, timestamp: -1 });
auditLogSchema.index({ ipAddress: 1, action: 1, timestamp: -1 });

// TTL index - automatically delete logs older than 90 days
auditLogSchema.index({ timestamp: 1 }, { expireAfterSeconds: 90 * 24 * 60 * 60 });

/**
 * Static method to log an action
 */
auditLogSchema.statics.log = async function (data) {
    try {
        const log = new this(data);
        await log.save();
        return log;
    } catch (error) {
        console.error('Failed to create audit log:', error.message);
        // Don't throw - audit logging should not break the app
    }
};

/**
 * Get recent login attempts for an IP
 */
auditLogSchema.statics.getRecentLoginAttempts = async function (ipAddress, minutes = 60) {
    const since = new Date(Date.now() - minutes * 60 * 1000);
    return this.find({
        ipAddress,
        action: { $in: ['LOGIN_SUCCESS', 'LOGIN_FAILED'] },
        timestamp: { $gte: since }
    }).sort({ timestamp: -1 });
};

/**
 * Get failed login count for an IP
 */
auditLogSchema.statics.getFailedLoginCount = async function (ipAddress, minutes = 60) {
    const since = new Date(Date.now() - minutes * 60 * 1000);
    return this.countDocuments({
        ipAddress,
        action: 'LOGIN_FAILED',
        timestamp: { $gte: since }
    });
};

/**
 * Check if an IP should be blocked
 */
auditLogSchema.statics.shouldBlockIP = async function (ipAddress, maxAttempts = 10, windowMinutes = 60) {
    const failedCount = await this.getFailedLoginCount(ipAddress, windowMinutes);
    return failedCount >= maxAttempts;
};

/**
 * Get user activity summary
 */
auditLogSchema.statics.getUserActivity = async function (userId, days = 30) {
    const since = new Date(Date.now() - days * 24 * 60 * 60 * 1000);
    return this.find({
        userId,
        timestamp: { $gte: since }
    })
        .sort({ timestamp: -1 })
        .limit(100);
};

export default mongoose.model("AuditLog", auditLogSchema);
