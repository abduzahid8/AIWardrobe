import mongoose from "mongoose";

/**
 * Subscription Model
 * Tracks user subscription status and billing information
 */
const subscriptionSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "User",
        required: true,
        index: true
    },

    // Subscription tier
    tier: {
        type: String,
        enum: ['free', 'premium', 'vip'],
        default: 'free',
        required: true
    },

    // Status
    status: {
        type: String,
        enum: ['active', 'cancelled', 'expired', 'pending', 'trial'],
        default: 'pending',
        required: true
    },

    // Platform
    platform: {
        type: String,
        enum: ['apple', 'google', 'stripe', 'manual'],
        required: true
    },

    // Dates
    startDate: {
        type: Date,
        required: true,
        default: Date.now
    },
    endDate: {
        type: Date,
        required: true
    },
    trialEndDate: Date,
    cancelledAt: Date,

    // Billing
    autoRenew: {
        type: Boolean,
        default: true
    },

    // Platform-specific IDs
    appleOriginalTransactionId: String,
    googlePurchaseToken: String,
    stripeSubscriptionId: String,
    stripeCustomerId: String,

    // Receipt data (for validation)
    lastReceiptData: String,
    lastReceiptValidatedAt: Date,

    // Price at time of purchase
    price: {
        type: Number,
        required: true
    },
    currency: {
        type: String,
        default: 'USD'
    },

    // Product info
    productId: {
        type: String,
        required: true
    },

    // Metadata
    createdAt: {
        type: Date,
        default: Date.now
    },
    updatedAt: {
        type: Date,
        default: Date.now
    }
}, {
    timestamps: true
});

// Compound index for efficient user subscription lookups
subscriptionSchema.index({ userId: 1, status: 1 });
subscriptionSchema.index({ endDate: 1 });
subscriptionSchema.index({ appleOriginalTransactionId: 1 });
subscriptionSchema.index({ stripeSubscriptionId: 1 });

/**
 * Check if subscription is currently active
 */
subscriptionSchema.methods.isActive = function () {
    return this.status === 'active' && this.endDate > new Date();
};

/**
 * Check if subscription is in trial period
 */
subscriptionSchema.methods.isInTrial = function () {
    return this.status === 'trial' && this.trialEndDate && this.trialEndDate > new Date();
};

/**
 * Get days remaining
 */
subscriptionSchema.methods.getDaysRemaining = function () {
    if (!this.isActive()) return 0;
    const now = new Date();
    const diff = this.endDate - now;
    return Math.ceil(diff / (1000 * 60 * 60 * 24));
};

/**
 * Static method to get user's active subscription
 */
subscriptionSchema.statics.getActiveSubscription = async function (userId) {
    return this.findOne({
        userId,
        status: { $in: ['active', 'trial'] },
        endDate: { $gt: new Date() }
    }).sort({ createdAt: -1 });
};

/**
 * Static method to check if user has premium access
 */
subscriptionSchema.statics.hasPremiumAccess = async function (userId) {
    const subscription = await this.getActiveSubscription(userId);
    return subscription && ['premium', 'vip'].includes(subscription.tier);
};

export default mongoose.model("Subscription", subscriptionSchema);
