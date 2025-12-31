import mongoose from "mongoose";

/**
 * Payment Model
 * Tracks payment transactions and history
 */
const paymentSchema = new mongoose.Schema({
    userId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "User",
        required: true,
        index: true
    },

    subscriptionId: {
        type: mongoose.Schema.Types.ObjectId,
        ref: "Subscription",
        index: true
    },

    // Transaction info
    amount: {
        type: Number,
        required: true
    },
    currency: {
        type: String,
        default: 'USD',
        required: true
    },

    // Status
    status: {
        type: String,
        enum: ['pending', 'completed', 'failed', 'refunded', 'disputed'],
        default: 'pending',
        required: true
    },

    // Payment type
    type: {
        type: String,
        enum: ['subscription', 'one_time', 'refund', 'upgrade', 'renewal'],
        required: true
    },

    // Platform
    platform: {
        type: String,
        enum: ['apple', 'google', 'stripe'],
        required: true
    },

    // Platform-specific transaction IDs
    appleTransactionId: String,
    appleOriginalTransactionId: String,
    googleOrderId: String,
    googlePurchaseToken: String,
    stripePaymentIntentId: String,
    stripeChargeId: String,

    // Product info
    productId: {
        type: String,
        required: true
    },
    productName: String,
    tier: {
        type: String,
        enum: ['premium', 'vip']
    },

    // Receipt data
    receiptData: String,
    receiptValidated: {
        type: Boolean,
        default: false
    },
    receiptValidatedAt: Date,

    // Metadata for debugging
    rawResponse: mongoose.Schema.Types.Mixed,
    errorDetails: String,

    // IP and device info (for fraud prevention)
    ipAddress: String,
    deviceInfo: String,

    // Timestamps
    completedAt: Date,
    refundedAt: Date,

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

// Indexes for efficient queries
paymentSchema.index({ userId: 1, createdAt: -1 });
paymentSchema.index({ status: 1, createdAt: -1 });
paymentSchema.index({ appleTransactionId: 1 });
paymentSchema.index({ stripePaymentIntentId: 1 });
paymentSchema.index({ googleOrderId: 1 });

/**
 * Mark payment as completed
 */
paymentSchema.methods.markCompleted = async function () {
    this.status = 'completed';
    this.completedAt = new Date();
    return this.save();
};

/**
 * Mark payment as refunded
 */
paymentSchema.methods.markRefunded = async function (reason) {
    this.status = 'refunded';
    this.refundedAt = new Date();
    this.errorDetails = reason;
    return this.save();
};

/**
 * Get user's payment history
 */
paymentSchema.statics.getUserPaymentHistory = async function (userId, limit = 20) {
    return this.find({ userId })
        .sort({ createdAt: -1 })
        .limit(limit)
        .populate('subscriptionId');
};

/**
 * Get total revenue for user
 */
paymentSchema.statics.getUserTotalRevenue = async function (userId) {
    const result = await this.aggregate([
        { $match: { userId: mongoose.Types.ObjectId(userId), status: 'completed' } },
        { $group: { _id: null, total: { $sum: '$amount' } } }
    ]);
    return result[0]?.total || 0;
};

export default mongoose.model("Payment", paymentSchema);
