import express from "express";
import Subscription from "../models/Subscription.js";
import Payment from "../models/Payment.js";
import User from "../models/user.js";
import { authenticateToken } from "../middleware/auth.js";
import { aiLimiter } from "../middleware/rateLimit.js";

const router = express.Router();

/**
 * GET /subscription/status
 * Get current user's subscription status
 */
router.get("/status", authenticateToken, async (req, res) => {
    try {
        const subscription = await Subscription.getActiveSubscription(req.user.id);

        if (!subscription) {
            return res.json({
                tier: 'free',
                status: 'none',
                hasActiveSubscription: false,
                features: {
                    maxUses: 5,
                    aiOutfits: 5,
                    wardrobeScans: 5,
                    tryOns: 3,
                    analytics: false,
                    prioritySupport: false,
                    unlimitedStorage: false,
                }
            });
        }

        const features = getFeaturesByTier(subscription.tier);

        res.json({
            tier: subscription.tier,
            status: subscription.status,
            hasActiveSubscription: subscription.isActive(),
            startDate: subscription.startDate,
            endDate: subscription.endDate,
            daysRemaining: subscription.getDaysRemaining(),
            autoRenew: subscription.autoRenew,
            platform: subscription.platform,
            features
        });
    } catch (error) {
        console.error("Subscription status error:", error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /subscription/verify-apple-receipt
 * Verify Apple App Store receipt and create/update subscription
 */
router.post("/verify-apple-receipt", authenticateToken, aiLimiter, async (req, res) => {
    try {
        const { receiptData, productId } = req.body;

        if (!receiptData) {
            return res.status(400).json({ error: "Receipt data is required" });
        }

        console.log(`🍎 Verifying Apple receipt for user ${req.user.id}`);

        // TODO: Implement actual Apple receipt validation
        // For now, simulating successful validation for development
        // In production, call Apple's verifyReceipt endpoint:
        // https://buy.itunes.apple.com/verifyReceipt (production)
        // https://sandbox.itunes.apple.com/verifyReceipt (sandbox)

        const isSandbox = process.env.NODE_ENV !== 'production';
        const appleVerifyUrl = isSandbox
            ? 'https://sandbox.itunes.apple.com/verifyReceipt'
            : 'https://buy.itunes.apple.com/verifyReceipt';

        // Simulated response structure (replace with actual Apple API call)
        const simulatedAppleResponse = {
            status: 0, // 0 = valid
            latest_receipt_info: [{
                product_id: productId || 'com.aiwardrobe.premium.monthly',
                original_transaction_id: `sim_${Date.now()}`,
                expires_date_ms: String(Date.now() + 30 * 24 * 60 * 60 * 1000), // 30 days
                transaction_id: `trans_${Date.now()}`
            }]
        };

        // In production, replace above with:
        // const response = await axios.post(appleVerifyUrl, {
        //     'receipt-data': receiptData,
        //     'password': process.env.APPLE_SHARED_SECRET,
        //     'exclude-old-transactions': true
        // });
        // const appleResponse = response.data;

        const appleResponse = simulatedAppleResponse;

        if (appleResponse.status !== 0) {
            return res.status(400).json({
                error: "Invalid receipt",
                appleStatus: appleResponse.status
            });
        }

        const latestReceipt = appleResponse.latest_receipt_info[0];
        const { product_id, original_transaction_id, expires_date_ms, transaction_id } = latestReceipt;

        // Determine tier from product ID
        const tier = product_id.includes('vip') ? 'vip' : 'premium';
        const price = tier === 'vip' ? 99.99 : 9.99;

        // Check for existing subscription
        let subscription = await Subscription.findOne({
            userId: req.user.id,
            appleOriginalTransactionId: original_transaction_id
        });

        if (subscription) {
            // Update existing subscription
            subscription.endDate = new Date(parseInt(expires_date_ms));
            subscription.lastReceiptData = receiptData;
            subscription.lastReceiptValidatedAt = new Date();
            subscription.status = 'active';
            await subscription.save();

            console.log(`✅ Updated existing Apple subscription for user ${req.user.id}`);
        } else {
            // Create new subscription
            subscription = new Subscription({
                userId: req.user.id,
                tier,
                status: 'active',
                platform: 'apple',
                startDate: new Date(),
                endDate: new Date(parseInt(expires_date_ms)),
                appleOriginalTransactionId: original_transaction_id,
                lastReceiptData: receiptData,
                lastReceiptValidatedAt: new Date(),
                price,
                currency: 'USD',
                productId: product_id
            });
            await subscription.save();

            console.log(`✅ Created new Apple subscription for user ${req.user.id}`);
        }

        // Record payment
        const payment = new Payment({
            userId: req.user.id,
            subscriptionId: subscription._id,
            amount: price,
            currency: 'USD',
            status: 'completed',
            type: 'subscription',
            platform: 'apple',
            appleTransactionId: transaction_id,
            appleOriginalTransactionId: original_transaction_id,
            productId: product_id,
            tier,
            receiptData,
            receiptValidated: true,
            receiptValidatedAt: new Date(),
            completedAt: new Date()
        });
        await payment.save();

        res.json({
            success: true,
            subscription: {
                tier: subscription.tier,
                status: subscription.status,
                endDate: subscription.endDate,
                daysRemaining: subscription.getDaysRemaining()
            }
        });
    } catch (error) {
        console.error("Apple receipt verification error:", error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /subscription/verify-google-receipt
 * Verify Google Play receipt and create/update subscription
 */
router.post("/verify-google-receipt", authenticateToken, aiLimiter, async (req, res) => {
    try {
        const { purchaseToken, productId, packageName } = req.body;

        if (!purchaseToken || !productId) {
            return res.status(400).json({ error: "Purchase token and product ID are required" });
        }

        console.log(`🤖 Verifying Google Play receipt for user ${req.user.id}`);

        // TODO: Implement actual Google Play receipt validation
        // In production, use Google Play Developer API:
        // https://developers.google.com/android-publisher/api-ref/rest/v3/purchases.subscriptions/get

        // Simulated response (replace with actual Google API call)
        const simulatedGoogleResponse = {
            expiryTimeMillis: String(Date.now() + 30 * 24 * 60 * 60 * 1000),
            orderId: `GPA.${Date.now()}`,
            paymentState: 1 // 1 = received
        };

        const googleResponse = simulatedGoogleResponse;

        // Determine tier from product ID
        const tier = productId.includes('vip') ? 'vip' : 'premium';
        const price = tier === 'vip' ? 99.99 : 9.99;

        // Check for existing subscription
        let subscription = await Subscription.findOne({
            userId: req.user.id,
            googlePurchaseToken: purchaseToken
        });

        if (subscription) {
            // Update existing
            subscription.endDate = new Date(parseInt(googleResponse.expiryTimeMillis));
            subscription.status = 'active';
            await subscription.save();
        } else {
            // Create new
            subscription = new Subscription({
                userId: req.user.id,
                tier,
                status: 'active',
                platform: 'google',
                startDate: new Date(),
                endDate: new Date(parseInt(googleResponse.expiryTimeMillis)),
                googlePurchaseToken: purchaseToken,
                price,
                currency: 'USD',
                productId
            });
            await subscription.save();
        }

        // Record payment
        const payment = new Payment({
            userId: req.user.id,
            subscriptionId: subscription._id,
            amount: price,
            currency: 'USD',
            status: 'completed',
            type: 'subscription',
            platform: 'google',
            googleOrderId: googleResponse.orderId,
            googlePurchaseToken: purchaseToken,
            productId,
            tier,
            receiptValidated: true,
            completedAt: new Date()
        });
        await payment.save();

        res.json({
            success: true,
            subscription: {
                tier: subscription.tier,
                status: subscription.status,
                endDate: subscription.endDate,
                daysRemaining: subscription.getDaysRemaining()
            }
        });
    } catch (error) {
        console.error("Google receipt verification error:", error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /subscription/cancel
 * Cancel subscription (disable auto-renew)
 */
router.post("/cancel", authenticateToken, async (req, res) => {
    try {
        const subscription = await Subscription.getActiveSubscription(req.user.id);

        if (!subscription) {
            return res.status(404).json({ error: "No active subscription found" });
        }

        subscription.autoRenew = false;
        subscription.cancelledAt = new Date();
        // Note: Subscription remains active until endDate
        await subscription.save();

        console.log(`🚫 Subscription cancelled for user ${req.user.id}`);

        res.json({
            success: true,
            message: "Subscription cancelled. You will retain access until the end of your billing period.",
            endDate: subscription.endDate,
            daysRemaining: subscription.getDaysRemaining()
        });
    } catch (error) {
        console.error("Subscription cancellation error:", error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /subscription/restore
 * Restore purchases (check for existing subscriptions)
 */
router.post("/restore", authenticateToken, async (req, res) => {
    try {
        const { platform, receiptData, purchaseToken } = req.body;

        console.log(`🔄 Restoring purchases for user ${req.user.id}`);

        // Find any existing active subscription
        const subscription = await Subscription.getActiveSubscription(req.user.id);

        if (subscription) {
            res.json({
                success: true,
                restored: true,
                subscription: {
                    tier: subscription.tier,
                    status: subscription.status,
                    endDate: subscription.endDate,
                    daysRemaining: subscription.getDaysRemaining()
                }
            });
        } else {
            res.json({
                success: true,
                restored: false,
                message: "No active subscription found to restore"
            });
        }
    } catch (error) {
        console.error("Restore purchases error:", error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * GET /subscription/history
 * Get payment history
 */
router.get("/history", authenticateToken, async (req, res) => {
    try {
        const payments = await Payment.getUserPaymentHistory(req.user.id, 50);

        res.json({
            payments: payments.map(p => ({
                id: p._id,
                amount: p.amount,
                currency: p.currency,
                status: p.status,
                type: p.type,
                platform: p.platform,
                tier: p.tier,
                createdAt: p.createdAt,
                completedAt: p.completedAt
            }))
        });
    } catch (error) {
        console.error("Payment history error:", error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /subscription/webhook/apple
 * Apple App Store Server Notifications (webhook)
 */
router.post("/webhook/apple", async (req, res) => {
    try {
        console.log("📱 Apple webhook received");

        // TODO: Verify webhook signature
        // TODO: Process notification type (CANCEL, DID_RENEW, etc.)

        const { notification_type, unified_receipt } = req.body;

        console.log(`   Type: ${notification_type}`);

        // Handle different notification types
        switch (notification_type) {
            case 'CANCEL':
            case 'DID_FAIL_TO_RENEW':
                // Find and update subscription
                if (unified_receipt?.latest_receipt_info?.[0]) {
                    const { original_transaction_id } = unified_receipt.latest_receipt_info[0];
                    await Subscription.findOneAndUpdate(
                        { appleOriginalTransactionId: original_transaction_id },
                        { autoRenew: false, status: 'cancelled' }
                    );
                }
                break;

            case 'DID_RENEW':
                // Extend subscription
                if (unified_receipt?.latest_receipt_info?.[0]) {
                    const { original_transaction_id, expires_date_ms } = unified_receipt.latest_receipt_info[0];
                    await Subscription.findOneAndUpdate(
                        { appleOriginalTransactionId: original_transaction_id },
                        {
                            endDate: new Date(parseInt(expires_date_ms)),
                            status: 'active'
                        }
                    );
                }
                break;
        }

        res.status(200).json({ success: true });
    } catch (error) {
        console.error("Apple webhook error:", error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * Helper: Get features by subscription tier
 */
function getFeaturesByTier(tier) {
    const features = {
        free: {
            maxUses: 5,
            aiOutfits: 5,
            wardrobeScans: 5,
            tryOns: 3,
            analytics: false,
            prioritySupport: false,
            unlimitedStorage: false,
        },
        premium: {
            maxUses: -1,
            aiOutfits: -1,
            wardrobeScans: -1,
            tryOns: 50,
            analytics: true,
            prioritySupport: false,
            unlimitedStorage: false,
        },
        vip: {
            maxUses: -1,
            aiOutfits: -1,
            wardrobeScans: -1,
            tryOns: -1,
            analytics: true,
            prioritySupport: true,
            unlimitedStorage: true,
        }
    };

    return features[tier] || features.free;
}

export default router;
