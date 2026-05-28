import express from "express";
import { supabase } from "../lib/supabase.js";
import { authenticateToken } from "../middleware/auth.js";
import { aiLimiter } from "../middleware/rateLimit.js";

import logger from '../utils/logger.js';
const router = express.Router();

/**
 * GET /subscription/status
 * Get current user's subscription status
 */
router.get("/status", authenticateToken, async (req, res) => {
    try {
        const userId = req.user.id;

        // Fetch active subscription
        const now = new Date().toISOString();
        const { data: subscription, error } = await supabase
            .from('subscriptions')
            .select('*')
            .eq('user_id', userId)
            .eq('status', 'active')
            .gte('end_date', now)
            .order('created_at', { ascending: false })
            .limit(1)
            .maybeSingle();

        if (error || !subscription) {
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

        const endDateObj = new Date(subscription.end_date);
        const daysRemaining = Math.max(0, Math.ceil((endDateObj.getTime() - Date.now()) / (1000 * 60 * 60 * 24)));

        res.json({
            tier: subscription.tier,
            status: subscription.status,
            hasActiveSubscription: true,
            startDate: subscription.start_date,
            endDate: subscription.end_date,
            daysRemaining,
            autoRenew: subscription.auto_renew ?? true,
            platform: subscription.platform,
            features
        });
    } catch (error) {
        logger.error("Subscription status error:", error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /subscription/verify-apple-receipt
 * Verify Apple App Store receipt and create/update subscription
 */
router.post("/verify-apple-receipt", authenticateToken, aiLimiter, async (req, res) => {
    // DISABLED: Apple receipt validation is handled exclusively by RevenueCat
    // (supabase/functions/revenuecat-webhook). This legacy stub previously
    // returned a fabricated successful validation, which is unsafe in production.
    return res.status(410).json({
        error: 'Endpoint retired. Receipt validation is handled by RevenueCat webhook.'
    });
    /* eslint-disable no-unreachable */
    try {
        const { receiptData, productId } = req.body;

        if (!receiptData) {
            return res.status(400).json({ error: "Receipt data is required" });
        }

        logger.info(`🍎 Verifying Apple receipt for user ${req.user.id}`);

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
        let { data: subscription, error } = await supabase
            .from('subscriptions')
            .select('*')
            .eq('user_id', req.user.id)
            .eq('apple_original_transaction_id', original_transaction_id)
            .maybeSingle();

        if (subscription) {
            // Update existing subscription
            const { data: updatedSub } = await supabase
                .from('subscriptions')
                .update({
                    end_date: new Date(parseInt(expires_date_ms)).toISOString(),
                    last_receipt_data: receiptData,
                    last_receipt_validated_at: new Date().toISOString(),
                    status: 'active'
                })
                .eq('id', subscription.id)
                .select()
                .single();

            subscription = updatedSub;
            logger.info(`✅ Updated existing Apple subscription for user ${req.user.id}`);
        } else {
            // Create new subscription
            const { data: newSub, error: insertError } = await supabase
                .from('subscriptions')
                .insert([{
                    user_id: req.user.id,
                    tier,
                    status: 'active',
                    platform: 'apple',
                    start_date: new Date().toISOString(),
                    end_date: new Date(parseInt(expires_date_ms)).toISOString(),
                    apple_original_transaction_id: original_transaction_id,
                    last_receipt_data: receiptData,
                    last_receipt_validated_at: new Date().toISOString(),
                    price,
                    currency: 'USD',
                    product_id: product_id
                }])
                .select()
                .single();

            if (insertError) throw insertError;
            subscription = newSub;
            logger.info(`✅ Created new Apple subscription for user ${req.user.id}`);
        }

        // Record payment
        await supabase.from('payments').insert([{
            user_id: req.user.id,
            subscription_id: subscription.id,
            amount: price,
            currency: 'USD',
            status: 'completed',
            type: 'subscription',
            platform: 'apple',
            apple_transaction_id: transaction_id,
            apple_original_transaction_id: original_transaction_id,
            product_id: product_id,
            tier,
            receipt_data: receiptData,
            receipt_validated: true,
            receipt_validated_at: new Date().toISOString(),
            completed_at: new Date().toISOString()
        }]);

        const endDateObj = new Date(subscription.end_date);
        const daysRemaining = Math.max(0, Math.ceil((endDateObj.getTime() - Date.now()) / (1000 * 60 * 60 * 24)));

        res.json({
            success: true,
            subscription: {
                tier: subscription.tier,
                status: subscription.status,
                endDate: subscription.end_date,
                daysRemaining
            }
        });
    } catch (error) {
        logger.error("Apple receipt verification error:", error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /subscription/verify-google-receipt
 * Verify Google Play receipt and create/update subscription
 */
router.post("/verify-google-receipt", authenticateToken, aiLimiter, async (req, res) => {
    // DISABLED: See note on /verify-apple-receipt. Receipt validation is
    // delegated to RevenueCat. The original simulated implementation below
    // is retained only for reference and is unreachable.
    return res.status(410).json({
        error: 'Endpoint retired. Receipt validation is handled by RevenueCat webhook.'
    });
    /* eslint-disable no-unreachable */
    try {
        const { purchaseToken, productId, packageName } = req.body;

        if (!purchaseToken || !productId) {
            return res.status(400).json({ error: "Purchase token and product ID are required" });
        }

        logger.info(`🤖 Verifying Google Play receipt for user ${req.user.id}`);

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
        let { data: subscription } = await supabase
            .from('subscriptions')
            .select('*')
            .eq('user_id', req.user.id)
            .eq('google_purchase_token', purchaseToken)
            .maybeSingle();

        if (subscription) {
            // Update existing
            const { data: updatedSub } = await supabase
                .from('subscriptions')
                .update({
                    end_date: new Date(parseInt(googleResponse.expiryTimeMillis)).toISOString(),
                    status: 'active'
                })
                .eq('id', subscription.id)
                .select().single();
            subscription = updatedSub;
        } else {
            // Create new
            const { data: newSub, error: insertError } = await supabase
                .from('subscriptions')
                .insert([{
                    user_id: req.user.id,
                    tier,
                    status: 'active',
                    platform: 'google',
                    start_date: new Date().toISOString(),
                    end_date: new Date(parseInt(googleResponse.expiryTimeMillis)).toISOString(),
                    google_purchase_token: purchaseToken,
                    price,
                    currency: 'USD',
                    product_id: productId
                }])
                .select().single();

            if (insertError) throw insertError;
            subscription = newSub;
        }

        // Record payment
        await supabase.from('payments').insert([{
            user_id: req.user.id,
            subscription_id: subscription.id,
            amount: price,
            currency: 'USD',
            status: 'completed',
            type: 'subscription',
            platform: 'google',
            google_order_id: googleResponse.orderId,
            google_purchase_token: purchaseToken,
            product_id: productId,
            tier,
            receipt_validated: true,
            completed_at: new Date().toISOString()
        }]);

        const endDateObj = new Date(subscription.end_date);
        const daysRemaining = Math.max(0, Math.ceil((endDateObj.getTime() - Date.now()) / (1000 * 60 * 60 * 24)));

        res.json({
            success: true,
            subscription: {
                tier: subscription.tier,
                status: subscription.status,
                endDate: subscription.end_date,
                daysRemaining
            }
        });
    } catch (error) {
        logger.error("Google receipt verification error:", error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /subscription/cancel
 * Cancel subscription (disable auto-renew)
 */
router.post("/cancel", authenticateToken, async (req, res) => {
    try {
        const userId = req.user.id;
        const now = new Date().toISOString();
        const { data: subscription } = await supabase
            .from('subscriptions')
            .select('*')
            .eq('user_id', userId)
            .eq('status', 'active')
            .gte('end_date', now)
            .order('created_at', { ascending: false })
            .limit(1)
            .maybeSingle();

        if (!subscription) {
            return res.status(404).json({ error: "No active subscription found" });
        }

        await supabase
            .from('subscriptions')
            .update({
                auto_renew: false,
                cancelled_at: new Date().toISOString()
            })
            .eq('id', subscription.id);

        logger.info(`🚫 Subscription cancelled for user ${userId}`);

        const endDateObj = new Date(subscription.end_date);
        const daysRemaining = Math.max(0, Math.ceil((endDateObj.getTime() - Date.now()) / (1000 * 60 * 60 * 24)));

        res.json({
            success: true,
            message: "Subscription cancelled. You will retain access until the end of your billing period.",
            endDate: subscription.end_date,
            daysRemaining
        });
    } catch (error) {
        logger.error("Subscription cancellation error:", error);
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

        logger.info(`🔄 Restoring purchases for user ${req.user.id}`);

        const userId = req.user.id;
        const now = new Date().toISOString();
        const { data: subscription } = await supabase
            .from('subscriptions')
            .select('*')
            .eq('user_id', userId)
            .eq('status', 'active')
            .gte('end_date', now)
            .order('created_at', { ascending: false })
            .limit(1)
            .maybeSingle();

        if (subscription) {
            const endDateObj = new Date(subscription.end_date);
            const daysRemaining = Math.max(0, Math.ceil((endDateObj.getTime() - Date.now()) / (1000 * 60 * 60 * 24)));

            res.json({
                success: true,
                restored: true,
                subscription: {
                    tier: subscription.tier,
                    status: subscription.status,
                    endDate: subscription.end_date,
                    daysRemaining
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
        logger.error("Restore purchases error:", error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * GET /subscription/history
 * Get payment history
 */
router.get("/history", authenticateToken, async (req, res) => {
    try {
        const userId = req.user.id;
        const { data: payments, error } = await supabase
            .from('payments')
            .select('*')
            .eq('user_id', userId)
            .order('created_at', { ascending: false })
            .limit(50);

        if (error) throw error;

        res.json({
            payments: (payments || []).map(p => ({
                id: p.id,
                amount: p.amount,
                currency: p.currency,
                status: p.status,
                type: p.type,
                platform: p.platform,
                tier: p.tier,
                createdAt: p.created_at,
                completedAt: p.completed_at
            }))
        });
    } catch (error) {
        logger.error("Payment history error:", error);
        res.status(500).json({ error: error.message });
    }
});

/**
 * POST /subscription/webhook/apple
 * Apple App Store Server Notifications (webhook)
 */
router.post("/webhook/apple", async (req, res) => {
    // DISABLED: Apple App Store Server Notifications are routed through
    // RevenueCat → supabase/functions/revenuecat-webhook. This legacy
    // unauthenticated stub is closed to prevent spoofed subscription writes.
    return res.status(410).json({ error: 'Endpoint retired.' });
    /* eslint-disable no-unreachable */
    try {
        logger.info("📱 Apple webhook received");

        // TODO: Verify webhook signature
        // TODO: Process notification type (CANCEL, DID_RENEW, etc.)

        const { notification_type, unified_receipt } = req.body;

        logger.info(`   Type: ${notification_type}`);

        // Handle different notification types
        switch (notification_type) {
            case 'CANCEL':
            case 'DID_FAIL_TO_RENEW':
                // Find and update subscription
                if (unified_receipt?.latest_receipt_info?.[0]) {
                    const { original_transaction_id } = unified_receipt.latest_receipt_info[0];
                    await supabase
                        .from('subscriptions')
                        .update({ auto_renew: false, status: 'cancelled' })
                        .eq('apple_original_transaction_id', original_transaction_id);
                }
                break;

            case 'DID_RENEW':
                // Extend subscription
                if (unified_receipt?.latest_receipt_info?.[0]) {
                    const { original_transaction_id, expires_date_ms } = unified_receipt.latest_receipt_info[0];
                    await supabase
                        .from('subscriptions')
                        .update({
                            end_date: new Date(parseInt(expires_date_ms)).toISOString(),
                            status: 'active'
                        })
                        .eq('apple_original_transaction_id', original_transaction_id);
                }
                break;
        }

        res.status(200).json({ success: true });
    } catch (error) {
        logger.error("Apple webhook error:", error);
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
