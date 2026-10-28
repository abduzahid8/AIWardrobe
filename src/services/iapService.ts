import readinessLogger from '../utils/logger';
/**
 * IAP Service — In-App Purchase integration via RevenueCat.
 *
 * RevenueCat handles:
 * - Apple App Store + Google Play purchase flow
 * - Server-side receipt validation (automatic)
 * - Subscription status tracking
 * - Webhooks for renewal/cancellation
 *
 * Setup:
 * 1. Create account at https://app.revenuecat.com
 * 2. Configure products in App Store Connect / Google Play Console
 * 3. Add API key to .env: EXPO_PUBLIC_REVENUECAT_API_KEY=your-key
 *
 * Products:
 * - 2.99 ($2.99/month)
 * - com.aiwardrobe.premium.monthly ($9.99/month)
 * - com.aiwardrobe.premium.yearly ($99.99/year)
 */
import { Platform } from 'react-native';
import useSubscriptionStore, { SubscriptionTier } from '../../store/subscriptionStore';
import analyticsService from './analyticsService';
import crashReporting from './crashReporting';
import Config from '../config/env';
import { supabase } from '../../lib/supabase';
import { authEvents } from '../../store/authEvents';
import { withRequestDeadline } from '../lib/requestDeadline';

export type ProductId = string;

const TIER_BY_PRODUCT_ID: Record<string, SubscriptionTier> = {
    '2.99': 'lite',
    'com.aiwardrobe.lite.monthly': 'lite',
    'com.aiwardrobe.premium.monthly': 'premium',
    'com.aiwardrobe.premium.yearly': 'vip',
};

function getDynamicTier(productId: string): SubscriptionTier {
    if (TIER_BY_PRODUCT_ID[productId]) return TIER_BY_PRODUCT_ID[productId];

    const lowerId = String(productId || '').toLowerCase();
    if (lowerId.includes('yearly') || lowerId.includes('annual')) return 'vip';
    if (lowerId.includes('lite')) return 'lite';
    if (lowerId.includes('premium') || lowerId.includes('pro') || lowerId.includes('monthly')) return 'premium';

    return 'free';
}

/**
 * Known App Store / Play Store product IDs to try even when RevenueCat
 * "Offerings" are empty or misconfigured. Keeps the paywall functional
 * as long as the products exist in the native store.
 */
const KNOWN_PRODUCT_IDS: string[] = [
    '2.99',
    'com.aiwardrobe.lite.monthly',
    'com.aiwardrobe.premium.monthly',
    'com.aiwardrobe.premium.yearly',
];

const ANALYTICS_PRICE_BY_TIER: Record<SubscriptionTier, number> = {
    free: 0,
    lite: 2.99,
    premium: 9.99,
    vip: 99.99,
};

/**
 * Convert verbose RevenueCat / StoreKit error text into a short,
 * user-friendly message safe to show inside an Alert.
 * Strips URLs and long diagnostic strings.
 */
function normalizeIapError(rawMessage: string): string {
    const msg = String(rawMessage || '').toLowerCase();

    if (msg.includes('no app store products registered') || msg.includes('offerings-empty')) {
        return 'Subscriptions are not available right now. Please try again later.';
    }
    if (msg.includes('invalid api key')) {
        return 'Subscriptions are not available in this build. Please update the app.';
    }
    if (msg.includes('network') || msg.includes('connection')) {
        return 'Network error. Please check your connection and try again.';
    }
    if (msg.includes('store problem') || msg.includes('storekit')) {
        return 'The App Store is temporarily unavailable. Please try again later.';
    }
    if (msg.includes('cancel')) {
        return 'Purchase was cancelled.';
    }

    // Fallback: strip URLs and truncate long text
    const cleaned = String(rawMessage)
        .replace(/https?:\/\/\S+/g, '')
        .replace(/more information:?/gi, '')
        .trim();

    if (!cleaned) {
        return 'Something went wrong. Please try again.';
    }

    return cleaned.length > 140 ? cleaned.slice(0, 140) + '…' : cleaned;
}

// RevenueCat lazy import — loaded when API key is configured
let Purchases: any = null;

interface PurchaseResult {
    success: boolean;
    productId?: string;
    transactionId?: string;
    receipt?: string;
    error?: string;
    verificationPending?: boolean;
    /** StoreKit still owns an unresolved native sheet or transaction request. */
    checkoutPending?: boolean;
}

// StoreKit can wait for Apple account approval without resolving its promise.
// Bound the UI wait but keep the native operation locked until Apple finishes.
export const NATIVE_STORE_PURCHASE_TIMEOUT_MS = 120_000;
const NATIVE_STORE_PURCHASE_PENDING = 'The App Store is still processing this purchase. Finish or cancel its prompt, then try Restore Purchases before buying again.';

class IAPService {
    private isInitialized = false;
    private isRevenueCatAvailable = false;
    /** Supabase user ID passed to RevenueCat via identify(). null = not yet identified. */
    private identifiedUserId: string | null = null;
    private serverSyncPromise: Promise<boolean> | null = null;
    private accountRevision = 0;
    private pendingNativeStoreOperation: Promise<unknown> | null = null;

    private async waitForNativeStoreOperation<T>(operation: () => Promise<T>): Promise<{ completed: true; value: T } | { completed: false }> {
        if (this.pendingNativeStoreOperation) return { completed: false };

        const pending = Promise.resolve().then(operation);
        this.pendingNativeStoreOperation = pending;
        const finish = () => {
            if (this.pendingNativeStoreOperation === pending) this.pendingNativeStoreOperation = null;
        };
        void pending.then(finish, finish);

        let timer: ReturnType<typeof setTimeout> | undefined;
        const settled = pending.then(
            value => ({ kind: 'completed' as const, value }),
            error => ({ kind: 'failed' as const, error }),
        );
        const deadline = new Promise<{ kind: 'pending' }>(resolve => {
            timer = setTimeout(() => resolve({ kind: 'pending' }), NATIVE_STORE_PURCHASE_TIMEOUT_MS);
        });
        try {
            const result = await Promise.race([settled, deadline]);
            if (result.kind === 'pending') return { completed: false };
            if (result.kind === 'failed') throw result.error;
            return { completed: true, value: result.value };
        } finally {
            if (timer) clearTimeout(timer);
        }
    }

    /**
     * Initialize IAP connection. Call once on app start.
     */
    async initialize(): Promise<void> {
        if (this.isInitialized) return;

        const apiKey = Config.revenueCat.apiKey;
        const keyKind = !apiKey
            ? 'missing'
            : apiKey.startsWith('appl_')
                ? 'production'
                : apiKey.startsWith('goog_')
                    ? 'production-android'
                    : apiKey.startsWith('test_')
                        ? 'test-store'
                        : 'unknown';
        readinessLogger.debug(String('[IAP] initialize'), [{ keyKind, platform: Platform.OS }]);

        if (!apiKey || apiKey === 'your-revenuecat-api-key') {
            console.warn('[IAP] Purchases unavailable: no RevenueCat API key configured for this platform');
            this.isInitialized = true;
            return;
        }

        try {
            // Dynamic import to avoid crash when SDK isn't installed
            const RC = require('react-native-purchases');
            Purchases = RC.default || RC.Purchases;

            if (Platform.OS === 'ios' || Platform.OS === 'android') {
                await Purchases.configure({ apiKey });
                this.isRevenueCatAvailable = true;
                readinessLogger.debug(String('[IAP] RevenueCat configured successfully'), []);

                // Listen for customer info changes (renewal, cancellation, etc.)
                Purchases.addCustomerInfoUpdateListener((info: any) => {
                    this.syncSubscriptionStatus(info);
                });
            }

            this.isInitialized = true;
        } catch (error: any) {
            console.warn('[IAP] RevenueCat SDK not available — using mock mode', error?.message || error);
            crashReporting.logBreadcrumb('IAP: RevenueCat not available, mock mode active');
            this.isInitialized = true;
        }
    }

    private async getAvailablePackages(): Promise<any[]> {
        if (!this.isInitialized) {
            await this.initialize();
        }

        if (!this.isRevenueCatAvailable) {
            return [];
        }

        try {
            const offerings = await Purchases.getOfferings();
            const currentPackages = offerings.current?.availablePackages || [];
            const allOfferingsCount = Object.keys(offerings.all || {}).length;
            readinessLogger.debug(String('[IAP] offerings fetched'), [{
                currentId: offerings.current?.identifier || null,
                currentPackagesCount: currentPackages.length,
                allOfferingsCount,
            }]);
            if (currentPackages.length > 0) {
                return currentPackages;
            }

            const allOfferings = Object.values(offerings.all || {}) as Array<{ availablePackages?: any[] }>;
            const seenProductIds = new Set<string>();

            return allOfferings
                .flatMap((offering) => offering.availablePackages || [])
                .filter((pkg: any) => {
                    const productId = pkg?.product?.identifier;
                    if (!productId || seenProductIds.has(productId)) {
                        return false;
                    }
                    seenProductIds.add(productId);
                    return true;
                });
        } catch (error: any) {
            console.warn('[IAP] getOfferings failed', error?.message || error);
            return [];
        }
    }

    /**
     * Get available products with pricing from the store.
     */
    async getProducts(): Promise<{
        id: string;
        title: string;
        price: string;
        packageType?: string;
        /** Introductory offer price string (e.g. "Free") if configured in App Store Connect. */
        introductoryPrice?: string;
        /** Introductory offer period description (e.g. "7 days"). */
        introductoryPeriod?: string;
        /** True when the product has an active introductory offer (free trial). */
        hasFreeTrial?: boolean;
    }[]> {
        if (!this.isInitialized) {
            await this.initialize();
        }

        if (!this.isRevenueCatAvailable) {
            // Mock products help local development, but must never make a
            // production paywall advertise plans that cannot be purchased.
            if (__DEV__) {
                return [
                    { id: '2.99', title: 'Lite Monthly', price: '$2.99' },
                    { id: 'com.aiwardrobe.premium.monthly', title: 'Premium Monthly', price: '$9.99' },
                    { id: 'com.aiwardrobe.premium.yearly', title: 'Premium Yearly', price: '$99.99' },
                ];
            }
            return [];
        }

        try {
            // Preferred path: RevenueCat offerings (has packages + pricing).
            const packages = await this.getAvailablePackages();
            if (packages.length) {
                return packages.map((pkg: any) => {
                    const product = pkg.product || pkg;
                    const introPrice = product.introPrice;
                    return {
                        id: product.identifier,
                        title: product.title,
                        price: product.priceString,
                        packageType: pkg.packageType,
                        introductoryPrice: introPrice?.priceString || undefined,
                        introductoryPeriod: introPrice?.period?.unit
                            ? `${introPrice.period.numberOfUnits} ${introPrice.period.unit}${introPrice.period.numberOfUnits > 1 ? 's' : ''}`
                            : undefined,
                        hasFreeTrial: introPrice?.price === 0 || undefined,
                    };
                });
            }

            // Fallback: fetch products directly by known IDs so the paywall
            // still shows real store prices even if no Offering is configured.
            readinessLogger.debug(String('[IAP] Offerings empty — falling back to getProducts by ID'), []);
            const storeProducts = await Purchases.getProducts(KNOWN_PRODUCT_IDS);
            if (Array.isArray(storeProducts) && storeProducts.length) {
                return storeProducts.map((p: any) => {
                    const introPrice = p.introPrice;
                    return {
                        id: p.identifier,
                        title: p.title || p.identifier,
                        price: p.priceString,
                        introductoryPrice: introPrice?.priceString || undefined,
                        introductoryPeriod: introPrice?.period?.unit
                            ? `${introPrice.period.numberOfUnits} ${introPrice.period.unit}${introPrice.period.numberOfUnits > 1 ? 's' : ''}`
                            : undefined,
                        hasFreeTrial: introPrice?.price === 0 || undefined,
                    };
                });
            }

            return [];
        } catch (error) {
            crashReporting.reportCrash(
                error instanceof Error ? error : new Error('Failed to get products'),
                { source: 'iapService.getProducts' }
            );
            return [];
        }
    }

    /**
     * Purchase a product via RevenueCat.
     */
    async purchase(productId: ProductId): Promise<PurchaseResult> {
        try {
            if (this.pendingNativeStoreOperation) {
                return { success: false, checkoutPending: true, error: NATIVE_STORE_PURCHASE_PENDING };
            }
            if (!this.isInitialized) {
                await this.initialize();
            }

            crashReporting.logBreadcrumb(`IAP purchase started: ${productId}`);

            if (!this.isRevenueCatAvailable) {
                return { success: false, error: 'In-app purchases are not configured. Please contact support.' };
            }
            const { data: { session } } = await supabase.auth.getSession();
            const userId = session?.user?.id;
            if (!userId) return { success: false, error: 'You must be logged in to purchase a subscription.' };
            if (this.identifiedUserId !== userId) await this.identify(userId);
            if (this.identifiedUserId !== userId) return { success: false, error: 'Purchase identity could not be verified. No payment was started.' };

            // Safety guard: detect test-store key in a production build. Apple's
            // sandbox will not return products through a test-store SDK key.
            const apiKey = Config.revenueCat.apiKey || '';
            if (apiKey.startsWith('test_') && !__DEV__) {
                console.warn('[IAP] Refusing purchase: using RevenueCat test-store key in a production build');
                return {
                    success: false,
                    error:
                        'Subscriptions are not yet available in this build. ' +
                        'The app is using a RevenueCat test-store key which cannot process real App Store purchases.',
                };
            }

            // Confirm the verification path works even for an account that is
            // currently free. Do not start a payment against a known-broken sync.
            const checkoutRevision = this.accountRevision;
            const checkoutAuthRevision = authEvents.getRevision();
            const ownerIsCurrent = () => this.accountRevision === checkoutRevision &&
                authEvents.getRevision() === checkoutAuthRevision &&
                this.identifiedUserId === userId && authEvents.canApplyToUser(userId);
            const checkoutReadiness = await this.isCheckoutVerificationReady(userId);
            const ownerStillCurrent = ownerIsCurrent();
            if (!checkoutReadiness.ready || !ownerStillCurrent) {
                const reason = checkoutReadiness.ready ? 'account_changed' : checkoutReadiness.reason;
                // Keep credentials, user IDs, and provider response data out of
                // telemetry. A short reason code is enough to diagnose a
                // checkout blocked before StoreKit starts.
                readinessLogger.warn(String('[IAP] Checkout blocked before StoreKit'), [{ reason }]);
                return { success: false, error: 'Subscriptions are temporarily unavailable. No payment was started.' };
            }

            // Preferred path: find a RevenueCat offering package.
            const packages = await this.getAvailablePackages();
            const exactPackage = packages.find((pkg: any) => pkg.product.identifier === productId);
            const targetPackage = exactPackage || null;

            let resolvedProductId: string;

            readinessLogger.debug(String('[IAP] purchase resolution'), [{
                requested: productId,
                availablePackages: packages.length,
                matchedExact: !!exactPackage,
                availableProductIds: packages.map((pkg: any) => pkg?.product?.identifier).filter(Boolean),
            }]);

            if (targetPackage) {
                resolvedProductId = targetPackage.product.identifier as string;
                if (!ownerIsCurrent()) return { success: false, error: 'Account changed. No payment was started.' };
                const operation = await this.waitForNativeStoreOperation(() => Purchases.purchasePackage(targetPackage));
                if (!operation.completed) return { success: false, checkoutPending: true, error: NATIVE_STORE_PURCHASE_PENDING };
            } else {
                // Fallback: buy the product directly from the store without an offering.
                readinessLogger.debug(String('[IAP] No offering package found — falling back to purchaseStoreProduct'), []);
                const uniqueIds = productId ? [productId] : KNOWN_PRODUCT_IDS;
                let storeProducts: any[] = [];
                try {
                    storeProducts = await Purchases.getProducts(uniqueIds);
                } catch (fetchErr: any) {
                    console.warn('[IAP] getProducts by ID failed', fetchErr?.message || fetchErr);
                }
                readinessLogger.debug(String('[IAP] store products fetched'), [{
                    requestedIds: uniqueIds,
                    returnedCount: storeProducts?.length || 0,
                    returnedIds: storeProducts?.map((p: any) => p.identifier) || [],
                }]);

                const storeProduct = storeProducts?.find((p: any) => p.identifier === productId);

                if (!storeProduct) {
                    return {
                        success: false,
                        error:
                            `App Store did not return ${productId}. ` +
                            'Check that this exact product ID exists in App Store Connect, is Ready to Submit, ' +
                            'is attached to the current app version, and is added to the active RevenueCat offering.',
                    };
                }

                resolvedProductId = storeProduct.identifier;
                if (!ownerIsCurrent()) return { success: false, error: 'Account changed. No payment was started.' };
                const operation = await this.waitForNativeStoreOperation(() => Purchases.purchaseStoreProduct(storeProduct));
                if (!operation.completed) return { success: false, checkoutPending: true, error: NATIVE_STORE_PURCHASE_PENDING };
            }

            // Confirm the receipt server-side before activating paid features.
            // A completed payment stays a completed payment if synchronization is pending.
            const verified = await this.verifyServerSubscription();
            const tier = this.getTierByProductId(resolvedProductId);
            if (verified && tier !== 'free') analyticsService.trackSubscriptionPurchased(tier, ANALYTICS_PRICE_BY_TIER[tier]);

            return {
                success: true,
                productId: resolvedProductId,
                verificationPending: !verified,
            };
        } catch (error: any) {
            const rawMessage = error.message || 'Purchase failed';

            // RevenueCat-specific error codes
            if (error.userCancelled) {
                return { success: false, error: 'Purchase was cancelled.' };
            }
            if (rawMessage.includes('PURCHASE_NOT_ALLOWED')) {
                return { success: false, error: 'Purchases are not allowed on this device.' };
            }
            if (rawMessage.includes('PRODUCT_ALREADY_PURCHASED')) {
                // Restore the existing purchase
                return { ...await this.restorePurchases(), productId };
            }

            crashReporting.reportCrash(
                error instanceof Error ? error : new Error(rawMessage),
                { source: 'iapService.purchase', productId }
            );

            return { success: false, error: normalizeIapError(rawMessage) };
        }
    }

    /**
     * Restore previous purchases via RevenueCat.
     */
    async restorePurchases(): Promise<PurchaseResult> {
        try {
            if (this.pendingNativeStoreOperation) {
                return await this.verifyServerSubscription()
                    ? { success: true }
                    : { success: false, checkoutPending: true, error: NATIVE_STORE_PURCHASE_PENDING };
            }
            if (!this.isInitialized) {
                await this.initialize();
            }

            const { data: { session } } = await supabase.auth.getSession();
            const userId = session?.user?.id;
            if (!userId) return { success: false, error: 'Sign in before restoring purchases.' };
            if (!this.isRevenueCatAvailable) {
                return await this.verifyServerSubscription() ? { success: true } : { success: false, error: 'No verified subscription found.' };
            }
            if (this.identifiedUserId !== userId) await this.identify(userId);
            if (this.identifiedUserId !== userId) return { success: false, error: 'Purchase identity could not be verified.' };

            const restoration = await this.waitForNativeStoreOperation<any>(() => Purchases.restorePurchases());
            if (!restoration.completed) return { success: false, checkoutPending: true, error: NATIVE_STORE_PURCHASE_PENDING };
            const customerInfo = restoration.value;
            const verified = await this.verifyServerSubscription();
            if (!verified && (Object.keys(customerInfo.entitlements?.active || {}).length > 0 || customerInfo.activeSubscriptions?.length > 0)) {
                return { success: true, verificationPending: true };
            }

            if (verified) {
                return { success: true };
            }

            return {
                success: false,
                error: 'No previous purchases found.',
            };
        } catch (error: any) {
            crashReporting.reportCrash(
                error instanceof Error ? error : new Error('Restore failed'),
                { source: 'iapService.restorePurchases' }
            );
            return {
                success: false,
                error: error.message || 'Failed to restore purchases.',
            };
        }
    }

    /**
     * Sync subscription state from RevenueCat customer info.
     * Called on purchase, restore, and customer info updates.
     */
    private async syncSubscriptionStatus(_customerInfo: any): Promise<void> {
        await this.verifyServerSubscription();
    }

    private async isCheckoutVerificationReady(owner: string): Promise<{ ready: true } | { ready: false; reason: string }> {
        try {
            const { data, error } = await withRequestDeadline(
                signal => supabase.functions.invoke('sync-subscription', { body: {}, signal }), 20_000,
            );
            if (error) return { ready: false, reason: 'sync_request_failed' };
            if (data?.verified !== true) return { ready: false, reason: 'sync_unverified' };
            if (!['free', 'lite', 'premium', 'vip'].includes(data.tier)) return { ready: false, reason: 'sync_invalid_tier' };
            const { data: identity } = await supabase.auth.getSession();
            if (identity.session?.user?.id !== owner) return { ready: false, reason: 'session_changed' };
            return { ready: true };
        } catch (error) {
            const reason = error instanceof Error && error.name === 'RequestTimeoutError' ? 'sync_timeout' : 'sync_check_failed';
            return { ready: false, reason };
        }
    }

    async verifyServerSubscription(): Promise<boolean> {
        if (this.serverSyncPromise) return this.serverSyncPromise;
        const revision = this.accountRevision;
        const authRevision = authEvents.getRevision();
        this.serverSyncPromise = (async () => {
            try {
                const { data: before } = await supabase.auth.getSession();
                const owner = before.session?.user?.id;
                if (!owner || !authEvents.canApplyToUser(owner)) return false;
                const { data, error } = await withRequestDeadline(
                    signal => supabase.functions.invoke('sync-subscription', { body: {}, signal }), 20_000,
                );
                const { data: after } = await supabase.auth.getSession();
                if (after.session?.user?.id !== owner || revision !== this.accountRevision || authRevision !== authEvents.getRevision()) return false;
                if (error || !data?.verified || !['free', 'lite', 'premium', 'vip'].includes(data.tier)) return false;
                if (data.tier !== 'free' && data.expiryDate != null &&
                    (!Number.isFinite(Date.parse(data.expiryDate)) || Date.parse(data.expiryDate) <= Date.now())) return false;
                await useSubscriptionStore.getState().setSubscription(data.tier, data.expiryDate ?? null);
                return data.tier !== 'free';
            } catch { return false; }
        })();
        const pending = this.serverSyncPromise;
        try { return await pending; }
        finally { if (this.serverSyncPromise === pending) this.serverSyncPromise = null; }
    }

    private getTierByProductId(productId: ProductId): SubscriptionTier {
        return getDynamicTier(productId);
    }

    /**
     * Identify the user with RevenueCat for cross-device subscription management.
     * Call after login.
     */
    async identify(userId: string): Promise<void> {
        if (!this.isInitialized) {
            await this.initialize();
        }

        if (!this.isRevenueCatAvailable) return;
        // Skip if already identified as this user
        if (this.identifiedUserId === userId) return;
        this.accountRevision++;
        this.serverSyncPromise = null;
        try {
            const { customerInfo } = await Purchases.logIn(userId);
            this.identifiedUserId = userId;
            await this.syncSubscriptionStatus(customerInfo);
            readinessLogger.debug(String('[IAP] identify succeeded'), [{ userId }]);
        } catch (error) {
            crashReporting.logBreadcrumb(`IAP identify failed: ${error}`);
        }
    }

    /**
     * Present Apple's native offer code redemption sheet (iOS 14+).
     * This is the Apple-compliant way to redeem offer codes for
     * free trials or discounted subscriptions. Replaces custom promo
     * code mechanisms per Guideline 3.1.1.
     *
     * After successful redemption, RevenueCat automatically syncs the
     * subscription and the webhook updates Supabase.
     */
    async presentCodeRedemptionSheet(): Promise<void> {
        if (Platform.OS !== 'ios') {
            // Offer codes are iOS-only; Android uses alternative flows.
            return;
        }
        
        const appStoreRedeemUrl = 'https://apps.apple.com/redeem?ctx=offercodes&id=6761912805';
        
        try {
            if (this.isRevenueCatAvailable && Purchases) {
                await Purchases.presentCodeRedemptionSheet();
                // After the sheet dismisses, refresh subscription status
                // (redemption may have succeeded or been cancelled)
                const { customerInfo } = await Purchases.getCustomerInfo();
                await this.syncSubscriptionStatus(customerInfo);
            } else {
                // Fallback: Open the official App Store redemption URL directly
                readinessLogger.debug(String('[IAP] RevenueCat not available, redirecting to App Store redemption URL'), []);
                const { Linking } = require('react-native');
                await Linking.openURL(appStoreRedeemUrl);
            }
        } catch (error) {
            console.warn('[IAP] Purchases.presentCodeRedemptionSheet failed, falling back to App Store URL:', error);
            try {
                const { Linking } = require('react-native');
                await Linking.openURL(appStoreRedeemUrl);
            } catch (linkErr) {
                console.error('[IAP] Failed to open App Store redemption link:', linkErr);
            }
        }
    }

    /**
     * Open the App Store subscription management page so the user can
     * cancel or change their subscription. Required by Apple Guideline 3.1.2(b).
     */
    async manageSubscriptions(): Promise<void> {
        if (!this.isRevenueCatAvailable) return;
        try {
            await Purchases.showManageSubscriptions();
        } catch (error) {
            // Fallback: open App Store subscriptions URL directly
            const Linking = require('react-native').Linking;
            await Linking.openURL('https://apps.apple.com/account/subscriptions');
        }
    }

    /**
     * Reset RevenueCat user on logout.
     */
    async logout(): Promise<void> {
        this.accountRevision++;
        this.serverSyncPromise = null;
        this.identifiedUserId = null;
        if (!this.isRevenueCatAvailable) return;
        try {
            await Purchases.logOut();
            this.identifiedUserId = null;
        } catch {
            // Non-critical
        }
    }
}

export const iapService = new IAPService();
export default iapService;
