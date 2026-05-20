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
 * - com.aiwardrobe.premium.monthly ($9.99/month)
 */
import { Platform } from 'react-native';
import useSubscriptionStore, { SubscriptionTier } from '../../store/subscriptionStore';
import analyticsService from './analyticsService';
import crashReporting from './crashReporting';
import Config from '../config/env';
import { supabase } from '../../lib/supabase';

export type ProductId = string;

const TIER_BY_PRODUCT_ID: Record<string, SubscriptionTier> = {
    'com.aiwardrobe.premium.monthly': 'premium',
    'com.aiwardrobe.premium.yearly': 'vip',
};

function getDynamicTier(productId: string): SubscriptionTier {
    if (TIER_BY_PRODUCT_ID[productId]) return TIER_BY_PRODUCT_ID[productId];

    const lowerId = String(productId || '').toLowerCase();
    if (lowerId.includes('yearly') || lowerId.includes('annual')) return 'vip';
    if (lowerId.includes('premium') || lowerId.includes('pro') || lowerId.includes('monthly')) return 'premium';

    return 'free';
}

/**
 * Known App Store / Play Store product IDs to try even when RevenueCat
 * "Offerings" are empty or misconfigured. Keeps the paywall functional
 * as long as the products exist in the native store.
 */
const KNOWN_PRODUCT_IDS: string[] = [
    'com.aiwardrobe.premium.monthly',
    'com.aiwardrobe.premium.yearly',
];

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
}

class IAPService {
    private isInitialized = false;
    private isRevenueCatAvailable = false;
    /** Supabase user ID passed to RevenueCat via identify(). null = not yet identified. */
    private identifiedUserId: string | null = null;

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
        console.log('[IAP] initialize', { keyKind, platform: Platform.OS });

        if (!apiKey || apiKey === 'your-revenuecat-api-key') {
            console.warn('[IAP] RevenueCat API key not configured — using mock mode');
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
                console.log('[IAP] RevenueCat configured successfully');

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
            console.log('[IAP] offerings fetched', {
                currentId: offerings.current?.identifier || null,
                currentPackagesCount: currentPackages.length,
                allOfferingsCount,
            });
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
            // Mock products for development
            return [
                { id: 'com.aiwardrobe.premium.monthly', title: 'Premium Monthly', price: '$9.99' },
                { id: 'com.aiwardrobe.premium.yearly', title: 'Premium Yearly', price: '$99.99' },
            ];
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
                        hasFreeTrial: introPrice?.price?.toNumber() === 0 || undefined,
                    };
                });
            }

            // Fallback: fetch products directly by known IDs so the paywall
            // still shows real store prices even if no Offering is configured.
            console.log('[IAP] Offerings empty — falling back to getProducts by ID');
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
                        hasFreeTrial: introPrice?.price?.toNumber() === 0 || undefined,
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
            if (!this.isInitialized) {
                await this.initialize();
            }

            crashReporting.logBreadcrumb(`IAP purchase started: ${productId}`);

            // CRITICAL: Ensure the user is identified with RevenueCat before
            // purchasing. If we proceed with an anonymous RC user ID, the
            // webhook won't be able to map the purchase back to a Supabase user.
            if (!this.identifiedUserId) {
                const { data: { session } } = await supabase.auth.getSession();
                const userId = session?.user?.id;
                if (userId) {
                    console.warn('[IAP] User not identified before purchase — calling identify() now');
                    await this.identify(userId);
                } else {
                    return {
                        success: false,
                        error: 'You must be logged in to purchase a subscription.',
                    };
                }
            }

            if (!this.isRevenueCatAvailable) {
                // Mock mode — for development/simulator only.
                // IMPORTANT: never sync mock purchases to the server.
                if (__DEV__) {
                    analyticsService.trackSubscriptionPurchased('premium', 9.99);
                    return {
                        success: true,
                        productId,
                        transactionId: `mock_dev_${Date.now()}`,
                    };
                }
                return {
                    success: false,
                    error: 'In-app purchases are not configured. Please contact support.',
                };
            }

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

            // Preferred path: find a RevenueCat offering package.
            const packages = await this.getAvailablePackages();
            const exactPackage = packages.find((pkg: any) => pkg.product.identifier === productId);
            const premiumLikePackage = packages.find((pkg: any) => {
                const identifier = String(pkg?.product?.identifier || '').toLowerCase();
                const title = String(pkg?.product?.title || '').toLowerCase();
                return identifier.includes('premium') || identifier.includes('pro') || title.includes('premium') || title.includes('pro');
            });
            const targetPackage = exactPackage || premiumLikePackage || (packages.length === 1 ? packages[0] : null);

            let customerInfo: any;
            let resolvedProductId: string;

            console.log('[IAP] purchase resolution', {
                requested: productId,
                availablePackages: packages.length,
                matchedExact: !!exactPackage,
                matchedPremiumLike: !!premiumLikePackage,
            });

            if (targetPackage) {
                resolvedProductId = targetPackage.product.identifier as string;
                const result = await Purchases.purchasePackage(targetPackage);
                customerInfo = result.customerInfo;
            } else {
                // Fallback: buy the product directly from the store without an offering.
                console.log('[IAP] No offering package found — falling back to purchaseStoreProduct');
                const idsToFetch = productId ? [productId, ...KNOWN_PRODUCT_IDS] : KNOWN_PRODUCT_IDS;
                const uniqueIds = Array.from(new Set(idsToFetch));
                let storeProducts: any[] = [];
                try {
                    storeProducts = await Purchases.getProducts(uniqueIds);
                } catch (fetchErr: any) {
                    console.warn('[IAP] getProducts by ID failed', fetchErr?.message || fetchErr);
                }
                console.log('[IAP] store products fetched', {
                    requestedIds: uniqueIds,
                    returnedCount: storeProducts?.length || 0,
                    returnedIds: storeProducts?.map((p: any) => p.identifier) || [],
                });

                const storeProduct =
                    storeProducts?.find((p: any) => p.identifier === productId) ||
                    (storeProducts?.length ? storeProducts[0] : undefined);

                if (!storeProduct) {
                    return {
                        success: false,
                        error:
                            'This subscription is not yet available on the App Store. ' +
                            'If you are the developer, ensure the product is in "Ready to Submit" ' +
                            'state and attached to the current app version.',
                    };
                }

                resolvedProductId = storeProduct.identifier;
                const result = await Purchases.purchaseStoreProduct(storeProduct);
                customerInfo = result.customerInfo;
            }

            // Sync subscription status from RevenueCat customer info first.
            // This picks up the entitlement keys RC assigned to this product.
            await this.syncSubscriptionStatus(customerInfo);

            let tier = this.getTierByProductId(resolvedProductId);

            // Safety net: the purchase itself succeeded (Apple has charged the
            // user and RevenueCat has validated the receipt), so we MUST grant
            // access even if entitlement detection in syncSubscriptionStatus
            // didn't match anything.
            if (tier === 'free') {
                console.warn(`[IAP] Paid product ID '${resolvedProductId}' did not match any tier strings, forcing premium tier to prevent taking user's money without granting access.`);
                tier = 'premium';
            }

            // ALWAYS call setSubscription after a successful purchase.
            // We do NOT gate this on currentTier === 'free' because:
            //   1. syncSubscriptionStatus may have silently failed to detect the entitlement.
            //   2. The user could be upgrading from premium → vip.
            //   3. Skipping this leaves a window where the local store shows free
            //      until the RevenueCat webhook fires (which can take minutes).
            const expiry =
                customerInfo?.allExpirationDatesByProduct?.[resolvedProductId] ??
                customerInfo?.allExpirationDates?.[resolvedProductId] ??
                undefined;
            const { setSubscription, verifySubscriptionFromServer } = useSubscriptionStore.getState();
            console.log('[IAP] Granting subscription after successful purchase', { tier, resolvedProductId, expiry });
            await setSubscription(tier, expiry, resolvedProductId);

            // Fire-and-forget server verification to pull the freshly inserted DB row.
            // Delayed 5 s to give Supabase time to commit+replicate the subscription row
            // written by setSubscription() above. Without the delay, the read can race
            // the write and return 'free', triggering an incorrect downgrade.
            // The race-condition guard in verifySubscriptionFromServer is the primary
            // defense; this delay is defense-in-depth.
            setTimeout(() => {
                verifySubscriptionFromServer().catch((err) =>
                    console.warn('[IAP] post-purchase server verification failed (non-critical):', err)
                );
            }, 5000);

            analyticsService.trackSubscriptionPurchased(tier, 9.99);

            return {
                success: true,
                productId: resolvedProductId,
                transactionId: customerInfo.originalAppUserId,
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
                await this.restorePurchases();
                return { success: true, productId };
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
            if (!this.isInitialized) {
                await this.initialize();
            }

            if (!this.isRevenueCatAvailable) {
                // Fallback: check server
                const { verifySubscriptionFromServer } = useSubscriptionStore.getState();
                await verifySubscriptionFromServer();
                const { tier } = useSubscriptionStore.getState();
                return tier !== 'free'
                    ? { success: true }
                    : { success: false, error: 'No previous purchases found.' };
            }

            const customerInfo = await Purchases.restorePurchases();
            await this.syncSubscriptionStatus(customerInfo);

            const { tier } = useSubscriptionStore.getState();
            if (tier !== 'free') {
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
    private async syncSubscriptionStatus(customerInfo: any): Promise<void> {
        try {
            const { setSubscription, clearSubscription } = useSubscriptionStore.getState();

            // Check active entitlements from RevenueCat
            const activeEntitlements = customerInfo.entitlements?.active || {};
            const entitlementEntries = Object.entries(activeEntitlements) as Array<[string, any]>;

            const findEntitlement = (aliases: string[]) => {
                const normalizedAliases = aliases.map((alias) => alias.toLowerCase());
                return entitlementEntries.find(([entitlementKey]) =>
                    normalizedAliases.includes(entitlementKey.toLowerCase())
                )?.[1];
            };

            // Fallback to active subscriptions by product ID in case
            // entitlement keys differ across environments.
            const activeProductIds = new Set<string>(customerInfo.activeSubscriptions || []);
            const findProductForTier = (tier: SubscriptionTier): string | null => {
                for (const productId of activeProductIds) {
                    if (getDynamicTier(productId) === tier) return productId;
                }
                return null;
            };

            const getExpiryForProduct = (productId: string | null): string | undefined => {
                if (!productId) return undefined;
                const byProduct = customerInfo.allExpirationDatesByProduct || customerInfo.allExpirationDates || {};
                return byProduct[productId] || undefined;
            };

            // Check VIP first (higher tier), then premium
            const vipEntitlement = findEntitlement(['vip', 'max', 'aiwardrobe_pro_yearly', 'com.aiwardrobe.premium.yearly']);
            const vipProductId = findProductForTier('vip');

            const premiumEntitlement = findEntitlement(['premium', 'pro', 'aiwardrobe pro', 'com.aiwardrobe.premium.monthly']);
            const premiumProductId = findProductForTier('premium');

            if (vipEntitlement || vipProductId) {
                const expiry = vipEntitlement?.expirationDate || getExpiryForProduct(vipProductId);
                await setSubscription('vip', expiry, vipProductId || undefined);
            } else if (premiumEntitlement || premiumProductId) {
                const expiry = premiumEntitlement?.expirationDate || getExpiryForProduct(premiumProductId);
                await setSubscription('premium', expiry, premiumProductId || undefined);
            } else if (entitlementEntries.length === 0 && activeProductIds.size === 0) {
                // Truly no active entitlements or subscriptions — safe to downgrade.
                await clearSubscription();
            } else {
                // RevenueCat reports active entitlements / subscriptions but none
                // matched our known aliases / product IDs. Do NOT clear the tier —
                // this would incorrectly downgrade a paying user. Log so we can
                // catch the mismatch in diagnostics.
                console.warn('[IAP] Active RevenueCat state did not match known tiers — keeping current subscription', {
                    entitlementKeys: Object.keys(activeEntitlements),
                    activeProductIds: Array.from(activeProductIds),
                });
            }
        } catch (error) {
            crashReporting.reportCrash(
                error instanceof Error ? error : new Error('Subscription sync failed'),
                { source: 'iapService.syncSubscriptionStatus' }
            );
        }
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
        try {
            const { customerInfo } = await Purchases.logIn(userId);
            this.identifiedUserId = userId;
            await this.syncSubscriptionStatus(customerInfo);
            console.log('[IAP] identify succeeded', { userId });
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
                console.log('[IAP] RevenueCat not available, redirecting to App Store redemption URL');
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
