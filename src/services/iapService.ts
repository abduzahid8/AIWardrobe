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

export type ProductId =
    | 'com.aiwardrobe.premium.monthly'
    | 'com.aiwardrobe.premium.yearly';

const TIER_BY_PRODUCT_ID: Record<ProductId, SubscriptionTier> = {
    'com.aiwardrobe.premium.monthly': 'premium',
    'com.aiwardrobe.premium.yearly': 'premium',
};

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

    /**
     * Initialize IAP connection. Call once on app start.
     */
    async initialize(): Promise<void> {
        if (this.isInitialized) return;

        const apiKey = Config.revenueCat.apiKey;
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

                // Listen for customer info changes (renewal, cancellation, etc.)
                Purchases.addCustomerInfoUpdateListener((info: any) => {
                    this.syncSubscriptionStatus(info);
                });
            }

            this.isInitialized = true;
        } catch (error) {
            console.warn('[IAP] RevenueCat SDK not available — using mock mode');
            crashReporting.logBreadcrumb('IAP: RevenueCat not available, mock mode active');
            this.isInitialized = true;
        }
    }

    /**
     * Get available products with pricing from the store.
     */
    async getProducts(): Promise<{ id: string; title: string; price: string; packageType?: string }[]> {
        if (!this.isRevenueCatAvailable) {
            // Mock products for development
            return [
                { id: 'com.aiwardrobe.premium.monthly', title: 'Premium Monthly', price: '$9.99' },
            ];
        }

        try {
            const offerings = await Purchases.getOfferings();
            if (!offerings.current?.availablePackages?.length) {
                return [];
            }

            return offerings.current.availablePackages.map((pkg: any) => ({
                id: pkg.product.identifier,
                title: pkg.product.title,
                price: pkg.product.priceString,
                packageType: pkg.packageType,
            }));
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
            crashReporting.logBreadcrumb(`IAP purchase started: ${productId}`);

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

            // Get the offering and find the matching package
            const offerings = await Purchases.getOfferings();
            const packages = offerings.current?.availablePackages || [];
            const targetPackage = packages.find(
                (pkg: any) => pkg.product.identifier === productId
            );

            if (!targetPackage) {
                return {
                    success: false,
                    error: 'Product not found. Please try again later.',
                };
            }

            // Execute the purchase through RevenueCat
            const { customerInfo } = await Purchases.purchasePackage(targetPackage);

            // Sync subscription status from RevenueCat customer info
            await this.syncSubscriptionStatus(customerInfo);

            const tier = this.getTierByProductId(productId);
            analyticsService.trackSubscriptionPurchased(tier, 9.99);

            return {
                success: true,
                productId,
                transactionId: customerInfo.originalAppUserId,
            };
        } catch (error: any) {
            const message = error.message || 'Purchase failed';

            // RevenueCat-specific error codes
            if (error.userCancelled) {
                return { success: false, error: 'Purchase was cancelled.' };
            }
            if (message.includes('PURCHASE_NOT_ALLOWED')) {
                return { success: false, error: 'Purchases are not allowed on this device.' };
            }
            if (message.includes('PRODUCT_ALREADY_PURCHASED')) {
                // Restore the existing purchase
                await this.restorePurchases();
                return { success: true, productId };
            }

            crashReporting.reportCrash(
                error instanceof Error ? error : new Error(message),
                { source: 'iapService.purchase', productId }
            );

            return { success: false, error: message };
        }
    }

    /**
     * Restore previous purchases via RevenueCat.
     */
    async restorePurchases(): Promise<PurchaseResult> {
        try {
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
                const productEntries = Object.entries(TIER_BY_PRODUCT_ID) as Array<[ProductId, SubscriptionTier]>;
                const matched = productEntries.find(([productId, mappedTier]) =>
                    mappedTier === tier && activeProductIds.has(productId)
                );
                return matched?.[0] || null;
            };

            const getExpiryForProduct = (productId: string | null): string | undefined => {
                if (!productId) return undefined;
                const byProduct = customerInfo.allExpirationDatesByProduct || customerInfo.allExpirationDates || {};
                return byProduct[productId] || undefined;
            };

            const premiumEntitlement = findEntitlement(['premium', 'pro']);
            const premiumProductId = findProductForTier('premium');

            if (premiumEntitlement || premiumProductId) {
                const expiry = premiumEntitlement?.expirationDate || getExpiryForProduct(premiumProductId);
                await setSubscription('premium', expiry);
            } else {
                await clearSubscription();
            }
        } catch (error) {
            crashReporting.reportCrash(
                error instanceof Error ? error : new Error('Subscription sync failed'),
                { source: 'iapService.syncSubscriptionStatus' }
            );
        }
    }

    private getTierByProductId(productId: ProductId): SubscriptionTier {
        return TIER_BY_PRODUCT_ID[productId] || 'premium';
    }

    /**
     * Identify the user with RevenueCat for cross-device subscription management.
     * Call after login.
     */
    async identify(userId: string): Promise<void> {
        if (!this.isRevenueCatAvailable) return;
        try {
            const { customerInfo } = await Purchases.logIn(userId);
            await this.syncSubscriptionStatus(customerInfo);
        } catch (error) {
            crashReporting.logBreadcrumb(`IAP identify failed: ${error}`);
        }
    }

    /**
     * Reset RevenueCat user on logout.
     */
    async logout(): Promise<void> {
        if (!this.isRevenueCatAvailable) return;
        try {
            await Purchases.logOut();
        } catch {
            // Non-critical
        }
    }
}

export const iapService = new IAPService();
export default iapService;
