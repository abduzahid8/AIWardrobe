const mockConfigure = jest.fn();
const mockAddCustomerInfoUpdateListener = jest.fn();
const mockGetOfferings = jest.fn();
const mockGetProducts = jest.fn();
const mockPurchasePackage = jest.fn();
const mockPurchaseStoreProduct = jest.fn();
const mockSetSubscription = jest.fn();
const mockVerifySubscriptionFromServer = jest.fn();

jest.mock('react-native-purchases', () => ({
    default: {
        configure: mockConfigure,
        addCustomerInfoUpdateListener: mockAddCustomerInfoUpdateListener,
        getOfferings: mockGetOfferings,
        getProducts: mockGetProducts,
        purchasePackage: mockPurchasePackage,
        purchaseStoreProduct: mockPurchaseStoreProduct,
    },
}));

jest.mock('../../src/config/env', () => ({
    __esModule: true,
    default: {
        revenueCat: { apiKey: 'appl_test_key' },
    },
    Config: {
        revenueCat: { apiKey: 'appl_test_key' },
    },
}));

jest.mock('../../lib/supabase', () => ({
    supabase: {
        auth: {
            getSession: jest.fn().mockResolvedValue({
                data: { session: { user: { id: 'user-1' } } },
            }),
        },
    },
}));

jest.mock('../../store/subscriptionStore', () => ({
    __esModule: true,
    default: {
        getState: () => ({
            setSubscription: mockSetSubscription,
            verifySubscriptionFromServer: mockVerifySubscriptionFromServer,
        }),
    },
}));

jest.mock('../../src/services/analyticsService', () => ({
    __esModule: true,
    default: {
        trackSubscriptionPurchased: jest.fn(),
    },
}));

jest.mock('../../src/services/crashReporting', () => ({
    __esModule: true,
    default: {
        logBreadcrumb: jest.fn(),
        reportCrash: jest.fn(),
    },
}));

describe('iapService purchase product resolution', () => {
    beforeEach(() => {
        jest.clearAllMocks();
        mockConfigure.mockResolvedValue(undefined);
        mockAddCustomerInfoUpdateListener.mockImplementation(() => undefined);
        mockSetSubscription.mockResolvedValue(undefined);
        mockVerifySubscriptionFromServer.mockResolvedValue(undefined);
    });

    it('buys only the requested Lite product when offerings contain Pro and Yearly packages', async () => {
        jest.useFakeTimers();
        const liteStoreProduct = {
            identifier: '2.99',
            title: 'Lite Monthly',
            priceString: '$2.99',
        };
        const proPackage = {
            product: {
                identifier: 'com.aiwardrobe.premium.monthly',
                title: 'Pro Monthly',
                priceString: '$9.99',
            },
        };
        const yearlyPackage = {
            product: {
                identifier: 'com.aiwardrobe.premium.yearly',
                title: 'Max Yearly',
                priceString: '$99.99',
            },
        };
        const customerInfo = {
            originalAppUserId: 'user-1',
            entitlements: { active: {} },
            activeSubscriptions: ['2.99'],
            allExpirationDatesByProduct: {
                '2.99': '2026-07-04T00:00:00.000Z',
            },
        };

        mockGetOfferings.mockResolvedValue({
            current: { identifier: 'default', availablePackages: [proPackage, yearlyPackage] },
            all: {},
        });
        mockGetProducts.mockResolvedValue([liteStoreProduct]);
        mockPurchaseStoreProduct.mockResolvedValue({ customerInfo });

        const { iapService } = require('../../src/services/iapService');
        const result = await iapService.purchase('2.99');

        expect(result.success).toBe(true);
        expect(mockPurchasePackage).not.toHaveBeenCalled();
        expect(mockGetProducts).toHaveBeenCalledWith(['2.99']);
        expect(mockPurchaseStoreProduct).toHaveBeenCalledWith(liteStoreProduct);
        expect(mockSetSubscription).toHaveBeenCalledWith(
            'lite',
            '2026-07-04T00:00:00.000Z',
            '2.99'
        );
        jest.useRealTimers();
    });
});
