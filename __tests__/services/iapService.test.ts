const mockConfigure = jest.fn();
const mockLogIn = jest.fn();
const mockRestore = jest.fn();
const mockSyncInvoke = jest.fn();
const mockSession = jest.fn();
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
        logIn: mockLogIn,
        restorePurchases: mockRestore,
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
        auth: { getSession: mockSession },
        functions: { invoke: mockSyncInvoke },
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
    afterEach(() => jest.useRealTimers());
    beforeEach(() => {
        jest.resetModules();
        jest.resetAllMocks();
        mockSession.mockResolvedValue({ data: { session: { user: { id: 'user-1' } } } });
        mockLogIn.mockResolvedValue({ customerInfo: { entitlements: { active: {} } } });
        mockSyncInvoke.mockResolvedValue({ data: { verified: true, tier: 'lite', expiryDate: '2030-07-04T00:00:00.000Z' }, error: null });
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
                '2.99': '2030-07-04T00:00:00.000Z',
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
            '2030-07-04T00:00:00.000Z'
        );
        jest.useRealTimers();
    });
    it('a completed payment with unavailable verification remains pending, never a fake activation', async () => {
        mockSyncInvoke.mockResolvedValue({ data: null, error: new Error('offline') })
            .mockResolvedValueOnce({ data: { verified: true, tier: 'free', expiryDate: null }, error: null })
            .mockResolvedValueOnce({ data: { verified: true, tier: 'free', expiryDate: null }, error: null });
        mockGetOfferings.mockResolvedValue({ current: { availablePackages: [] }, all: {} });
        const product = { identifier: '2.99', title: 'Lite', priceString: '$2.99' };
        mockGetProducts.mockResolvedValue([product]);
        mockPurchaseStoreProduct.mockResolvedValue({ customerInfo: { entitlements: { active: { pro: {} } } } });
        const { iapService } = require('../../src/services/iapService');
        const result = await iapService.purchase('2.99');
        expect(result).toMatchObject({ success: true, verificationPending: true });
        expect(mockSetSubscription.mock.calls.every(([tier]) => tier === 'free')).toBe(true);
        expect(result.transactionId).toBeUndefined();
    });

    it('identity verification failure stops before charging the store', async () => {
        mockLogIn.mockRejectedValue(new Error('identity unavailable'));
        const { iapService } = require('../../src/services/iapService');
        const result = await iapService.purchase('2.99');
        expect(result.success).toBe(false);
        expect(mockPurchaseStoreProduct).not.toHaveBeenCalled();
        expect(mockPurchasePackage).not.toHaveBeenCalled();
    });

    it('a server response for a logged-out owner cannot reactivate that account', async () => {
        let resolveSync: (value: unknown) => void = () => undefined;
        mockSyncInvoke.mockReturnValue(new Promise(resolve => { resolveSync = resolve; }));
        const { iapService } = require('../../src/services/iapService');
        const pending = iapService.verifyServerSubscription();
        await Promise.resolve();
        mockSession.mockResolvedValue({ data: { session: null } });
        resolveSync({ data: { verified: true, tier: 'premium', expiryDate: '2030-07-04T00:00:00Z' }, error: null });
        expect(await pending).toBe(false);
        expect(mockSetSubscription).not.toHaveBeenCalled();
    });

    it('restored store entitlements remain pending until the server verifies them', async () => {
        mockSyncInvoke.mockResolvedValue({ data: null, error: new Error('offline') });
        mockRestore.mockResolvedValue({ entitlements: { active: { premium: {} } }, activeSubscriptions: ['com.aiwardrobe.premium.monthly'] });
        const { iapService } = require('../../src/services/iapService');
        expect(await iapService.restorePurchases()).toMatchObject({ success: true, verificationPending: true });
        expect(mockSetSubscription).not.toHaveBeenCalled();
    });

    it('logout invalidates in-flight verification even before the Auth session finishes clearing', async () => {
        let resolveSync: (value: unknown) => void = () => undefined;
        mockSyncInvoke.mockReturnValue(new Promise(resolve => { resolveSync = resolve; }));
        const { iapService } = require('../../src/services/iapService');
        const pending = iapService.verifyServerSubscription();
        await Promise.resolve();
        await iapService.logout();
        resolveSync({ data: { verified: true, tier: 'premium', expiryDate: '2030-07-04T00:00:00Z' }, error: null });
        expect(await pending).toBe(false);
        expect(mockSetSubscription).not.toHaveBeenCalled();
    });

    it('a late SDK update cannot begin verification during logout with a still-cached Auth session', async () => {
        const { iapService } = require('../../src/services/iapService');
        require('../../store/authEvents').authEvents.emitLogout();
        expect(await iapService.verifyServerSubscription()).toBe(false);
        expect(mockSyncInvoke).not.toHaveBeenCalled();
        expect(mockSetSubscription).not.toHaveBeenCalled();
    });

    it('verification service unavailable stops before any store payment', async () => {
        mockGetOfferings.mockResolvedValue({ current: { availablePackages: [{ product: { identifier: '2.99' } }] }, all: {} });
        mockPurchasePackage.mockResolvedValue({});
        mockSyncInvoke.mockResolvedValue({ data: null, error: new Error('unavailable') });
        const { iapService } = require('../../src/services/iapService');
        const result = await iapService.purchase('2.99');
        expect(result.success).toBe(false);
        expect(mockPurchasePackage).not.toHaveBeenCalled();
        expect(mockPurchaseStoreProduct).not.toHaveBeenCalled();
    });
    it('a verified free account can start a purchase when reconciliation is ready', async () => {
        mockSyncInvoke.mockResolvedValue({ data: { verified: true, tier: 'free', expiryDate: null }, error: null });
        mockGetOfferings.mockResolvedValue({ current: { availablePackages: [] }, all: {} });
        mockGetProducts.mockResolvedValue([{ identifier: '2.99' }]);
        mockPurchaseStoreProduct.mockResolvedValue({});
        const { iapService } = require('../../src/services/iapService');
        expect((await iapService.purchase('2.99')).success).toBe(true);
        expect(mockPurchaseStoreProduct).toHaveBeenCalledTimes(1);
    });

    it('a stalled checkout preflight times out without starting payment, even if it resolves late', async () => {
        const { iapService } = require('../../src/services/iapService');
        await iapService.identify('user-1');
        let resolveProbe: (value: unknown) => void = () => undefined;
        mockSyncInvoke.mockReturnValue(new Promise(resolve => { resolveProbe = resolve; }));
        jest.useFakeTimers();
        const pending = iapService.purchase('2.99');
        for (let n = 0; n < 12; n++) await Promise.resolve();
        await jest.advanceTimersByTimeAsync(20_000);
        expect((await pending).success).toBe(false);
        resolveProbe({ data: { verified: true, tier: 'free' }, error: null });
        for (let n = 0; n < 4; n++) await Promise.resolve();
        expect(mockPurchasePackage).not.toHaveBeenCalled();
        expect(mockPurchaseStoreProduct).not.toHaveBeenCalled();
    });
    it('logout while offerings load prevents store checkout', async () => {
        let resolveOfferings: (value: unknown) => void = () => undefined;
        mockGetOfferings.mockReturnValue(new Promise(resolve => { resolveOfferings = resolve; }));
        const { iapService } = require('../../src/services/iapService');
        const pending = iapService.purchase('2.99');
        for (let n = 0; n < 24; n++) await Promise.resolve();
        expect(mockGetOfferings).toHaveBeenCalled();
        require('../../store/authEvents').authEvents.emitLogout();
        resolveOfferings({ current: { availablePackages: [{ product: { identifier: '2.99' } }] }, all: {} });
        expect((await pending).success).toBe(false);
        expect(mockPurchasePackage).not.toHaveBeenCalled();
        expect(mockPurchaseStoreProduct).not.toHaveBeenCalled();
    });
    it.each([null, { verified: false, tier: 'free' }, { verified: true, tier: 'administrator' }])('invalid verification response %j cannot start payment', async data => {
        mockGetOfferings.mockResolvedValue({ current: { availablePackages: [{ product: { identifier: '2.99' } }] }, all: {} });
        mockSyncInvoke.mockResolvedValue({ data, error: null });
        const { iapService } = require('../../src/services/iapService');
        expect((await iapService.purchase('2.99')).success).toBe(false);
        expect(mockPurchasePackage).not.toHaveBeenCalled();
        expect(mockPurchaseStoreProduct).not.toHaveBeenCalled();
    });

});
