import AsyncStorage from '@react-native-async-storage/async-storage';

beforeEach(() => {
  jest.clearAllMocks();
  (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
  (AsyncStorage.setItem as jest.Mock).mockResolvedValue(undefined);
  (AsyncStorage.removeItem as jest.Mock).mockResolvedValue(undefined);
});

const getStore = () => {
  const store = require('../../store/subscriptionStore').default;
  store.setState({
    tier: 'free',
    effectiveTier: 'free',
    hasActiveSubscription: false,
    isPremium: false,
    expiryDate: null,
  });
  return store;
};

describe('subscriptionStore (Free / Pro matrix)', () => {
  it('initializes with free tier when AsyncStorage is empty', async () => {
    const useSubscriptionStore = getStore();
    await useSubscriptionStore.getState().initializeSubscription();
    expect(useSubscriptionStore.getState().tier).toBe('free');
    expect(useSubscriptionStore.getState().hasActiveSubscription).toBe(false);
    expect(useSubscriptionStore.getState().isPremium).toBe(false);
  });

  it('loads active premium (Pro) subscription from AsyncStorage', async () => {
    const futureDate = new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString();
    (AsyncStorage.getItem as jest.Mock).mockImplementation((key: string) => {
      if (key === 'subscription_tier') return Promise.resolve('premium');
      if (key === 'subscription_expiry') return Promise.resolve(futureDate);
      return Promise.resolve(null);
    });
    const useSubscriptionStore = getStore();
    await useSubscriptionStore.getState().initializeSubscription();
    expect(useSubscriptionStore.getState().tier).toBe('premium');
    expect(useSubscriptionStore.getState().isPremium).toBe(true);
    expect(useSubscriptionStore.getState().hasActiveSubscription).toBe(true);
  });

  it('resets to free when stored subscription is expired', async () => {
    const pastDate = new Date(Date.now() - 1000).toISOString();
    (AsyncStorage.getItem as jest.Mock).mockImplementation((key: string) => {
      if (key === 'subscription_tier') return Promise.resolve('premium');
      if (key === 'subscription_expiry') return Promise.resolve(pastDate);
      return Promise.resolve(null);
    });
    const useSubscriptionStore = getStore();
    await useSubscriptionStore.getState().initializeSubscription();
    expect(useSubscriptionStore.getState().tier).toBe('free');
    expect(useSubscriptionStore.getState().hasActiveSubscription).toBe(false);
  });

  it('setSubscription persists tier and expiry to AsyncStorage', async () => {
    const useSubscriptionStore = getStore();
    await useSubscriptionStore.getState().setSubscription('premium');
    expect(AsyncStorage.setItem).toHaveBeenCalledWith('subscription_tier', 'premium');
    expect(useSubscriptionStore.getState().tier).toBe('premium');
    expect(useSubscriptionStore.getState().isPremium).toBe(true);
  });

  it('clearSubscription resets to free and removes AsyncStorage keys', async () => {
    const useSubscriptionStore = getStore();
    await useSubscriptionStore.getState().setSubscription('premium');
    await useSubscriptionStore.getState().clearSubscription();
    expect(useSubscriptionStore.getState().tier).toBe('free');
    expect(useSubscriptionStore.getState().hasActiveSubscription).toBe(false);
    expect(AsyncStorage.removeItem).toHaveBeenCalledWith('subscription_tier');
    expect(AsyncStorage.removeItem).toHaveBeenCalledWith('subscription_expiry');
  });

  describe('feature gates', () => {
    it('free tier: analytics is locked', () => {
      const useSubscriptionStore = getStore();
      expect(useSubscriptionStore.getState().checkFeatureAccess('analytics')).toBe(false);
    });

    it('free tier: trip planner is locked', () => {
      const useSubscriptionStore = getStore();
      expect(useSubscriptionStore.getState().checkFeatureAccess('tripPlanner')).toBe(false);
    });

    it('free tier: virtual try-on is locked', () => {
      const useSubscriptionStore = getStore();
      expect(useSubscriptionStore.getState().checkFeatureAccess('tryOns')).toBe(false);
    });

    it('free tier: AI outfits are available (daily limited)', () => {
      const useSubscriptionStore = getStore();
      expect(useSubscriptionStore.getState().checkFeatureAccess('aiOutfits')).toBe(true);
    });

    it('Pro (premium): all features are unlocked', async () => {
      const useSubscriptionStore = getStore();
      await useSubscriptionStore.getState().setSubscription('premium');
      expect(useSubscriptionStore.getState().checkFeatureAccess('analytics')).toBe(true);
      expect(useSubscriptionStore.getState().checkFeatureAccess('tripPlanner')).toBe(true);
      expect(useSubscriptionStore.getState().checkFeatureAccess('tryOns')).toBe(true);
      expect(useSubscriptionStore.getState().checkFeatureAccess('earlyAccess')).toBe(true);
      expect(useSubscriptionStore.getState().checkFeatureAccess('prioritySupport')).toBe(true);
    });
  });

  describe('getTriesRemaining (lifetime shim — daily tracking lives in dailyUsageStore)', () => {
    it('free tier AI outfits: 10 - used', () => {
      const useSubscriptionStore = getStore();
      expect(useSubscriptionStore.getState().getTriesRemaining('aiOutfits', 3)).toBe(7);
    });

    it('free tier wardrobe items: 20 - used', () => {
      const useSubscriptionStore = getStore();
      expect(useSubscriptionStore.getState().getTriesRemaining('wardrobeItems', 15)).toBe(5);
    });

    it('Pro tier: unlimited (-1) everywhere', async () => {
      const useSubscriptionStore = getStore();
      await useSubscriptionStore.getState().setSubscription('premium');
      expect(useSubscriptionStore.getState().getTriesRemaining('aiOutfits', 500)).toBe(-1);
      expect(useSubscriptionStore.getState().getTriesRemaining('tryOns', 500)).toBe(-1);
    });
  });
});
