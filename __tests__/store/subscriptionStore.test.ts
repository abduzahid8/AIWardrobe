import AsyncStorage from '@react-native-async-storage/async-storage';

beforeEach(() => {
  jest.resetAllMocks();
  (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
  (AsyncStorage.setItem as jest.Mock).mockResolvedValue(undefined);
  (AsyncStorage.removeItem as jest.Mock).mockResolvedValue(undefined);
});

const getStore = () => {
  const store = require('../../store/subscriptionStore').default;
  store.setState({ tier: 'free', hasActiveSubscription: false, isPremium: false, isVIP: false, expiresAt: null });
  return store;
};

describe('subscriptionStore', () => {
  it('initializes with free tier when AsyncStorage is empty', async () => {
    const useSubscriptionStore = getStore();
    await useSubscriptionStore.getState().initializeSubscription();
    expect(useSubscriptionStore.getState().tier).toBe('free');
    expect(useSubscriptionStore.getState().hasActiveSubscription).toBe(false);
    expect(useSubscriptionStore.getState().isPremium).toBe(false);
    expect(useSubscriptionStore.getState().isVIP).toBe(false);
  });

  it('loads active premium subscription from AsyncStorage', async () => {
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
    await useSubscriptionStore.getState().setSubscription('vip');
    expect(AsyncStorage.setItem).toHaveBeenCalledWith('subscription_tier', 'vip');
    expect(useSubscriptionStore.getState().tier).toBe('vip');
    expect(useSubscriptionStore.getState().isVIP).toBe(true);
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

  it('checkFeatureAccess returns false for analytics on free tier', () => {
    const useSubscriptionStore = getStore();
    expect(useSubscriptionStore.getState().checkFeatureAccess('analytics')).toBe(false);
  });

  it('checkFeatureAccess returns true for analytics on premium tier', async () => {
    const useSubscriptionStore = getStore();
    await useSubscriptionStore.getState().setSubscription('premium');
    expect(useSubscriptionStore.getState().checkFeatureAccess('analytics')).toBe(true);
  });

  it('getTriesRemaining returns correct count for free tier', () => {
    const useSubscriptionStore = getStore();
    expect(useSubscriptionStore.getState().getTriesRemaining(2)).toBe(3);
  });

  it('getTriesRemaining returns -1 (unlimited) for premium tier', async () => {
    const useSubscriptionStore = getStore();
    await useSubscriptionStore.getState().setSubscription('premium');
    expect(useSubscriptionStore.getState().getTriesRemaining(100)).toBe(-1);
  });
});
