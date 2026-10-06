import AsyncStorage from '@react-native-async-storage/async-storage';

beforeEach(() => {
  jest.resetModules();
  jest.resetAllMocks();
  (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
  (AsyncStorage.setItem as jest.Mock).mockResolvedValue(undefined);
  (AsyncStorage.removeItem as jest.Mock).mockResolvedValue(undefined);
});

function loadStores() {
  const useSubscriptionStore = require('../../store/subscriptionStore').default;
  const useDailyUsageStore = require('../../store/dailyUsageStore').default;
  useSubscriptionStore.setState({
    tier: 'free',
    hasActiveSubscription: false,
    isPremium: false,
    isVIP: false,
    expiryDate: null,
  });
  useDailyUsageStore.setState({
    snapshot: { date: '2099-01-01', counts: {} },
    hydrated: false,
  });
  return { useSubscriptionStore, useDailyUsageStore };
}

describe('dailyUsageStore', () => {
  it('free tier (trial expired): aiOutfits locked at 0', async () => {
    const { useDailyUsageStore } = loadStores();
    await useDailyUsageStore.getState().hydrate();
    expect(useDailyUsageStore.getState().getRemaining('aiOutfits')).toBe(0);
    expect(useDailyUsageStore.getState().canUse('aiOutfits')).toBe(false);
  });

  it('free tier (trial expired): starts at 3 remaining for homeSwipes', async () => {
    const { useDailyUsageStore } = loadStores();
    await useDailyUsageStore.getState().hydrate();
    expect(useDailyUsageStore.getState().getRemaining('homeSwipes')).toBe(3);
    expect(useDailyUsageStore.getState().canUse('homeSwipes')).toBe(true);
  });

  it('consume decrements remaining (homeSwipes)', async () => {
    const { useDailyUsageStore } = loadStores();
    await useDailyUsageStore.getState().hydrate();
    await useDailyUsageStore.getState().consume('homeSwipes');
    await useDailyUsageStore.getState().consume('homeSwipes');
    expect(useDailyUsageStore.getState().getUsed('homeSwipes')).toBe(2);
    expect(useDailyUsageStore.getState().getRemaining('homeSwipes')).toBe(1);
  });

  it('free tier: blocks after 3 home swipes', async () => {
    const { useDailyUsageStore } = loadStores();
    await useDailyUsageStore.getState().hydrate();
    for (let i = 0; i < 3; i++) {
      await useDailyUsageStore.getState().consume('homeSwipes');
    }
    expect(useDailyUsageStore.getState().getRemaining('homeSwipes')).toBe(0);
    expect(useDailyUsageStore.getState().canUse('homeSwipes')).toBe(false);
  });

  it('Pro tier: unlimited (-1)', async () => {
    const { useSubscriptionStore, useDailyUsageStore } = loadStores();
    await useSubscriptionStore.getState().setSubscription('premium');
    await useDailyUsageStore.getState().hydrate();
    expect(useDailyUsageStore.getState().getRemaining('aiOutfits')).toBe(-1);
  });

  it('Max tier: unlimited (-1)', async () => {
    const { useSubscriptionStore, useDailyUsageStore } = loadStores();
    await useSubscriptionStore.getState().setSubscription('vip');
    await useDailyUsageStore.getState().hydrate();
    expect(useDailyUsageStore.getState().getRemaining('aiOutfits')).toBe(-1);
    expect(useDailyUsageStore.getState().canUse('aiOutfits')).toBe(true);
  });

  it('tryOns on free tier is blocked (limit = 0)', async () => {
    const { useDailyUsageStore } = loadStores();
    await useDailyUsageStore.getState().hydrate();
    expect(useDailyUsageStore.getState().getRemaining('tryOns')).toBe(0);
    expect(useDailyUsageStore.getState().canUse('tryOns')).toBe(false);
  });

  it('hydrate rolls over stored snapshot from a previous day', async () => {
    const yesterdaySnapshot = JSON.stringify({
      date: '1999-01-01',
      counts: { homeSwipes: 2 },
    });
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(yesterdaySnapshot);
    const { useDailyUsageStore } = loadStores();
    await useDailyUsageStore.getState().hydrate();
    expect(useDailyUsageStore.getState().getUsed('homeSwipes')).toBe(0);
    expect(useDailyUsageStore.getState().getRemaining('homeSwipes')).toBe(3);
  });

  it('resetToday clears counters', async () => {
    const { useDailyUsageStore } = loadStores();
    await useDailyUsageStore.getState().hydrate();
    await useDailyUsageStore.getState().consume('homeSwipes', 2);
    await useDailyUsageStore.getState().resetToday();
    expect(useDailyUsageStore.getState().getUsed('homeSwipes')).toBe(0);
  });
});
