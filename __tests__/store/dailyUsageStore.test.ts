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
  it('free tier: starts at 10 remaining for aiOutfits', async () => {
    const { useDailyUsageStore } = loadStores();
    await useDailyUsageStore.getState().hydrate();
    expect(useDailyUsageStore.getState().getRemaining('aiOutfits')).toBe(10);
    expect(useDailyUsageStore.getState().canUse('aiOutfits')).toBe(true);
  });

  it('consume decrements remaining', async () => {
    const { useDailyUsageStore } = loadStores();
    await useDailyUsageStore.getState().hydrate();
    await useDailyUsageStore.getState().consume('aiOutfits');
    await useDailyUsageStore.getState().consume('aiOutfits');
    expect(useDailyUsageStore.getState().getUsed('aiOutfits')).toBe(2);
    expect(useDailyUsageStore.getState().getRemaining('aiOutfits')).toBe(8);
  });

  it('free tier: blocks after 10 AI outfits', async () => {
    const { useDailyUsageStore } = loadStores();
    await useDailyUsageStore.getState().hydrate();
    for (let i = 0; i < 10; i++) {
      await useDailyUsageStore.getState().consume('aiOutfits');
    }
    expect(useDailyUsageStore.getState().getRemaining('aiOutfits')).toBe(0);
    expect(useDailyUsageStore.getState().canUse('aiOutfits')).toBe(false);
  });

  it('Pro tier: gets 100 per day', async () => {
    const { useSubscriptionStore, useDailyUsageStore } = loadStores();
    await useSubscriptionStore.getState().setSubscription('premium');
    await useDailyUsageStore.getState().hydrate();
    expect(useDailyUsageStore.getState().getRemaining('aiOutfits')).toBe(100);
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
      counts: { aiOutfits: 9 },
    });
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(yesterdaySnapshot);
    const { useDailyUsageStore } = loadStores();
    await useDailyUsageStore.getState().hydrate();
    expect(useDailyUsageStore.getState().getUsed('aiOutfits')).toBe(0);
    expect(useDailyUsageStore.getState().getRemaining('aiOutfits')).toBe(10);
  });

  it('resetToday clears counters', async () => {
    const { useDailyUsageStore } = loadStores();
    await useDailyUsageStore.getState().hydrate();
    await useDailyUsageStore.getState().consume('aiOutfits', 5);
    await useDailyUsageStore.getState().resetToday();
    expect(useDailyUsageStore.getState().getUsed('aiOutfits')).toBe(0);
  });
});
