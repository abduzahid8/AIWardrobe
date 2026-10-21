import { act, renderHook } from '@testing-library/react-native';
import { AppState, AppStateStatus } from 'react-native';
import useSubscriptionStore from '../../store/subscriptionStore';
import { useSubscriptionClock } from '../../src/hooks/useSubscriptionClock';
import { useTrialStatus } from '../../hooks/useTrialStatus';

const end = Date.parse('2026-10-21T12:00:00Z');
const start = new Date(end - 7 * 86400000).toISOString();
function activeTrial() {
  useSubscriptionStore.setState({ tier: 'free', expiryDate: null, trialStartedAt: start,
    effectiveTier: 'premium', isPremium: true, hasActiveSubscription: true,
    isTrialActive: true, isTrialExpired: false, isTrialPending: false, trialDaysRemaining: 1 });
}
beforeEach(() => { jest.useFakeTimers(); jest.setSystemTime(end - 1000); activeTrial(); });
afterEach(() => { jest.restoreAllMocks(); jest.useRealTimers(); });

test('an open app loses trial access and updates its countdown at the exact boundary', () => {
  const { result, unmount } = renderHook(() => { useSubscriptionClock(); return useTrialStatus(); });
  expect(result.current.isTrialActive).toBe(true);
  act(() => jest.advanceTimersByTime(1000));
  expect(result.current.isTrialActive).toBe(false);
  expect(result.current.isTrialExpired).toBe(true);
  expect(result.current.daysRemaining).toBe(0);
  expect(useSubscriptionStore.getState().effectiveTier).toBe('free');
  expect(useSubscriptionStore.getState().checkFeatureAccess('aiOutfits')).toBe(false);
  unmount();
});

test('resuming after suspended timers immediately reconciles trial access', () => {
  let resume!: (state: AppStateStatus) => void;
  const remove = jest.fn();
  jest.spyOn(AppState, 'addEventListener').mockImplementation((_event, listener) => {
    resume = listener; return { remove };
  });
  const { unmount } = renderHook(useSubscriptionClock);
  act(() => { jest.setSystemTime(end + 60000); resume('active'); });
  expect(useSubscriptionStore.getState().isTrialExpired).toBe(true);
  unmount(); expect(remove).toHaveBeenCalledTimes(1);
});

test('feature checks reconcile elapsed time even before any timer is delivered', () => {
  jest.setSystemTime(end);
  expect(useSubscriptionStore.getState().checkFeatureAccess('aiOutfits')).toBe(false);
  expect(useSubscriptionStore.getState().getTriesRemaining('aiOutfits', 0)).toBe(0);
});

test('an old scheduled boundary cannot restore the previous account trial', () => {
  const { unmount } = renderHook(useSubscriptionClock);
  act(() => useSubscriptionStore.setState({ trialStartedAt: null, isTrialPending: true,
    isTrialActive: false, isTrialExpired: false, effectiveTier: 'free' }));
  act(() => jest.advanceTimersByTime(2000));
  expect(useSubscriptionStore.getState().trialStartedAt).toBeNull();
  expect(useSubscriptionStore.getState().isTrialPending).toBe(true);
  expect(useSubscriptionStore.getState().effectiveTier).toBe('free');
  unmount();
});

test('unchanged clock ticks do not publish repeated store updates, and unmount removes timers', () => {
  const { unmount } = renderHook(useSubscriptionClock);
  const changed = jest.fn(); const unsubscribe = useSubscriptionStore.subscribe(changed);
  act(() => jest.advanceTimersByTime(500));
  expect(changed).not.toHaveBeenCalled();
  unmount(); unsubscribe();
  expect(jest.getTimerCount()).toBe(0);
});

test.each(['lite', 'premium', 'vip'] as const)('an open %s plan loses paid access at its deadline', tier => {
  useSubscriptionStore.setState({ tier, expiryDate: new Date(end).toISOString(), trialStartedAt: null,
    effectiveTier: tier, isTrialActive: false, isTrialExpired: false, isPremium: tier === 'premium' });
  const { unmount } = renderHook(useSubscriptionClock);
  expect(useSubscriptionStore.getState().effectiveTier).toBe(tier);
  act(() => jest.advanceTimersByTime(1000));
  const state = useSubscriptionStore.getState();
  expect(state.effectiveTier).toBe('free');
  expect(state.isPremium).toBe(false);
  expect(state.isSubscriptionExpired).toBe(true);
  expect(state.hasActiveSubscription).toBe(false);
  expect(state.tier).toBe(tier); // Preserve the last verified tier for reconciliation.
  expect(state.checkFeatureAccess('aiOutfits')).toBe(false);
  unmount();
});

test('paid expiry preserves any remaining original trial, then expires that trial at its own boundary', () => {
  const trialEnd = end + 2000;
  useSubscriptionStore.setState({ tier: 'premium', expiryDate: new Date(end).toISOString(),
    trialStartedAt: new Date(trialEnd - 7 * 86400000).toISOString(), effectiveTier: 'premium', isTrialActive: false });
  const { unmount } = renderHook(useSubscriptionClock);
  act(() => jest.advanceTimersByTime(1000));
  expect(useSubscriptionStore.getState().isTrialActive).toBe(true);
  expect(useSubscriptionStore.getState().effectiveTier).toBe('premium');
  act(() => jest.advanceTimersByTime(2000));
  expect(useSubscriptionStore.getState().effectiveTier).toBe('free');
  expect(useSubscriptionStore.getState().isTrialExpired).toBe(true);
  unmount();
});
