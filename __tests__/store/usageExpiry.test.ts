const mockRpc = jest.fn();
jest.mock('../../lib/supabase', () => ({ supabase: { rpc: (...args: unknown[]) => mockRpc(...args) } }));
import useSubscriptionStore from '../../store/subscriptionStore';
import useDailyUsageStore from '../../store/dailyUsageStore';
import { authEvents } from '../../store/authEvents';
const end = Date.parse('2026-10-21T12:00:00Z');
beforeEach(() => {
  jest.useFakeTimers(); jest.setSystemTime(end - 1000); jest.clearAllMocks();
  authEvents.emitLogin('usage-expiry-owner');
  useSubscriptionStore.setState({ tier: 'premium', expiryDate: new Date(end).toISOString(),
    trialStartedAt: null, effectiveTier: 'premium', isTrialActive: false, isTrialPending: false });
  useDailyUsageStore.setState({ snapshot: { date: '2026-10-21', counts: {} } });
  mockRpc.mockResolvedValue({ data: null, error: { message: 'offline fixture' } });
});
afterEach(() => jest.useRealTimers());
test('direct quota reads reconcile an expired plan without a mounted clock or screen gate', () => {
  jest.setSystemTime(end);
  expect(useDailyUsageStore.getState().getRemaining('aiOutfits')).toBe(0);
  expect(useDailyUsageStore.getState().canUse('aiOutfits')).toBe(false);
  expect(useDailyUsageStore.getState().getRemaining('homeSwipes')).toBe(3);
  expect(useDailyUsageStore.getState().getRemaining('wardrobeScans')).toBe(5);
});
test('direct offline consumption cannot retain an expired paid allowance', async () => {
  jest.setSystemTime(end);
  expect((await useDailyUsageStore.getState().consume('aiOutfits')).allowed).toBe(false);
});
test('expiry while a failed quota request is pending cannot authorize from its earlier tier', async () => {
  let finish!: (value: unknown) => void;
  mockRpc.mockReturnValue(new Promise(resolve => { finish = resolve; }));
  const pending = useDailyUsageStore.getState().consume('aiOutfits');
  jest.setSystemTime(end);
  finish({ data: null, error: { message: 'offline fixture' } });
  expect((await pending).allowed).toBe(false);
});
