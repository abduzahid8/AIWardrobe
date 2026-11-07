import AsyncStorage from '@react-native-async-storage/async-storage';
const mockReadStatus = jest.fn();
jest.mock('../../lib/supabase', () => ({
  supabase: {
    auth: { getSession: async () => ({ data: { session: { user: { id: 'owner' } } } }) },
    rpc: (...args: unknown[]) => mockReadStatus(...args),
  },
}));
beforeEach(() => {
  jest.resetModules();
  jest.clearAllMocks();
  (AsyncStorage.setItem as jest.Mock).mockResolvedValue(undefined);
  (AsyncStorage.removeItem as jest.Mock).mockResolvedValue(undefined);
  mockReadStatus.mockResolvedValue({ data: [{ tier: 'free', expiry_date: null, trial_started_at: '2020-01-01T00:00:00Z' }], error: null });
});
const store = () => require('../../store/subscriptionStore').default;
test('an authoritative refund/free state overrides a future cached paid expiry', async () => {
  const s = store();
  await s.getState().setSubscription('vip', '2030-01-01T00:00:00Z');
  await s.getState().verifySubscriptionFromServer();
  expect(s.getState().tier).toBe('free');
  expect(s.getState().effectiveTier).toBe('free');
  expect(mockReadStatus).toHaveBeenCalledWith('get_subscription_status');
});
test('database read failure does not masquerade as a verified downgrade', async () => {
  const s = store();
  await s.getState().setSubscription('premium', '2030-01-01T00:00:00Z');
  mockReadStatus.mockResolvedValue({ data: null, error: new Error('unavailable') });
  await s.getState().verifySubscriptionFromServer();
  expect(s.getState().tier).toBe('premium');
});
test('verified paid access is applied from the server status', async () => {
  const s = store();
  mockReadStatus.mockResolvedValue({ data: [{ tier: 'premium', expiry_date: '2030-01-01T00:00:00Z', trial_started_at: null }], error: null });
  await s.getState().verifySubscriptionFromServer();
  expect(s.getState().tier).toBe('premium');
});
test('a verified lifetime entitlement preserves its actual tier and null expiry', async () => {
  const s = store();
  await s.getState().setSubscription('vip', '2030-01-01T00:00:00Z');
  mockReadStatus.mockResolvedValue({ data: [{ tier: 'premium', expiry_date: null, trial_started_at: '2020-01-01T00:00:00Z' }], error: null });
  await s.getState().verifySubscriptionFromServer();
  expect(s.getState().tier).toBe('premium');
  expect(s.getState().expiryDate).toBeNull();
  await s.getState().setSubscription('premium', null);
  expect(s.getState().expiryDate).toBeNull();
});


test('logout invalidates a pending authoritative status response before it can restore paid state', async () => {
  const s = store();
  let resolveRead: (value: unknown) => void = () => undefined;
  mockReadStatus.mockReturnValue(new Promise(resolve => { resolveRead = resolve; }));
  const pending = s.getState().verifySubscriptionFromServer();
  await Promise.resolve();
  const { authEvents } = require('../../store/authEvents');
  authEvents.emitLogout();
  s.setState({ tier: 'free', effectiveTier: 'free' });
  resolveRead({ data: [{ tier: 'vip', expiry_date: '2030-01-01T00:00:00Z', trial_started_at: null }], error: null });
  await pending;
  expect(s.getState().tier).toBe('free');
  expect(AsyncStorage.setItem).not.toHaveBeenCalledWith('subscription_tier', 'vip');
});

test('concurrent startup subscription checks share one authoritative request', async () => {
  const s = store();
  let finish: (value: unknown) => void = () => undefined;
  mockReadStatus.mockReturnValue(new Promise(resolve => { finish = resolve; }));
  const first = s.getState().verifySubscriptionFromServer();
  const second = s.getState().verifySubscriptionFromServer();
  await Promise.resolve();
  expect(mockReadStatus).toHaveBeenCalledTimes(1);
  finish({ data: [{ tier: 'premium', expiry_date: null, trial_started_at: '2020-01-01T00:00:00Z' }], error: null });
  await Promise.all([first, second]);
  expect(s.getState().tier).toBe('premium');
});

test('a network outage retains the last verified tier and is not logged as a crash', async () => {
  const s = store();
  await s.getState().setSubscription('premium', '2030-01-01T00:00:00Z');
  const errors = jest.spyOn(console, 'error').mockImplementation(() => {});
  const warnings = jest.spyOn(console, 'warn').mockImplementation(() => {});
  try {
    mockReadStatus.mockResolvedValue({data:null,error:{message:'fetch failed: The network connection was lost'}});
    await s.getState().verifySubscriptionFromServer();
    expect(s.getState().tier).toBe('premium');
    expect(errors).not.toHaveBeenCalled();
    expect(warnings).toHaveBeenCalled();
  } finally { errors.mockRestore(); warnings.mockRestore(); }
});


test('reconnect retries a failed in-flight subscription check after it settles', async () => {
  const s = store();
  let finish: (value: unknown) => void = () => undefined;
  mockReadStatus.mockReturnValueOnce(new Promise(resolve => { finish = resolve; }))
    .mockResolvedValueOnce({data:[{tier:'premium',expiry_date:null,trial_started_at:'2020-01-01T00:00:00Z'}],error:null});
  const initial = s.getState().verifySubscriptionFromServer();
  await Promise.resolve();
  const recovery = require('../../store/subscriptionStore').recoverSubscriptionAfterNetworkChange();
  finish({data:null,error:{message:'network connection was lost'}});
  await Promise.all([initial,recovery]);
  expect(mockReadStatus).toHaveBeenCalledTimes(2);
  expect(s.getState().tier).toBe('premium');
});

test('reconnect does not duplicate a successful in-flight subscription check', async () => {
  const s = store();
  let finish: (value: unknown) => void = () => undefined;
  mockReadStatus.mockReturnValueOnce(new Promise(resolve => { finish = resolve; }));
  const initial = s.getState().verifySubscriptionFromServer();
  await Promise.resolve();
  const recovery = require('../../store/subscriptionStore').recoverSubscriptionAfterNetworkChange();
  finish({data:[{tier:'premium',expiry_date:null,trial_started_at:'2020-01-01T00:00:00Z'}],error:null});
  await Promise.all([initial,recovery]);
  expect(mockReadStatus).toHaveBeenCalledTimes(1);
});

test('reconnect does not retry a pending subscription request after logout', async () => {
  const s = store();
  let finish: (value: unknown) => void = () => undefined;
  mockReadStatus.mockReturnValueOnce(new Promise(resolve => { finish = resolve; }));
  const initial = s.getState().verifySubscriptionFromServer();
  await Promise.resolve();
  const recovery = require('../../store/subscriptionStore').recoverSubscriptionAfterNetworkChange();
  require('../../store/authEvents').authEvents.emitLogout();
  finish({data:null,error:{message:'network connection was lost'}});
  await Promise.all([initial,recovery]);
  expect(mockReadStatus).toHaveBeenCalledTimes(1);
});

test('a pre-purchase free snapshot cannot overwrite a newly verified paid plan', async () => {
  const s = store();
  let finish!: (value: unknown) => void;
  mockReadStatus.mockReturnValueOnce(new Promise(resolve => { finish = resolve; }));
  const startup = s.getState().verifySubscriptionFromServer();
  await Promise.resolve();
  await s.getState().setSubscription('premium', '2030-01-01T00:00:00Z');
  finish({data:[{tier:'free',expiry_date:null,trial_started_at:new Date().toISOString()}],error:null});
  await startup;
  expect(s.getState().tier).toBe('premium');
  expect(s.getState().isTrialActive).toBe(false);
  expect(AsyncStorage.setItem).not.toHaveBeenCalledWith('subscription_tier', 'free');
});
