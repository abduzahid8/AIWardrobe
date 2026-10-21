const mockGetItem = jest.fn();
const mockSetItem = jest.fn();
jest.mock('@react-native-async-storage/async-storage', () => ({
  getItem: (...args: unknown[]) => mockGetItem(...args),
  setItem: (...args: unknown[]) => mockSetItem(...args),
}));
const mockRead = jest.fn();
const mockWrite = jest.fn();
jest.mock('../../lib/supabase', () => ({
  supabase: {
    from: () => ({
      select: () => ({ eq: () => ({ single: () => mockRead() }) }),
      update: () => ({ eq: () => ({ select: () => ({ single: () => mockWrite() }) }) }),
    }),
  },
}));
const confirmed = '2026-10-19T00:00:00.000Z';
beforeEach(() => {
  jest.resetModules();
  jest.clearAllMocks();
  mockGetItem.mockResolvedValue(null);
  mockSetItem.mockResolvedValue(undefined);
  mockRead.mockResolvedValue({ data: { trial_started_at: null }, error: null });
  mockWrite.mockResolvedValue({ data: { trial_started_at: confirmed }, error: null });
  jest.spyOn(console, 'error').mockImplementation(() => undefined);
});
afterEach(() => jest.restoreAllMocks());
const store = () => require('../../store/subscriptionStore').default;
test('persists the server-confirmed trial date rather than the device clock', async () => {
  const s = store();
  await s.getState().initializeTrial('owner');
  expect(mockWrite).toHaveBeenCalledTimes(1);
  expect(s.getState().trialStartedAt).toBe(confirmed);
  expect(mockSetItem).toHaveBeenCalledWith('trial_started_at_v1', confirmed);
});
test('failed server write cannot create a local trial', async () => {
  mockWrite.mockResolvedValue({ data: null, error: { message: 'unavailable' } });
  const s = store();
  await s.getState().initializeTrial('owner');
  expect(s.getState().trialStartedAt).toBeNull();
  expect(mockSetItem).not.toHaveBeenCalled();
});
test('failed server read cannot trigger a new trial write', async () => {
  mockRead.mockResolvedValue({ data: null, error: { message: 'unavailable' } });
  await store().getState().initializeTrial('owner');
  expect(mockWrite).not.toHaveBeenCalled();
  expect(mockSetItem).not.toHaveBeenCalled();
});
test('a stale device cache cannot override the recorded server trial', async () => {
  mockGetItem.mockResolvedValue('2099-01-01T00:00:00Z');
  mockRead.mockResolvedValue({ data: { trial_started_at: '2020-01-01T00:00:00Z' }, error: null });
  const s = store();
  await s.getState().initializeTrial('owner');
  expect(s.getState().trialStartedAt).toBe('2020-01-01T00:00:00Z');
  expect(mockWrite).not.toHaveBeenCalled();
});
test('logout during the server read prevents trial activation and writes', async () => {
  let resolveRead: (value: unknown) => void = () => undefined;
  mockRead.mockReturnValue(new Promise(resolve => { resolveRead = resolve; }));
  const s = store();
  const pending = s.getState().initializeTrial('owner');
  await Promise.resolve();
  require('../../store/authEvents').authEvents.emitLogout();
  resolveRead({ data: { trial_started_at: null }, error: null });
  await pending;
  expect(mockWrite).not.toHaveBeenCalled();
  expect(mockSetItem).not.toHaveBeenCalled();
});
test.each([null, 'invalid-date'])('an unconfirmed server date %s cannot create a trial', async trialDate => {
  mockWrite.mockResolvedValue({ data: { trial_started_at: trialDate }, error: null });
  const s = store();
  await s.getState().initializeTrial('owner');
  expect(s.getState().trialStartedAt).toBeNull();
  expect(mockSetItem).not.toHaveBeenCalled();
});
test('logout while the write is pending cannot activate its returned trial', async () => {
  let resolveWrite: (value: unknown) => void = () => undefined;
  mockWrite.mockReturnValue(new Promise(resolve => { resolveWrite = resolve; }));
  const s = store();
  const pending = s.getState().initializeTrial('owner');
  await Promise.resolve();
  expect(mockWrite).toHaveBeenCalled();
  require('../../store/authEvents').authEvents.emitLogout();
  resolveWrite({ data: { trial_started_at: confirmed }, error: null });
  await pending;
  expect(s.getState().trialStartedAt).toBeNull();
  expect(mockSetItem).not.toHaveBeenCalled();
});
