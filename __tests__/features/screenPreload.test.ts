import { scheduleScreenPreloads } from '../../navigation/screenPreload';

beforeEach(() => jest.useFakeTimers());
afterEach(() => { jest.useRealTimers(); });

test('defers evaluation, loads serially, and yields between modules', async () => {
  let finish!: () => void;
  const first = jest.fn(() => new Promise<void>(resolve => { finish = resolve; }));
  const second = jest.fn();
  const cancel = scheduleScreenPreloads([first, second]);
  expect(first).not.toHaveBeenCalled();
  await jest.advanceTimersByTimeAsync(1000);
  expect(first).toHaveBeenCalledTimes(1);
  await jest.advanceTimersByTimeAsync(3000);
  expect(second).not.toHaveBeenCalled();
  finish();
  await jest.advanceTimersByTimeAsync(499);
  expect(second).not.toHaveBeenCalled();
  await jest.advanceTimersByTimeAsync(1);
  expect(second).toHaveBeenCalledTimes(1);
  cancel();
});

test('cancelling prevents queued screens from loading after logout', async () => {
  const load = jest.fn();
  const cancel = scheduleScreenPreloads([load]);
  cancel();
  await jest.runAllTimersAsync();
  expect(load).not.toHaveBeenCalled();
});

test('a failed module does not stop later preloads', async () => {
  const next = jest.fn();
  const cancel = scheduleScreenPreloads([() => Promise.reject(new Error('offline')), next]);
  await jest.runAllTimersAsync();
  expect(next).toHaveBeenCalledTimes(1);
  cancel();
});
