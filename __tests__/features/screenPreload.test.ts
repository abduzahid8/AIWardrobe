import { scheduleScreenPreloads } from '../../navigation/screenPreload';
beforeEach(() => jest.useFakeTimers());
afterEach(() => jest.useRealTimers());

test('preloads serially with an idle gap, without blocking the first render', async () => {
  let finish!: () => void;
  const first = jest.fn(() => new Promise<void>(resolve => { finish = resolve; }));
  const second = jest.fn();
  const cancel = scheduleScreenPreloads([first, second]);
  expect(first).not.toHaveBeenCalled();
  await jest.advanceTimersByTimeAsync(1000);
  expect(first).toHaveBeenCalledTimes(1);
  await jest.advanceTimersByTimeAsync(2000);
  expect(second).not.toHaveBeenCalled();
  finish();
  await jest.advanceTimersByTimeAsync(500);
  expect(second).toHaveBeenCalledTimes(1);
  cancel();
});
test('cancellation prevents queued modules from loading', async () => {
  const load = jest.fn(), cancel = scheduleScreenPreloads([load]);
  cancel();
  await jest.runAllTimersAsync();
  expect(load).not.toHaveBeenCalled();
});
test('startup preloading respects a quiet period after opening', async () => {
  const load = jest.fn(), cancel = scheduleScreenPreloads([load], 4000);
  await jest.advanceTimersByTimeAsync(1000);
  expect(load).not.toHaveBeenCalled();
  await jest.advanceTimersByTimeAsync(2999);
  expect(load).not.toHaveBeenCalled();
  await jest.advanceTimersByTimeAsync(1);
  expect(load).toHaveBeenCalledTimes(1);
  cancel();
});
test('failed imports do not block later modules', async () => {
  const next = jest.fn(), cancel = scheduleScreenPreloads([() => Promise.reject(new Error('offline')), next]);
  await jest.runAllTimersAsync();
  expect(next).toHaveBeenCalledTimes(1);
  cancel();
});
