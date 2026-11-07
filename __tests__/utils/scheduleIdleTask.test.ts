import { scheduleIdleTask } from '../../src/lib/scheduleIdleTask';

const originalIdle = global.requestIdleCallback;
const originalCancel = global.cancelIdleCallback;
beforeEach(() => jest.useFakeTimers());
afterEach(() => {
  global.requestIdleCallback = originalIdle;
  global.cancelIdleCallback = originalCancel;
  jest.useRealTimers();
});

test('an available idle callback cannot bypass the minimum startup delay', () => {
  const task = jest.fn();
  global.requestIdleCallback = jest.fn(callback => {
    callback({ didTimeout: false, timeRemaining: () => 16 });
    return 1;
  });
  scheduleIdleTask(task, 1500);
  jest.advanceTimersByTime(1000);
  expect(task).not.toHaveBeenCalled();
  expect(global.requestIdleCallback).not.toHaveBeenCalled();
  jest.advanceTimersByTime(500);
  expect(task).toHaveBeenCalledTimes(1);
});

test('a busy frame postpones the task and cancellation removes the pending callback', () => {
  let callback!: IdleRequestCallback;
  global.requestIdleCallback = jest.fn(fn => { callback = fn; return 7; });
  global.cancelIdleCallback = jest.fn();
  const task = jest.fn(), cancel = scheduleIdleTask(task, 1000);
  jest.advanceTimersByTime(1000);
  callback({ didTimeout: false, timeRemaining: () => 1 });
  expect(task).not.toHaveBeenCalled();
  jest.advanceTimersByTime(50);
  cancel();
  expect(global.cancelIdleCallback).toHaveBeenCalledWith(7);
  callback({ didTimeout: false, timeRemaining: () => 16 });
  expect(task).not.toHaveBeenCalled();
});

test('cancelling before the delay prevents idle work from being registered', () => {
  global.requestIdleCallback = jest.fn();
  const cancel = scheduleIdleTask(jest.fn(), 1000);
  cancel();
  jest.runAllTimers();
  expect(global.requestIdleCallback).not.toHaveBeenCalled();
});
