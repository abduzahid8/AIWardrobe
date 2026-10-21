import { createDebouncedStorage } from '../../src/lib/debouncedStorage';
const backend = { getItem: jest.fn(), setItem: jest.fn(), removeItem: jest.fn() };
beforeEach(() => { jest.useFakeTimers(); jest.clearAllMocks(); backend.setItem.mockResolvedValue(undefined); backend.removeItem.mockResolvedValue(undefined); });
afterEach(() => { jest.useRealTimers(); });

test('coalesces writes but settles every caller after writing the latest value', async () => {
  const storage = createDebouncedStorage(backend);
  const first = storage.setItem('wardrobe', 'old'), second = storage.setItem('wardrobe', 'new');
  await jest.advanceTimersByTimeAsync(500); await Promise.all([first, second]);
  expect(backend.setItem.mock.calls).toEqual([['wardrobe', 'new']]);
});

test('remove cancels a pending private write and settles its promise', async () => {
  const storage = createDebouncedStorage(backend);
  const pending = storage.setItem('wardrobe', 'private-old-account');
  await storage.removeItem('wardrobe'); await pending; await jest.runAllTimersAsync();
  expect(backend.setItem).not.toHaveBeenCalled(); expect(backend.removeItem).toHaveBeenCalledWith('wardrobe');
});

test('removal waits behind an already running write so old data cannot reappear', async () => {
  let finish!: () => void;
  backend.setItem.mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; }));
  const storage = createDebouncedStorage(backend);
  const pending = storage.setItem('wardrobe', 'private-old-account');
  await jest.advanceTimersByTimeAsync(500);
  const removal = storage.removeItem('wardrobe');
  expect(backend.removeItem).not.toHaveBeenCalled();
  finish(); await pending; await removal;
  expect(backend.removeItem).toHaveBeenCalledTimes(1);
});

test('write errors reject awaited callers and do not stall later writes', async () => {
  backend.setItem.mockRejectedValueOnce(new Error('disk full'));
  const onError = jest.fn(), storage = createDebouncedStorage(backend, 500, onError);
  const pending = storage.setItem('wardrobe', 'failed');
  const rejection = expect(pending).rejects.toThrow('disk full');
  await jest.advanceTimersByTimeAsync(500); await rejection;
  expect(onError).toHaveBeenCalled();
  const next = storage.setItem('wardrobe', 'saved'); await storage.flush(); await next;
  expect(backend.setItem).toHaveBeenLastCalledWith('wardrobe', 'saved');
});

test('reads see the latest pending value and explicit flush persists without waiting for the timer', async () => {
  const storage = createDebouncedStorage(backend);
  const pending = storage.setItem('wardrobe', 'latest');
  expect(await storage.getItem('wardrobe')).toBe('latest');
  expect(backend.setItem).not.toHaveBeenCalled();
  await storage.flush(); await pending;
  expect(backend.setItem).toHaveBeenCalledWith('wardrobe', 'latest');
});

test('a read resolving after removal cannot return the previous account cache', async () => {
  let finish!: (value: string | null) => void;
  backend.getItem.mockImplementationOnce(() => new Promise<string | null>(resolve => { finish = resolve; }));
  const storage = createDebouncedStorage(backend);
  const pending = storage.getItem('wardrobe'); await Promise.resolve();
  await storage.removeItem('wardrobe'); finish('private-old-account');
  expect(await pending).toBeNull();
});

test('initial hydration reads disk at invocation instead of a later optimistic snapshot', async () => {
  backend.getItem.mockResolvedValueOnce(null);
  const storage = createDebouncedStorage(backend);
  const initial = storage.getItem('wardrobe');
  const pending = storage.setItem('wardrobe', 'intermediate-optimistic-state');
  expect(await initial).toBeNull();
  await storage.removeItem('wardrobe'); await pending;
});

test('flush reports a failed background write until the key is successfully removed', async () => {
  backend.setItem.mockRejectedValueOnce(new Error('background disk failure'));
  const storage = createDebouncedStorage(backend);
  const write = storage.setItem('private-cache', 'sensitive');
  const rejected = expect(write).rejects.toThrow('background disk failure');
  await jest.advanceTimersByTimeAsync(500); await rejected;
  await expect(storage.flush()).rejects.toThrow('background disk failure');
  await storage.removeItem('private-cache');
  await expect(storage.flush()).resolves.toBeUndefined();
});

test('flush started during a write preserves its later rejection', async () => {
  let fail!: (error: Error) => void;
  backend.setItem.mockImplementationOnce(() => new Promise<void>((_resolve, reject) => { fail = reject; }));
  const storage = createDebouncedStorage(backend);
  const write = storage.setItem('private-cache', 'sensitive');
  const rejected = expect(write).rejects.toThrow('in-flight disk failure');
  await jest.advanceTimersByTimeAsync(500);
  const flush = storage.flush();
  const flushRejected = expect(flush).rejects.toThrow('in-flight disk failure');
  fail(new Error('in-flight disk failure'));
  await Promise.all([rejected, flushRejected]);
});

test('flush drains every key before reporting a failure', async () => {
  let finish!: () => void;
  backend.setItem.mockImplementationOnce(async () => { throw new Error('first key failed'); })
    .mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; }));
  const storage = createDebouncedStorage(backend);
  const failed = storage.setItem('failed-key', 'old');
  const held = storage.setItem('held-key', 'new');
  const rejected = expect(failed).rejects.toThrow('first key failed');
  let settled = false;
  const flush = storage.flush().then(() => { settled = true; }, error => { settled = true; throw error; });
  const flushRejected = expect(flush).rejects.toThrow('first key failed');
  await jest.advanceTimersByTimeAsync(0);
  try { expect(settled).toBe(false); }
  finally { finish(); await Promise.all([held, rejected, flushRejected]); }
});
