import { authEvents } from '../../store/authEvents';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { clearAllPersistedUserData, resetInMemoryUserState, ensurePreviousAccountCleanup, ACCOUNT_CLEANUP_PENDING_KEY } from '../../src/lib/persistence';
import useSavedShopStore, { resetSavedShopState, flushSavedShopPersistence } from '../../store/savedShopStore';
import { clearAllPendingWardrobeImages } from '../../src/services/pendingWardrobeMedia';

jest.mock('../../src/services/pendingWardrobeMedia', () => ({ clearAllPendingWardrobeImages: jest.fn().mockResolvedValue(undefined) }));

jest.mock('../../src/utils/secureStorage', () => ({ deleteSecureItem: jest.fn().mockResolvedValue(undefined) }));
jest.mock('../../store/stylePreferenceStore', () => ({ useStylePreferenceStore: { getState: () => ({ clearAll: jest.fn() }) } }));
jest.mock('../../store/avatarStore', () => ({ __esModule: true, default: { setState: jest.fn() } }));
jest.mock('../../store/wardrobeStore', () => ({ __esModule: true, default: { getState: () => ({}), setState: jest.fn() }, clearWardrobePersistence: jest.fn().mockResolvedValue(undefined), flushWardrobePersistence: jest.fn().mockResolvedValue(undefined) }));
jest.mock('../../store/subscriptionStore', () => ({ __esModule: true, default: { setState: jest.fn() } }));
jest.mock('../../store/slices/wardrobeSyncSlice', () => ({ invalidateRehydrateState: jest.fn() }));
jest.mock('../../store/dailyUsageStore', () => ({ __esModule: true, drainDailyUsageWrites: jest.fn().mockResolvedValue(undefined), default: { setState: jest.fn() } }));
jest.mock('../../store/promoCodeStore', () => ({ __esModule: true, default: { setState: jest.fn() } }));
jest.mock('../../store/shopCatalogStore', () => ({ __esModule: true, default: { setState: jest.fn() } }));
jest.mock('../../store/bodyProfileStore', () => ({ __esModule: true, default: { getState: () => ({ reset: jest.fn() }) } }));
jest.mock('../../src/services/uploadQueue', () => ({ uploadQueue: { destroy: jest.fn(), clearQueue: jest.fn() } }));
jest.mock('../../src/lib/api', () => ({ clearCachedToken: jest.fn() }));
jest.mock('../../src/lib/queryClient', () => ({ clearQueryCache: jest.fn() }));

beforeEach(async () => {
  jest.clearAllMocks();
  await AsyncStorage.clear();
  authEvents.emitLogin('owner-a');
  resetSavedShopState();
  await flushSavedShopPersistence();
});

test('account cleanup awaits durable photo removal before acknowledging completion', async () => {
  let finish!: () => void;
  jest.mocked(clearAllPendingWardrobeImages).mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; }));
  let complete = false;
  const cleanup = clearAllPersistedUserData().then(() => { complete = true; });
  for (let tick = 0; tick < 100 && !finish && !complete; tick++) await new Promise(resolve => setTimeout(resolve, 0));
  expect(finish).toBeDefined();
  expect(complete).toBe(false);
  expect(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY)).toBe('1');
  finish();
  await cleanup;
  expect(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY)).toBeNull();
});

test('durable photo deletion failure retains the cleanup marker and retries', async () => {
  jest.mocked(clearAllPendingWardrobeImages).mockRejectedValueOnce(new Error('synthetic photo deletion failure'));
  await expect(clearAllPersistedUserData()).rejects.toThrow('Local account cleanup failed');
  expect(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY)).toBe('1');
  expect(require('../../src/utils/secureStorage').deleteSecureItem).toHaveBeenCalled();
  await ensurePreviousAccountCleanup();
  expect(clearAllPendingWardrobeImages).toHaveBeenCalledTimes(2);
  expect(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY)).toBeNull();
});

test('logout removes personal calendar, conversations and likes while retaining device preferences', async () => {
  const personalKeys = ['outfitLogs::owner-a', 'outfitLogs::owner-b', 'outfitLogs', '@outfit_logs', '@ai_chat_history', '@selected_style', 'savedInspo',
    'aiRecentShownItemIds', 'aiExcludeShownItemIds', 'aiExcludeShownItemIds::user-a', 'notificationLastScheduled', 'wardrobe-upload-queue'];
  for (const key of personalKeys) await AsyncStorage.setItem(key, 'private-user-a-data');
  await AsyncStorage.setItem('@app_language', 'uz');
  useSavedShopStore.setState({ items: [{ id: 'private-like' } as never], hydrated: true });
  await clearAllPersistedUserData();
  const orderedClear = require('../../store/wardrobeStore').clearWardrobePersistence as jest.Mock;
  expect(orderedClear).toHaveBeenCalled();
  const multiRemove = AsyncStorage.multiRemove as jest.Mock;
  const registryRemoval = multiRemove.mock.calls.findIndex(([keys]) => keys.includes('wardrobe-storage'));
  expect(registryRemoval).toBeGreaterThanOrEqual(0);
  expect(orderedClear.mock.invocationCallOrder[0]).toBeLessThan(multiRemove.mock.invocationCallOrder[registryRemoval]);
  for (const key of personalKeys) {
    const value = await AsyncStorage.getItem(key);
    if (key === 'wardrobe-upload-queue') expect(value).not.toContain('private-user-a-data');
    else expect(value).toBeNull();
  }
  expect(require('../../src/store/uploadQueueStore').default.getState().pendingUploads).toEqual([]);
  expect(await AsyncStorage.getItem('@app_language')).toBe('uz');
  expect(useSavedShopStore.getState()).toMatchObject({ items: [], hydrated: false });
});

test('a pending old-account hydration cannot restore likes after logout', async () => {
  let resolve!: (value: string | null) => void;
  (AsyncStorage.getItem as jest.Mock).mockImplementationOnce(() => new Promise<string | null>(done => { resolve = done; }));
  const pending = useSavedShopStore.getState().hydrate();
  await Promise.resolve();
  resetSavedShopState();
  resolve(JSON.stringify([{ id: 'old-account-like' }]));
  await pending;
  expect(useSavedShopStore.getState()).toMatchObject({ items: [], hydrated: false });
});

test('a fresh account hydration loads only its current storage', async () => {
  await AsyncStorage.setItem('savedInspo', JSON.stringify({version:1,ownerId:'owner-a',items:[{ id: 'current-account-like' }]}));
  await useSavedShopStore.getState().hydrate();
  expect(useSavedShopStore.getState().items.map(item => item.id)).toEqual(['current-account-like']);
});

test('logout waits for a late likes write and clears it after the registry removal', async () => {
  const set = AsyncStorage.setItem as jest.Mock, originalSet = set.getMockImplementation()!;
  const remove = AsyncStorage.multiRemove as jest.Mock, originalRemove = remove.getMockImplementation()!;
  let release!: () => void, started!: () => void, removed!: () => void;
  const active = new Promise<void>(resolve => { started = resolve; });
  const registryRemoved = new Promise<void>(resolve => { removed = resolve; });
  set.mockImplementationOnce((key: string, value: string) => new Promise<void>(resolve => {
    release = () => { Promise.resolve(originalSet(key, value)).then(() => resolve()); }; started();
  }));
  remove.mockImplementation(async (keys: string[]) => {
    await originalRemove(keys);
    if (keys.includes('savedInspo')) removed();
  });
  try {
    const write = useSavedShopStore.getState().toggleSave({ id: 'late-private-like' } as never);
    await active;
    const cleanup = clearAllPersistedUserData();
    await registryRemoved;
    release(); await Promise.all([write, cleanup]);
    expect(await AsyncStorage.getItem('savedInspo')).toBeNull();
    expect(useSavedShopStore.getState().items).toEqual([]);
  } finally { remove.mockImplementation(originalRemove); }
});

test('cleanup prepared for an older identity does not remove the current account data', async () => {
  const revision=authEvents.getRevision();
  authEvents.emitLogin('owner-b');
  await AsyncStorage.setItem('outfitLogs::owner-b','new-private-plan');
  jest.clearAllMocks();
  await clearAllPersistedUserData(revision);
  expect(AsyncStorage.multiRemove).not.toHaveBeenCalled();
  expect(await AsyncStorage.getItem('outfitLogs::owner-b')).toBe('new-private-plan');
});

test('an account change while cleanup scans keys prevents bulk deletion and memory reset', async () => {
  let finish!:(keys:string[])=>void, started!:()=>void;
  const active=new Promise<void>(resolve=>{started=resolve;});
  (AsyncStorage.getAllKeys as jest.Mock).mockImplementationOnce(()=>new Promise<string[]>(resolve=>{finish=resolve;started();}));
  const cleanup=clearAllPersistedUserData();await active;
  authEvents.emitLogin('owner-b');
  await useSavedShopStore.getState().toggleSave({id:'new-account-like'} as never);
  finish(['savedInspo','outfitLogs::owner-b']);await cleanup;
  expect((AsyncStorage.multiRemove as jest.Mock).mock.calls.some(([keys])=>keys.includes('wardrobe-storage'))).toBe(false);
  expect(useSavedShopStore.getState().items.map(item=>item.id)).toEqual(['new-account-like']);
  expect(JSON.parse((await AsyncStorage.getItem('savedInspo'))!).ownerId).toBe('owner-b');
});

test('cleanup does not reset new memory or delete secure keys after an already-issued removal finishes', async () => {
  const remove=AsyncStorage.multiRemove as jest.Mock, original=remove.getMockImplementation()!;
  let finish!:()=>void, started!:()=>void;
  const active=new Promise<void>(resolve=>{started=resolve;});
  remove.mockImplementation((keys:string[])=>{
    if (!keys.includes('wardrobe-storage')) return original(keys);
    return new Promise<void>(resolve=>{finish=()=>{Promise.resolve(original(keys)).then(()=>resolve());};started();});
  });
  const cleanup=clearAllPersistedUserData();await active;
  authEvents.emitLogin('owner-b');
  useSavedShopStore.setState({items:[{id:'new-memory'} as never],hydrated:true});
  finish();await cleanup;
  remove.mockImplementation(original);
  expect(useSavedShopStore.getState().items.map(item=>item.id)).toEqual(['new-memory']);
  expect(require('../../src/utils/secureStorage').deleteSecureItem).not.toHaveBeenCalled();
});

test('account cleanup waits for the upload-service deletion started by memory reset', async () => {
  const queue = require('../../src/services/uploadQueue').uploadQueue;
  let finish!: () => void, started!: () => void;
  const active = new Promise<void>(resolve => { started = resolve; });
  queue.clearQueue.mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; started(); }));
  let settled = false;
  const cleanup = clearAllPersistedUserData().then(() => { settled = true; });
  await active;
  // Drain queued microtasks without relying on a latency threshold.
  await new Promise<void>(resolve => setImmediate(resolve));
  try { expect(settled).toBe(false); }
  finally { finish(); await cleanup; }
  expect(settled).toBe(true);
});


test.each(['wardrobe', 'upload'])('cleanup waits for %s reset persistence after the registry wipe', async kind => {
  const module = kind === 'wardrobe' ? require('../../store/wardrobeStore') : require('../../src/store/uploadQueueStore');
  const key = kind === 'wardrobe' ? 'flushWardrobePersistence' : 'flushUploadQueuePersistence';
  let finish!: () => void, started!: () => void;
  const active = new Promise<void>(resolve => { started = resolve; });
  const flush = jest.spyOn(module, key).mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; started(); }));
  let settled = false;
  const cleanup = clearAllPersistedUserData().then(() => { settled = true; });
  await active;
  await new Promise<void>(resolve => setImmediate(resolve));
  try { expect(settled).toBe(false); }
  finally { finish(); await cleanup; flush.mockRestore(); }
  expect(settled).toBe(true);
});

test('memory reset reports a failed deletion after draining its other persistence work', async () => {
  const queue = require('../../src/services/uploadQueue').uploadQueue;
  queue.clearQueue.mockRejectedValueOnce(new Error('synthetic deletion failure'));
  const wardrobe = require('../../store/wardrobeStore');
  let finish!: () => void, started!: () => void;
  const active = new Promise<void>(resolve => { started = resolve; });
  wardrobe.flushWardrobePersistence.mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; started(); }));
  let settled = false;
  const reset = resetInMemoryUserState().finally(() => { settled = true; });
  const rejection = expect(reset).rejects.toThrow('Local account cleanup failed');
  await active;
  await new Promise<void>(resolve => setImmediate(resolve));
  try { expect(settled).toBe(false); }
  finally { finish(); await rejection; }
});

test('cleanup still attempts secure deletion before reporting a reset persistence failure', async () => {
  require('../../src/services/uploadQueue').uploadQueue.clearQueue.mockRejectedValueOnce(new Error('synthetic removal failure'));
  await expect(clearAllPersistedUserData()).rejects.toThrow('Local account cleanup failed');
  const secure = require('../../src/utils/secureStorage').deleteSecureItem;
  expect(secure).toHaveBeenCalledWith('supabase.auth.token');
  expect(secure).toHaveBeenCalledWith('supabase.auth.refreshToken');
});


test('failed registry removal retains the cleanup marker and a successful retry clears it', async () => {
  await AsyncStorage.setItem('@ai_chat_history', 'old-private-chat');
  const remove = jest.mocked(AsyncStorage.multiRemove), original = remove.getMockImplementation()!;
  let injected = false;
  remove.mockImplementation(keys => {
    if (keys.includes('wardrobe-storage') && !injected) {
      injected = true; return Promise.reject(new Error('synthetic registry failure'));
    }
    return original(keys);
  });
  try {
    await expect(clearAllPersistedUserData()).rejects.toThrow('Local account cleanup failed');
    expect(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY)).toBe('1');
    expect(await AsyncStorage.getItem('@ai_chat_history')).toBe('old-private-chat');
    await ensurePreviousAccountCleanup();
    expect(await AsyncStorage.getItem('@ai_chat_history')).toBeNull();
    expect(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY)).toBeNull();
  } finally { remove.mockImplementation(original); }
});

test('failed prefix scan still removes static keys and secure data and remains retryable', async () => {
  await AsyncStorage.setItem('outfitLogs::owner-a', 'old-private-plan');
  jest.mocked(AsyncStorage.getAllKeys).mockRejectedValueOnce(new Error('synthetic scan failure'));
  await expect(clearAllPersistedUserData()).rejects.toThrow('Local account cleanup failed');
  expect(AsyncStorage.multiRemove).toHaveBeenCalled();
  expect(require('../../src/utils/secureStorage').deleteSecureItem).toHaveBeenCalled();
  expect(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY)).toBe('1');
  await ensurePreviousAccountCleanup();
  expect(await AsyncStorage.getItem('outfitLogs::owner-a')).toBeNull();
});

test('secure deletion failure is reported after both keys are attempted', async () => {
  const secure = require('../../src/utils/secureStorage').deleteSecureItem as jest.Mock;
  secure.mockRejectedValueOnce(new Error('synthetic secure deletion failure'));
  await expect(clearAllPersistedUserData()).rejects.toThrow('Local account cleanup failed');
  expect(secure).toHaveBeenCalledWith('supabase.auth.token');
  expect(secure).toHaveBeenCalledWith('supabase.auth.refreshToken');
  expect(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY)).toBe('1');
  await ensurePreviousAccountCleanup();
  expect(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY)).toBeNull();
});

test('a fresh persistence module honors the saved incomplete-cleanup marker', async () => {
  await AsyncStorage.setItem(ACCOUNT_CLEANUP_PENDING_KEY, '1');
  await AsyncStorage.setItem('@ai_chat_history', 'previous-process-private-chat');
  let restarted!: typeof import('../../src/lib/persistence');
  jest.isolateModules(() => { restarted = require('../../src/lib/persistence'); });
  await restarted.ensurePreviousAccountCleanup();
  expect(await AsyncStorage.getItem('@ai_chat_history')).toBeNull();
  expect(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY)).toBeNull();
});

test('failure to read the cleanup marker is never treated as a clean account boundary', async () => {
  jest.mocked(AsyncStorage.getItem).mockRejectedValueOnce(new Error('synthetic marker read failure'));
  await expect(ensurePreviousAccountCleanup()).rejects.toThrow('synthetic marker read failure');
});


test('the marker is retained if its final removal fails', async () => {
  const remove = jest.mocked(AsyncStorage.removeItem), original = remove.getMockImplementation()!;
  let injected = false;
  remove.mockImplementation(key => {
    if (key === ACCOUNT_CLEANUP_PENDING_KEY && !injected) {
      injected = true; return Promise.reject(new Error('synthetic marker removal failure'));
    }
    return original(key);
  });
  try {
    await expect(clearAllPersistedUserData()).rejects.toThrow('Local account cleanup failed');
    expect(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY)).toBe('1');
    await ensurePreviousAccountCleanup();
    expect(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY)).toBeNull();
  } finally { remove.mockImplementation(original); }
});

test('failure to write the marker still requires an in-process cleanup retry', async () => {
  const set = jest.mocked(AsyncStorage.setItem), original = set.getMockImplementation()!;
  let injected = false, markerWrites = 0;
  set.mockImplementation((key, value) => {
    if (key === ACCOUNT_CLEANUP_PENDING_KEY) {
      markerWrites++;
      if (!injected) { injected = true; return Promise.reject(new Error('synthetic marker write failure')); }
    }
    return original(key, value);
  });
  try {
    await expect(clearAllPersistedUserData()).rejects.toThrow('Local account cleanup failed');
    expect(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY)).toBeNull();
    await ensurePreviousAccountCleanup();
    expect(markerWrites).toBe(2);
    expect(await AsyncStorage.getItem(ACCOUNT_CLEANUP_PENDING_KEY)).toBeNull();
  } finally { set.mockImplementation(original); }
});


test('cleanup drains delayed daily-usage writes before removing owner keys', async () => {
  const drain=require('../../store/dailyUsageStore').drainDailyUsageWrites;
  let finish!:()=>void;
  drain.mockImplementationOnce(()=>new Promise<void>(resolve=>{finish=resolve;}));
  let done=false;
  const pending=clearAllPersistedUserData().then(()=>{done=true;});
  for(let i=0;i<100&&!finish;i++)await new Promise(resolve=>setTimeout(resolve,0));
  expect(finish).toBeDefined();expect(done).toBe(false);
  await AsyncStorage.setItem('daily_usage_v2::owner-a','old account counts');
  finish();await pending;
  expect(await AsyncStorage.getItem('daily_usage_v2::owner-a')).toBeNull();
});
