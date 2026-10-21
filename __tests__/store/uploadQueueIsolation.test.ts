import { authEvents } from '../../store/authEvents';
import AsyncStorage from '@react-native-async-storage/async-storage';
import useUploadQueueStore, { resetUploadQueueState, flushUploadQueuePersistence } from '../../src/store/uploadQueueStore';

beforeEach(async () => {
  authEvents.emitLogin('owner');
  resetUploadQueueState();
  await flushUploadQueuePersistence();
  await AsyncStorage.clear();
});

afterEach(async () => { await flushUploadQueuePersistence(); });

test('account reset removes queued private media from memory and disk', async () => {
  useUploadQueueStore.getState().addUpload('file:///private-user-a-video.mov', 'video');
  await flushUploadQueuePersistence();
  expect(await AsyncStorage.getItem('wardrobe-upload-queue')).toContain('private-user-a-video');
  resetUploadQueueState();
  await flushUploadQueuePersistence();
  expect(useUploadQueueStore.getState().pendingUploads).toEqual([]);
  expect(await AsyncStorage.getItem('wardrobe-upload-queue')).not.toContain('private-user-a-video');
});

test('an old pending hydration cannot restore media after account reset', async () => {
  let finish!: (value: string | null) => void;
  (AsyncStorage.getItem as jest.Mock).mockImplementationOnce(() => new Promise(resolve => { finish = resolve; }));
  const pending = useUploadQueueStore.persist.rehydrate();
  resetUploadQueueState();
  finish(JSON.stringify({ state: { pendingUploads: [{ id: 'old', uri: 'private-a', type: 'image' }] }, version: 0 }));
  await pending;
  expect(useUploadQueueStore.getState().pendingUploads).toEqual([]);
});

test('reset cancels an old queued write while preserving new-account uploads', async () => {
  useUploadQueueStore.getState().addUpload('private-a', 'image');
  resetUploadQueueState();
  useUploadQueueStore.getState().addUpload('private-b', 'image');
  await flushUploadQueuePersistence();
  expect(useUploadQueueStore.getState().pendingUploads.map(item => item.uri)).toEqual(['private-b']);
  const raw = await AsyncStorage.getItem('wardrobe-upload-queue');
  expect(raw).toContain('private-b');
  expect(raw).not.toContain('private-a');
});


test('entries retain their owner and another account cannot mutate or count them', () => {
  expect(useUploadQueueStore.getState().addUpload('private-a', 'image')).toBe(true);
  const entry = useUploadQueueStore.getState().pendingUploads[0];
  expect(entry.ownerId).toBe('owner');
  authEvents.emitLogin('other');
  const store = useUploadQueueStore.getState();
  expect(store.getPendingCount()).toBe(0);
  store.updateStatus(entry.id, 'failed'); store.incrementRetry(entry.id); store.removeUpload(entry.id);
  expect(useUploadQueueStore.getState().pendingUploads[0]).toEqual(entry);
});
test('signed-out input is not accepted or reported as saved', () => {
  authEvents.emitLogout();
  expect(useUploadQueueStore.getState().addUpload('private', 'image')).toBe(false);
  expect(useUploadQueueStore.getState().pendingUploads).toEqual([]);
});
