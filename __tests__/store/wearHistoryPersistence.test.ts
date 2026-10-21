import AsyncStorage from '@react-native-async-storage/async-storage';
import useWardrobeStore, { flushWardrobePersistence } from '../../store/wardrobeStore';
import type { WearLog } from '../../src/types/domain';

const history = () => Array.from({ length: 1500 }, (_, i) => ({
  id: `history-${i}`, userId: 'owner', itemIds: ['owned'], date: '2026-10-16',
  createdAt: '2026-10-16T00:00:00.000Z',
} as WearLog));
beforeEach(async () => {
  await flushWardrobePersistence();
  await useWardrobeStore.persist.rehydrate();
  useWardrobeStore.setState({ wearLogs: [], items: [], outfits: [], pendingActions: [] });
  await flushWardrobePersistence();
});
afterEach(async () => { await flushWardrobePersistence(); });

test('wear logging immediately attempts cloud sync and retains the queued log on failure', async () => {
  const original = useWardrobeStore.getState().syncToServer;
  const sync = jest.fn().mockRejectedValue(new Error('offline'));
  useWardrobeStore.setState({ syncToServer: sync });
  try {
    useWardrobeStore.getState().logWear(['owned']);
    await Promise.resolve();
    expect(sync).toHaveBeenCalledTimes(1);
    expect(useWardrobeStore.getState().pendingActions).toHaveLength(1);
    expect(useWardrobeStore.getState().wearLogs).toHaveLength(1);
  } finally {
    useWardrobeStore.setState({ syncToServer: original });
  }
});

test('logging another wear retains every older history entry', () => {
  const before = history();
  useWardrobeStore.setState({ wearLogs: before });
  useWardrobeStore.getState().logWear(['owned']);
  const after = useWardrobeStore.getState().wearLogs;
  expect(after).toHaveLength(1501);
  expect(after.slice(1)).toEqual(before);
  expect(useWardrobeStore.getState().pendingActions).toHaveLength(1);
});

test('full history survives persistence and rehydration beyond 1000 entries', async () => {
  const before = history();
  useWardrobeStore.setState({ wearLogs: before });
  await flushWardrobePersistence();
  const stored = await AsyncStorage.getItem('wardrobe-storage');
  expect(JSON.parse(stored!).state.wearLogs).toEqual(before);
  // Recreate the saved disk snapshot after clearing this store's memory.
  useWardrobeStore.setState({ wearLogs: [] });
  await flushWardrobePersistence();
  await AsyncStorage.setItem('wardrobe-storage', stored!);
  await useWardrobeStore.persist.rehydrate();
  expect(useWardrobeStore.getState().wearLogs).toEqual(before);
});
