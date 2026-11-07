/** @jest-environment node */
const mockDisk = new Map<string, string>();
jest.mock('@react-native-async-storage/async-storage', () => ({
  getItem: jest.fn(async (key: string) => mockDisk.get(key) || null),
  setItem: jest.fn(async (key: string, value: string) => {
    mockDisk.set(key, value);
  }),
  removeItem: jest.fn(async (key: string) => {
    mockDisk.delete(key);
  }),
}));
import { authEvents } from '../../store/authEvents';
import {
  usePurchaseIntelligenceStore as store,
  resetPurchaseIntelligence,
} from '../../store/purchaseIntelligenceStore';
const answer = {
  productId: 'garment',
  brand: 'Brand',
  category: 'top',
  sizeLabel: 'M',
  overallFit: 'perfect' as const,
  source: 'owned' as const,
  confidence: 0.8,
  kept: true,
};
beforeEach(async () => {
  await resetPurchaseIntelligence();
  mockDisk.clear();
  authEvents.emitLogin('user-a');
  await store.getState().hydrate();
});
test('one tap persists owner-tagged evidence and suppresses repeated prompts; optional zones and reference remain editable', async () => {
  expect(store.getState().canPrompt('garment')).toBe(true);
  const result = await store.getState().recordFit(answer);
  expect(store.getState().canPrompt('garment')).toBe(false);
  await store.getState().updateFit(result.id, { zones: { shoulders: 'tight' }, reference: true });
  expect(store.getState().evidence[0]).toMatchObject({
    userId: 'user-a',
    zones: { shoulders: 'tight' },
    reference: true,
  });
  expect([...mockDisk.values()][0]).toContain('"ownerId":"user-a"');
});
test('dismiss cooldown and disabled preference do not block any core wardrobe action', async () => {
  const now = Date.now();
  await store.getState().dismiss('garment', now);
  expect(store.getState().canPrompt('garment', now + 1000)).toBe(false);
  expect(store.getState().canPrompt('garment', now + 15 * 86400000)).toBe(true);
  await store.getState().setFeedbackEnabled(false);
  expect(store.getState().canPrompt('other')).toBe(false);
});
test('another account cannot hydrate the previous private evidence, and reset drains pending writes', async () => {
  await store.getState().recordFit(answer);
  authEvents.emitLogin('user-b');
  await store.getState().hydrate();
  expect(store.getState().evidence).toEqual([]);
  const pending = store.getState().recordFit({ ...answer, productId: 'b' });
  const reset = resetPurchaseIntelligence();
  await pending;
  await reset;
  expect(mockDisk.size).toBe(0);
  authEvents.emitLogout();
  expect(store.getState().evidence).toEqual([]);
});
test('local history and price watches remain bounded and owner-specific', async () => {
  for (let i = 0; i < 165; i++)
    await store.getState().recordFit({ ...answer, productId: String(i) });
  expect(store.getState().evidence).toHaveLength(160);
  await expect(
    store.getState().saveWatch({ productId: 'p', currency: 'USD', threshold: -1, status: 'active' })
  ).rejects.toThrow();
  await store
    .getState()
    .saveWatch({ productId: 'p', currency: 'MYR', threshold: 350, status: 'active' });
  expect(store.getState().watches[0].userId).toBe('user-a');
});

test('a slow hydration cannot publish fit evidence into a different owner', async () => {
  const storage = require('@react-native-async-storage/async-storage');
  await store.getState().recordFit(answer);
  const old = [...mockDisk.values()][0];
  let finish!: (value: string) => void;
  store.setState({ hydrated: false });
  storage.getItem.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finish = resolve;
      })
  );
  const pending = store.getState().hydrate();
  await Promise.resolve();
  authEvents.emitLogin('user-b');
  finish(old);
  await pending;
  expect(store.getState().ownerId).toBe('user-b');
  expect(store.getState().evidence).toEqual([]);
});

test('manual watch evaluation records local status without pretending to send a notification', async () => {
  await store
    .getState()
    .saveWatch({ productId: 'p', currency: 'USD', threshold: 80, status: 'active' });
  const id = store.getState().watches[0].id;
  await store.getState().markWatchChecked(id, true);
  expect(store.getState().watches[0]).toMatchObject({ status: 'triggered' });
  expect(store.getState().watches[0].lastCheckedAt).toBeDefined();
  expect(store.getState().watches[0].lastNotifiedOfferId).toBeUndefined();
});
