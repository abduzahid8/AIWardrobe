import { createStore } from 'zustand/vanilla';
import { createItemsSlice } from '../../store/slices/wardrobeItemsSlice';
import type { WardrobeState } from '../../store/wardrobeStore';
import type { ClothingItem } from '../../src/types/domain';
import { authEvents } from '../../store/authEvents';
const mockInsert = jest.fn();
jest.mock('../../lib/supabase', () => ({ supabase: { from: () => ({ insert: mockInsert }), auth: { getSession: async () => ({ data: { session: { user: { id: 'owner' } } } }) } } }));
jest.mock('../../store/auth', () => ({ __esModule: true, default: { getState: () => ({ user: { id: 'owner' } }) } }));
jest.mock('../../store/subscriptionStore', () => ({ __esModule: true, default: { getState: () => ({ tier: 'premium', effectiveTier: 'premium', refreshTimeState: () => {} }) }, TIER_FEATURES: { premium: { wardrobeItems: -1 } } }));
function fixture() {
  return createStore<WardrobeState>()((set, get, store) => ({ ...createItemsSlice(set, get, store),
    items: [{ id: 'item', isFavorite: false } as ClothingItem], outfits: [], pendingActions: [], syncToServer: jest.fn().mockResolvedValue(undefined),
  } as unknown as WardrobeState));
}
beforeEach(() => { jest.clearAllMocks(); authEvents.emitLogin('owner'); });

test('rapid favorite taps queue explicit ordered values, not toggles', () => {
  const state = fixture();
  state.getState().toggleFavorite('item'); state.getState().toggleFavorite('item');
  expect(state.getState().pendingActions.map(action => action.payload)).toEqual([{ itemId: 'item', isFavorite: true }, { itemId: 'item', isFavorite: false }]);
  expect(state.getState().items[0].isFavorite).toBe(false);
  expect(state.getState().syncToServer).toHaveBeenCalledTimes(2);
});

test('creation queues a stable UUID before starting sync and never inserts directly', async () => {
  const state = fixture();
  const sync = jest.fn().mockImplementation(async () => {
    const item = state.getState().items[0];
    expect(item.id).toMatch(/^[0-9a-f-]{36}$/i);
    expect(state.getState().pendingActions[0]).toMatchObject({ type: 'add_item', payload: { id: item.id } });
  });
  state.setState({ syncToServer: sync });
  await state.getState().addItem({ category: 'top', imageUrl: 'https://catalog.test/shirt.png', userId: 'owner' } as never);
  expect(sync).toHaveBeenCalledTimes(1);
  expect(mockInsert).not.toHaveBeenCalled();
});

test('a deletion failing after logout cannot enqueue work for the next account', async () => {
  let fail!: (error: Error) => void;
  const state = fixture();
  state.setState({ syncToServer: jest.fn().mockResolvedValue(undefined).mockImplementationOnce(() => new Promise((_, reject) => { fail = reject; })) });
  const pending = state.getState().removeItem('item');
  authEvents.emitLogout(); authEvents.emitLogin('next-owner');
  state.setState({ items: [{ id: 'next-item' } as ClothingItem], pendingActions: [] });
  fail(new Error('offline'));
  await pending;
  expect(state.getState().items.map(item => item.id)).toEqual(['next-item']);
  expect(state.getState().pendingActions).toEqual([]);
});

test('deleting during a queued save keeps ordered creation and removal for serialized sync', async () => {
  const state = fixture();
  let finish!: () => void;
  state.setState({ syncToServer: jest.fn().mockResolvedValue(undefined).mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; })) });
  const pending = state.getState().addItem({ category: 'top', imageUrl: 'https://catalog.test/shirt.png', userId: 'owner' } as never);
  const id = state.getState().items[0].id;
  await state.getState().removeItem(id);
  finish(); await pending;
  expect(state.getState().items.some(item => item.id === id)).toBe(false);
  expect(state.getState().pendingActions.map(action => action.type)).toEqual(['add_item', 'remove_item']);
  expect(state.getState().pendingActions[1].payload.itemId).toBe(id);
  expect(mockInsert).not.toHaveBeenCalled();
});

test('a favorite changed during queued creation remains local and queues after creation', async () => {
  const state = fixture();
  let finish!: () => void;
  state.setState({ syncToServer: jest.fn().mockResolvedValue(undefined).mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve; })) });
  const pending = state.getState().addItem({ category: 'top', imageUrl: 'https://catalog.test/shirt.png', userId: 'owner' } as never);
  state.getState().toggleFavorite(state.getState().items[0].id);
  finish(); await pending;
  expect(state.getState().items[0].isFavorite).toBe(true);
  expect(state.getState().pendingActions.map(action => action.type)).toEqual(['add_item', 'update_item']);
});
