/**
 * store/savedShopStore.ts
 *
 * Shared source of truth for liked / saved shop catalog items (Inspo → Shop tab heart).
 *
 * Previously InspoScreen kept this list in local component state + AsyncStorage
 * key `savedInspo`, so ProfileScreen's Saved Looks never saw shop likes.
 * Both screens now read/write here, so a heart in Shop instantly appears in
 * Profile → Saved Looks.
 *
 * Storage: reuses the legacy `savedInspo` AsyncStorage key so existing users
 * keep their saves. Writes are debounced through the toggle/remove actions.
 */

import { create } from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';
import type { ShopCatalogItem } from '../src/types/shop';

const STORAGE_KEY = 'savedInspo';

function normalizeItems(raw: unknown): ShopCatalogItem[] {
  if (!Array.isArray(raw)) return [];
  return raw.filter(
    (item): item is ShopCatalogItem =>
      !!item && typeof item === 'object' && typeof (item as { id?: unknown }).id !== 'undefined',
  ) as ShopCatalogItem[];
}

interface SavedShopState {
  items: ShopCatalogItem[];
  hydrated: boolean;
  /** Load from AsyncStorage (idempotent). Called on screen focus. */
  hydrate: () => Promise<void>;
  /** Toggle a save. Returns true when the item is now saved. */
  toggleSave: (item: ShopCatalogItem) => Promise<boolean>;
  removeSave: (id: string) => Promise<void>;
  isSaved: (id: string) => boolean;
  setItems: (items: ShopCatalogItem[]) => Promise<void>;
  clearAll: () => Promise<void>;
}

async function persistItems(items: ShopCatalogItem[]) {
  try {
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(items));
  } catch {
    // Best-effort — saves stay in memory for the session.
  }
}

const useSavedShopStore = create<SavedShopState>()((set, get) => ({
  items: [],
  hydrated: false,

  hydrate: async () => {
    try {
      const raw = await AsyncStorage.getItem(STORAGE_KEY);
      if (raw) {
        const parsed = normalizeItems(JSON.parse(raw));
        set({ items: parsed, hydrated: true });
        return;
      }
    } catch {
      // Fall through to hydrated=true with in-memory items.
    }
    set({ hydrated: true });
  },

  toggleSave: async (item) => {
    const { items } = get();
    const id = String(item.id);
    const alreadySaved = items.some((s) => String(s.id) === id);
    const next = alreadySaved
      ? items.filter((s) => String(s.id) !== id)
      : [...items, item];
    set({ items: next, hydrated: true });
    await persistItems(next);
    return !alreadySaved;
  },

  removeSave: async (id) => {
    const next = get().items.filter((s) => String(s.id) !== String(id));
    set({ items: next, hydrated: true });
    await persistItems(next);
  },

  isSaved: (id) => get().items.some((s) => String(s.id) === String(id)),

  setItems: async (items) => {
    const normalized = normalizeItems(items);
    set({ items: normalized, hydrated: true });
    await persistItems(normalized);
  },

  clearAll: async () => {
    set({ items: [], hydrated: true });
    await persistItems([]);
  },
}));

export default useSavedShopStore;
export { STORAGE_KEY as SAVED_SHOP_STORAGE_KEY };
