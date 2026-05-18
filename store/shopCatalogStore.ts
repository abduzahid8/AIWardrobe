/**
 * store/shopCatalogStore.ts
 *
 * Lightweight in-memory store that holds the shop catalog items fetched by
 * HomeScreen (and InspoScreen).  ProfileScreen reads from here instead of
 * calling useShopCatalog() directly, eliminating the redundant third
 * Supabase query on app startup (Defect 1.5).
 *
 * This store is intentionally NOT persisted — the catalog is always
 * re-fetched from Supabase on app launch by HomeScreen, so there is no
 * value in writing it to AsyncStorage.
 */

import { create } from 'zustand';
import type { ShopCatalogItem } from '../features/try-on/types';

interface ShopCatalogState {
  /** All catalog items loaded so far (may grow as HomeScreen paginates). */
  items: ShopCatalogItem[];
  /** True while the initial page is being fetched. */
  loading: boolean;
  /** Replace the full item list (called after each page load). */
  setItems: (items: ShopCatalogItem[]) => void;
  /** Append additional pages without discarding existing items. */
  appendItems: (items: ShopCatalogItem[]) => void;
  /** Update loading flag. */
  setLoading: (loading: boolean) => void;
}

const useShopCatalogStore = create<ShopCatalogState>()((set) => ({
  items: [],
  loading: true,

  setItems: (items) => set({ items }),
  appendItems: (newItems) =>
    set((state) => {
      const existingIds = new Set(state.items.map((i) => i.id));
      const deduped = newItems.filter((i) => !existingIds.has(i.id));
      return { items: [...state.items, ...deduped] };
    }),
  setLoading: (loading) => set({ loading }),
}));

export default useShopCatalogStore;
