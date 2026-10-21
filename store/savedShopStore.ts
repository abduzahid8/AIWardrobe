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
 * Storage: owner-tagged records use the legacy `savedInspo` key. Untagged
 * records cannot safely be attributed to an account and are not hydrated.
 * Writes and account cleanup share one ordered queue.
 */

import { create } from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { authEvents } from './authEvents';
import type { ShopCatalogItem } from '../src/types/shop';

const STORAGE_KEY = 'savedInspo';
let ownerId = authEvents.getUserId();
let hydrationRevision = 0;
let mutationRevision = 0;
let diskTrusted = true;
let persistenceWork: Promise<void> = Promise.resolve();
let hydrationWork: { revision: number; promise: Promise<void> } | null = null;

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
  /** Commit a toggle to storage. Returns true when saved; rejects on write failure. */
  toggleSave: (item: ShopCatalogItem) => Promise<boolean>;
  removeSave: (id: string) => Promise<void>;
  isSaved: (id: string) => boolean;
  setItems: (items: ShopCatalogItem[]) => Promise<void>;
  clearAll: () => Promise<void>;
}

function syncOwner() {
  const nextOwner = authEvents.getUserId();
  if (nextOwner === ownerId) return;
  ownerId = nextOwner;
  hydrationRevision++;
  mutationRevision++;
  diskTrusted = true;
  useSavedShopStore.setState({ items: [], hydrated: false });
}

/** Commit each mutation from the latest confirmed list, in storage order. */
function commitItems(change: (items: ShopCatalogItem[]) => ShopCatalogItem[], revision: number) {
  const owner = ownerId;
  const isCurrent = () => !!owner && revision === hydrationRevision && owner === authEvents.getUserId();
  mutationRevision++;
  const operation = persistenceWork.then(async () => {
    if (!isCurrent()) return null;
    const next = change(useSavedShopStore.getState().items);
    await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 1, ownerId: owner, items: next }));
    if (!isCurrent()) return null;
    diskTrusted = true;
    useSavedShopStore.setState({ items: next, hydrated: true });
    return next;
  });
  // Keep the queue usable after a failure, but return the original rejection
  // to the caller so UI feedback reflects the actual persistence result.
  persistenceWork = operation.then(() => undefined, () => undefined);
  return operation;
}

const useSavedShopStore = create<SavedShopState>()((set, get) => ({
  items: [],
  hydrated: false,

  hydrate: async () => {
    syncOwner();
    const owner = ownerId;
    if (!owner || get().hydrated) return;
    const revision = hydrationRevision;
    if (hydrationWork?.revision === revision) return hydrationWork.promise;
    const mutation = mutationRevision;
    const promise = (async () => {
      try {
        await persistenceWork;
        if (revision !== hydrationRevision || mutation !== mutationRevision) return;
        if (!diskTrusted) { set({ hydrated: true }); return; }
        const raw = await AsyncStorage.getItem(STORAGE_KEY);
        if (revision !== hydrationRevision || mutation !== mutationRevision) return;
        let items: ShopCatalogItem[] = [];
        if (raw) {
          const record = JSON.parse(raw);
          items = record?.version === 1 && record.ownerId === owner
            ? normalizeItems(record.items) : [];
        }
        set({ items, hydrated: true });
      } catch {
        // Leave initial hydration incomplete: a mutation must not overwrite an
        // unread cache after an I/O or parsing failure. A later focus can retry.
      }
    })();
    hydrationWork = { revision, promise };
    try { await promise; }
    finally { if (hydrationWork?.promise === promise) hydrationWork = null; }
  },

  toggleSave: async (item) => {
    syncOwner();
    if (!ownerId) return false;
    const revision = hydrationRevision;
    if (!get().hydrated) {
      await get().hydrate();
      if (revision !== hydrationRevision) return false;
      if (!get().hydrated) throw new Error('Saved items could not be loaded');
    }
    let alreadySaved = false;
    const committed = await commitItems((items) => {
      const id = String(item.id);
      alreadySaved = items.some((saved) => String(saved.id) === id);
      return alreadySaved ? items.filter((saved) => String(saved.id) !== id) : [...items, item];
    }, revision);
    return committed !== null && !alreadySaved;
  },

  removeSave: async (id) => {
    syncOwner();
    if (!ownerId) return;
    const revision = hydrationRevision;
    if (!get().hydrated) {
      await get().hydrate();
      if (revision !== hydrationRevision) return;
      if (!get().hydrated) throw new Error('Saved items could not be loaded');
    }
    await commitItems((items) => items.filter((saved) => String(saved.id) !== String(id)), revision);
  },

  isSaved: (id) => get().items.some((s) => String(s.id) === String(id)),

  setItems: async (items) => {
    syncOwner();
    if (!ownerId) return;
    const revision = hydrationRevision;
    const normalized = normalizeItems(items);
    await commitItems(() => normalized, revision);
  },

  clearAll: async () => {
    syncOwner();
    if (!ownerId) return;
    const revision = hydrationRevision;
    await commitItems(() => [], revision);
  },
}));

export function resetSavedShopState(): void {
  hydrationRevision++;
  mutationRevision++;
  diskTrusted = false;
  const revision = hydrationRevision;
  const cleanup = persistenceWork.then(async () => {
    await AsyncStorage.removeItem(STORAGE_KEY);
    if (revision === hydrationRevision) diskTrusted = true;
  });
  // A failed removal must not allow a new account to hydrate the old cache.
  persistenceWork = cleanup.catch(() => undefined);
  // Subscribers may synchronously hydrate or write for the new identity.
  // Publish the cleanup barrier before notifying them.
  useSavedShopStore.setState({ items: [], hydrated: false });
}

// Identity events clear memory synchronously even when a screen never hydrates.
// Disk ownership survives process death; an old in-flight write retains its old tag.
authEvents.onLogin(syncOwner);
authEvents.onLogout(syncOwner);

export const flushSavedShopPersistence = () => persistenceWork;

export default useSavedShopStore;
export { STORAGE_KEY as SAVED_SHOP_STORAGE_KEY };
