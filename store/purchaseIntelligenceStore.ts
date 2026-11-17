import { create } from 'zustand';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { authEvents } from './authEvents';
import type { ShopCatalogItem } from '../src/types/shop';
import type { FitEvidence, PriceWatch } from '../src/types/purchase';
import { newPurchaseRecordId } from '../src/services/purchase/localId';
import { deriveFitProfile } from '../src/services/purchase/fitEvidence';
const KEY = 'purchase-intelligence-v1',
  DAY = 86400000;
interface PrivateData {
  bookmarks: ShopCatalogItem[];
  evidence: FitEvidence[];
  watches: PriceWatch[];
  feedbackEnabled: boolean;
  dismissed: Record<string, number>;
}
const empty = (): PrivateData => ({
  bookmarks: [],
  evidence: [],
  watches: [],
  feedbackEnabled: true,
  dismissed: {},
});
interface State extends PrivateData {
  ownerId: string | null;
  hydrated: boolean;
  saveCandidate: (item: ShopCatalogItem) => Promise<void>;
  hydrate: () => Promise<void>;
  recordFit: (input: Omit<FitEvidence, 'id' | 'userId' | 'createdAt'>) => Promise<FitEvidence>;
  updateFit: (
    id: string,
    patch: Pick<
      Partial<FitEvidence>,
      'zones' | 'reference' | 'kept' | 'returned' | 'returnReason' | 'sizeLabel' | 'preferredFit'
    >
  ) => Promise<void>;
  setFeedbackEnabled: (enabled: boolean) => Promise<void>;
  dismiss: (productId: string, now?: number) => Promise<void>;
  canPrompt: (productId: string, now?: number) => boolean;
  saveWatch: (input: Omit<PriceWatch, 'id' | 'userId' | 'createdAt'>) => Promise<void>;
  updateWatch: (id: string, status: PriceWatch['status']) => Promise<void>;
  markWatchChecked: (id: string, triggered: boolean) => Promise<void>;
  reset: () => Promise<void>;
}
let queue: Promise<unknown> = Promise.resolve(),
  dataRevision = 0;
function safeData(raw: unknown, owner: string): PrivateData {
  const record = raw as { ownerId?: string; version?: number; data?: Partial<PrivateData> };
  if (!record || record.ownerId !== owner || record.version !== 1) return empty();
  const data = record.data || {};
  const evidence = (Array.isArray(data.evidence) ? data.evidence : [])
    .filter(
      (e) =>
        e &&
        e.userId === owner &&
        typeof e.id === 'string' &&
        typeof e.category === 'string' &&
        ['too_small', 'snug', 'perfect', 'relaxed', 'too_large', 'unknown'].includes(
          e.overallFit
        ) &&
        [
          'owned',
          'purchase',
          'try_on',
          'return',
          'manual',
          'body_profile',
          'reference_garment',
        ].includes(e.source) &&
        (!e.materials ||
          (Array.isArray(e.materials) &&
            e.materials.length <= 20 &&
            e.materials.every((m) => typeof m === 'string'))) &&
        Number.isFinite(e.confidence) &&
        Number.isFinite(Date.parse(e.createdAt))
    )
    .slice(-160);
  const watches = (Array.isArray(data.watches) ? data.watches : [])
    .filter(
      (w) =>
        w &&
        w.userId === owner &&
        typeof w.id === 'string' &&
        Number.isFinite(w.threshold) &&
        w.threshold >= 0
    )
    .slice(-40);
  const dismissed = Object.fromEntries(
    Object.entries(data.dismissed || {})
      .filter(
        ([key, value]) => key.length < 512 && typeof value === 'number' && Number.isFinite(value)
      )
      .slice(-160)
  );
  const bookmarks = (Array.isArray(data.bookmarks) ? data.bookmarks : [])
    .filter((item) => item && typeof item.id === 'string' && typeof item.name === 'string')
    .slice(-20);
  return {
    bookmarks,
    evidence,
    watches,
    feedbackEnabled: data.feedbackEnabled !== false,
    dismissed,
  };
}
function snapshot(state: State): PrivateData {
  return {
    bookmarks: state.bookmarks,
    evidence: state.evidence,
    watches: state.watches,
    feedbackEnabled: state.feedbackEnabled,
    dismissed: state.dismissed,
  };
}
function syncOwner() {
  const owner = authEvents.getUserId();
  if (usePurchaseIntelligenceStore.getState().ownerId !== owner)
    usePurchaseIntelligenceStore.setState({ ...empty(), ownerId: owner, hydrated: false });
  return owner;
}
async function mutate(change: (data: PrivateData, owner: string) => PrivateData): Promise<void> {
  dataRevision++;
  const owner = syncOwner(),
    revision = authEvents.getRevision();
  if (!owner) throw new Error('Sign in to save purchase preferences');
  const work = queue.then(async () => {
    if (owner !== authEvents.getUserId() || revision !== authEvents.getRevision()) return;
    let data = snapshot(usePurchaseIntelligenceStore.getState());
    if (!usePurchaseIntelligenceStore.getState().hydrated) {
      const raw = await AsyncStorage.getItem(KEY);
      data = raw ? safeData(JSON.parse(raw), owner) : empty();
    }
    if (owner !== authEvents.getUserId() || revision !== authEvents.getRevision()) return;
    const next = change(data, owner);
    await AsyncStorage.setItem(KEY, JSON.stringify({ version: 1, ownerId: owner, data: next }));
    if (owner === authEvents.getUserId() && revision === authEvents.getRevision())
      usePurchaseIntelligenceStore.setState({ ...next, ownerId: owner, hydrated: true });
  });
  queue = work.catch(() => undefined);
  await work;
}
export const usePurchaseIntelligenceStore = create<State>((set, get) => ({
  ...empty(),
  ownerId: null,
  hydrated: false,
  saveCandidate: async (item) =>
    mutate((data) => ({
      ...data,
      bookmarks: [
        ...data.bookmarks.filter((row) => row.id !== item.id),
        {
          id: item.id,
          name: item.name,
          brand: item.brand,
          price: item.price,
          currency: item.currency,
          garmentType: item.garmentType,
          category: item.category,
          sourceUrl: item.sourceUrl,
          source: item.source,
          imageUrl:
            typeof item.imageUrl === 'string' &&
            /^https?:\/\//i.test(item.imageUrl) &&
            item.imageUrl.length <= 2048
              ? item.imageUrl
              : '',
        },
      ].slice(-20),
    })),
  hydrate: async () => {
    const owner = syncOwner(),
      revision = authEvents.getRevision(),
      dataVersion = dataRevision;
    if (!owner || get().hydrated) return;
    await queue;
    const raw = await AsyncStorage.getItem(KEY);
    let data = empty();
    try {
      data = raw ? safeData(JSON.parse(raw), owner) : empty();
    } catch {
      /* Corrupt private cache is not attributed to another account. */
    }
    if (
      owner === authEvents.getUserId() &&
      revision === authEvents.getRevision() &&
      dataVersion === dataRevision
    )
      set({ ...data, hydrated: true });
  },
  recordFit: async (input) => {
    const owner = authEvents.getUserId();
    if (!owner) throw new Error('Sign in to save fit feedback');
    if (
      !input.category ||
      !Number.isFinite(input.confidence) ||
      input.confidence < 0 ||
      input.confidence > 1
    )
      throw new Error('Invalid fit evidence');
    const allowed = Object.fromEntries(
      Object.entries(input).filter(([key]) =>
        [
          'productId',
          'productFamilyKey',
          'variantId',
          'brand',
          'category',
          'subcategory',
          'sizeLabel',
          'cut',
          'materials',
          'stretch',
          'source',
          'overallFit',
          'zones',
          'kept',
          'returned',
          'sold',
          'returnReason',
          'confidence',
          'preferredFit',
          'sourceVersion',
          'reference',
          'physicalProfile',
        ].includes(key)
      )
    ) as typeof input;
    const evidence: FitEvidence = {
      ...allowed,
      id: newPurchaseRecordId('fit'),
      userId: owner,
      createdAt: new Date().toISOString(),
    };
    await mutate((data) => ({ ...data, evidence: [...data.evidence, evidence].slice(-160) }));
    if (owner !== authEvents.getUserId()) throw new Error('Account changed');
    return evidence;
  },
  updateFit: async (id, patch) =>
    mutate((data) => ({
      ...data,
      evidence: data.evidence.map((e) =>
        e.id === id
          ? {
              ...e,
              ...Object.fromEntries(
                Object.entries(patch).filter(([key]) =>
                  [
                    'zones',
                    'reference',
                    'kept',
                    'returned',
                    'returnReason',
                    'sizeLabel',
                    'preferredFit',
                  ].includes(key)
                )
              ),
            }
          : e
      ),
    })),
  setFeedbackEnabled: async (feedbackEnabled) => mutate((data) => ({ ...data, feedbackEnabled })),
  dismiss: async (productId, now = Date.now()) =>
    mutate((data) => ({
      ...data,
      dismissed: Object.fromEntries(
        [
          ...Object.entries(data.dismissed).filter(([id]) => id !== productId),
          [productId, now],
        ].slice(-160)
      ),
    })),
  canPrompt: (productId, now = Date.now()) => {
    syncOwner();
    const data = get();
    return (
      !!data.ownerId &&
      data.hydrated &&
      data.feedbackEnabled &&
      !data.evidence.some((e) => e.productId === productId) &&
      (data.dismissed[productId] === undefined || now - data.dismissed[productId] >= 14 * DAY)
    );
  },
  saveWatch: async (input) => {
    if (
      !Number.isFinite(input.threshold) ||
      input.threshold < 0 ||
      !input.currency.trim() ||
      !input.productId
    )
      throw new Error('Invalid watch threshold');
    await mutate((data, owner) => ({
      ...data,
      watches: [
        ...data.watches.filter(
          (w) =>
            !(
              w.productId === input.productId &&
              w.selectedSize === input.selectedSize &&
              w.merchant === input.merchant
            )
        ),
        {
          ...input,
          id: newPurchaseRecordId('watch'),
          userId: owner,
          createdAt: new Date().toISOString(),
        },
      ].slice(-40),
    }));
  },
  updateWatch: async (id, status) =>
    mutate((data) => ({
      ...data,
      watches: data.watches.map((w) => (w.id === id ? { ...w, status } : w)),
    })),
  markWatchChecked: async (id, triggered) =>
    mutate((data) => ({
      ...data,
      watches: data.watches.map((watch) =>
        watch.id === id
          ? {
              ...watch,
              lastCheckedAt: new Date().toISOString(),
              status: triggered && watch.status === 'active' ? 'triggered' : watch.status,
            }
          : watch
      ),
    })),
  reset: async () => {
    dataRevision++;
    set({ ...empty(), ownerId: null, hydrated: false });
    const work = queue.then(() => AsyncStorage.removeItem(KEY));
    queue = work.catch(() => undefined);
    await work;
  },
}));
authEvents.onLogin(() => {
  syncOwner();
});
authEvents.onLogout(() => {
  usePurchaseIntelligenceStore.setState({ ...empty(), ownerId: null, hydrated: false });
});
export const flushPurchaseIntelligence = () => queue;
export const resetPurchaseIntelligence = () => usePurchaseIntelligenceStore.getState().reset();
export const currentFitProfile = () =>
  deriveFitProfile(usePurchaseIntelligenceStore.getState().evidence);
