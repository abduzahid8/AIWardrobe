/**
 * Wardrobe Items Slice — CRUD operations for clothing items
 */

import type { StateCreator } from 'zustand';
import { Platform } from 'react-native';
import useAuthStore from '../auth';
import { stageNativeWardrobeImage, removePendingWardrobeImage } from '../../src/services/pendingWardrobeMedia';
import { WardrobeMediaError } from '../../src/services/wardrobeMedia';
import { authEvents } from '../authEvents';
import { normalizeCategory } from '../../src/utils/categoryMapper';
import type { ClothingItem, ClothingCategory } from '../../src/types/domain';
import type { WardrobeState } from '../wardrobeStore';
import { generateId } from './helpers';
import useSubscriptionStore, { TIER_FEATURES } from '../subscriptionStore';

/**
 * Thrown by addItem when the user's tier has a wardrobe cap and
 * adding one more item would exceed it. UI surfaces that call addItem
 * should catch this error by checking `err.code === 'WARDROBE_LIMIT_REACHED'`
 * and then navigate to the Paywall.
 */
export class WardrobeLimitError extends Error {
    code = 'WARDROBE_LIMIT_REACHED' as const;
    limit: number;
    constructor(limit: number) {
        super(`Wardrobe limit reached (${limit} items).`);
        this.limit = limit;
    }
}

export interface ItemsSlice {
    items: ClothingItem[];
    isLoading: boolean;

    addItem: (item: Omit<ClothingItem, 'id' | 'createdAt' | 'updatedAt' | 'wearCount' | 'lastWornAt' | 'isFavorite'>, options?: { idempotencyId: string }) => Promise<void>;
    removeItem: (id: string) => Promise<void>;
    updateItem: (id: string, updates: Partial<ClothingItem>) => void;
    toggleFavorite: (id: string) => void;
    getItemsByCategory: (category: ClothingCategory) => ClothingItem[];
    getItemsBySeason: (season: string) => ClothingItem[];
}

const stableScanSaves = new Map<string, Promise<void>>();

export const createItemsSlice: StateCreator<WardrobeState, [], [], ItemsSlice> = (set, get) => ({
    items: [],
    isLoading: false,

    addItem: (itemInput, options) => {
      const confirmCapacity = (itemId: string) => {
        const issue = get().wardrobeCapacityIssue;
        if (issue && get().pendingActions.some(action => issue.actionIds.includes(action.id)
            && action.type === 'add_item' && action.payload.id === itemId)) {
            throw Object.assign(new WardrobeLimitError(issue.limit), { savedLocally: true, itemId });
        }
      };
      const perform = async () => {
        if (options && (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(options.idempotencyId) || authEvents.getUserId() !== itemInput.userId)) throw new Error('Invalid scan save identity');
        const previous = options && get().items.find(item => item.id === options.idempotencyId);
        if (previous) {
            if (previous.userId !== itemInput.userId) throw new Error('Scan save owner mismatch');
            await get().syncToServer(); confirmCapacity(previous.id); return;
        }
        const revision = authEvents.getRevision();
        // Enforce free-tier wardrobe cap before touching state or Supabase.
        useSubscriptionStore.getState().refreshTimeState();
        const tier = useSubscriptionStore.getState().effectiveTier;
        const cap = TIER_FEATURES[tier].wardrobeItems;
        if (cap !== -1 && get().items.length >= cap) {
            throw new WardrobeLimitError(cap);
        }

        const now = new Date().toISOString();
        const tempId = options?.idempotencyId || generateId();
        const normalizedCategory = normalizeCategory(itemInput.category || '');

        let optimisticItem: ClothingItem = {
            ...itemInput,
            id: tempId,
            category: normalizedCategory,
            wearCount: 0,
            lastWornAt: null,
            isFavorite: false,
            createdAt: now,
            updatedAt: now,
        };

        const needsStaging = Platform.OS !== 'web' && [itemInput.imageUrl, itemInput.thumbnailUrl]
            .some(uri => uri && !/^https?:\/\//i.test(uri));
        if (needsStaging) {
            const owner = useAuthStore.getState().user?.id ?? useAuthStore.getState().session?.user?.id;
            if (!owner) throw new WardrobeMediaError('access_denied');
            const staged: string[] = [];
            try {
                const imageUrl = await stageNativeWardrobeImage(owner, tempId, itemInput.imageUrl);
                staged.push(imageUrl);
                let thumbnailUrl: string;
                if (itemInput.thumbnailUrl && itemInput.thumbnailUrl !== itemInput.imageUrl) {
                    thumbnailUrl = await stageNativeWardrobeImage(owner, generateId(), itemInput.thumbnailUrl);
                } else {
                    // Load native codec code only for local saves, after the store
                    // and authenticated identity have finished initializing.
                    if (revision !== authEvents.getRevision()) throw new WardrobeMediaError('account_changed');
                    // Metro's conditional native-module loading keeps startup free of codec initialization.
                    // eslint-disable-next-line @typescript-eslint/no-var-requires
                    const { createNativeWardrobeThumbnail } = require('../../src/services/wardrobeThumbnail') as typeof import('../../src/services/wardrobeThumbnail');
                    thumbnailUrl = await createNativeWardrobeThumbnail(owner, generateId(), imageUrl);
                }
                staged.push(thumbnailUrl);
                if (revision !== authEvents.getRevision() || !authEvents.canApplyToUser(owner)) throw new WardrobeMediaError('account_changed');
                // Another save or a tier refresh may have completed during file IO.
                useSubscriptionStore.getState().refreshTimeState();
                const currentCap = TIER_FEATURES[useSubscriptionStore.getState().effectiveTier].wardrobeItems;
                if (currentCap !== -1 && get().items.length >= currentCap) throw new WardrobeLimitError(currentCap);
                optimisticItem = { ...optimisticItem, userId: owner, imageUrl, thumbnailUrl };
            } catch (error) {
                // Nothing from this attempt has been published or queued yet.
                // Never roll back files after sync starts: retries still need them.
                const cleanup = await Promise.allSettled(staged.map(uri => removePendingWardrobeImage(owner, uri)));
                const failures = cleanup.filter((result): result is PromiseRejectedResult => result.status === 'rejected');
                if (failures.length) {
                    throw Object.assign(new Error('Unpublished photo cleanup failed'), {
                        code: 'photo_cleanup_failed', cause: error, cleanupErrors: failures.map(result => result.reason),
                    });
                }
                throw error;
            }
        }
        if (revision !== authEvents.getRevision()) return;
        // Queue creation atomically with the optimistic item. The shared sync
        // path persists this snapshot before issuing any cloud mutation.
        set((state) => ({
            items: [optimisticItem, ...state.items],
            pendingActions: [...state.pendingActions, {
                id: generateId(), type: 'add_item',
                payload: optimisticItem as unknown as Record<string, unknown>, createdAt: now,
            }],
        }));
        await get().syncToServer();
        confirmCapacity(tempId);
      };
      if (!options) return perform();
      const key = `${authEvents.getRevision()}:${itemInput.userId}:${options.idempotencyId}`;
      const previous = stableScanSaves.get(key);
      if (previous) return previous;
      const pending = perform(); stableScanSaves.set(key, pending);
      const clear = () => { if (stableScanSaves.get(key) === pending) stableScanSaves.delete(key); };
      void pending.then(clear, clear);
      return pending;
    },

    removeItem: async (id) => {
        const now = new Date().toISOString();
        set((state) => ({
            items: state.items.filter((item) => item.id !== id),
            outfits: state.outfits.map((outfit) => ({
                ...outfit,
                itemIds: outfit.itemIds.filter((itemId) => itemId !== id),
            })),
            pendingActions: [...state.pendingActions, {
                id: generateId(), type: 'remove_item', payload: { itemId: id }, createdAt: now,
            }],
        }));
        await get().syncToServer().catch(() => undefined);
    },

    updateItem: (id, updates) => {
        set((state) => ({
            items: state.items.map((item) =>
                item.id === id ? { ...item, ...updates, updatedAt: new Date().toISOString() } : item
            ),
        }));
    },

    toggleFavorite: (id) => {
        const item = get().items.find(existing => existing.id === id);
        if (!item) return;
        const isFavorite = !item.isFavorite;
        const now = new Date().toISOString();
        set((state) => ({
            items: state.items.map(existing => existing.id === id ? { ...existing, isFavorite, updatedAt: now } : existing),
            pendingActions: [...state.pendingActions, {
                id: generateId(), type: 'update_item', payload: { itemId: id, isFavorite }, createdAt: now,
            }],
        }));
        // Queue persistence owns retries; replay writes a value, never a toggle.
        void get().syncToServer().catch(() => undefined);
    },

    getItemsByCategory: (category) => get().items.filter((item) => item.category === category),
    getItemsBySeason: (season) => get().items.filter((item) => item.seasons.includes(season as any)),
});
