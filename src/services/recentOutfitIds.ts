/**
 * recentOutfitIds.ts
 *
 * Rolling window of item ids the user has ACTUALLY SEEN in generated
 * decks. Sent to the generate-outfits edge function as `excludeItemIds`
 * so every regenerate tap surfaces NEW pieces instead of recombining the
 * same ~40 candidates forever (measured Jaccard 1.0 across taps).
 *
 * Distinct from AIOutfitmaker's `aiRecentShownItemIds` (soft sort-last
 * via prioritizeUnseen): this store drives HARD exclusion at the edge and
 * is shared by every generation entry point, persisted so rotation also
 * survives app restarts (the daily-outfit path especially).
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { productIdentityKeys } from '../../features/outfit-generator/utils/outfitDiversity';

const MAX_RECENT_IDS = 150;
const MAX_RECENT_DECKS = 8;
const STORAGE_KEY = 'aiExcludeShownItemIds';
const PRODUCT_KEY_STORAGE_KEY = `${STORAGE_KEY}::productKeys`;
const DECK_STORAGE_KEY = `${STORAGE_KEY}::decks`;

export interface RotationHistory {
    previousDeckProductKeys: string[];
    olderRecentProductKeys: string[];
}

interface RecencyState {
    recentIds: string[];
    recentProductKeys: string[];
    recentDecks: string[][];
    hydrated: boolean;
    hydratePromise: Promise<void> | null;
    writeChain: Promise<void>;
}
const states = new Map<string, RecencyState>();
function stateFor(scope?: string): { key: string; state: RecencyState } {
    const key = scope ? `${STORAGE_KEY}::${encodeURIComponent(scope)}` : STORAGE_KEY;
    let state = states.get(key);
    if (!state) {
        state = { recentIds: [], recentProductKeys: [], recentDecks: [], hydrated: false, hydratePromise: null, writeChain: Promise.resolve() };
        states.set(key, state);
    }
    return { key, state };
}

function isRealPoolId(id: string): boolean {
    if (!id) return false;
    // Placeholder / synthesized ids are not pool members — recording them
    // would burn window space without excluding anything real.
    if (id === 'placeholder_shoes') return false;
    if (/^(item_|local_)/.test(id)) return false;
    return true;
}

function ensureHydrated(state: RecencyState, key: string): Promise<void> {
    if (state.hydrated) return Promise.resolve();
    if (!state.hydratePromise) {
        const productKey = key === STORAGE_KEY ? PRODUCT_KEY_STORAGE_KEY : `${key}::productKeys`;
        const decksKey = key === STORAGE_KEY ? DECK_STORAGE_KEY : `${key}::decks`;
        state.hydratePromise = Promise.all([AsyncStorage.getItem(key), AsyncStorage.getItem(productKey), AsyncStorage.getItem(decksKey)])
            .then(([raw, productKeysRaw, decksRaw]) => {
                try {
                    const stored = raw ? JSON.parse(raw) : [];
                    const ids = Array.isArray(stored) ? stored : stored?.recentIds;
                    const productKeys = productKeysRaw ? JSON.parse(productKeysRaw) : stored?.productKeys;
                    const decks = decksRaw ? JSON.parse(decksRaw) : stored?.recentDecks;
                    if (Array.isArray(ids)) {
                        const valid = ids.map((v: unknown) => String(v ?? '')).filter(isRealPoolId);
                        state.recentIds = valid.slice(Math.max(0, valid.length - MAX_RECENT_IDS));
                    }
                    if (Array.isArray(productKeys)) {
                        state.recentProductKeys = productKeys.map((v: unknown) => String(v ?? '').trim()).filter(Boolean)
                            .slice(Math.max(0, productKeys.length - MAX_RECENT_IDS * 4));
                    }
                    if (Array.isArray(decks)) {
                        state.recentDecks = decks
                            .filter((deck: unknown) => Array.isArray(deck))
                            .map((deck: unknown[]) => deck.map((v) => String(v ?? '').trim()).filter(Boolean))
                            .filter((deck: string[]) => deck.length > 0)
                            .slice(-MAX_RECENT_DECKS);
                    }
                } catch (_) {
                    // Corrupt cache — start fresh.
                }
            })
            .catch(() => undefined)
            .then(() => {
                state.hydrated = true;
            });
    }
    return state.hydratePromise;
}

/**
 * Snapshot of ids to send as `excludeItemIds` on the next generate call.
 * Awaits hydration AND any in-flight record so the very next request
 * already excludes the deck the user just saw.
 */
export async function getRecentShownIds(scope?: string): Promise<string[]> {
    const { state, key } = stateFor(scope);
    await Promise.all([ensureHydrated(state, key), state.writeChain]);
    return [...state.recentIds];
}

/** Product identities in oldest-first order for cross-generation rotation. */
export async function getRecentShownProductKeys(scope?: string): Promise<string[]> {
    const { state, key } = stateFor(scope);
    await Promise.all([ensureHydrated(state, key), state.writeChain]);
    return [...state.recentProductKeys];
}

/** Returns the immediately previous deck separately from older history. */
export async function getRotationHistory(scope?: string): Promise<RotationHistory> {
    const { state, key } = stateFor(scope);
    await Promise.all([ensureHydrated(state, key), state.writeChain]);
    const previousDeckProductKeys = state.recentDecks[state.recentDecks.length - 1] || [];
    const olderRecentProductKeys = state.recentDecks
        .slice(0, -1)
        .flat()
        .filter((productKey, index, keys) => !previousDeckProductKeys.includes(productKey) && keys.indexOf(productKey) === index);
    return { previousDeckProductKeys: [...previousDeckProductKeys], olderRecentProductKeys };
}

/** Stable rotation scope shared by repeated generations of the same intent. */
export function makeRotationScope(
    userId: string,
    source: string | null | undefined,
    style: string | null | undefined,
    occasion: string | null | undefined,
): string {
    const normalize = (value: string | null | undefined) => String(value || '').trim().toLowerCase().replace(/[\s-]+/g, '_');
    return `${userId}|${normalize(source) || 'wardrobe'}|${normalize(style) || 'casual'}|${normalize(occasion) || 'everyday'}`;
}

/**
 * Record the decks the user is now looking at (raw edge response or any
 * `outfits` array with `items[].id`). FIFO-capped, persisted. Fire-and-
 * forget for callers; internally ordered so hydration always runs first
 * and the first record can never clobber stored ids.
 */
export function recordShownOutfits(outfits: unknown, scope?: string): void {
    const { state, key } = stateFor(scope);
    state.writeChain = state.writeChain.then(async () => {
        await ensureHydrated(state, key);
        if (!Array.isArray(outfits)) return;
        const seen = new Set(state.recentIds);
        const seenProductKeys = new Set(state.recentProductKeys);
        let added = false;
        const deckProductKeys: string[] = [];
        const deckKeySet = new Set<string>();
        for (const outfit of outfits) {
            const items = (outfit as { items?: unknown } | null)?.items;
            if (!Array.isArray(items)) continue;
            for (const raw of items) {
                const id = String((raw as { id?: unknown } | null)?.id ?? '').trim();
                const keys = productIdentityKeys(raw).filter(Boolean);
                if (isRealPoolId(id)) {
                    for (const productKey of keys) {
                        if (!deckKeySet.has(productKey)) {
                            deckKeySet.add(productKey);
                            deckProductKeys.push(productKey);
                        }
                    }
                }
                const isNewId = isRealPoolId(id) && !seen.has(id);
                if (isNewId) {
                    seen.add(id);
                    state.recentIds.push(id);
                    added = true;
                    for (const productKey of keys) {
                        if (seenProductKeys.has(productKey)) continue;
                        seenProductKeys.add(productKey);
                        state.recentProductKeys.push(productKey);
                    }
                }
            }
        }
        if (state.recentIds.length > MAX_RECENT_IDS) {
            state.recentIds = state.recentIds.slice(state.recentIds.length - MAX_RECENT_IDS);
        }
        if (state.recentProductKeys.length > MAX_RECENT_IDS * 4) {
            state.recentProductKeys = state.recentProductKeys.slice(state.recentProductKeys.length - MAX_RECENT_IDS * 4);
        }
        const latestDeck = state.recentDecks[state.recentDecks.length - 1];
        const isNewDeck = deckProductKeys.length > 0 &&
            (!latestDeck || latestDeck.length !== deckProductKeys.length || latestDeck.some((key, index) => key !== deckProductKeys[index]));
        if (isNewDeck) {
            state.recentDecks = [...state.recentDecks, deckProductKeys].slice(-MAX_RECENT_DECKS);
            added = true;
        }
        if (added) {
            const productKey = key === STORAGE_KEY ? PRODUCT_KEY_STORAGE_KEY : `${key}::productKeys`;
            const decksKey = key === STORAGE_KEY ? DECK_STORAGE_KEY : `${key}::decks`;
            await Promise.all([
                AsyncStorage.setItem(key, JSON.stringify(state.recentIds)),
                AsyncStorage.setItem(productKey, JSON.stringify(state.recentProductKeys)),
                AsyncStorage.setItem(decksKey, JSON.stringify(state.recentDecks)),
            ]).catch(() => undefined);
        }
    }).catch(() => {
        // Recency is best-effort — never break generation.
    });
}
