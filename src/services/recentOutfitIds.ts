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

const MAX_RECENT_IDS = 150;
const STORAGE_KEY = 'aiExcludeShownItemIds';

interface RecencyState {
    recentIds: string[];
    hydrated: boolean;
    hydratePromise: Promise<void> | null;
    writeChain: Promise<void>;
}
const states = new Map<string, RecencyState>();
function stateFor(scope?: string): { key: string; state: RecencyState } {
    const key = scope ? `${STORAGE_KEY}::${encodeURIComponent(scope)}` : STORAGE_KEY;
    let state = states.get(key);
    if (!state) {
        state = { recentIds: [], hydrated: false, hydratePromise: null, writeChain: Promise.resolve() };
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
        state.hydratePromise = AsyncStorage.getItem(key)
            .then((raw) => {
                if (raw) {
                    try {
                        const ids = JSON.parse(raw);
                        if (Array.isArray(ids)) {
                            const valid = ids.map((v: unknown) => String(v ?? '')).filter(isRealPoolId);
                            // Stored oldest-first — keep the newest window.
                            state.recentIds = valid.slice(Math.max(0, valid.length - MAX_RECENT_IDS));
                        }
                    } catch (_) {
                        // Corrupt cache — start fresh.
                    }
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
        let added = false;
        for (const outfit of outfits) {
            const items = (outfit as { items?: unknown } | null)?.items;
            if (!Array.isArray(items)) continue;
            for (const raw of items) {
                const id = String((raw as { id?: unknown } | null)?.id ?? '').trim();
                if (!isRealPoolId(id) || seen.has(id)) continue;
                seen.add(id);
                state.recentIds.push(id);
                added = true;
            }
        }
        if (state.recentIds.length > MAX_RECENT_IDS) {
            state.recentIds = state.recentIds.slice(state.recentIds.length - MAX_RECENT_IDS);
        }
        if (added) {
            await AsyncStorage.setItem(key, JSON.stringify(state.recentIds)).catch(() => undefined);
        }
    }).catch(() => {
        // Recency is best-effort — never break generation.
    });
}
