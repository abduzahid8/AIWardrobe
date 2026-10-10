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

let recentIds: string[] = [];
let hydrated = false;
let hydratePromise: Promise<void> | null = null;
let writeChain: Promise<void> = Promise.resolve();

function isRealPoolId(id: string): boolean {
    if (!id) return false;
    // Placeholder / synthesized ids are not pool members — recording them
    // would burn window space without excluding anything real.
    if (id === 'placeholder_shoes') return false;
    if (/^(item_|local_)/.test(id)) return false;
    return true;
}

function ensureHydrated(): Promise<void> {
    if (hydrated) return Promise.resolve();
    if (!hydratePromise) {
        hydratePromise = AsyncStorage.getItem(STORAGE_KEY)
            .then((raw) => {
                if (raw) {
                    try {
                        const ids = JSON.parse(raw);
                        if (Array.isArray(ids)) {
                            const valid = ids.map((v: unknown) => String(v ?? '')).filter(isRealPoolId);
                            // Stored oldest-first — keep the newest window.
                            recentIds = valid.slice(Math.max(0, valid.length - MAX_RECENT_IDS));
                        }
                    } catch (_) {
                        // Corrupt cache — start fresh.
                    }
                }
            })
            .catch(() => undefined)
            .then(() => {
                hydrated = true;
            });
    }
    return hydratePromise;
}

/**
 * Snapshot of ids to send as `excludeItemIds` on the next generate call.
 * Awaits hydration AND any in-flight record so the very next request
 * already excludes the deck the user just saw.
 */
export async function getRecentShownIds(): Promise<string[]> {
    await Promise.all([ensureHydrated(), writeChain]);
    return [...recentIds];
}

/**
 * Record the decks the user is now looking at (raw edge response or any
 * `outfits` array with `items[].id`). FIFO-capped, persisted. Fire-and-
 * forget for callers; internally ordered so hydration always runs first
 * and the first record can never clobber stored ids.
 */
export function recordShownOutfits(outfits: unknown): void {
    writeChain = writeChain.then(async () => {
        await ensureHydrated();
        if (!Array.isArray(outfits)) return;
        const seen = new Set(recentIds);
        let added = false;
        for (const outfit of outfits) {
            const items = (outfit as { items?: unknown } | null)?.items;
            if (!Array.isArray(items)) continue;
            for (const raw of items) {
                const id = String((raw as { id?: unknown } | null)?.id ?? '').trim();
                if (!isRealPoolId(id) || seen.has(id)) continue;
                seen.add(id);
                recentIds.push(id);
                added = true;
            }
        }
        if (recentIds.length > MAX_RECENT_IDS) {
            recentIds = recentIds.slice(recentIds.length - MAX_RECENT_IDS);
        }
        if (added) {
            await AsyncStorage.setItem(STORAGE_KEY, JSON.stringify(recentIds)).catch(() => undefined);
        }
    }).catch(() => {
        // Recency is best-effort — never break generation.
    });
}
