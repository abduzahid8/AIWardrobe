/**
 * deckDedupe — pure deck-level deduplication for AI outfit decks.
 *
 * A "deck" is the 3-card result set shown after Generate. Every card must
 * differ from every other card by >= 2 items, otherwise the user sees
 * "2 of 3 outfits are the same".
 *
 * Rules (executable spec — covered by __tests__/e2e/aiDecks.e2e.test.ts):
 *   - exact duplicate  = same sorted id signature already in the deck
 *   - near duplicate   = shares >= NEAR_DUPE_THRESHOLD non-anchor ids
 *                        with any deck already kept
 *
 * Default threshold is 2: two cards that share shoes AND pants look like
 * the same outfit. Sharing a single hero piece (e.g. one blazer) is fine.
 *   - the locked anchor item (baseItemId) is excluded from the shared
 *     count: all wardrobe-mode decks share it by design
 *
 * Pure + deterministic: safe to run on client and (via copy) in Deno edge
 * functions. No imports — zero dependency surface.
 */

export const NEAR_DUPE_THRESHOLD = 2;

/** Canonical sorted signature for an outfit's item ids. */
export function deckSignature(ids: Array<string | number>): string {
    return ids.map(String).sort().join('|');
}

/**
 * Count ids shared with a kept deck, skipping anchor ids.
 * `skip` is typically { baseItemId } — empty/undefined counts everything.
 */
export function countSharedIds(
    ids: Array<string | number>,
    kept: Set<string>,
    skip?: Set<string>,
): number {
    let shared = 0;
    for (const raw of ids) {
        const id = String(raw);
        if (skip?.has(id)) continue;
        if (kept.has(id)) shared++;
    }
    return shared;
}

export interface DeckDedupe {
    /** Number of decks tracked so far. */
    readonly size: number;
    /** True when the exact signature was already kept. */
    hasExact(ids: Array<string | number>): boolean;
    /**
     * True when the candidate is an exact OR near duplicate of anything kept.
     * `skip` excludes ids (e.g. the locked anchor) from the shared count.
     */
    isDupe(ids: Array<string | number>, skip?: Set<string>): boolean;
    /** Record a deck as kept. */
    track(ids: Array<string | number>): void;
}

/** Stateful per-generation tracker (one instance per Generate tap). */
export function createDeckDedupe(threshold: number = NEAR_DUPE_THRESHOLD): DeckDedupe {
    const sigs = new Set<string>();
    const sets: Array<Set<string>> = [];

    return {
        get size() {
            return sets.length;
        },
        hasExact(ids) {
            return sigs.has(deckSignature(ids));
        },
        isDupe(ids, skip) {
            const sorted = ids.map(String).sort();
            if (sigs.has(sorted.join('|'))) return true;
            for (const kept of sets) {
                if (countSharedIds(sorted, kept, skip) >= threshold) return true;
            }
            return false;
        },
        track(ids) {
            const sorted = ids.map(String).sort();
            sigs.add(sorted.join('|'));
            sets.push(new Set(sorted));
        },
    };
}

/**
 * Cross-tap recency: order a candidate pool so items shown in recent decks
 * sort last. Stable — relative order inside the unseen/seen groups is kept,
 * so the style rank still decides within each group. Downstream top-N trims
 * (edge keeps 40) then naturally drop recently-shown items first, which is
 * what breaks the "same pants every tap" loop without risking starvation:
 * seen items are deprioritized, never deleted. `idsOf` may return one id or
 * several keys (id + name + image) — an item is "seen" if ANY key matches.
 *
 * @param idsOf  extracts the stable id (or keys) from a pool entry
 */
export function prioritizeUnseen<T>(pool: T[], seenIds: Set<string>, idsOf: (item: T) => string | string[]): T[] {
    if (seenIds.size === 0) return pool;
    const isSeen = (item: T) => {
        const keys = idsOf(item);
        const list = Array.isArray(keys) ? keys : [keys];
        return list.some((k) => k && seenIds.has(String(k)));
    };
    const unseen: T[] = [];
    const seen: T[] = [];
    for (const item of pool) {
        (isSeen(item) ? seen : unseen).push(item);
    }
    return [...unseen, ...seen];
}

// ── Seeded shuffle (mulberry32) ─────────────────────────────────────────
// Deterministic shuffle per Generate tap. Moved here from
// screens/AIOutfitmaker.tsx so slot ordering is unit-testable.
export function mulberry32(seed: number): () => number {
    let a = seed >>> 0;
    return function () {
        a |= 0; a = (a + 0x6d2b79f5) | 0;
        let t = Math.imul(a ^ (a >>> 15), 1 | a);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

export function shuffleWithSeed<T>(array: T[], seed: number): T[] {
    const arr = [...array];
    const rng = mulberry32(seed);
    for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
}

export interface SlotPickOrderOptions {
    /** Seed for the pool shuffle (already includes any slot offset). */
    seed: number;
    /** Ids already placed in an earlier card of this tap's deck. */
    deckUsedIds?: Set<string> | ReadonlySet<string>;
    /** Defaults to String(item?.id). */
    idOf?: (item: any) => string;
    /** Extra keys (name / image) that count as "already used" in this deck. */
    keysOf?: (item: any) => string[];
}

function itemIsUsedInDeck<T>(
    item: T,
    deckUsedIds: Set<string> | ReadonlySet<string> | undefined,
    idOf: (item: T) => string,
    keysOf?: (item: T) => string[],
): boolean {
    if (!deckUsedIds || deckUsedIds.size === 0) return false;
    if (deckUsedIds.has(idOf(item))) return true;
    if (!keysOf) return false;
    return keysOf(item).some((k) => k && deckUsedIds.has(k));
}

/**
 * orderSlotCandidates — deterministic pick ordering for one outfit slot.
 *
 * Production incident: AIOutfitmaker shuffled edge picks + pool picks
 * TOGETHER, so the edge's careful choices (blazer + linen shirt) were
 * replaced by random pool items (pajama top in an old_money/work deck).
 * And with no memory across the 3 normalize calls, thin formal pools
 * repeated the same pants + shoes in 2 of 3 cards.
 *
 * Order produced:
 *   1. incoming matches in edge/AI order (stylistic rank wins when fresh),
 *   2. pool matches shuffled by seed,
 *   each group stable-partitioned deck-unused-first when deckUsedIds is
 *   provided (falls back to used ids when the pool is exhausted, so decks
 *   never shrink — repetition is minimized, never forced).
 */
export function orderSlotCandidates<T>(
    incomingMatches: T[],
    poolMatches: T[],
    opts: SlotPickOrderOptions,
): T[] {
    const idOf = opts.idOf ?? ((item: any) => String(item?.id));
    const preferUnused = (list: T[]): T[] => {
        if (!opts.deckUsedIds || opts.deckUsedIds.size === 0) return list;
        const fresh = list.filter((c) => !itemIsUsedInDeck(c, opts.deckUsedIds, idOf, opts.keysOf));
        if (fresh.length === 0) return list;
        const used = list.filter((c) => itemIsUsedInDeck(c, opts.deckUsedIds, idOf, opts.keysOf));
        return [...fresh, ...used];
    };
    const shuffledPool =
        poolMatches.length > 1 ? shuffleWithSeed(poolMatches, opts.seed) : [...poolMatches];
    return [...preferUnused(incomingMatches), ...preferUnused(shuffledPool)];
}

export interface SlotPickSelection {
    rotationOffset?: number;
    slotOffset?: number;
    deckUsedIds?: Set<string> | ReadonlySet<string>;
    idOf?: (item: any) => string;
    keysOf?: (item: any) => string[];
}

/**
 * selectSlotPick — index into an orderSlotCandidates list.
 *
 * Indexing a preference-ordered list by raw rotation would defeat the
 * preference (position 1 of [fresh, used] is still used). So the rotation
 * lands inside the FRESH partition while one exists, and only falls back
 * to the full list when the pool is exhausted — decks then repeat as
 * little as the pool allows, never less.
 */
export function selectSlotPick<T>(
    ordered: T[],
    sel: SlotPickSelection,
): T | undefined {
    if (ordered.length === 0) return undefined;
    const idOf = sel.idOf ?? ((item: any) => String(item?.id));
    const fresh =
        sel.deckUsedIds && sel.deckUsedIds.size > 0
            ? ordered.filter((c) => !itemIsUsedInDeck(c, sel.deckUsedIds, idOf, sel.keysOf))
            : [];
    const pool = fresh.length > 0 ? fresh : ordered;
    return pool[((sel.rotationOffset ?? 0) + (sel.slotOffset ?? 0)) % pool.length];
}
