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
import { productIdentityKeys } from '../../backend/supabase/functions/_shared/outfitDiversity';

const MAX_RECENT_IDS = 150;
const STORAGE_KEY = 'aiExcludeShownItemIds';

interface RecencyState {
    recentIds: string[];
    productKeys: string[];
    hydrated: boolean;
    hydratePromise: Promise<void> | null;
    writeChain: Promise<void>;
}
const states = new Map<string, RecencyState>();
function stateFor(scope?: string): { key: string; state: RecencyState } {
    const key = scope ? `${STORAGE_KEY}::${encodeURIComponent(scope)}` : STORAGE_KEY;
    let state = states.get(key);
    if (!state) {
        state = { recentIds: [], productKeys: [], hydrated: false, hydratePromise: null, writeChain: Promise.resolve() };
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
        state.hydratePromise = Promise.all([AsyncStorage.getItem(key), AsyncStorage.getItem(`${key}::products`)])
            .then(([raw, products]) => {
                if (products) {
                    try {
                        const keys = JSON.parse(products);
                        if (Array.isArray(keys)) state.productKeys = keys.filter((key: unknown) => typeof key === 'string' && /^(id:|n:|i:)/.test(key) && key.length <= 2048).slice(-MAX_RECENT_IDS * 3);
                    } catch { /* Ignore a corrupt optional identity cache. */ }
                }
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

/** The same displayed-product history, including duplicate catalog listings. */
export async function getRecentShownProductKeys(scope?: string): Promise<string[]> {
    const {state, key} = stateFor(scope);
    await Promise.all([ensureHydrated(state, key), state.writeChain]);
    return [...state.productKeys];
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
        const seenKeys = new Set(state.productKeys);
        let keysAdded = false;
        let added = false;
        for (const outfit of outfits) {
            const items = (outfit as { items?: unknown } | null)?.items;
            if (!Array.isArray(items)) continue;
            for (const raw of items) {
                const id = String((raw as { id?: unknown } | null)?.id ?? '').trim();
                if (!isRealPoolId(id)) continue;
                for (const key of productIdentityKeys(raw)) {
                    if (key.length > 2048 || seenKeys.has(key)) continue;
                    seenKeys.add(key);
                    state.productKeys.push(key);
                    keysAdded = true;
                }
                if (seen.has(id)) continue;
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
        if (keysAdded) {
            state.productKeys = state.productKeys.slice(-MAX_RECENT_IDS * 3);
            await AsyncStorage.setItem(`${key}::products`, JSON.stringify(state.productKeys)).catch(() => undefined);
        }
    }).catch(() => {
        // Recency is best-effort — never break generation.
    });
}

export interface RotationDeck { requestId: string; productKeys: string[] }
const deckStates = new Map<string, {decks: RotationDeck[]; ready: Promise<void>; writes: Promise<void>}>();
let requestSequence = 0;
/** Stable semantic identity; request IDs deliberately live outside this scope. */
export function rotationScope(userId: string, source: string, style: string, occasion: string): string {
    const canonical=(value:string)=>value.trim().toLowerCase().replace(/[\s-]+/g,'_');
    return [userId, source === 'new_clothing' ? 'shop' : canonical(source), canonical(style), canonical(occasion).replace(/^date_night$/,'date')].join('|');
}
export function generationRequestId(): string { return `${Date.now().toString(36)}-${(++requestSequence).toString(36)}`; }
function decksFor(scope: string) {
    let state=deckStates.get(scope);
    if (!state) {
        state={decks:[],ready:Promise.resolve(),writes:Promise.resolve()};
        const captured=state;
        captured.ready=AsyncStorage.getItem(`stylistRotationDecks::${scope}`).then(raw=>{
            try { const decks=JSON.parse(raw||'[]');if(Array.isArray(decks))captured.decks=decks.filter(d=>typeof d?.requestId==='string'&&Array.isArray(d.productKeys)&&d.productKeys.every((k:unknown)=>typeof k==='string')).slice(-12); } catch { /* An unreadable cache is empty. */ }
        }).catch(()=>undefined);
        deckStates.set(scope,state);
    }
    return state;
}
export async function getRotationHistory(scope:string) {
    const state=decksFor(scope);await state.ready;
    return {previousDeckProductKeys:[...(state.decks.at(-1)?.productKeys||[])],olderRecentProductKeys:[...new Set(state.decks.slice(0,-1).flatMap(d=>d.productKeys))],rotationDecks:state.decks.map(d=>({...d,productKeys:[...d.productKeys]}))};
}
/** Call after final acceptance, before unlocking the next Generate press. */
export async function recordDisplayedDeck(outfits:unknown,scope:string,requestId:string):Promise<void> {
    const state=decksFor(scope);await state.ready;
    if(!Array.isArray(outfits)||state.decks.some(d=>d.requestId===requestId))return;
    const productKeys=[...new Set<string>(outfits.flatMap(o=>Array.isArray(o?.items)?o.items.flatMap(productIdentityKeys):[]))];
    if(!productKeys.length)return;
    state.decks=[...state.decks,{requestId,productKeys}].slice(-12);
    const snapshot=JSON.stringify(state.decks);
    state.writes=state.writes.then(()=>AsyncStorage.setItem(`stylistRotationDecks::${scope}`,snapshot)).catch(()=>undefined);
}
