import {outfitVisualSignature} from '../domain/outfitVisualSignature';
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
import {withRequestDeadline} from '../lib/requestDeadline';
import { productIdentityKeys, diversitySlotOf } from '../domain/outfitDiversity';

const MAX_RECENT_IDS = 150;
const STORAGE_KEY = 'aiExcludeShownItemIds';

interface RecencyState {
    recentIds: string[];
    productKeys: string[];
    lookSignatures:string[];
    hydrated: boolean;
    hydratePromise: Promise<void> | null;
    writeChain: Promise<void>;
}
const states = new Map<string, RecencyState>();
function stateFor(scope?: string): { key: string; state: RecencyState } {
    const key = scope ? `${STORAGE_KEY}::${encodeURIComponent(scope)}` : STORAGE_KEY;
    let state = states.get(key);
    if (!state) {
        state = { recentIds: [], productKeys: [], lookSignatures:[], hydrated: false, hydratePromise: null, writeChain: Promise.resolve() };
        states.set(key, state);
        if(states.size>32)states.delete(states.keys().next().value!);
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
        state.hydratePromise = Promise.all([AsyncStorage.getItem(key), AsyncStorage.getItem(`${key}::products`),AsyncStorage.getItem(`${key}::looks`)])
            .then(([raw, products,looks]) => {
                try{const values=JSON.parse(looks||'[]');if(Array.isArray(values))state.lookSignatures=values.filter(v=>typeof v==='string'&&v.startsWith('sq1:')&&v.length<=1000).slice(-36);}catch{/* Optional malformed history stays empty. */}
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

export async function getRecentShownLookSignatures(scope?:string):Promise<string[]>{const {state,key}=stateFor(scope);await Promise.all([ensureHydrated(state,key),state.writeChain]);return [...state.lookSignatures];}

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
        const signatures=outfits.filter(o=>Array.isArray(o?.items)&&o.items.length===4).map(o=>outfitVisualSignature(o.items));
        if(signatures.length){state.lookSignatures=[...state.lookSignatures,...signatures].slice(-36);await AsyncStorage.setItem(`${key}::looks`,JSON.stringify(state.lookSignatures)).catch(()=>undefined);}
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

export interface RotationDeck { requestId: string; productKeys: string[]; lookSignatures?: string[] }
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
        if(deckStates.size>64)deckStates.delete(deckStates.keys().next().value!);
    }
    return state;
}
export async function getRotationHistory(scope:string) {
    const state=decksFor(scope);await state.ready;
    const owner=scope.split('|')[0];
    const products=decksFor(`products:${owner}`);await products.ready;
    const shoes=decksFor(`shoes:${owner}`);await shoes.ready;
    // Older clients persisted an account window but only kept deck boundaries
    // per occasion. Carry that evidence forward without copying another user.
    const legacy=await getRecentShownProductKeys(owner);
    const decks=products.decks.length ? products.decks : state.decks;
    const previous=[...(decks.at(-1)?.productKeys||[])],previousSet=new Set(previous);
    const semanticDecks=[...new Map([...state.decks,...decks].map(deck=>[deck.requestId,deck])).values()].slice(-12);
    return {recentLookSignatures:semanticDecks.flatMap(d=>d.lookSignatures||[]).filter(s=>typeof s==='string'&&s.startsWith('sq1:')&&s.length<=1000).slice(-36),previousDeckProductKeys:previous,olderRecentProductKeys:[...new Set([...legacy,...decks.slice(0,-1).flatMap(d=>d.productKeys)])].filter(key=>!previousSet.has(key)).slice(-(450-previous.length)),rotationDecks:decks.map(d=>({...d,productKeys:[...d.productKeys]})),
        ...(shoes.decks.length ? {previousShoeProductKeys:[...shoes.decks.at(-1)!.productKeys],
            recentShoeProductKeys:[...new Set(shoes.decks.flatMap(d=>d.productKeys))],
            shoeRotationDecks:shoes.decks.map(d=>({...d,productKeys:[...d.productKeys]}))} : {})};
}
/** Call after final acceptance, before unlocking the next Generate press. */
export async function recordDisplayedDeck(outfits:unknown,scope:string,requestId:string):Promise<void> {
    const state=decksFor(scope);await state.ready;
    if(!Array.isArray(outfits)||state.decks.some(d=>d.requestId===requestId))return;
    const productKeys=[...new Set<string>(outfits.flatMap(o=>Array.isArray(o?.items)?o.items.flatMap(productIdentityKeys):[]))];
    if(!productKeys.length)return;
    const lookSignatures=outfits.filter(o=>Array.isArray(o?.items)&&o.items.length===4).map(o=>outfitVisualSignature(o.items)).slice(0,15);
    state.decks=[...state.decks,{requestId,productKeys,lookSignatures}].slice(-12);
    const snapshot=JSON.stringify(state.decks);
    state.writes=state.writes.then(()=>AsyncStorage.setItem(`stylistRotationDecks::${scope}`,snapshot)).catch(()=>undefined);
    // All physical products rotate across occasions and styles. Context still
    // identifies the request; it must not reset the displayed-product window.
    const productScope=`products:${scope.split('|')[0]}`,products=decksFor(productScope);await products.ready;
    if(!products.decks.some(d=>d.requestId===requestId)) {
        products.decks=[...products.decks,{requestId,productKeys,lookSignatures}].slice(-12);
        const productSnapshot=JSON.stringify(products.decks);
        products.writes=products.writes.then(()=>AsyncStorage.setItem(`stylistRotationDecks::${productScope}`,productSnapshot)).catch(()=>undefined);
    }
    // Keep the explicit footwear schedule for compatible older Edge versions.
    const shoeScope=`shoes:${scope.split('|')[0]}`,shoes=decksFor(shoeScope);await shoes.ready;
    const shoeKeys=[...new Set<string>(outfits.flatMap(o=>Array.isArray(o?.items)?o.items.filter((item:any)=>diversitySlotOf(item)==='shoes').flatMap(productIdentityKeys):[]))];
    if(shoeKeys.length && !shoes.decks.some(d=>d.requestId===requestId)) {
        shoes.decks=[...shoes.decks,{requestId,productKeys:shoeKeys}].slice(-12);
        const shoeSnapshot=JSON.stringify(shoes.decks);
        shoes.writes=shoes.writes.then(()=>AsyncStorage.setItem(`stylistRotationDecks::${shoeScope}`,shoeSnapshot)).catch(()=>undefined);
    }
    // Normal restarts must see the displayed deck on disk. Bound storage
    // faults so an already visible success can still unlock Generate.
    await withRequestDeadline(() => Promise.all([state.writes, products.writes, shoes.writes]), 1500).catch(() => undefined);
}
