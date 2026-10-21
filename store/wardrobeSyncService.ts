import { withRequestDeadline } from '../src/lib/requestDeadline';
/**
 * Wardrobe Sync Service
 *
 * Handles Supabase synchronization for the wardrobe store:
 * - Fetching items from server
 * - Syncing pending actions (offline-first queue)
 * - Conflict resolution (server-wins strategy)
 *
 * Extracted from wardrobeStore.ts to reduce store complexity.
 */

import { authEvents } from './authEvents';
import { prepareWardrobeItemMedia } from '../src/services/wardrobeItemMedia';
import { supabase } from '../lib/supabase';
import type {
    ClothingItem,
    ClothingCategory,
    WearLog,
    Occasion,
    Season,
} from '../src/types/domain';

/** Shape of a pending sync action */
export interface PendingAction {
    id: string;
    type: 'add_item' | 'remove_item' | 'update_item' | 'add_wear_log';
    payload: Record<string, unknown>;
    createdAt: string;
}

export const WARDROBE_READ_TIMEOUT_MS = 30_000;

async function withOwnedReadDeadline<T>(read: (signal: AbortSignal) => Promise<T | null>): Promise<T | null> {
    try { return await withRequestDeadline(read, WARDROBE_READ_TIMEOUT_MS); }
    catch { return null; } // No replacement: callers retain the owned cache.
}

/** Map a Supabase row to our domain ClothingItem */
function mapRowToItem(row: Record<string, unknown>): ClothingItem {
    return {
        id: row.id as string,
        userId: row.user_id as string,
        imageUrl: row.image_url as string,
        thumbnailUrl: row.thumbnail_url as string | undefined,
        category: (row.category as ClothingCategory) || 'top',
        subCategory: (row.sub_category as string) || '',
        primaryColor: (row.primary_color as string) || '',
        colorHex: (row.color_hex as string) || '#000000',
        pattern: typeof row.pattern === 'string' ? row.pattern : '',
        material: (row.material as string) || '',
        brand: row.brand as string | undefined,
        name: row.name as string | undefined,
        seasons: (row.seasons as Season[]) || [],
        occasions: (row.occasions as Occasion[]) || [],
        wearCount: (row.wear_count as number) || 0,
        lastWornAt: row.last_worn_at as string | null,
        isFavorite: (row.is_favorite as boolean) || false,
        createdAt: row.created_at as string,
        updatedAt: row.updated_at as string,
        detectionConfidence: row.detection_confidence as number | undefined,
    };
}

/**
 * Fetch all clothing items for the current user from Supabase.
 * Returns null if no session or on error (cached items remain available).
 */
export async function fetchItemsFromServer(): Promise<ClothingItem[] | null> {
    return withOwnedReadDeadline(async signal => {
        const revision = authEvents.getRevision();
        const { data: session } = await supabase.auth.getSession();
        if (signal.aborted || !session?.session?.user || revision !== authEvents.getRevision()) return null;

        const items: ClothingItem[] = [];
        const pageSize = 250;
        let cursor: string | undefined;
        for (;;) {
            let query = supabase.from('clothing_items')
                .select('id, user_id, image_url, thumbnail_url, category, sub_category, primary_color, color_hex, pattern, material, brand, name, seasons, occasions, wear_count, last_worn_at, is_favorite, created_at, updated_at, detection_confidence')
                .eq('user_id', session.session.user.id)
                .order('id', { ascending: false }).abortSignal(signal);
            if (cursor) query = query.lt('id', cursor);
            const { data, error } = await query.limit(pageSize);
            if (signal.aborted || error || revision !== authEvents.getRevision()) return null;
            if (!data) return null;
            items.push(...data.map(mapRowToItem));
            if (data.length < pageSize) return items.sort((a, b) => (b.createdAt ?? '').localeCompare(a.createdAt ?? '') || b.id.localeCompare(a.id));
            const next = data[data.length - 1].id;
            if (typeof next !== 'string' || (cursor && next >= cursor)) return null;
            cursor = next;
        }
    });
}

export async function fetchWearLogsFromServer(): Promise<WearLog[] | null> {
    return withOwnedReadDeadline(async signal => {
        const revision = authEvents.getRevision();
        const { data: session } = await supabase.auth.getSession();
        if (signal.aborted || !session?.session?.user || revision !== authEvents.getRevision()) return null;

        const logs: WearLog[] = [];
        const pageSize = 250;
        let cursor: string | undefined;
        for (;;) {
            let query = supabase.from('wear_logs')
                .select('id, user_id, outfit_id, item_ids, date, occasion, weather_temp, weather_condition, created_at')
                .eq('user_id', session.session.user.id)
                .order('id', { ascending: false }).abortSignal(signal);
            if (cursor) query = query.lt('id', cursor);
            const { data, error } = await query.limit(pageSize);
            // Return no replacement on any failed/stale page: cached history stays intact.
            if (signal.aborted || error || !data || revision !== authEvents.getRevision()) return null;
            logs.push(...data.map((row): WearLog => ({
                id: row.id,
                userId: row.user_id,
                outfitId: row.outfit_id ?? undefined,
                itemIds: Array.isArray(row.item_ids) ? row.item_ids : [],
                date: row.date,
                occasion: row.occasion ?? undefined,
                weatherTemp: row.weather_temp ?? undefined,
                weatherCondition: row.weather_condition ?? undefined,
                createdAt: row.created_at,
            })));
            if (data.length < pageSize) return logs.sort((a, b) => b.date.localeCompare(a.date) || b.id.localeCompare(a.id));
            const next = data[data.length - 1].id;
            if (typeof next !== 'string' || (cursor && next >= cursor)) return null;
            cursor = next;
        }
    });
}

/**
 * Process pending sync actions against Supabase.
 * Uses conflict resolution: server version wins if newer.
 *
 * `add_item` actions are batched into a single upsert call to reduce
 * N sequential round-trips to 1 (Defect 2.5 fix).
 *
 * Persistent callers must checkpoint prepared favorite versions before mutation.
 * Unacknowledged prepared versions are also returned for in-memory callers.
 * @returns IDs of successfully processed actions and confirmed item state
 */
export type WardrobeCapacityIssue = { limit: number; actionIds: string[] };

export async function processPendingActions(
    pendingActions: PendingAction[],
    localItems: ClothingItem[],
    expectedUserId?: string,
    checkpointFavorite?: (actions: PendingAction[]) => Promise<void>
): Promise<{ processedIds: string[]; updatedItems: ClothingItem[]; updatedActions?: PendingAction[]; capacityIssue?: WardrobeCapacityIssue }> {
    if (pendingActions.length === 0) {
        return { processedIds: [], updatedItems: localItems };
    }

    const revision = authEvents.getRevision();
    const { data: session } = await supabase.auth.getSession();
    if (!session?.session?.user || revision !== authEvents.getRevision() || (expectedUserId && session.session.user.id !== expectedUserId)) {
        return { processedIds: [], updatedItems: localItems };
    }

    const userId = session.session.user.id;
    const assertAccount = () => {
        if (revision !== authEvents.getRevision()) throw new Error('Wardrobe sync account changed');
    };
    let capacityIssue: WardrobeCapacityIssue | undefined;
    const processed: string[] = [];
    const preparedFavorites = new Map<string, PendingAction>();
    let items = [...localItems];

    // ── Separate add_item actions from other action types ──────────────────
    const deletionGroups = new Map<string, PendingAction[]>();
    const superseded = new Set<string>();
    for (let index = pendingActions.length - 1; index >= 0; index--) {
        const action = pendingActions[index];
        if (action.type === 'add_wear_log') continue;
        const itemId = action.type === 'add_item' ? action.payload.id : action.payload.itemId;
        if (typeof itemId !== 'string') continue;
        const group = deletionGroups.get(itemId);
        if (group) {
            group.push(action);
            superseded.add(action.id);
        } else if (action.type === 'remove_item') deletionGroups.set(itemId, [action]);
    }
    // Retain obsolete entries until deletion is confirmed, but never upload
    // them again: a failed upload must not resurrect a subsequently deleted row.
    const effectiveActions = pendingActions.filter(action => !superseded.has(action.id));
    const addItemActions = effectiveActions.filter((a) => a.type === 'add_item');
    const otherActions = effectiveActions.filter((a) => a.type !== 'add_item');
    const favoriteGroups = new Map<string, PendingAction[]>();
    for (const action of otherActions) {
        if (action.type !== 'update_item' || typeof action.payload.itemId !== 'string'
            || typeof action.payload.isFavorite !== 'boolean') continue;
        const group = favoriteGroups.get(action.payload.itemId) ?? [];
        group.push(action);
        favoriteGroups.set(action.payload.itemId, group);
    }

    // ── Batch-process all add_item actions ─────────────────────────────────
    if (addItemActions.length > 0) {
        try {
            // Step 1: Conflict-resolution — fetch server timestamps for all items
            // in a single query instead of N individual selects.
            const addItemIds = addItemActions.map(
                (a) => (a.payload as unknown as ClothingItem).id
            );

            const { data: serverRows, error: timestampError } = await supabase
                .from('clothing_items')
                .select('id, updated_at')
                .in('id', addItemIds);
            if (timestampError) throw timestampError;
            assertAccount();

            const serverTimestampById: Record<string, string> = {};
            if (serverRows) {
                for (const row of serverRows) {
                    serverTimestampById[row.id as string] = row.updated_at as string;
                }
            }

            // Step 2: Partition into "server wins" vs "local wins" groups.
            const serverWinsActions: PendingAction[] = [];
            const localWinsActions: PendingAction[] = [];

            for (const action of addItemActions) {
                const item = action.payload as unknown as ClothingItem;
                const serverUpdatedAt = serverTimestampById[item.id];

                if (
                    serverUpdatedAt &&
                    item.updatedAt &&
                    new Date(serverUpdatedAt) > new Date(item.updatedAt)
                ) {
                    serverWinsActions.push(action);
                } else {
                    localWinsActions.push(action);
                }
            }

            // Step 3: For server-wins items, re-fetch each one individually
            // (these are genuine conflicts — unavoidable per-item round-trips).
            for (const action of serverWinsActions) {
                try {
                    const item = action.payload as unknown as ClothingItem;
                    assertAccount();
                    const { data: freshItem, error: fetchError } = await supabase
                        .from('clothing_items')
                        .select('*')
                        .eq('id', item.id)
                        .single();
                    if (fetchError || !freshItem) throw fetchError || new Error('Conflict row missing');
                    assertAccount();

                    if (freshItem) {
                        items = items.map((existing) =>
                            existing.id === item.id ? mapRowToItem(freshItem) : existing
                        );
                    }
                    processed.push(action.id);
                } catch (err) {
                    console.error(
                        '[WardrobeSyncService] Server-wins re-fetch failed for action',
                        action.id,
                        err
                    );
                    // Keep in queue for retry — do not push to processed
                }
            }

            // Step 4: Batch upsert all local-wins items in a single round-trip.
            if (localWinsActions.length > 0) {
                const preparedItems: ClothingItem[] = [];
                for (const action of localWinsActions) {
                    assertAccount();
                    preparedItems.push(await prepareWardrobeItemMedia(userId, action.payload as unknown as ClothingItem));
                    assertAccount();
                }
                const rows = preparedItems.map((item) => {
                    return {
                        id: item.id,
                        user_id: userId,
                        image_url: item.imageUrl,
                        thumbnail_url: item.thumbnailUrl,
                        category: item.category,
                        sub_category: item.subCategory,
                        type: item.subCategory,
                        detection_confidence: item.detectionConfidence,
                        primary_color: item.primaryColor,
                        color_hex: item.colorHex,
                        pattern: item.pattern,
                        material: item.material,
                        brand: item.brand,
                        name: item.name,
                        seasons: item.seasons,
                        occasions: item.occasions,
                        wear_count: item.wearCount,
                        is_favorite: item.isFavorite,
                        created_at: item.createdAt,
                        updated_at: item.updatedAt,
                    };
                });

                assertAccount();
                const { error: batchError } = await supabase
                    .from('clothing_items')
                    .upsert(rows);

                if (batchError) {
                    if (batchError.code === 'P0001' && batchError.message === 'WARDROBE_LIMIT_REACHED') {
                        try {
                            const detail = JSON.parse(batchError.details ?? 'null');
                            if (detail?.limit === 20 || detail?.limit === 100) {
                                capacityIssue = { limit: detail.limit, actionIds: localWinsActions.map(action => action.id) };
                            }
                        } catch { /* Other failures stay retryable without inventing a capacity decision. */ }
                    }
                    // Batch failed — keep all local-wins actions in the queue for retry.
                    console.error(
                        '[WardrobeSyncService] Batch upsert failed:',
                        batchError
                    );
                } else {
                    assertAccount();
                    const preparedById = new Map(preparedItems.map(item => [item.id, item]));
                    items = items.map(item => preparedById.get(item.id) ?? item);
                    // All rows written successfully — mark every action as processed.
                    for (const action of localWinsActions) {
                        processed.push(action.id);
                    }
                }
            }
        } catch (err) {
            console.error('[WardrobeSyncService] add_item batch processing failed:', err);
            // Keep all add_item actions in the queue for retry
        }
    }

    const unconfirmedItemIds = new Set(addItemActions.filter(action => !processed.includes(action.id)).map(action => action.payload.id));
    // ── Process remaining action types sequentially ────────────────────────
    for (const action of otherActions) {
        try {
            assertAccount();
            switch (action.type) {
                case 'remove_item': {
                    const { itemId } = action.payload as { itemId: string };
                    const { error } = await supabase.from('clothing_items').delete().eq('id', itemId).eq('user_id', userId);
                    if (error) throw error;
                    assertAccount();
                    processed.push(...(deletionGroups.get(itemId) ?? [action]).map(queued => queued.id));
                    break;
                }
                case 'update_item': {
                    const { itemId, isFavorite } = action.payload;
                    if (typeof itemId !== 'string' || typeof isFavorite !== 'boolean') throw new Error('Invalid favorite action');
                    const group = favoriteGroups.get(itemId)!;
                    if (group[group.length - 1] !== action) break;
                    const readFavorite = () => withRequestDeadline(signal => supabase.from('clothing_items')
                        .select('id, is_favorite, updated_at').eq('id', itemId).eq('user_id', userId)
                        .abortSignal(signal).single(), WARDROBE_READ_TIMEOUT_MS);
                    const { data: snapshot, error: versionError } = await readFavorite();
                    if (versionError || snapshot?.id !== itemId ||
                        !(typeof snapshot.updated_at === 'string' || snapshot.updated_at === null)) {
                        throw versionError || new Error('Favorite version not confirmed');
                    }
                    assertAccount();
                    const acceptServerFavorite = (row: typeof snapshot) => {
                        if (!row || row.id !== itemId || typeof row.is_favorite !== 'boolean' ||
                            !(typeof row.updated_at === 'string' || row.updated_at === null)) throw new Error('Favorite conflict not confirmed');
                        assertAccount();
                        items = items.map(item => item.id === itemId
                            ? { ...item, isFavorite: row.is_favorite, updatedAt: row.updated_at ?? undefined } : item);
                        processed.push(...group.map(queued => queued.id));
                    };
                    const hasVersion = Object.prototype.hasOwnProperty.call(action.payload, 'favoriteVersion');
                    const version = hasVersion ? action.payload.favoriteVersion : snapshot.updated_at;
                    if (!(typeof version === 'string' || version === null)) throw new Error('Invalid favorite version');
                    if (hasVersion && version !== snapshot.updated_at) {
                        acceptServerFavorite(snapshot);
                        break;
                    }
                    if (!hasVersion) {
                        const prepared = { ...action, payload: { ...action.payload, favoriteVersion: version } };
                        preparedFavorites.set(action.id, prepared);
                        // Persistent callers must save this version before mutation.
                        await checkpointFavorite?.([prepared]);
                        assertAccount();
                    }
                    let update = supabase.from('clothing_items')
                        .update({ is_favorite: isFavorite, updated_at: action.createdAt })
                        .eq('id', itemId).eq('user_id', userId);
                    // Write even if the boolean already matches: the server's
                    // version change must invalidate older delayed requests.
                    // An older request cannot overwrite a row changed since its read.
                    update = version === null
                        ? update.is('updated_at', null) : update.eq('updated_at', version);
                    const { data, error } = await update.select('id, is_favorite, updated_at').single();
                    if (error?.code === 'PGRST116') {
                        // The primary-key/owner/version predicate matched no row.
                        // Confirm the current server value before retiring the
                        // conflicted action; blindly retrying could undo it later.
                        const { data: current, error: currentError } = await readFavorite();
                        if (currentError) throw currentError;
                        acceptServerFavorite(current);
                        break;
                    }
                    if (error || data?.id !== itemId || data.is_favorite !== isFavorite ||
                        !(typeof data.updated_at === 'string' || data.updated_at === null)) {
                        throw error || new Error('Favorite update not confirmed');
                    }
                    assertAccount();
                    items = items.map(item => item.id === itemId
                        ? { ...item, isFavorite: data.is_favorite, updatedAt: data.updated_at ?? undefined } : item);
                    // Acknowledge superseded taps only with the final value.
                    processed.push(...group.map(queued => queued.id));
                    break;
                }
                case 'add_wear_log': {
                    const log = action.payload as unknown as WearLog;
                    if (log.itemIds?.some(id => unconfirmedItemIds.has(id))) break;
                    const { error } = await supabase.from('wear_logs').upsert({
                        id: log.id,
                        user_id: userId,
                        outfit_id: log.outfitId,
                        item_ids: log.itemIds,
                        date: log.date,
                        occasion: log.occasion,
                        weather_temp: log.weatherTemp,
                        weather_condition: log.weatherCondition,
                        created_at: log.createdAt,
                    });
                    if (error) throw error;
                    assertAccount();
                    processed.push(action.id);
                    break;
                }
            }
        } catch (err) {
            console.error(`[WardrobeSyncService] Action ${action.type} failed:`, err);
            // Keep in queue for retry
        }
    }

    const updatedActions = [...preparedFavorites.values()].filter(action => !processed.includes(action.id));
    return { processedIds: processed, updatedItems: items, ...(updatedActions.length ? { updatedActions } : {}), ...(capacityIssue ? { capacityIssue } : {}) };
}


/** Reconcile a fetch without undoing local work completed while it was in flight. */
export function reconcileCloudItems(remote: ClothingItem[], before: ClothingItem[], current: ClothingItem[], pending: PendingAction[]): ClothingItem[] {
    const baseline = new Map(before.map(item => [item.id, item]));
    const local = new Map(current.map(item => [item.id, item]));
    const merged = new Map(remote.map(item => [item.id, item]));
    for (const id of baseline.keys()) if (!local.has(id)) merged.delete(id);
    for (const item of current) if (baseline.get(item.id) !== item) merged.set(item.id, item);
    for (const action of pending) {
        if (action.type === 'remove_item') merged.delete(String(action.payload.itemId));
        if (action.type === 'add_item') {
            const item = action.payload as unknown as ClothingItem;
            merged.set(item.id, local.get(item.id) ?? item);
        }
        if (action.type === 'update_item') {
            const id = String(action.payload.itemId), item = merged.get(id);
            if (item && typeof action.payload.isFavorite === 'boolean') merged.set(id, { ...item, isFavorite: action.payload.isFavorite });
        }
        if (action.type === 'add_wear_log' && Array.isArray(action.payload.itemIds)) {
            // The local increment already happened when this log was queued.
            // Retain those optimistic fields until acknowledgement; adding again
            // here would double-count a committed request whose response was lost.
            for (const id of action.payload.itemIds) {
                if (typeof id !== 'string') continue;
                const item = merged.get(id), optimistic = local.get(id);
                if (!item || !optimistic) continue;
                const localTime = Date.parse(optimistic.lastWornAt ?? '');
                const remoteTime = Date.parse(item.lastWornAt ?? '');
                merged.set(id, {
                    ...item,
                    wearCount: Math.max(item.wearCount ?? 0, optimistic.wearCount ?? 0),
                    lastWornAt: Number.isFinite(localTime) && (!Number.isFinite(remoteTime) || localTime > remoteTime)
                        ? optimistic.lastWornAt : item.lastWornAt,
                });
            }
        }
    }
    return [...merged.values()];
}
