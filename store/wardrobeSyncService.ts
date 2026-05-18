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
        pattern: (row.pattern as string) || 'solid',
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
    const { data: session } = await supabase.auth.getSession();
    if (!session?.session?.user) return null;

    const { data, error } = await supabase
        .from('clothing_items')
        .select('*')
        .eq('user_id', session.session.user.id)
        .order('created_at', { ascending: false });

    if (error) {
        console.error('[WardrobeSyncService] Fetch error:', error);
        return null;
    }

    return data ? data.map(mapRowToItem) : null;
}

export async function fetchWearLogsFromServer(): Promise<WearLog[] | null> {
    const { data: session } = await supabase.auth.getSession();
    if (!session?.session?.user) return null;

    const { data, error } = await supabase
        .from('wear_logs')
        .select('*')
        .eq('user_id', session.session.user.id)
        .order('date', { ascending: false });

    if (error) {
        console.error('[WardrobeSyncService] Wear log fetch error:', error);
        return null;
    }

    return (data || []).map((row: any): WearLog => ({
        id: row.id,
        userId: row.user_id,
        outfitId: row.outfit_id ?? undefined,
        itemIds: Array.isArray(row.item_ids) ? row.item_ids : [],
        date: row.date,
        occasion: row.occasion ?? undefined,
        weatherTemp: row.weather_temp ?? undefined,
        weatherCondition: row.weather_condition ?? undefined,
        createdAt: row.created_at,
    }));
}

/**
 * Process pending sync actions against Supabase.
 * Uses conflict resolution: server version wins if newer.
 *
 * `add_item` actions are batched into a single upsert call to reduce
 * N sequential round-trips to 1 (Defect 2.5 fix).
 *
 * @returns IDs of successfully processed actions
 */
export async function processPendingActions(
    pendingActions: PendingAction[],
    localItems: ClothingItem[]
): Promise<{ processedIds: string[]; updatedItems: ClothingItem[] }> {
    if (pendingActions.length === 0) {
        return { processedIds: [], updatedItems: localItems };
    }

    const { data: session } = await supabase.auth.getSession();
    if (!session?.session?.user) {
        return { processedIds: [], updatedItems: localItems };
    }

    const userId = session.session.user.id;
    const processed: string[] = [];
    let items = [...localItems];

    // ── Separate add_item actions from other action types ──────────────────
    const addItemActions = pendingActions.filter((a) => a.type === 'add_item');
    const otherActions = pendingActions.filter((a) => a.type !== 'add_item');

    // ── Batch-process all add_item actions ─────────────────────────────────
    if (addItemActions.length > 0) {
        try {
            // Step 1: Conflict-resolution — fetch server timestamps for all items
            // in a single query instead of N individual selects.
            const addItemIds = addItemActions.map(
                (a) => (a.payload as unknown as ClothingItem).id
            );

            const { data: serverRows } = await supabase
                .from('clothing_items')
                .select('id, updated_at')
                .in('id', addItemIds);

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
                    const { data: freshItem } = await supabase
                        .from('clothing_items')
                        .select('*')
                        .eq('id', item.id)
                        .single();

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
                const rows = localWinsActions.map((action) => {
                    const item = action.payload as unknown as ClothingItem;
                    return {
                        id: item.id,
                        user_id: userId,
                        image_url: item.imageUrl,
                        thumbnail_url: item.thumbnailUrl,
                        category: item.category,
                        sub_category: item.subCategory,
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

                const { error: batchError } = await supabase
                    .from('clothing_items')
                    .upsert(rows);

                if (batchError) {
                    // Batch failed — keep all local-wins actions in the queue for retry.
                    console.error(
                        '[WardrobeSyncService] Batch upsert failed:',
                        batchError
                    );
                } else {
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

    // ── Process remaining action types sequentially ────────────────────────
    for (const action of otherActions) {
        try {
            switch (action.type) {
                case 'remove_item': {
                    const { itemId } = action.payload as { itemId: string };
                    await supabase.from('clothing_items').delete().eq('id', itemId);
                    processed.push(action.id);
                    break;
                }
                case 'add_wear_log': {
                    const log = action.payload as unknown as WearLog;
                    await supabase.from('wear_logs').upsert({
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
                    processed.push(action.id);
                    break;
                }
            }
        } catch (err) {
            console.error(`[WardrobeSyncService] Action ${action.type} failed:`, err);
            // Keep in queue for retry
        }
    }

    return { processedIds: processed, updatedItems: items };
}
