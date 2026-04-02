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

/**
 * Fetch all wear logs for the current user from Supabase.
 * Returns null if no session or on error.
 */
export async function fetchWearLogsFromServer(): Promise<WearLog[] | null> {
    const { data: session } = await supabase.auth.getSession();
    if (!session?.session?.user) return null;

    const { data, error } = await supabase
        .from('wear_logs')
        .select('*')
        .eq('user_id', session.session.user.id)
        .order('date', { ascending: false })
        .limit(500);

    if (error) {
        console.error('[WardrobeSyncService] Fetch wear logs error:', error);
        return null;
    }

    if (!data) return null;

    return data.map((row: Record<string, unknown>): WearLog => ({
        id: row.id as string,
        userId: row.user_id as string,
        outfitId: row.outfit_id as string | undefined,
        itemIds: (row.item_ids as string[]) || [],
        date: row.date as string,
        occasion: row.occasion as Occasion | string | undefined,
        weatherTemp: row.weather_temp as number | undefined,
        weatherCondition: row.weather_condition as string | undefined,
        createdAt: row.created_at as string,
    }));
}

/**
 * Process pending sync actions against Supabase.
 * Uses conflict resolution: server version wins if newer.
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

    const processed: string[] = [];
    let items = [...localItems];

    try {
        // --- 1. Batch Add Items ---
        const addActions = pendingActions.filter(a => a.type === 'add_item');
        if (addActions.length > 0) {
            const itemsToUpsert = addActions.map(action => {
                const item = action.payload as unknown as ClothingItem;
                return {
                    id: item.id,
                    user_id: session.session.user.id,
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
            const { error } = await supabase.from('clothing_items').upsert(itemsToUpsert);
            if (!error) {
                processed.push(...addActions.map(a => a.id));
            } else {
                console.error('[WardrobeSyncService] Batch add_item failed:', error);
            }
        }

        // --- 2. Batch Remove Items ---
        const removeActions = pendingActions.filter(a => a.type === 'remove_item');
        if (removeActions.length > 0) {
            const idsToDelete = removeActions.map(a => (a.payload as { itemId: string }).itemId);
            const { error } = await supabase.from('clothing_items').delete().in('id', idsToDelete);
            if (!error) {
                processed.push(...removeActions.map(a => a.id));
                // Update local items to reflect deletions in memory
                items = items.filter(item => !idsToDelete.includes(item.id));
            } else {
                console.error('[WardrobeSyncService] Batch remove_item failed:', error);
            }
        }

        // --- 3. Batch Add Wear Logs ---
        const logActions = pendingActions.filter(a => a.type === 'add_wear_log');
        if (logActions.length > 0) {
            const logsToUpsert = logActions.map(action => {
                const log = action.payload as unknown as WearLog;
                return {
                    id: log.id,
                    user_id: session.session.user.id,
                    outfit_id: log.outfitId,
                    item_ids: log.itemIds,
                    date: log.date,
                    occasion: log.occasion,
                    weather_temp: log.weatherTemp,
                    weather_condition: log.weatherCondition,
                    created_at: log.createdAt,
                };
            });
            const { error } = await supabase.from('wear_logs').upsert(logsToUpsert);
            if (!error) {
                processed.push(...logActions.map(a => a.id));
            } else {
                console.error('[WardrobeSyncService] Batch add_wear_log failed:', error);
            }
        }
    } catch (err) {
        console.error('[WardrobeSyncService] Batch process failed:', err);
    }

    return { processedIds: processed, updatedItems: items };
}
