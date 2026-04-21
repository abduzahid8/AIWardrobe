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

    for (const action of pendingActions) {
        try {
            switch (action.type) {
                case 'add_item': {
                    const item = action.payload as unknown as ClothingItem;

                    // Conflict resolution: check server version
                    const { data: serverItem } = await supabase
                        .from('clothing_items')
                        .select('updated_at')
                        .eq('id', item.id)
                        .maybeSingle();

                    if (
                        serverItem?.updated_at &&
                        item.updatedAt &&
                        new Date(serverItem.updated_at) > new Date(item.updatedAt)
                    ) {
                        // Server wins — re-fetch this item
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
                        break;
                    }

                    await supabase.from('clothing_items').upsert({
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
                    });
                    processed.push(action.id);
                    break;
                }
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
                        user_id: session.session.user.id,
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
