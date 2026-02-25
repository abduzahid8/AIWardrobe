/**
 * useWardrobeQuery — React Query hook for fetching wardrobe items
 *
 * Replaces manual useState + useEffect + try/catch patterns with
 * automatic caching, deduplication, retry, and background refetch.
 */

import { useQuery } from '@tanstack/react-query';
import { supabase } from '../../../lib/supabase';
import useAuthStore from '../../../store/auth';

export interface WardrobeQueryItem {
    id: string;
    type: string;
    category: string;
    color: string;
    imageUrl: string;
    style?: string;
    season?: string;
    description?: string;
    createdAt?: string;
}

async function fetchWardrobeItems(userId: string): Promise<WardrobeQueryItem[]> {
    const { data, error } = await supabase
        .from('clothing_items')
        .select('*')
        .eq('user_id', userId)
        .order('created_at', { ascending: false });

    if (error) throw error;

    return (data ?? []).map((item) => ({
        id: item.id,
        type: item.type || '',
        category: item.category || '',
        color: item.color || '',
        imageUrl: item.image_url || '',
        style: item.style,
        season: item.season,
        description: item.description,
        createdAt: item.created_at,
    }));
}

export function useWardrobeQuery() {
    const { user } = useAuthStore();

    return useQuery({
        queryKey: ['wardrobe-items', user?.id],
        queryFn: () => fetchWardrobeItems(user!.id),
        enabled: !!user?.id,
    });
}
