import { useCallback, useEffect, useState } from 'react';
import { supabase } from '../lib/supabase';
import { getMediumImageUrl } from '../src/utils/imageUrl';

export interface FeaturedCapsule {
    id: string;
    title: string;
    subtitle?: string;
    imageUrl: string;
    linkUrl?: string;
}

export interface UseFeaturedCapsulesResult {
    items: FeaturedCapsule[];
    loading: boolean;
    error: string | null;
    refresh: () => void;
}

function rowToCapsule(row: Record<string, any>): FeaturedCapsule {
    return {
        id:       row.id,
        title:    row.title,
        subtitle: row.subtitle ?? undefined,
        imageUrl: getMediumImageUrl(row.image_url),
        linkUrl:  row.link_url ?? undefined,
    };
}

/**
 * Fetches active Featured Capsules from Supabase ordered by `sort_order`.
 * The `featured_capsules` table is admin-editable from the Supabase dashboard.
 */
export function useFeaturedCapsules(): UseFeaturedCapsulesResult {
    const [items, setItems]     = useState<FeaturedCapsule[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError]     = useState<string | null>(null);

    const fetchCapsules = useCallback(async () => {
        setLoading(true);
        setError(null);

        const { data, error: fetchError } = await supabase
            .from('featured_capsules')
            .select('id, title, subtitle, image_url, link_url, sort_order')
            .eq('is_active', true)
            .order('sort_order', { ascending: true });

        if (fetchError) {
            setError(fetchError.message);
        } else {
            setItems((data ?? []).map(rowToCapsule));
        }

        setLoading(false);
    }, []);

    useEffect(() => {
        fetchCapsules();
    }, [fetchCapsules]);

    return {
        items,
        loading,
        error,
        refresh: fetchCapsules,
    };
}
