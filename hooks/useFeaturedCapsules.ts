import { useCallback, useEffect, useState } from 'react';
import { supabase } from '../lib/supabase';
import { getMediumImageUrl } from '../src/utils/imageUrl';

// ── Module-level cache ───────────────────────────────────────────────────
// Capsule data rarely changes — persist across tab switches to avoid
// a Supabase round-trip on every Inspo tab visit.
let capsulesCache: { items: FeaturedCapsule[]; fetchedAt: number } | null = null;
const CAPSULES_CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

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
 * Results are cached in-memory (5 min TTL) to avoid refetching on tab revisits.
 */
export function useFeaturedCapsules(): UseFeaturedCapsulesResult {
    // Seed from cache immediately so the screen shows content without waiting
    const [items, setItems]     = useState<FeaturedCapsule[]>(capsulesCache?.items ?? []);
    const [loading, setLoading] = useState(!capsulesCache);
    const [error, setError]     = useState<string | null>(null);

    const fetchCapsules = useCallback(async (silent = false) => {
        if (!silent) setLoading(true);
        setError(null);

        const { data, error: fetchError } = await supabase
            .from('featured_capsules')
            .select('id, title, subtitle, image_url, link_url, sort_order')
            .eq('is_active', true)
            .order('sort_order', { ascending: true });

        if (fetchError) {
            setError(fetchError.message);
        } else {
            const mapped = (data ?? []).map(rowToCapsule);
            setItems(mapped);
            capsulesCache = { items: mapped, fetchedAt: Date.now() };
        }

        setLoading(false);
    }, []);

    useEffect(() => {
        if (capsulesCache) {
            // Already seeded from cache in useState — only refresh if stale
            if (Date.now() - capsulesCache.fetchedAt > CAPSULES_CACHE_TTL_MS) {
                fetchCapsules(true); // silent background refresh
            }
        } else {
            fetchCapsules(false);
        }
    }, [fetchCapsules]);

    return {
        items,
        loading,
        error,
        refresh: () => fetchCapsules(false),
    };
}
