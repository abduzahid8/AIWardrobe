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

// ── Deleted-object repair shim (Oct 2026) ────────────────────────────────────
// The three capsule rows below were still pointing at `shop-catalog/admin/`
// storage objects that no longer exist (verified 404 NoSuchKey — the whole
// `admin/` prefix is empty), so every phone showed blank placeholder cards.
// Each entry maps a known-deleted object path to a verified-good image that
// actually exists in the bucket (all return 200).
//
// This is self-retiring: as soon as the DB rows are repaired (see migration
// 025_repair_featured_capsule_images.sql), the stored URLs no longer match
// these keys and the map below becomes a no-op. Safe to delete afterwards.
const DELETED_OBJECT_REPAIRS: Record<string, string> = {
    'admin/capsule-1781186443576.jpg': 'items/shop-inspo-1.png', // Summer Outfits
    'admin/capsule-1781186470011.jpg': 'items/shop-inspo-3.png', // Summer 2026
    'admin/capsule-1781186613082.png': 'items/shop-inspo-5.png', // Massimo Dutti
};

export function repairDeletedCapsuleUrl(rawUrl: string): string {
    if (typeof rawUrl !== 'string' || !rawUrl) return rawUrl;
    const marker = '/shop-catalog/';
    const bucketIdx = rawUrl.indexOf(marker);
    if (bucketIdx === -1) return rawUrl;
    for (const [deletedPath, goodPath] of Object.entries(DELETED_OBJECT_REPAIRS)) {
        if (rawUrl.includes(deletedPath)) {
            return `${rawUrl.slice(0, bucketIdx + marker.length)}${goodPath}`;
        }
    }
    return rawUrl;
}

function rowToCapsule(row: Record<string, any>): FeaturedCapsule {
    // A row whose storage object was deleted (or never uploaded) resolves to
    // a 404 at render time and shows as a blank placeholder card. Warn loudly
    // in dev so broken admin content is caught before it ships to phones —
    // production users only see the silent shirt icon.
    if (!row.image_url && __DEV__) {
        console.warn(`[useFeaturedCapsules] capsule "${row.title ?? row.id}" has no image_url — it will render as a placeholder.`);
    }
    return {
        id:       row.id,
        title:    row.title,
        subtitle: row.subtitle ?? undefined,
        imageUrl: getMediumImageUrl(repairDeletedCapsuleUrl(row.image_url)),
        linkUrl:  row.link_url ?? undefined,
    };
}

// ── Shared load + in-flight dedup ────────────────────────────────────────
// The hook and idle prewarming share one flight so a first-time Inspo visit
// after prewarm finds a warm cache instead of firing a cold query mid-tab-switch.
let capsulesInflight: Promise<FeaturedCapsule[]> | null = null;

async function loadFeaturedCapsules(): Promise<FeaturedCapsule[]> {
    if (capsulesInflight) return capsulesInflight;
    capsulesInflight = (async () => {
        const { data, error: fetchError } = await supabase
            .from('featured_capsules')
            .select('id, title, subtitle, image_url, link_url, sort_order')
            .eq('is_active', true)
            .order('sort_order', { ascending: true });
        if (fetchError) throw new Error(fetchError.message);
        const mapped = (data ?? []).map(rowToCapsule);
        capsulesCache = { items: mapped, fetchedAt: Date.now() };
        return mapped;
    })();
    try {
        return await capsulesInflight;
    } finally {
        capsulesInflight = null;
    }
}

/**
 * Idle prewarm entry point — warms the capsules cache without touching any
 * component state. No-op when fresh. Never throws.
 */
export async function prefetchFeaturedCapsules(): Promise<void> {
    try {
        if (capsulesCache && Date.now() - capsulesCache.fetchedAt <= CAPSULES_CACHE_TTL_MS) return;
        await loadFeaturedCapsules();
    } catch {
        // Non-critical — the screen will fetch on mount as before.
    }
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

        try {
            const mapped = await loadFeaturedCapsules();
            setItems(mapped);
        } catch (err: any) {
            setError(err?.message ?? 'Failed to load capsules');
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
