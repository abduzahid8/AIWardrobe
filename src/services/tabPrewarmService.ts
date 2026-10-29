/**
 * tabPrewarmService — eliminates first-visit tab freezes.
 *
 * Tabs are lazy: each screen mounts (and pays its full cold cost — Supabase
 * queries, O(n²) diversification, collage resolution, image decode) the first
 * time the user opens it. After that, module caches + the native image cache
 * make revisits instant — which is exactly the reported symptom: "freezes
 * when switching pages the first time, smooth after".
 *
 * This service warms those caches on idle AFTER Home is interactive:
 *   1. Data: Inspo's `all|outfits` catalog page (Home already fetches
 *      `all|all`), featured capsules, guide hero row.
 *   2. Images: above-the-fold thumbnails for Closet + Inspo via
 *      expo-image prefetch (same memory-disk policy CachedImage reads).
 *
 * Everything is staggered, fire-and-forget, and never throws. One run per
 * app launch — caches have their own TTLs and refresh silently afterwards.
 */


import { getCachedGuideContent, loadGuideContent } from './guideContent';
import {
    prefetchShopCatalog,
    getCachedShopCatalog,
} from '../../hooks/useShopCatalog';
import { prefetchFeaturedCapsules } from '../../hooks/useFeaturedCapsules';
import useWardrobeStore from '../../store/wardrobeStore';
import { createLogger } from '../utils/logger';
import { ExpoImage } from '../components/MediaImage';
import { loadStylistCatalogInventory } from './stylistCatalogInventory';


const logger = createLogger('TabPrewarm');

let dataPrewarmed = false;
let imagesPrewarmed = false;

const isValidImageUrl = (v: unknown): v is string =>
    typeof v === 'string' && v.length > 0 && v.startsWith('http');

/** InspoScreen checks this before firing its own `guide_page` query. */
export function getPrewarmedGuideHero() {
    return getCachedGuideContent();
}

async function prewarmGuideHero(): Promise<void> {
    try { await loadGuideContent(); } catch { /* The guide can retry on open. */ }
}

async function prewarmImages(): Promise<void> {
    try {
        const urls: string[] = [];
        const seen = new Set<string>();
        const push = (u: unknown) => {
            if (urls.length >= 24) return;
            if (isValidImageUrl(u) && !seen.has(u)) {
                seen.add(u);
                urls.push(u);
            }
        };

        // Closet grid — first cells the user sees.
        for (const item of useWardrobeStore.getState().items.slice(0, 8)) {
            push((item as any)?.thumbnailUrl ?? (item as any)?.imageUrl);
        }
        // Inspo capsules + shop grid head.
        for (const item of getCachedShopCatalog({ category: 'all', source: 'all' }).slice(0, 8)) {
            push((item as any)?.imageUrl);
        }
        for (const item of getCachedShopCatalog({ category: 'outfits', source: 'all' }).slice(0, 4)) {
            push((item as any)?.imageUrl);
        }

        if (urls.length === 0) return;
        // Same memory-disk policy CachedImage reads, so first render hits cache.
        await ExpoImage.prefetch(urls, 'memory-disk');
        logger.debug('Prewarmed images', { count: urls.length });
    } catch (err) {
        logger.warn('Image prewarm skipped', err);
    }
}

const delay = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/**
 * Warm all non-Home tab caches. Call once, deferred until after first
 * frame (RootNavigator schedules it ~2.5s after app-ready). Safe to call
 * repeatedly — subsequent calls are no-ops.
 */
export async function prewarmTabs(): Promise<void> {
    if (dataPrewarmed && imagesPrewarmed) return;

    // The stylist's balanced inventory is independent of the generic shop page.
    // Start it before navigation so verified categories are ready on first open.
    void loadStylistCatalogInventory().catch(() => {});

    try {
        // Step 1: data — parallel small queries, one flight each.
        // (May already be warm via on-press prewarm — prefetchers no-op.)
        if (!dataPrewarmed) {
            dataPrewarmed = true;
            await Promise.all([
                prefetchShopCatalog({ category: 'outfits', source: 'all' }),
                prefetchFeaturedCapsules(),
                prewarmGuideHero(),
            ]);
            logger.debug('Tab data prewarmed');
        }

        // Step 2: images — after data lands, off the critical path.
        if (!imagesPrewarmed) {
            imagesPrewarmed = true;
            await delay(1200);
            await prewarmImages();
        }
    } catch (err) {
        logger.warn('Tab prewarm skipped', err);
    }
}

/**
 * Per-tab head-start, called from the tab bar onPress BEFORE navigate, so
 * the destination's data fetch is already in flight during the 50ms mount
 * defer + transition. Covers users who tap a tab before the idle prewarm
 * above has run (the usual case in the first seconds after launch).
 * Data only — no image storm, never throws, never blocks the press.
 */
export function prewarmTab(routeName: string): void {
    try {
        if (routeName === 'Inspo') {
            dataPrewarmed = true;
            void prefetchShopCatalog({ category: 'outfits', source: 'all' });
            void prefetchFeaturedCapsules();
            void prewarmGuideHero();
        }
        // Closet + Profile are store-driven (wardrobe already loaded at
        // startup) — nothing to fetch; their first-paint cost is render,
        // handled by progressive rendering + memoization in the screens.
    } catch {
        // Never break a tab press.
    }
}
