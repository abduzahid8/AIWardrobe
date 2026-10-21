import { Config } from '../config/env';

const SUPABASE_URL = Config.supabase.url;

interface OptimizeOptions {
    width?: number;
    height?: number;
    quality?: number;
}

/**
 * Check if a URL is a Supabase Storage URL (from this project).
 */
export function isSupabaseStorageUrl(url: string): boolean {
    if (!url || !url.startsWith('http')) return false;
    if (!SUPABASE_URL || SUPABASE_URL.includes('placeholder')) return false;
    return url.includes(SUPABASE_URL) && url.includes('/storage/v1/object/public/');
}

/**
 * Check if a URL is an Unsplash image.
 */
export function isUnsplashUrl(url: string): boolean {
    return typeof url === 'string' && url.includes('images.unsplash.com');
}

/**
 * Append Supabase image transformation params to resize/optimize images server-side.
 * Falls back to Unsplash width param, otherwise returns the original URL.
 *
 * NOTE: Supabase transformations only work through the Render endpoint
 * (`/storage/v1/render/image/public/...`) and only when the project has the
 * Image Transformations addon enabled. Plain public-object URLs
 * (`/storage/v1/object/public/...`) ignore — or on some CDN configs reject —
 * query params, so they are returned as-is. Appending `?width=` to them only
 * created cache-busting variants that made deleted/broken objects harder to
 * debug (every card showed a unique URL for the same missing file).
 */
export function getOptimizedImageUrl(
    url: string | undefined | null,
    options: OptimizeOptions = {}
): string {
    if (!url) return '';
    const trimmed = url.trim();
    if (!trimmed) return '';

    const { width = 400, quality = 80 } = options;

    // Supabase Storage public-object URL — return as-is (see NOTE above).
    // A `/render/image/` URL already carries its own transform params.
    if (isSupabaseStorageUrl(trimmed) || trimmed.includes('/storage/v1/render/image/')) {
        return trimmed;
    }

    // Unsplash - they support w=/q= params. Don't duplicate them when the
    // stored URL already carries sizing (e.g. seeded `?w=800&q=80` rows) —
    // just return as-is instead of appending a second `&w=`.
    if (isUnsplashUrl(trimmed)) {
        if (/[?&][wq]=/.test(trimmed)) return trimmed;
        const separator = trimmed.includes('?') ? '&' : '?';
        return `${trimmed}${separator}w=${width}&q=${quality}`;
    }

    // External / other URLs - return as-is (can't resize server-side)
    return trimmed;
}

/**
 * Get a small thumbnail URL suitable for grid/list cards.
 */
export function getThumbnailUrl(url: string | undefined | null): string {
    return getOptimizedImageUrl(url, { width: 300, height: 400, quality: 75 });
}

/**
 * Get a medium image URL suitable for detail/capsule views.
 */
export function getMediumImageUrl(url: string | undefined | null): string {
    return getOptimizedImageUrl(url, { width: 600, height: 800, quality: 80 });
}

/**
 * Get a small avatar/mini image URL.
 */
export function getMiniImageUrl(url: string | undefined | null): string {
    return getOptimizedImageUrl(url, { width: 150, height: 150, quality: 70 });
}
