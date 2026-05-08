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
 */
export function getOptimizedImageUrl(
    url: string | undefined | null,
    options: OptimizeOptions = {}
): string {
    if (!url) return '';

    const { width = 400, height, quality = 80 } = options;

    // Supabase Storage - use native transforms
    if (isSupabaseStorageUrl(url)) {
        const separator = url.includes('?') ? '&' : '?';
        const params = [`width=${width}`, `quality=${quality}`];
        if (height) params.push(`height=${height}`);
        params.push('resize=cover');
        return `${url}${separator}${params.join('&')}`;
    }

    // Unsplash - they support w= param
    if (isUnsplashUrl(url)) {
        const separator = url.includes('?') ? '&' : '?';
        return `${url}${separator}w=${width}&q=${quality}`;
    }

    // External / other URLs - return as-is (can't resize server-side)
    return url;
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
