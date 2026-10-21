import AsyncStorage from '@react-native-async-storage/async-storage';


const CACHE_INDEX_KEY = '@image_cache_index';
const MAX_CACHE_ENTRIES = 200;

interface CacheEntry {
    uri: string;
    timestamp: number;
    prefetched: boolean;
}

interface CacheIndex {
    entries: Record<string, CacheEntry>;
}

class ImageCacheManager {
    private index: CacheIndex = { entries: {} };
    private initialized = false;
    private saveTimer: ReturnType<typeof setTimeout> | null = null;
    private savePending = false;

    async initialize(): Promise<void> {
        if (this.initialized) return;

        try {
            // Load cache index
            const indexData = await AsyncStorage.getItem(CACHE_INDEX_KEY);
            if (indexData) {
                this.index = JSON.parse(indexData);
            }

            // Clean old entries
            await this.cleanOldEntries();

            this.initialized = true;
        } catch (error) {
            console.error('Image cache initialization error:', error);
        }
    }

    private async saveIndex(immediate = false): Promise<void> {
        // Debounced writes — getCachedImage is called per cell during scroll;
        // awaiting AsyncStorage on every cell blocked the JS thread (freeze).
        if (!immediate) {
            this.savePending = true;
            if (this.saveTimer) return;
            this.saveTimer = setTimeout(() => {
                this.saveTimer = null;
                if (!this.savePending) return;
                this.savePending = false;
                AsyncStorage.setItem(CACHE_INDEX_KEY, JSON.stringify(this.index)).catch(() => {});
            }, 2000);
            return;
        }
        try {
            await AsyncStorage.setItem(CACHE_INDEX_KEY, JSON.stringify(this.index));
        } catch {
            // Best-effort — never block image rendering on index persistence.
        }
    }

    private generateCacheKey(uri: string): string {
        let hash = 0;
        for (let i = 0; i < uri.length; i++) {
            const char = uri.charCodeAt(i);
            hash = ((hash << 5) - hash) + char;
            hash = hash & hash;
        }
        return `img_${Math.abs(hash).toString(16)}`;
    }

    async getCachedImage(uri: string): Promise<string> {
        await this.initialize();

        const cacheKey = this.generateCacheKey(uri);
        const entry = this.index.entries[cacheKey];

        if (entry) {
            // Update timestamp for LRU — debounced, never awaited on scroll.
            entry.timestamp = Date.now();
            void this.saveIndex();
        } else {
            // Add to index and prefetch
            await this.addToCache(uri, cacheKey);
        }

        // Return original URI - expo-image handles memory-disk caching
        return uri;
    }

    private async addToCache(uri: string, cacheKey: string): Promise<void> {
        try {
            await this.ensureSpace();

            // Prefetch via expo-image disk cache (RN Image.prefetch bypassed it).
            // Fire-and-forget with timeout so scroll never awaits network.
            void this.prefetchImage(uri);

            this.index.entries[cacheKey] = {
                uri,
                timestamp: Date.now(),
                prefetched: true,
            };

            void this.saveIndex();
        } catch {
            // Best-effort.
        }
    }

    private async prefetchImage(uri: string): Promise<void> {
        try {
            await Promise.race([
                ExpoImage.prefetch(uri),
                new Promise((resolve) => setTimeout(resolve, 5000)),
            ]);
        } catch {
            // Ignore errors — image will load on demand via expo-image.
        }
    }

    private async ensureSpace(): Promise<void> {
        const entries = Object.entries(this.index.entries);

        if (entries.length >= MAX_CACHE_ENTRIES) {
            // Remove oldest entries
            const sorted = entries.sort((a, b) => a[1].timestamp - b[1].timestamp);
            const toRemove = sorted.slice(0, Math.floor(MAX_CACHE_ENTRIES / 4));

            for (const [key] of toRemove) {
                delete this.index.entries[key];
            }
        }
    }

    private async cleanOldEntries(): Promise<void> {
        const maxAge = 7 * 24 * 60 * 60 * 1000; // 7 days
        const now = Date.now();

        const expiredKeys = Object.entries(this.index.entries)
            .filter(([_, entry]) => now - entry.timestamp > maxAge)
            .map(([key]) => key);

        for (const key of expiredKeys) {
            delete this.index.entries[key];
        }

        if (expiredKeys.length > 0) {
            await this.saveIndex();
        }
    }

    async clearCache(): Promise<void> {
        this.index = { entries: {} };
        await AsyncStorage.removeItem(CACHE_INDEX_KEY);
    }

    getCacheStats(): { count: number } {
        return {
            count: Object.keys(this.index.entries).length,
        };
    }

    // Prefetch multiple images — concurrency-limited so a grid mount
    // doesn't fire 50 parallel prefetches and freeze the JS thread.
    async prefetch(uris: string[]): Promise<void> {
        await this.initialize();

        const CONCURRENCY = 6;
        const queue = [...new Set(uris.filter(Boolean))];
        const workers = Array.from({ length: Math.min(CONCURRENCY, queue.length) }, async () => {
            while (queue.length > 0) {
                const uri = queue.shift();
                if (!uri) break;
                try {
                    await this.getCachedImage(uri);
                } catch {
                    // Ignore per-image failures.
                }
            }
        });
        await Promise.all(workers);
    }
}

// Singleton instance
export const imageCache = new ImageCacheManager();

// Helper function for components
export const getCachedImageUri = async (uri: string): Promise<string> => {
    if (!uri || !uri.startsWith('http')) return uri;
    return imageCache.getCachedImage(uri);
};

// React hook for cached images
import { useState, useEffect } from 'react';
import { ExpoImage } from '../components/MediaImage';


export const useCachedImage = (uri: string): { cachedUri: string; loading: boolean } => {
    const [cachedUri, setCachedUri] = useState(uri);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        let mounted = true;

        const loadCached = async () => {
            if (!uri || !uri.startsWith('http')) {
                setCachedUri(uri);
                setLoading(false);
                return;
            }

            try {
                const cached = await imageCache.getCachedImage(uri);
                if (mounted) {
                    setCachedUri(cached);
                }
            } catch (error) {
                if (mounted) {
                    setCachedUri(uri);
                }
            } finally {
                if (mounted) {
                    setLoading(false);
                }
            }
        };

        setLoading(true);
        loadCached();

        return () => {
            mounted = false;
        };
    }, [uri]);

    return { cachedUri, loading };
};

export default imageCache;
