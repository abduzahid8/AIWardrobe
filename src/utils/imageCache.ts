import AsyncStorage from '@react-native-async-storage/async-storage';
import { Image } from 'react-native';

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
    private prefetchQueue: string[] = [];

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

    private async saveIndex(): Promise<void> {
        try {
            await AsyncStorage.setItem(CACHE_INDEX_KEY, JSON.stringify(this.index));
        } catch (error) {
            console.error('Error saving cache index:', error);
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
            // Update timestamp for LRU
            entry.timestamp = Date.now();
            await this.saveIndex();
        } else {
            // Add to index and prefetch
            await this.addToCache(uri, cacheKey);
        }

        // Return original URI - React Native Image component handles caching
        return uri;
    }

    private async addToCache(uri: string, cacheKey: string): Promise<void> {
        try {
            await this.ensureSpace();

            // Prefetch the image
            await this.prefetchImage(uri);

            this.index.entries[cacheKey] = {
                uri,
                timestamp: Date.now(),
                prefetched: true,
            };

            await this.saveIndex();
        } catch (error) {
            console.error('Error adding to cache:', error);
        }
    }

    private async prefetchImage(uri: string): Promise<void> {
        return new Promise((resolve) => {
            Image.prefetch(uri)
                .then(() => resolve())
                .catch(() => resolve()); // Ignore errors
        });
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

    // Prefetch multiple images
    async prefetch(uris: string[]): Promise<void> {
        await this.initialize();

        const promises = uris.map(uri => this.getCachedImage(uri).catch(() => null));
        await Promise.all(promises);
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
