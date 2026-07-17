import { useState, useCallback, useEffect, useMemo } from 'react';
import AsyncStorage from '@react-native-async-storage/async-storage';
import * as Haptics from 'expo-haptics';
import { Alert } from 'react-native';
import { useFocusEffect } from '@react-navigation/native';
import { createLogger } from '../utils/logger';
import { useTranslation } from 'react-i18next';

const logger = createLogger('useWardrobeItems');

// Storage keys
const WARDROBE_KEY = 'myWardrobeItems';
const FAVORITES_KEY = 'wardrobeFavorites';

export interface WardrobeItem {
    id: string | number;
    itemType?: string;
    type?: string;
    color?: string;
    style?: string;
    description?: string;
    imageUrl?: string;
    image?: string;
    uniqueId?: string;
    isLocal?: boolean;
    isSaved?: boolean;
}

export type FilterCategory = 'All' | 'Tops' | 'Bottoms' | 'Shoes' | 'Accessories' | 'Favorites';

interface UseWardrobeItemsOptions {
    includePopularItems?: boolean;
    popularItems?: WardrobeItem[];
}

interface UseWardrobeItemsReturn {
    items: WardrobeItem[];
    filteredItems: WardrobeItem[];
    favorites: string[];
    loading: boolean;
    filter: FilterCategory;
    setFilter: (filter: FilterCategory) => void;
    deleteItem: (item: WardrobeItem) => Promise<void>;
    toggleFavorite: (itemId: string) => void;
    refreshItems: () => Promise<void>;
    updateItem: (item: WardrobeItem, updates: Partial<WardrobeItem>) => Promise<void>;
    itemCount: number;
}

/**
 * Shared hook for wardrobe item management
 * Consolidates duplicate logic from CuratedClosetScreen and ProfileScreen
 */
export const useWardrobeItems = (options: UseWardrobeItemsOptions = {}): UseWardrobeItemsReturn => {
    const { includePopularItems = false, popularItems = [] } = options;

    const { t } = useTranslation();

    const [items, setItems] = useState<WardrobeItem[]>([]);
    const [favorites, setFavorites] = useState<string[]>([]);
    const [loading, setLoading] = useState(true);
    const [filter, setFilter] = useState<FilterCategory>('All');

    // Load wardrobe items from AsyncStorage
    const loadItems = useCallback(async () => {
        try {
            setLoading(true);
            const data = await AsyncStorage.getItem(WARDROBE_KEY);
            if (data) {
                const parsedItems = JSON.parse(data);
                // Normalize item format with unique IDs
                const normalizedItems = parsedItems.map((item: WardrobeItem, idx: number) => ({
                    ...item,
                    id: item.id || `local-${idx}-${Date.now()}`,
                    uniqueId: item.uniqueId || `saved-${item.id || idx}-${Date.now()}`,
                    image: item.image || item.imageUrl,
                    type: item.type || item.itemType,
                    isLocal: true,
                    isSaved: true,
                }));
                setItems(normalizedItems);
                logger.debug(`📦 Loaded ${normalizedItems.length} items`);
            } else {
                setItems([]);
            }
        } catch (error) {
            console.error('Error loading wardrobe items:', error);
            setItems([]);
        } finally {
            setLoading(false);
        }
    }, []);

    // Load favorites from AsyncStorage
    const loadFavorites = useCallback(async () => {
        try {
            const data = await AsyncStorage.getItem(FAVORITES_KEY);
            if (data) {
                setFavorites(JSON.parse(data));
            }
        } catch (error) {
            console.error('Error loading favorites:', error);
        }
    }, []);

    // Refresh data when screen is focused
    useFocusEffect(
        useCallback(() => {
            let mounted = true;
            loadItems().then(() => {
                if (mounted) loadFavorites();
            });
            return () => { mounted = false; };
        }, [loadItems, loadFavorites])
    );

    // Save favorites to AsyncStorage
    const saveFavorites = async (newFavorites: string[]) => {
        try {
            await AsyncStorage.setItem(FAVORITES_KEY, JSON.stringify(newFavorites));
            setFavorites(newFavorites);
        } catch (error) {
            console.error('Error saving favorites:', error);
        }
    };

    // Toggle favorite status
    const toggleFavorite = useCallback((itemId: string) => {
        Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
        const newFavorites = favorites.includes(itemId)
            ? favorites.filter(id => id !== itemId)
            : [...favorites, itemId];
        saveFavorites(newFavorites);
    }, [favorites]);

    // Delete an item from wardrobe
    const deleteItem = useCallback(async (item: WardrobeItem) => {
        return new Promise<void>((resolve) => {
            const label = item.type || item.itemType || t('confirmations.thisItem');
            Alert.alert(
                t('confirmations.removeItem'),
                t('confirmations.removeItemConfirm', { label }),
                [
                    { text: t('common.cancel'), style: 'cancel', onPress: () => resolve() },
                    {
                        text: t('common.delete'),
                        style: 'destructive',
                        onPress: async () => {
                            try {
                                const newItems = items.filter(i => i.id !== item.id);
                                await AsyncStorage.setItem(WARDROBE_KEY, JSON.stringify(newItems));
                                setItems(newItems);

                                // Also remove from favorites if present
                                const itemUniqueId = String(item.uniqueId || item.id);
                                if (favorites.includes(itemUniqueId)) {
                                    saveFavorites(favorites.filter(id => id !== itemUniqueId));
                                }

                                Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
                                resolve();
                            } catch (error) {
                                console.error('Error deleting item:', error);
                                Alert.alert(t('common.error'), t('wardrobeItems.failedDeleteItem'));
                                resolve();
                            }
                        },
                    },
                ]
            );
        });
    }, [items, favorites, t]);

    // Update an item in wardrobe
    const updateItem = useCallback(async (item: WardrobeItem, updates: Partial<WardrobeItem>) => {
        try {
            const updatedItems = items.map(i =>
                i.id === item.id ? { ...i, ...updates } : i
            );
            await AsyncStorage.setItem(WARDROBE_KEY, JSON.stringify(updatedItems));
            setItems(updatedItems);
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        } catch (error) {
            console.error('Error updating item:', error);
            Alert.alert(t('common.error'), t('wardrobeItems.failedSaveChanges'));
        }
    }, [items, t]);

    // Refresh items manually
    const refreshItems = useCallback(async () => {
        await loadItems();
        await loadFavorites();
    }, [loadItems, loadFavorites]);

    // Apply filter logic
    const getFilteredItems = useCallback(() => {
        // Combine local items with popular items if requested
        let allItems = [...items];

        if (includePopularItems && popularItems.length > 0) {
            const normalizedPopular = popularItems.map((item, idx) => ({
                ...item,
                uniqueId: item.uniqueId || `popular-${item.type || 'item'}-${idx}`,
                isLocal: false,
                isSaved: false,
            }));
            allItems = [...items, ...normalizedPopular];
        }

        // Apply category filter
        if (filter === 'Favorites') {
            return allItems.filter(item => favorites.includes(String(item.uniqueId || item.id)));
        }

        if (filter === 'All') {
            return allItems;
        }

        return allItems.filter(item => {
            const type = (item.type || item.itemType || '').toLowerCase();

            switch (filter) {
                case 'Tops':
                    return type.includes('shirt') || type.includes('top') || type.includes('blouse') || type.includes('jacket');
                case 'Bottoms':
                    return type.includes('pant') || type.includes('jean') || type.includes('skirt') || type.includes('shorts');
                case 'Shoes':
                    return type.includes('shoe') || type.includes('sneaker') || type.includes('boot') || type.includes('heel');
                case 'Accessories':
                    return type.includes('watch') || type.includes('bag') || type.includes('hat') || type.includes('belt') || type.includes('scarf');
                default:
                    return true;
            }
        });
    }, [items, filter, favorites, includePopularItems, popularItems]);

    const filteredItems = useMemo(() => getFilteredItems(), [getFilteredItems]);

    return {
        items,
        filteredItems,
        favorites,
        loading,
        filter,
        setFilter,
        deleteItem,
        toggleFavorite,
        refreshItems,
        updateItem,
        itemCount: items.length,
    };
};

export default useWardrobeItems;
