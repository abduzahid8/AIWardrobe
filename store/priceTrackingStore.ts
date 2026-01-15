/**
 * Price Tracking Store for AIWardrobe
 * Track prices of wishlist items and get alerts on price drops
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';

export interface PricePoint {
    price: number;
    currency: string;
    date: string;
    source: string;
}

export interface TrackedItem {
    id: string;
    name: string;
    brand: string;
    imageUrl: string;
    category: string;
    currentPrice: number;
    originalPrice?: number;
    currency: string;
    priceHistory: PricePoint[];
    targetPrice?: number;
    url?: string;
    store?: string;
    dateAdded: string;
    lastChecked: string;
    priceDropPercent?: number;
    isOnSale: boolean;
    notes?: string;
}

export interface PriceAlert {
    id: string;
    itemId: string;
    itemName: string;
    previousPrice: number;
    newPrice: number;
    dropPercent: number;
    date: string;
    seen: boolean;
}

interface PriceTrackingState {
    trackedItems: TrackedItem[];
    priceAlerts: PriceAlert[];
    isLoading: boolean;

    // Actions
    addItem: (item: Omit<TrackedItem, 'id' | 'dateAdded' | 'lastChecked' | 'priceHistory' | 'isOnSale'>) => void;
    removeItem: (id: string) => void;
    updatePrice: (id: string, newPrice: number, source?: string) => void;
    setTargetPrice: (id: string, targetPrice: number) => void;
    markAlertSeen: (alertId: string) => void;
    clearAllAlerts: () => void;
    getItemById: (id: string) => TrackedItem | undefined;
    getItemsByBrand: (brand: string) => TrackedItem[];
    getItemsOnSale: () => TrackedItem[];
    getUnseenAlerts: () => PriceAlert[];
    getTotalSavings: () => number;
}

const usePriceTrackingStore = create<PriceTrackingState>()(
    persist(
        (set, get) => ({
            trackedItems: [],
            priceAlerts: [],
            isLoading: false,

            addItem: (item) => {
                const newItem: TrackedItem = {
                    ...item,
                    id: `price_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                    dateAdded: new Date().toISOString(),
                    lastChecked: new Date().toISOString(),
                    priceHistory: [{
                        price: item.currentPrice,
                        currency: item.currency,
                        date: new Date().toISOString(),
                        source: item.store || 'manual',
                    }],
                    isOnSale: item.originalPrice ? item.currentPrice < item.originalPrice : false,
                    priceDropPercent: item.originalPrice
                        ? Math.round((1 - item.currentPrice / item.originalPrice) * 100)
                        : undefined,
                };

                set((state) => ({
                    trackedItems: [newItem, ...state.trackedItems],
                }));
            },

            removeItem: (id) => {
                set((state) => ({
                    trackedItems: state.trackedItems.filter((item) => item.id !== id),
                    priceAlerts: state.priceAlerts.filter((alert) => alert.itemId !== id),
                }));
            },

            updatePrice: (id, newPrice, source = 'check') => {
                set((state) => {
                    const item = state.trackedItems.find((i) => i.id === id);
                    if (!item) return state;

                    const previousPrice = item.currentPrice;
                    const priceDropped = newPrice < previousPrice;
                    const dropPercent = priceDropped
                        ? Math.round((1 - newPrice / previousPrice) * 100)
                        : 0;

                    // Create alert if price dropped
                    const newAlerts = priceDropped ? [{
                        id: `alert_${Date.now()}`,
                        itemId: id,
                        itemName: item.name,
                        previousPrice,
                        newPrice,
                        dropPercent,
                        date: new Date().toISOString(),
                        seen: false,
                    }, ...state.priceAlerts] : state.priceAlerts;

                    // Update item
                    const updatedItems = state.trackedItems.map((i) => {
                        if (i.id !== id) return i;

                        const newPricePoint: PricePoint = {
                            price: newPrice,
                            currency: i.currency,
                            date: new Date().toISOString(),
                            source,
                        };

                        return {
                            ...i,
                            currentPrice: newPrice,
                            lastChecked: new Date().toISOString(),
                            priceHistory: [newPricePoint, ...i.priceHistory].slice(0, 30), // Keep last 30
                            isOnSale: i.originalPrice ? newPrice < i.originalPrice : priceDropped,
                            priceDropPercent: i.originalPrice
                                ? Math.round((1 - newPrice / i.originalPrice) * 100)
                                : dropPercent,
                        };
                    });

                    return {
                        trackedItems: updatedItems,
                        priceAlerts: newAlerts,
                    };
                });
            },

            setTargetPrice: (id, targetPrice) => {
                set((state) => ({
                    trackedItems: state.trackedItems.map((item) =>
                        item.id === id ? { ...item, targetPrice } : item
                    ),
                }));
            },

            markAlertSeen: (alertId) => {
                set((state) => ({
                    priceAlerts: state.priceAlerts.map((alert) =>
                        alert.id === alertId ? { ...alert, seen: true } : alert
                    ),
                }));
            },

            clearAllAlerts: () => {
                set({ priceAlerts: [] });
            },

            getItemById: (id) => {
                return get().trackedItems.find((item) => item.id === id);
            },

            getItemsByBrand: (brand) => {
                return get().trackedItems.filter(
                    (item) => item.brand.toLowerCase() === brand.toLowerCase()
                );
            },

            getItemsOnSale: () => {
                return get().trackedItems.filter((item) => item.isOnSale);
            },

            getUnseenAlerts: () => {
                return get().priceAlerts.filter((alert) => !alert.seen);
            },

            getTotalSavings: () => {
                return get().trackedItems.reduce((total, item) => {
                    if (item.originalPrice && item.currentPrice < item.originalPrice) {
                        return total + (item.originalPrice - item.currentPrice);
                    }
                    return total;
                }, 0);
            },
        }),
        {
            name: 'price-tracking-storage',
            storage: createJSONStorage(() => AsyncStorage),
        }
    )
);

export default usePriceTrackingStore;
