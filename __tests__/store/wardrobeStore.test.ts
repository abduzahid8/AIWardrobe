/**
 * Tests for wardrobeStore
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

// Mock supabase
jest.mock('../../lib/supabase', () => ({
    supabase: {
        auth: {
            getSession: jest.fn().mockResolvedValue({ data: { session: null } }),
            onAuthStateChange: jest.fn().mockReturnValue({ data: { subscription: { unsubscribe: jest.fn() } } }),
        },
        from: jest.fn().mockReturnValue({
            select: jest.fn().mockReturnThis(),
            eq: jest.fn().mockReturnThis(),
            order: jest.fn().mockReturnThis(),
            upsert: jest.fn().mockResolvedValue({ data: null, error: null }),
            delete: jest.fn().mockReturnThis(),
        }),
    },
}));

beforeEach(() => {
    jest.resetModules();
    jest.clearAllMocks();
    (AsyncStorage.getItem as jest.Mock).mockResolvedValue(null);
    (AsyncStorage.setItem as jest.Mock).mockResolvedValue(undefined);
    (AsyncStorage.removeItem as jest.Mock).mockResolvedValue(undefined);
});

const getStore = () => {
    const store = require('../../store/wardrobeStore').default;
    store.setState({
        items: [],
        outfits: [],
        wearLogs: [],
        dailySuggestion: null,
        streak: 0,
        lastWearDate: null,
        isLoading: false,
        isSyncing: false,
        lastSyncedAt: null,
        pendingActions: [],
    });
    return store;
};

describe('wardrobeStore', () => {
    describe('initial state', () => {
        it('should have empty arrays and zero streak', () => {
            const store = getStore();
            const state = store.getState();
            expect(state.items).toEqual([]);
            expect(state.outfits).toEqual([]);
            expect(state.wearLogs).toEqual([]);
            expect(state.streak).toBe(0);
            expect(state.isLoading).toBe(false);
        });
    });

    describe('addItem', () => {
        it('should add an item with generated id and timestamps', async () => {
            const store = getStore();
            await store.getState().addItem({
                userId: 'user1',
                imageUrl: 'https://example.com/shirt.jpg',
                category: 'top',
                subCategory: 't-shirt',
                primaryColor: 'Blue',
                colorHex: '#2563EB',
                pattern: 'solid',
                material: 'cotton',
                seasons: ['summer'],
                occasions: ['casual'],
            });

            const state = store.getState();
            expect(state.items).toHaveLength(1);
            expect(state.items[0].id).toBeDefined();
            expect(state.items[0].category).toBe('top');
            expect(state.items[0].wearCount).toBe(0);
            expect(state.items[0].isFavorite).toBe(false);
            expect(state.items[0].createdAt).toBeDefined();
        });

        it('should queue a pending action for sync', async () => {
            const store = getStore();
            await store.getState().addItem({
                userId: 'user1',
                imageUrl: 'https://example.com/pants.jpg',
                category: 'bottom',
                subCategory: 'jeans',
                primaryColor: 'Navy',
                colorHex: '#1B2A4A',
                pattern: 'solid',
                material: 'denim',
                seasons: ['fall', 'winter'],
                occasions: ['casual'],
            });

            const state = store.getState();
            expect(state.pendingActions.length).toBeGreaterThanOrEqual(1);
            expect(state.pendingActions.find((a: any) => a.type === 'add_item')).toBeDefined();
        });
    });

    describe('logWear', () => {
        it('should create a wear log and increment item wear counts', async () => {
            const store = getStore();

            // First add an item
            await store.getState().addItem({
                userId: 'user1',
                imageUrl: 'https://example.com/shirt.jpg',
                category: 'top',
                subCategory: 't-shirt',
                primaryColor: 'Blue',
                colorHex: '#2563EB',
                pattern: 'solid',
                material: 'cotton',
                seasons: ['summer'],
                occasions: ['casual'],
            });

            const itemId = store.getState().items[0].id;

            // Log wear
            store.getState().logWear([itemId], 'casual');

            const state = store.getState();
            expect(state.wearLogs).toHaveLength(1);
            expect(state.wearLogs[0].itemIds).toContain(itemId);
            expect(state.items[0].wearCount).toBe(1);
            expect(state.items[0].lastWornAt).toBeDefined();
        });

        it('should calculate streak correctly', async () => {
            const store = getStore();

            // Add an item
            await store.getState().addItem({
                userId: 'user1',
                imageUrl: 'https://example.com/shirt.jpg',
                category: 'top',
                subCategory: 't-shirt',
                primaryColor: 'Blue',
                colorHex: '#2563EB',
                pattern: 'solid',
                material: 'cotton',
                seasons: ['summer'],
                occasions: ['casual'],
            });

            const itemId = store.getState().items[0].id;
            store.getState().logWear([itemId], 'casual');

            // After logging today, streak should be 1
            expect(store.getState().streak).toBe(1);
        });
    });

    describe('getClosetUtilization', () => {
        it('should return 0 for empty closet', () => {
            const store = getStore();
            expect(store.getState().getClosetUtilization()).toBe(0);
        });

        it('should calculate utilization correctly', async () => {
            const store = getStore();

            // Add 2 items
            await store.getState().addItem({
                userId: 'user1',
                imageUrl: 'https://example.com/a.jpg',
                category: 'top',
                subCategory: 'shirt',
                primaryColor: 'White',
                colorHex: '#FFFFFF',
                pattern: 'solid',
                material: 'cotton',
                seasons: ['summer'],
                occasions: ['casual'],
            });
            await store.getState().addItem({
                userId: 'user1',
                imageUrl: 'https://example.com/b.jpg',
                category: 'bottom',
                subCategory: 'pants',
                primaryColor: 'Black',
                colorHex: '#000000',
                pattern: 'solid',
                material: 'cotton',
                seasons: ['summer'],
                occasions: ['casual'],
            });

            // Wear only the first item
            const firstItemId = store.getState().items[0].id;
            store.getState().logWear([firstItemId], 'casual');

            // Utilization = 1/2 = 50%
            expect(store.getState().getClosetUtilization()).toBe(50);
        });
    });

    describe('toggleFavorite', () => {
        it('should toggle item favorite status', async () => {
            const store = getStore();
            await store.getState().addItem({
                userId: 'user1',
                imageUrl: 'https://example.com/a.jpg',
                category: 'shoes',
                subCategory: 'sneakers',
                primaryColor: 'White',
                colorHex: '#FFFFFF',
                pattern: 'solid',
                material: 'leather',
                seasons: ['spring', 'summer'],
                occasions: ['casual', 'sport'],
            });

            const itemId = store.getState().items[0].id;
            expect(store.getState().items[0].isFavorite).toBe(false);

            store.getState().toggleFavorite(itemId);
            expect(store.getState().items[0].isFavorite).toBe(true);

            store.getState().toggleFavorite(itemId);
            expect(store.getState().items[0].isFavorite).toBe(false);
        });
    });

    describe('removeItem', () => {
        it('should remove an item and queue sync action', async () => {
            const store = getStore();
            await store.getState().addItem({
                userId: 'user1',
                imageUrl: 'https://example.com/a.jpg',
                category: 'top',
                subCategory: 'shirt',
                primaryColor: 'White',
                colorHex: '#FFFFFF',
                pattern: 'solid',
                material: 'cotton',
                seasons: ['summer'],
                occasions: ['casual'],
            });

            expect(store.getState().items).toHaveLength(1);
            const itemId = store.getState().items[0].id;

            await store.getState().removeItem(itemId);
            expect(store.getState().items).toHaveLength(0);
        });
    });
});
