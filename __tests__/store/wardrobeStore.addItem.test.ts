/**
 * wardrobeStore — addItem critical flow test
 *
 * Tests the full optimistic add → server insert → fallback pipeline.
 */

import { normalizeCategory } from '../../src/utils/categoryMapper';

// Must mock before importing the store
jest.mock('../../lib/supabase', () => ({
    supabase: {
        auth: {
            getSession: jest.fn().mockResolvedValue({ data: { session: null } }),
        },
        from: jest.fn(() => ({
            insert: jest.fn(() => ({
                select: jest.fn(() => ({
                    single: jest.fn().mockResolvedValue({ data: null, error: { message: 'offline' } }),
                })),
            })),
        })),
        channel: jest.fn(() => ({ on: jest.fn().mockReturnThis(), subscribe: jest.fn() })),
        removeChannel: jest.fn(),
    },
}));

jest.mock('../../src/lib/api', () => ({
    wardrobeApi: {
        list: jest.fn().mockResolvedValue([]),
        remove: jest.fn().mockResolvedValue(undefined),
        toggleFavorite: jest.fn().mockResolvedValue(undefined),
    },
    wearLogApi: {
        list: jest.fn().mockResolvedValue([]),
    },
}));

jest.mock('../../store/wardrobeSyncService', () => ({
    fetchItemsFromServer: jest.fn().mockResolvedValue([]),
    fetchWearLogsFromServer: jest.fn().mockResolvedValue([]),
    processPendingActions: jest.fn().mockResolvedValue({ processedIds: [], updatedItems: [] }),
}));

jest.mock('@react-native-async-storage/async-storage', () =>
    require('@react-native-async-storage/async-storage/jest/async-storage-mock')
);

describe('wardrobeStore.addItem', () => {
    let useWardrobeStore: typeof import('../../store/wardrobeStore').default;

    beforeEach(() => {
        jest.resetModules();
        useWardrobeStore = require('../../store/wardrobeStore').default;
        useWardrobeStore.setState({ items: [], pendingActions: [] });
    });

    it('adds item optimistically with a UUID id', async () => {
        await useWardrobeStore.getState().addItem({
            userId: '',
            imageUrl: 'https://example.com/image.jpg',
            category: 'top',
            subCategory: 't-shirt',
            primaryColor: 'blue',
            colorHex: '#0000FF',
            pattern: 'solid',
            material: 'cotton',
            seasons: ['summer'],
            occasions: ['casual'],
        });

        const items = useWardrobeStore.getState().items;
        expect(items.length).toBe(1);
        expect(items[0].category).toBe('top');
        expect(items[0].id).toMatch(/^[0-9a-f-]{36}$/i);
    });

    it('normalizes PascalCase categories to lowercase', async () => {
        await useWardrobeStore.getState().addItem({
            userId: '',
            imageUrl: 'https://example.com/img.jpg',
            category: 'Tops' as any,
            subCategory: 'polo',
            primaryColor: 'white',
            colorHex: '#FFFFFF',
            pattern: 'solid',
            material: 'cotton',
            seasons: [],
            occasions: [],
        });

        expect(useWardrobeStore.getState().items[0].category).toBe('top');
    });

    it('queues a pending action when no session exists', async () => {
        await useWardrobeStore.getState().addItem({
            userId: '',
            imageUrl: 'https://example.com/img.jpg',
            category: 'shoes',
            subCategory: 'sneakers',
            primaryColor: 'black',
            colorHex: '#000',
            pattern: 'solid',
            material: 'leather',
            seasons: [],
            occasions: [],
        });

        const { pendingActions } = useWardrobeStore.getState();
        expect(pendingActions.length).toBe(1);
        expect(pendingActions[0].type).toBe('add_item');
    });
});

describe('normalizeCategory', () => {
    it.each([
        ['top', 'top'], ['Tops', 'top'], ['tops', 'top'],
        ['bottom', 'bottom'], ['Bottoms', 'bottom'],
        ['shoe', 'shoes'], ['Shoes', 'shoes'],
        ['outerwear', 'outerwear'], ['Outerwear', 'outerwear'],
        ['dress', 'dress'], ['Dresses', 'dress'],
        ['accessory', 'accessory'], ['Accessories', 'accessory'],
        ['garbage', 'other'],
    ])('normalizeCategory("%s") → "%s"', (input, expected) => {
        expect(normalizeCategory(input)).toBe(expected);
    });
});
