/**
 * wardrobeStore — outfit dedup + dislike cap regression test.
 *
 * Every Home swipe-right calls addOutfit + saveOutfit. Without dedup the
 * `outfits` array (persisted in full on every write, re-scanned by Profile
 * memos on every swipe) grew by one object per swipe forever — the
 * slow-burn freeze. dislikedOutfitKeys had the same unbounded growth.
 */

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

describe('wardrobeStore outfits bounded growth', () => {
    let useWardrobeStore: typeof import('../../store/wardrobeStore').default;

    beforeEach(() => {
        jest.resetModules();
        useWardrobeStore = require('../../store/wardrobeStore').default;
        useWardrobeStore.setState({ items: [], outfits: [], dislikedOutfitKeys: [] });
    });

    const swipeCombo = (suffix = '') => ({
        userId: 'u1',
        itemIds: [`top-1${suffix}`, `bottom-1${suffix}`, `shoes-1${suffix}`],
        occasion: 'casual' as const,
        generatedBy: 'ai' as const,
    });

    it('does not duplicate an identical liked combo', () => {
        const first = useWardrobeStore.getState().addOutfit(swipeCombo());
        const second = useWardrobeStore.getState().addOutfit(swipeCombo());
        expect(second).toBe(first);
        expect(useWardrobeStore.getState().outfits).toHaveLength(1);
    });

    it('treats reordered itemIds as the same combo', () => {
        const a = swipeCombo();
        const b = { ...swipeCombo(), itemIds: [...swipeCombo().itemIds].reverse() };
        const first = useWardrobeStore.getState().addOutfit(a);
        const second = useWardrobeStore.getState().addOutfit(b);
        expect(second).toBe(first);
        expect(useWardrobeStore.getState().outfits).toHaveLength(1);
    });

    it('keeps distinct combos as separate outfits', () => {
        useWardrobeStore.getState().addOutfit(swipeCombo('-a'));
        useWardrobeStore.getState().addOutfit(swipeCombo('-b'));
        expect(useWardrobeStore.getState().outfits).toHaveLength(2);
    });

    it('caps dislikedOutfitKeys at 500', () => {
        const { dislikeOutfit } = useWardrobeStore.getState();
        for (let i = 0; i < 600; i++) {
            useWardrobeStore.getState().dislikeOutfit([`item-${i}`]);
        }
        const keys = useWardrobeStore.getState().dislikedOutfitKeys;
        expect(keys).toHaveLength(500);
        // Keeps the most recent, drops the oldest
        expect(keys[keys.length - 1]).toContain('item-599');
        expect(dislikeOutfit).toBeDefined();
    });
});
