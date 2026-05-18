/**
 * wardrobeSyncService — processPendingActions batch upsert tests
 *
 * Verifies that add_item actions are batched into a single upsert call
 * instead of N sequential round-trips (Defect 2.5 fix).
 */

import type { PendingAction } from '../../store/wardrobeSyncService';
import type { ClothingItem } from '../../src/types/domain';

// ── Supabase mock ──────────────────────────────────────────────────────────
const mockUpsert = jest.fn();
const mockDelete = jest.fn();
const mockSelect = jest.fn();
const mockIn = jest.fn();
const mockEq = jest.fn();
const mockSingle = jest.fn();

// We need to track which table is being queried
let currentTable = '';

jest.mock('../../lib/supabase', () => ({
    supabase: {
        auth: {
            getSession: jest.fn().mockResolvedValue({
                data: {
                    session: {
                        user: { id: 'user-123' },
                    },
                },
            }),
        },
        from: jest.fn((table: string) => {
            currentTable = table;
            return {
                select: mockSelect,
                upsert: mockUpsert,
                delete: mockDelete,
            };
        }),
    },
}));

// ── Helpers ────────────────────────────────────────────────────────────────
function makeItem(id: string, updatedAt = '2024-01-01T00:00:00Z'): ClothingItem {
    return {
        id,
        userId: 'user-123',
        imageUrl: `https://example.com/${id}.jpg`,
        thumbnailUrl: undefined,
        category: 'top',
        subCategory: 't-shirt',
        primaryColor: 'blue',
        colorHex: '#0000FF',
        pattern: 'solid',
        material: 'cotton',
        brand: undefined,
        name: undefined,
        seasons: ['summer'],
        occasions: ['casual'],
        wearCount: 0,
        lastWornAt: null,
        isFavorite: false,
        createdAt: '2024-01-01T00:00:00Z',
        updatedAt,
        detectionConfidence: undefined,
    };
}

function makeAddAction(item: ClothingItem): PendingAction {
    return {
        id: `action-${item.id}`,
        type: 'add_item',
        payload: item as unknown as Record<string, unknown>,
        createdAt: '2024-01-01T00:00:00Z',
    };
}

// ── Tests ──────────────────────────────────────────────────────────────────
describe('processPendingActions — batch add_item upsert', () => {
    let processPendingActions: typeof import('../../store/wardrobeSyncService').processPendingActions;

    beforeEach(() => {
        jest.clearAllMocks();
        jest.resetModules();

        // Re-import after clearing modules so mock state is fresh
        processPendingActions =
            require('../../store/wardrobeSyncService').processPendingActions;
    });

    it('returns empty results when there are no pending actions', async () => {
        const localItems = [makeItem('item-1')];
        const result = await processPendingActions([], localItems);

        expect(result.processedIds).toEqual([]);
        expect(result.updatedItems).toEqual(localItems);
    });

    it('issues a single batch upsert for multiple add_item actions', async () => {
        const item1 = makeItem('item-1');
        const item2 = makeItem('item-2');
        const item3 = makeItem('item-3');

        const actions = [
            makeAddAction(item1),
            makeAddAction(item2),
            makeAddAction(item3),
        ];

        // No server rows exist yet (new items) — conflict check returns empty
        mockSelect.mockReturnValue({
            in: jest.fn().mockResolvedValue({ data: [], error: null }),
        });

        // Batch upsert succeeds
        mockUpsert.mockResolvedValue({ data: null, error: null });

        const result = await processPendingActions(actions, [item1, item2, item3]);

        // All three actions should be marked processed
        expect(result.processedIds).toEqual([
            'action-item-1',
            'action-item-2',
            'action-item-3',
        ]);

        // upsert should have been called exactly ONCE (batch), not 3 times
        expect(mockUpsert).toHaveBeenCalledTimes(1);

        // The single upsert call should contain all 3 rows
        const upsertArg = mockUpsert.mock.calls[0][0];
        expect(Array.isArray(upsertArg)).toBe(true);
        expect(upsertArg).toHaveLength(3);
        expect(upsertArg.map((r: any) => r.id)).toEqual(
            expect.arrayContaining(['item-1', 'item-2', 'item-3'])
        );
    });

    it('does NOT mark add_item actions as processed when batch upsert fails', async () => {
        const item1 = makeItem('item-1');
        const item2 = makeItem('item-2');
        const actions = [makeAddAction(item1), makeAddAction(item2)];

        // No conflicts
        mockSelect.mockReturnValue({
            in: jest.fn().mockResolvedValue({ data: [], error: null }),
        });

        // Batch upsert fails
        mockUpsert.mockResolvedValue({
            data: null,
            error: { message: 'network error' },
        });

        const result = await processPendingActions(actions, [item1, item2]);

        // No actions should be marked processed — they stay in the queue for retry
        expect(result.processedIds).toEqual([]);
        expect(mockUpsert).toHaveBeenCalledTimes(1);
    });

    it('handles server-wins conflict: re-fetches conflicted item individually', async () => {
        const localItem = makeItem('item-conflict', '2024-01-01T00:00:00Z');
        const action = makeAddAction(localItem);

        // Server has a newer version of this item
        mockSelect.mockReturnValue({
            in: jest.fn().mockResolvedValue({
                data: [{ id: 'item-conflict', updated_at: '2024-06-01T00:00:00Z' }],
                error: null,
            }),
        });

        // Individual re-fetch for the conflicted item
        const freshServerItem = {
            id: 'item-conflict',
            user_id: 'user-123',
            image_url: 'https://example.com/server.jpg',
            thumbnail_url: null,
            category: 'top',
            sub_category: 't-shirt',
            primary_color: 'red',
            color_hex: '#FF0000',
            pattern: 'solid',
            material: 'cotton',
            brand: null,
            name: null,
            seasons: [],
            occasions: [],
            wear_count: 5,
            last_worn_at: null,
            is_favorite: false,
            created_at: '2024-01-01T00:00:00Z',
            updated_at: '2024-06-01T00:00:00Z',
            detection_confidence: null,
        };

        // The second .select() call is the individual re-fetch
        mockSelect
            .mockReturnValueOnce({
                in: jest.fn().mockResolvedValue({
                    data: [{ id: 'item-conflict', updated_at: '2024-06-01T00:00:00Z' }],
                    error: null,
                }),
            })
            .mockReturnValueOnce({
                eq: jest.fn().mockReturnValue({
                    single: jest.fn().mockResolvedValue({
                        data: freshServerItem,
                        error: null,
                    }),
                }),
            });

        const result = await processPendingActions([action], [localItem]);

        // Action is marked processed (server version accepted)
        expect(result.processedIds).toContain('action-item-conflict');

        // Local item should be replaced with server version
        const updatedItem = result.updatedItems.find((i) => i.id === 'item-conflict');
        expect(updatedItem?.primaryColor).toBe('red');
        expect(updatedItem?.wearCount).toBe(5);

        // No batch upsert should have been called (server wins, nothing to write)
        expect(mockUpsert).not.toHaveBeenCalled();
    });

    it('processes remove_item and add_wear_log actions sequentially (unchanged)', async () => {
        const removeAction: PendingAction = {
            id: 'action-remove-1',
            type: 'remove_item',
            payload: { itemId: 'item-to-delete' },
            createdAt: '2024-01-01T00:00:00Z',
        };

        const wearLogAction: PendingAction = {
            id: 'action-log-1',
            type: 'add_wear_log',
            payload: {
                id: 'log-1',
                userId: 'user-123',
                outfitId: undefined,
                itemIds: ['item-1'],
                date: '2024-01-15',
                occasion: 'casual',
                weatherTemp: 20,
                weatherCondition: 'sunny',
                createdAt: '2024-01-15T10:00:00Z',
            },
            createdAt: '2024-01-01T00:00:00Z',
        };

        // delete().eq() chain
        mockDelete.mockReturnValue({
            eq: jest.fn().mockResolvedValue({ data: null, error: null }),
        });

        // upsert for wear_log
        mockUpsert.mockResolvedValue({ data: null, error: null });

        const result = await processPendingActions(
            [removeAction, wearLogAction],
            []
        );

        expect(result.processedIds).toContain('action-remove-1');
        expect(result.processedIds).toContain('action-log-1');
    });

    it('batches only add_item actions; other types are processed separately', async () => {
        const item1 = makeItem('item-1');
        const item2 = makeItem('item-2');

        const addAction1 = makeAddAction(item1);
        const addAction2 = makeAddAction(item2);
        const removeAction: PendingAction = {
            id: 'action-remove-1',
            type: 'remove_item',
            payload: { itemId: 'item-old' },
            createdAt: '2024-01-01T00:00:00Z',
        };

        // No conflicts for add_item actions
        mockSelect.mockReturnValue({
            in: jest.fn().mockResolvedValue({ data: [], error: null }),
        });

        // Batch upsert succeeds
        mockUpsert.mockResolvedValue({ data: null, error: null });

        // delete().eq() chain
        mockDelete.mockReturnValue({
            eq: jest.fn().mockResolvedValue({ data: null, error: null }),
        });

        const result = await processPendingActions(
            [addAction1, addAction2, removeAction],
            [item1, item2]
        );

        expect(result.processedIds).toEqual(
            expect.arrayContaining([
                'action-item-1',
                'action-item-2',
                'action-remove-1',
            ])
        );

        // upsert called once for the batch of 2 add_item rows
        expect(mockUpsert).toHaveBeenCalledTimes(1);
        const upsertArg = mockUpsert.mock.calls[0][0];
        expect(upsertArg).toHaveLength(2);
    });
});
