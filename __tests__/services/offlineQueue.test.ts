/**
 * Tests for offlineQueue service.
 *
 * Covers: enqueue, getQueue, flush (success, client-error drop,
 * server-error retry, max-retry drop), getPendingCount, clearQueue.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { enqueue, getQueue, getPendingCount, flush, clearQueue } from '../../src/services/offlineQueue';

// Reset AsyncStorage between tests
beforeEach(async () => {
    await AsyncStorage.clear();
});

describe('offlineQueue', () => {
    describe('enqueue', () => {
        it('should add an action to the queue', async () => {
            await enqueue({
                type: 'ADD_ITEM',
                endpoint: '/clothing-items',
                method: 'POST',
                payload: { name: 'Test Item' },
            });

            const queue = await getQueue();
            expect(queue).toHaveLength(1);
            expect(queue[0].type).toBe('ADD_ITEM');
            expect(queue[0].endpoint).toBe('/clothing-items');
            expect(queue[0].method).toBe('POST');
            expect(queue[0].retryCount).toBe(0);
            expect(queue[0].id).toMatch(/^q_/);
            expect(queue[0].createdAt).toBeTruthy();
        });

        it('should append to existing queue', async () => {
            await enqueue({ type: 'A', endpoint: '/a', method: 'POST', payload: {} });
            await enqueue({ type: 'B', endpoint: '/b', method: 'PUT', payload: {} });

            const queue = await getQueue();
            expect(queue).toHaveLength(2);
            expect(queue[0].type).toBe('A');
            expect(queue[1].type).toBe('B');
        });
    });

    describe('getQueue', () => {
        it('should return empty array when no queue exists', async () => {
            const queue = await getQueue();
            expect(queue).toEqual([]);
        });
    });

    describe('getPendingCount', () => {
        it('should return 0 for empty queue', async () => {
            expect(await getPendingCount()).toBe(0);
        });

        it('should return correct count', async () => {
            await enqueue({ type: 'A', endpoint: '/a', method: 'POST', payload: {} });
            await enqueue({ type: 'B', endpoint: '/b', method: 'POST', payload: {} });
            expect(await getPendingCount()).toBe(2);
        });
    });

    describe('flush', () => {
        beforeEach(() => {
            (global.fetch as jest.Mock) = jest.fn();
        });

        afterEach(() => {
            jest.restoreAllMocks();
        });

        it('should return { processed: 0, failed: 0 } for empty queue', async () => {
            const result = await flush('https://api.test.com');
            expect(result).toEqual({ processed: 0, failed: 0 });
        });

        it('should process successful actions', async () => {
            await enqueue({ type: 'A', endpoint: '/items', method: 'POST', payload: { x: 1 } });

            (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: true, status: 200 });

            const result = await flush('https://api.test.com');
            expect(result).toEqual({ processed: 1, failed: 0 });

            // Queue should be empty after successful flush
            expect(await getPendingCount()).toBe(0);
        });

        it('should drop actions that get client errors (4xx)', async () => {
            await enqueue({ type: 'A', endpoint: '/items', method: 'POST', payload: {} });

            (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: false, status: 400 });

            const result = await flush('https://api.test.com');
            expect(result).toEqual({ processed: 0, failed: 1 });

            // Queue should be empty — client errors are not retried
            expect(await getPendingCount()).toBe(0);
        });

        it('should keep actions in queue on server errors (5xx)', async () => {
            await enqueue({ type: 'A', endpoint: '/items', method: 'POST', payload: {} });

            (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: false, status: 500 });

            const result = await flush('https://api.test.com');
            expect(result).toEqual({ processed: 0, failed: 1 });

            // Action should remain with incremented retryCount
            const queue = await getQueue();
            expect(queue).toHaveLength(1);
            expect(queue[0].retryCount).toBe(1);
        });

        it('should drop actions that exceed max retries', async () => {
            // Manually set a high retry count
            const queue = [{
                id: 'q_test',
                type: 'A',
                endpoint: '/items',
                method: 'POST' as const,
                payload: {},
                createdAt: new Date().toISOString(),
                retryCount: 5, // Already at max
            }];
            await AsyncStorage.setItem('offline_queue', JSON.stringify(queue));

            (global.fetch as jest.Mock).mockRejectedValueOnce(new Error('Network error'));

            const result = await flush('https://api.test.com');
            expect(result).toEqual({ processed: 0, failed: 1 });

            // Should be dropped (exceeded max retries)
            expect(await getPendingCount()).toBe(0);
        });
    });

    describe('clearQueue', () => {
        it('should remove all queued actions', async () => {
            await enqueue({ type: 'A', endpoint: '/a', method: 'POST', payload: {} });
            await enqueue({ type: 'B', endpoint: '/b', method: 'POST', payload: {} });

            await clearQueue();
            expect(await getPendingCount()).toBe(0);
        });
    });
});
