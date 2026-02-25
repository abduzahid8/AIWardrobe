/**
 * Offline Queue — Deferred operation queue for offline support
 *
 * Stores pending server operations in AsyncStorage.
 * When connectivity returns, replays them in order.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

const QUEUE_KEY = 'offline_queue';

export interface QueuedAction {
    id: string;
    type: string;
    endpoint: string;
    method: 'POST' | 'PUT' | 'DELETE';
    payload: Record<string, unknown>;
    createdAt: string;
    retryCount: number;
}

/**
 * Add an action to the offline queue.
 */
export async function enqueue(action: Omit<QueuedAction, 'id' | 'createdAt' | 'retryCount'>): Promise<void> {
    try {
        const queue = await getQueue();

        const newAction: QueuedAction = {
            ...action,
            id: `q_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
            createdAt: new Date().toISOString(),
            retryCount: 0,
        };

        queue.push(newAction);
        await AsyncStorage.setItem(QUEUE_KEY, JSON.stringify(queue));
    } catch (error) {
        console.error('[OfflineQueue] Failed to enqueue:', error);
    }
}

/**
 * Get all pending actions.
 */
export async function getQueue(): Promise<QueuedAction[]> {
    try {
        const raw = await AsyncStorage.getItem(QUEUE_KEY);
        return raw ? JSON.parse(raw) : [];
    } catch {
        return [];
    }
}

/**
 * Get count of pending actions.
 */
export async function getPendingCount(): Promise<number> {
    const queue = await getQueue();
    return queue.length;
}

/**
 * Flush the queue — replay all pending actions to the server.
 * Returns the number of successfully processed actions.
 */
export async function flush(
    baseUrl: string,
    headers: Record<string, string> = {}
): Promise<{ processed: number; failed: number }> {
    const queue = await getQueue();
    if (queue.length === 0) return { processed: 0, failed: 0 };

    let processed = 0;
    let failed = 0;
    const remaining: QueuedAction[] = [];

    for (const action of queue) {
        try {
            const response = await fetch(`${baseUrl}${action.endpoint}`, {
                method: action.method,
                headers: {
                    'Content-Type': 'application/json',
                    ...headers,
                },
                body: action.method !== 'DELETE' ? JSON.stringify(action.payload) : undefined,
            });

            if (response.ok) {
                processed++;
            } else if (response.status >= 400 && response.status < 500) {
                // Client error — don't retry
                failed++;
                console.warn(`[OfflineQueue] Action ${action.id} failed with ${response.status}, dropping`);
            } else {
                // Server error — retry later
                remaining.push({ ...action, retryCount: action.retryCount + 1 });
                failed++;
            }
        } catch {
            // Network error — keep in queue
            if (action.retryCount < 5) {
                remaining.push({ ...action, retryCount: action.retryCount + 1 });
            } else {
                // Too many retries, drop
                console.warn(`[OfflineQueue] Action ${action.id} exceeded max retries, dropping`);
            }
            failed++;
        }
    }

    // Save remaining actions
    await AsyncStorage.setItem(QUEUE_KEY, JSON.stringify(remaining));

    return { processed, failed };
}

/**
 * Clear all pending actions.
 */
export async function clearQueue(): Promise<void> {
    await AsyncStorage.removeItem(QUEUE_KEY);
}

export default { enqueue, getQueue, getPendingCount, flush, clearQueue };
