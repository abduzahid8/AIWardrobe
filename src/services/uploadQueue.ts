/**
 * src/services/uploadQueue.ts — Offline Upload Queue
 *
 * Handles clothing photo uploads that require the AliceVision microservice.
 * When offline, images are queued in AsyncStorage and processed when connectivity returns.
 *
 * Flow:
 *   1. User photographs item → enqueue() called
 *   2. If network unavailable: item saved to AsyncStorage queue
 *   3. On AppState 'active' event: check network → process all pending items in order
 *   4. After MAX_RETRIES failures: item marked 'failed' with ⚠️ badge — never silently discarded
 *
 * Dependencies:
 *   - @react-native-async-storage/async-storage
 *   - @react-native-community/netinfo
 *   - React Native AppState
 *   - src/services/aiProviderService (processUpload)
 */

import AsyncStorage from '@react-native-async-storage/async-storage';
import { AppState, AppStateStatus } from 'react-native';
import NetInfo from '@react-native-community/netinfo';
import type { ProcessUploadResult } from './aiProviderService';

// ============================================
// CONSTANTS
// ============================================

const UPLOAD_QUEUE_KEY = 'upload_queue_v1';
const MAX_RETRIES = 3;

// ============================================
// TYPES
// ============================================

/** Lifecycle status of a queued upload. */
export type UploadStatus = 'pending' | 'processing' | 'failed' | 'succeeded';

/** A single item waiting in the upload queue. */
export interface QueuedUpload {
    /** Temporary client-side identifier used until the server assigns a real ID. */
    tempId: string;
    /** Base64-encoded image data (may be very large — stored only until processed). */
    imageBase64: string;
    /** ISO timestamp when item was queued. */
    timestamp: string;
    /** Number of times this item has been attempted. */
    retryCount: number;
    /** Current lifecycle status. */
    status: UploadStatus;
    /** Optional low-res thumbnail for displaying in the closet while pending. */
    thumbnailBase64?: string;
    /** Human-readable label for UI (e.g. "Blue Jacket"). */
    label?: string;
}

/** Snapshot of the entire queue state broadcast to subscribers. */
export interface UploadQueueState {
    queue: QueuedUpload[];
    isProcessing: boolean;
    pendingCount: number;
    failedCount: number;
}

/** Callback invoked when an upload succeeds. Caller must add item to wardrobeStore. */
export type UploadSuccessHandler = (
    tempId: string,
    result: ProcessUploadResult
) => Promise<void>;

type QueueListener = (state: UploadQueueState) => void;

// ============================================
// UPLOAD QUEUE SERVICE
// ============================================

class UploadQueueService {
    private listeners: Set<QueueListener> = new Set();
    private isProcessing = false;
    private appStateSubscription: ReturnType<typeof AppState.addEventListener> | null = null;
    private onSuccessHandler: UploadSuccessHandler | null = null;

    /**
     * Initialize the service.
     * Registers AppState listener so the queue is flushed each time the app
     * comes to the foreground. Call once at app startup (e.g. in App.tsx).
     */
    init(): void {
        if (this.appStateSubscription) return;

        this.appStateSubscription = AppState.addEventListener(
            'change',
            (nextState: AppStateStatus) => {
                if (nextState === 'active') {
                    void this.processQueue();
                }
            }
        );

        // Attempt to drain any leftover items from a previous session.
        void this.processQueue();
    }

    /**
     * Tear down the AppState listener.
     * Call on logout to prevent memory leaks.
     */
    destroy(): void {
        this.appStateSubscription?.remove();
        this.appStateSubscription = null;
    }

    /**
     * Register the callback that handles a successfully processed upload.
     * The handler is responsible for calling wardrobeStore.addItem().
     */
    setSuccessHandler(handler: UploadSuccessHandler): void {
        this.onSuccessHandler = handler;
    }

    /**
     * Subscribe to queue state changes for real-time UI updates.
     * Returns an unsubscribe function.
     */
    subscribe(listener: QueueListener): () => void {
        this.listeners.add(listener);
        return () => this.listeners.delete(listener);
    }

    /**
     * Add a photo to the upload queue.
     * Immediately attempts processing if network is available.
     * Returns the tempId for tracking the item in the UI.
     */
    async enqueue(imageBase64: string, thumbnailBase64?: string, label?: string): Promise<string> {
        const tempId = `upload_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`;

        const item: QueuedUpload = {
            tempId,
            imageBase64,
            timestamp: new Date().toISOString(),
            retryCount: 0,
            status: 'pending',
            thumbnailBase64,
            label,
        };

        const queue = await this.loadQueue();
        queue.push(item);
        await this.saveQueue(queue);
        this.notify(queue);

        // Attempt immediate processing (no-op if already processing or offline).
        void this.processQueue();

        return tempId;
    }

    /**
     * Return the current queue snapshot.
     */
    async getQueue(): Promise<QueuedUpload[]> {
        return this.loadQueue();
    }

    /**
     * Return count of items currently pending or failed.
     */
    async getPendingCount(): Promise<number> {
        const queue = await this.loadQueue();
        return queue.filter((u) => u.status === 'pending' || u.status === 'failed').length;
    }

    /**
     * Manually trigger a retry for a specific failed item.
     * Resets its retryCount to 0 so it gets MAX_RETRIES fresh attempts.
     */
    async retryItem(tempId: string): Promise<void> {
        const queue = await this.loadQueue();
        const idx = queue.findIndex((u) => u.tempId === tempId);
        if (idx < 0) return;

        queue[idx] = { ...queue[idx], status: 'pending', retryCount: 0 };
        await this.saveQueue(queue);
        void this.processQueue();
    }

    /**
     * Retry ALL failed items at once.
     */
    async retryAll(): Promise<void> {
        const queue = await this.loadQueue();
        const updated = queue.map((u) =>
            u.status === 'failed' ? { ...u, status: 'pending' as UploadStatus, retryCount: 0 } : u
        );
        await this.saveQueue(updated);
        this.notify(updated);
        void this.processQueue();
    }

    /**
     * Remove a specific item from the queue (e.g. user explicitly cancels).
     */
    async removeItem(tempId: string): Promise<void> {
        const queue = await this.loadQueue();
        const updated = queue.filter((u) => u.tempId !== tempId);
        await this.saveQueue(updated);
        this.notify(updated);
    }

    /**
     * Drain the queue — process all pending items sequentially.
     * Guards against concurrent runs. No-ops if offline.
     */
    async processQueue(): Promise<void> {
        if (this.isProcessing) return;

        const netState = await NetInfo.fetch();
        if (!netState.isConnected) return;

        const queue = await this.loadQueue();
        const pending = queue.filter((u) => u.status === 'pending');
        if (pending.length === 0) return;

        this.isProcessing = true;
        this.notify(queue);

        for (const item of pending) {
            // Re-check connectivity before each item
            const net = await NetInfo.fetch();
            if (!net.isConnected) break;

            await this.processItem(item);
        }

        this.isProcessing = false;
        const finalQueue = await this.loadQueue();
        this.notify(finalQueue);
    }

    // ── Private helpers ────────────────────────────────────────────────

    /** Process a single queued item: call AliceVision, handle success/failure. */
    private async processItem(item: QueuedUpload): Promise<void> {
        const queue = await this.loadQueue();
        const idx = queue.findIndex((u) => u.tempId === item.tempId);
        if (idx < 0) return;

        // Mark as processing so UI can show spinner
        queue[idx].status = 'processing';
        await this.saveQueue(queue);
        this.notify(queue);

        try {
            // Lazy import avoids circular dependency at module load time
            const { aiProvider } = await import('./aiProviderService');
            const result = await aiProvider.processUpload(item.imageBase64);

            if (!result.imageUrl) {
                throw new Error('processUpload returned no imageUrl — service may be down');
            }

            // Invoke success handler so caller can add item to wardrobeStore
            if (this.onSuccessHandler) {
                await this.onSuccessHandler(item.tempId, result);
            }

            // Remove from queue on success
            const freshQueue = await this.loadQueue();
            const updatedQueue = freshQueue.filter((u) => u.tempId !== item.tempId);
            await this.saveQueue(updatedQueue);
            this.notify(updatedQueue);
        } catch {
            const freshQueue = await this.loadQueue();
            const freshIdx = freshQueue.findIndex((u) => u.tempId === item.tempId);
            if (freshIdx < 0) return;

            const newRetryCount = (freshQueue[freshIdx].retryCount ?? 0) + 1;
            freshQueue[freshIdx] = {
                ...freshQueue[freshIdx],
                retryCount: newRetryCount,
                // Only mark 'failed' after exhausting all retries — otherwise stays 'pending'
                status: newRetryCount >= MAX_RETRIES ? 'failed' : 'pending',
            };

            await this.saveQueue(freshQueue);
            this.notify(freshQueue);
        }
    }

    /** Load queue from AsyncStorage. Returns empty array on error. */
    private async loadQueue(): Promise<QueuedUpload[]> {
        try {
            const raw = await AsyncStorage.getItem(UPLOAD_QUEUE_KEY);
            return raw ? (JSON.parse(raw) as QueuedUpload[]) : [];
        } catch {
            return [];
        }
    }

    /** Persist queue to AsyncStorage. */
    private async saveQueue(queue: QueuedUpload[]): Promise<void> {
        try {
            // Strip imageBase64 from failed items to avoid bloating AsyncStorage
            const stripped = queue.map((u) =>
                u.status === 'failed'
                    ? { ...u, imageBase64: '' }
                    : u
            );
            await AsyncStorage.setItem(UPLOAD_QUEUE_KEY, JSON.stringify(stripped));
        } catch (err) {
            console.error('[UploadQueue] Failed to persist queue:', err);
        }
    }

    /** Broadcast current state to all subscribers. */
    private notify(queue: QueuedUpload[]): void {
        const state: UploadQueueState = {
            queue,
            isProcessing: this.isProcessing,
            pendingCount: queue.filter((u) => u.status === 'pending' || u.status === 'processing').length,
            failedCount: queue.filter((u) => u.status === 'failed').length,
        };
        this.listeners.forEach((l) => l(state));
    }
}

/** Singleton instance — import this directly throughout the app. */
export const uploadQueue = new UploadQueueService();
export default uploadQueue;
