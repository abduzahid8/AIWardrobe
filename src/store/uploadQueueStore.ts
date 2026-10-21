import { authEvents } from '../../store/authEvents';
import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { createDebouncedStorage } from '../lib/debouncedStorage';

const STORAGE_KEY = 'wardrobe-upload-queue';
const queueStorage = createDebouncedStorage(AsyncStorage, 0);
export const clearUploadQueuePersistence = () => queueStorage.removeItem(STORAGE_KEY);
export const flushUploadQueuePersistence = () => queueStorage.flush();

export interface PendingUpload {
    id: string;
    ownerId: string;
    uri: string;
    type: 'image' | 'video';
    addedAt: number;
    status: 'pending' | 'processing' | 'failed';
    retryCount: number;
    nextRetryAt?: number;
    error?: string;
}

interface UploadQueueState {
    pendingUploads: PendingUpload[];
    addUpload: (uri: string, type: 'image' | 'video') => boolean;
    removeUpload: (id: string) => void;
    retryUpload: (id: string) => boolean;
    updateStatus: (id: string, status: PendingUpload['status'], error?: string) => void;
    incrementRetry: (id: string) => void;
    getPendingCount: () => number;
}

export const useUploadQueueStore = create<UploadQueueState>()(
    persist(
        (set, get) => ({
            pendingUploads: [],

            addUpload: (uri, type) => {
                const ownerId = authEvents.getUserId();
                if (!ownerId) return false;
                const newUpload: PendingUpload = {
                    id: Date.now().toString() + Math.random().toString(36).substring(7),
                    ownerId,
                    uri,
                    type,
                    addedAt: Date.now(),
                    status: 'pending',
                    retryCount: 0,
                };
                set((state) => ({
                    pendingUploads: [...state.pendingUploads, newUpload],
                }));
                return true;
            },

            retryUpload: (id) => {
                const owner = authEvents.getUserId();
                const entry = get().pendingUploads.find(item => item.id === id && item.ownerId === owner);
                if (!owner || !entry || entry.status === 'processing') return false;
                set(state => ({ pendingUploads: state.pendingUploads.map(item => item === entry
                    ? { ...item, status: 'pending', retryCount: 0, nextRetryAt: undefined, error: undefined } : item) }));
                return true;
            },

            removeUpload: (id) => {
                set((state) => ({
                    pendingUploads: state.pendingUploads.filter((item) => item.id !== id || item.ownerId !== authEvents.getUserId()),
                }));
            },

            updateStatus: (id, status, error) => {
                set((state) => ({
                    pendingUploads: state.pendingUploads.map((item) =>
                        item.id === id && item.ownerId === authEvents.getUserId() ? { ...item, status, error } : item
                    ),
                }));
            },

            incrementRetry: (id) => {
                set((state) => ({
                    pendingUploads: state.pendingUploads.map((item) =>
                        item.id === id && item.ownerId === authEvents.getUserId()
                            ? { ...item, retryCount: (item.retryCount ?? 0) + 1,
                                nextRetryAt: Date.now() + Math.min(300_000, 30_000 * 4 ** Math.min(item.retryCount ?? 0, 2)) }
                            : item
                    ),
                }));
            },

            getPendingCount: () => {
                return get().pendingUploads.filter(u => u.ownerId === authEvents.getUserId() && (u.status === 'pending' || u.status === 'failed')).length;
            }
        }),
        {
            name: STORAGE_KEY,
            storage: createJSONStorage(() => queueStorage),
        }
    )
);

/** Invalidate pending hydration before clearing memory on an account boundary. */
export function resetUploadQueueState(): void {
    void clearUploadQueuePersistence().catch(() => undefined);
    useUploadQueueStore.setState({ pendingUploads: [] });
}

export default useUploadQueueStore;
