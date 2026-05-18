import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';

export interface PendingUpload {
    id: string;
    uri: string;
    type: 'image' | 'video';
    addedAt: number;
    status: 'pending' | 'processing' | 'failed';
    retryCount: number;
    error?: string;
}

interface UploadQueueState {
    pendingUploads: PendingUpload[];
    addUpload: (uri: string, type: 'image' | 'video') => void;
    removeUpload: (id: string) => void;
    updateStatus: (id: string, status: PendingUpload['status'], error?: string) => void;
    incrementRetry: (id: string) => void;
    getPendingCount: () => number;
}

export const useUploadQueueStore = create<UploadQueueState>()(
    persist(
        (set, get) => ({
            pendingUploads: [],

            addUpload: (uri, type) => {
                const newUpload: PendingUpload = {
                    id: Date.now().toString() + Math.random().toString(36).substring(7),
                    uri,
                    type,
                    addedAt: Date.now(),
                    status: 'pending',
                    retryCount: 0,
                };
                set((state) => ({
                    pendingUploads: [...state.pendingUploads, newUpload],
                }));
            },

            removeUpload: (id) => {
                set((state) => ({
                    pendingUploads: state.pendingUploads.filter((item) => item.id !== id),
                }));
            },

            updateStatus: (id, status, error) => {
                set((state) => ({
                    pendingUploads: state.pendingUploads.map((item) =>
                        item.id === id ? { ...item, status, error } : item
                    ),
                }));
            },

            incrementRetry: (id) => {
                set((state) => ({
                    pendingUploads: state.pendingUploads.map((item) =>
                        item.id === id
                            ? { ...item, retryCount: (item.retryCount ?? 0) + 1 }
                            : item
                    ),
                }));
            },

            getPendingCount: () => {
                return get().pendingUploads.filter(u => u.status === 'pending' || u.status === 'failed').length;
            }
        }),
        {
            name: 'wardrobe-upload-queue',
            storage: createJSONStorage(() => AsyncStorage),
        }
    )
);

export default useUploadQueueStore;
