/**
 * store/tryOnLooksStore.ts
 * Persisted store for try-on looks saved from AITryOnScreen.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';

export interface TryOnLook {
    id: string;
    resultUrl: string;          // AI-generated result image URL / data URI
    garmentName: string;
    garmentBrand?: string;
    garmentType: string;        // upper_body | lower_body | shoes | etc.
    garmentImageUrl?: string;   // thumbnail of the garment worn
    savedAt: string;            // ISO date string
}

interface TryOnLooksState {
    looks: TryOnLook[];
    saveLook: (look: Omit<TryOnLook, 'id' | 'savedAt'>) => TryOnLook;
    removeLook: (id: string) => void;
    clearAll: () => void;
}

const generateId = () =>
    'tryon_' + Math.random().toString(36).substring(2, 10) + '_' + Date.now();

const useTryOnLooksStore = create<TryOnLooksState>()(
    persist(
        (set, get) => ({
            looks: [],

            saveLook: (input) => {
                const look: TryOnLook = {
                    ...input,
                    id: generateId(),
                    savedAt: new Date().toISOString(),
                };
                set((state) => ({ looks: [look, ...state.looks] }));
                return look;
            },

            removeLook: (id) => {
                set((state) => ({ looks: state.looks.filter((l) => l.id !== id) }));
            },

            clearAll: () => set({ looks: [] }),
        }),
        {
            name: 'try-on-looks-storage',
            storage: createJSONStorage(() => AsyncStorage),
        }
    )
);

export default useTryOnLooksStore;
