import { create } from 'zustand';

interface AppContextState {
    weather: any;
    todaysOutfit: any;
    setContext: (weather: any, todaysOutfit: any) => void;
}

export const useAppContextStore = create<AppContextState>((set) => ({
    weather: null,
    todaysOutfit: null,
    setContext: (weather, todaysOutfit) => set({ weather, todaysOutfit }),
}));

export default useAppContextStore;
