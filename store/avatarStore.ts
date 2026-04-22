import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { BodyTypeId } from '../features/try-on/utils/mannequin3D';

interface AvatarState {
  heightCm: string;
  weightKg: string;
  bodyType: BodyTypeId;
  gender: 'male' | 'female';
  setMeasurements: (height: string, weight: string, bodyType: BodyTypeId) => void;
  setGender: (gender: 'male' | 'female') => void;
}

const useAvatarStore = create<AvatarState>()(
  persist(
    (set) => ({
      heightCm: '175',
      weightKg: '70',
      bodyType: 'average',
      gender: 'male',
      setMeasurements: (heightCm, weightKg, bodyType) =>
        set({ heightCm, weightKg, bodyType }),
      setGender: (gender) => set({ gender }),
    }),
    {
      name: 'avatar-storage',
      storage: createJSONStorage(() => AsyncStorage),
    }
  )
);

export default useAvatarStore;
