/**
 * Body profile store (Zustand + AsyncStorage persistence).
 *
 * Holds the user's calibrated digital body. One profile is `active` at a time
 * — that one is what the mannequin renders and the fit engine consumes.
 *
 * Coexists with the legacy `avatarStore` (which only stores height/weight/
 * bodyType/gender for the basic mannequin). New screens should read from
 * this store; old screens keep working via a thin compatibility shim
 * (`getLegacyAvatarSnapshot`).
 *
 * Persistence key: `body-profile-storage-v1`. Registered in
 * src/lib/persistence.ts so account deletion + logout wipe it.
 */

import { create } from 'zustand';
import { persist, createJSONStorage } from 'zustand/middleware';
import AsyncStorage from '@react-native-async-storage/async-storage';
import {
  BodyProfile,
  BodyProfileStatus,
  BodyTypeId,
  GenderOption,
  createEmptyBodyProfile,
} from '../src/types/bodyProfile';
import { GarmentPhysicalProfile } from '../src/types/garment';

const STORAGE_KEY = 'body-profile-storage-v1';
const FIT_ENGINE_VERSION = 'fit-engine/v1';

interface BodyProfileState {
  profiles: Record<string, BodyProfile>;
  activeProfileId: string | null;
  // Cache of garment physical profiles (Month 1 uses seed; Month 5 ingests size charts).
  garmentProfiles: Record<string, GarmentPhysicalProfile>;

  // ── Profile CRUD ─────────────────────────────────────────────────────────
  /** Create a new draft profile. If no profile exists yet, marks it active. */
  createProfile: (userId: string) => BodyProfile;
  /** Mark a profile active (deselects others). */
  setActive: (id: string) => void;
  /** Patch any subset of a profile. Bumps `version` and `updatedAt`. */
  updateProfile: (id: string, patch: Partial<BodyProfile>) => void;
  /** Convenience setters for the most common fields. */
  setHeight: (id: string, valueCm: number, source?: BodyProfile['height']['source']) => void;
  setWeight: (id: string, weightKg: number) => void;
  setBodyType: (id: string, bodyType: BodyTypeId) => void;
  setGender: (id: string, gender: GenderOption) => void;
  setMeasurement: (
    id: string,
    zone: keyof BodyProfile['measurements'],
    valueCm: number,
  ) => void;
  setStatus: (id: string, status: BodyProfileStatus) => void;
  deleteProfile: (id: string) => void;

  // ── Garment physical profile cache ───────────────────────────────────────
  setGarmentProfile: (profile: GarmentPhysicalProfile) => void;
  getGarmentProfile: (garmentId: string, sizeLabel: string) => GarmentPhysicalProfile | undefined;

  // ── Reset (called by account deletion / logout) ──────────────────────────
  reset: () => void;
}

function bumpVersion(profile: BodyProfile): BodyProfile {
  return {
    ...profile,
    version: (profile.version || 1) + 1,
    updatedAt: new Date().toISOString(),
  };
}

export const useBodyProfileStore = create<BodyProfileState>()(
  persist(
    (set, get) => ({
      profiles: {},
      activeProfileId: null,
      garmentProfiles: {},

      createProfile: (userId) => {
        const profile = createEmptyBodyProfile(userId);
        set((state) => ({
          profiles: { ...state.profiles, [profile.id]: profile },
          activeProfileId: state.activeProfileId ?? profile.id,
        }));
        return profile;
      },

      setActive: (id) => {
        const { profiles } = get();
        if (!profiles[id]) return;
        set({ activeProfileId: id });
      },

      updateProfile: (id, patch) => {
        set((state) => {
          const existing = state.profiles[id];
          if (!existing) return state;
          return {
            profiles: { ...state.profiles, [id]: bumpVersion({ ...existing, ...patch }) },
          };
        });
      },

      setHeight: (id, valueCm, source = 'manual') => {
        set((state) => {
          const existing = state.profiles[id];
          if (!existing) return state;
          return {
            profiles: {
              ...state.profiles,
              [id]: bumpVersion({
                ...existing,
                height: {
                  valueCm,
                  confidence: source === 'apple_measure' || source === 'arkit_height' ? 'high' : 'medium',
                  source,
                  updatedAt: new Date().toISOString(),
                },
              }),
            },
          };
        });
      },

      setWeight: (id, weightKg) => {
        get().updateProfile(id, { weightKg });
      },

      setBodyType: (id, bodyType) => {
        get().updateProfile(id, { bodyType });
      },

      setGender: (id, gender) => {
        get().updateProfile(id, { gender });
      },

      setMeasurement: (id, zone, valueCm) => {
        set((state) => {
          const existing = state.profiles[id];
          if (!existing) return state;
          return {
            profiles: {
              ...state.profiles,
              [id]: bumpVersion({
                ...existing,
                measurements: {
                  ...existing.measurements,
                  [zone]: {
                    valueCm,
                    confidence: 'medium',
                    source: 'manual',
                    updatedAt: new Date().toISOString(),
                  },
                },
              }),
            },
          };
        });
      },

      setStatus: (id, status) => {
        get().updateProfile(id, { status });
      },

      deleteProfile: (id) => {
        set((state) => {
          const { [id]: _removed, ...rest } = state.profiles;
          const nextActive = state.activeProfileId === id
            ? Object.keys(rest)[0] ?? null
            : state.activeProfileId;
          return { profiles: rest, activeProfileId: nextActive };
        });
      },

      setGarmentProfile: (profile) => {
        set((state) => ({
          garmentProfiles: { ...state.garmentProfiles, [profile.id]: profile },
        }));
      },

      getGarmentProfile: (garmentId, sizeLabel) => {
        const { garmentProfiles } = get();
        return Object.values(garmentProfiles).find(
          (p) => p.garmentId === garmentId && p.sizeLabel === sizeLabel,
        );
      },

      reset: () => {
        set({ profiles: {}, activeProfileId: null, garmentProfiles: {} });
      },
    }),
    {
      name: STORAGE_KEY,
      storage: createJSONStorage(() => AsyncStorage),
      version: 1,
    },
  ),
);

// ─── Selectors ────────────────────────────────────────────────────────────────

/** Return the active body profile (or undefined if none / still loading). */
export function selectActiveProfile(state: BodyProfileState): BodyProfile | undefined {
  if (!state.activeProfileId) return undefined;
  return state.profiles[state.activeProfileId];
}

/**
 * Compatibility shim for the legacy `avatarStore` shape. Used by old
 * CreateAvatarScreen + AITryOnScreen until they migrate. Prefer reading
 * from `selectActiveProfile` directly in new code.
 */
export function getLegacyAvatarSnapshot() {
  const state = useBodyProfileStore.getState();
  const profile = selectActiveProfile(state);
  return {
    heightCm: profile ? String(profile.height.valueCm) : '175',
    weightKg: profile && profile.weightKg != null ? String(profile.weightKg) : '70',
    bodyType: (profile?.bodyType ?? 'average') as BodyTypeId,
    gender: (profile?.gender === 'other' || profile?.gender === 'prefer_not_to_say'
      ? 'male'
      : (profile?.gender ?? 'male')) as 'male' | 'female',
  };
}

export { FIT_ENGINE_VERSION };

export default useBodyProfileStore;
