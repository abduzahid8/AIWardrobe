/**
 * Outfit Generator Slice — ephemeral state for the anchor-item outfit generation flow.
 *
 * This slice is intentionally excluded from `partialize` in wardrobeStore so it is
 * never persisted to AsyncStorage. Generation state is ephemeral and must not survive
 * app restarts (Requirement 7.5).
 *
 * Valid status transitions:
 *   idle → selecting_items → composing_image → complete | error | fallback_active
 *
 * Requirements: 6.1, 7.5
 */

import type { StateCreator } from 'zustand';
import type { GenerationStatus, ClosetClothingItem, GeneratedOutfit } from '../../features/outfit-generator/types';
import type { WardrobeState } from '../wardrobeStore';

// ============================================
// VALID STATUS TRANSITIONS
// ============================================

/**
 * Defines which statuses can be transitioned to from a given current status.
 * Enforces the pipeline flow: idle → selecting_items → composing_image → complete | error | fallback_active
 */
const VALID_TRANSITIONS: Record<GenerationStatus, GenerationStatus[]> = {
  idle: ['selecting_items'],
  selecting_items: ['composing_image', 'fallback_active', 'error', 'idle'],
  composing_image: ['complete', 'error', 'idle'],
  fallback_active: ['complete', 'error', 'idle'],
  complete: ['idle'],
  error: ['idle'],
};

// ============================================
// STATUS MESSAGES
// ============================================

const DEFAULT_STATUS_MESSAGES: Record<GenerationStatus, string> = {
  idle: '',
  selecting_items: 'Building your outfit…',
  composing_image: 'Composing the image…',
  fallback_active: 'Showing a simplified result',
  complete: '',
  error: '',
};

// ============================================
// SLICE INTERFACE
// ============================================

export interface OutfitGeneratorSlice {
  // Anchor item
  anchorItem: ClosetClothingItem | null;

  // Generation state
  status: GenerationStatus;
  statusMessage: string;
  isFallbackActive: boolean;

  // Result
  currentOutfit: GeneratedOutfit | null;
  generatedImageUrl: string | null;

  // Error
  errorMessage: string | null;

  // Actions
  setAnchorItem: (item: ClosetClothingItem | null) => void;
  setStatus: (status: GenerationStatus, message?: string) => void;
  setCurrentOutfit: (outfit: GeneratedOutfit | null) => void;
  setGeneratedImageUrl: (url: string | null) => void;
  setFallbackActive: (active: boolean) => void;
  setError: (message: string) => void;
  cancelGeneration: () => void;
  reset: () => void;
}

// ============================================
// INITIAL STATE
// ============================================

const initialState = {
  anchorItem: null,
  status: 'idle' as GenerationStatus,
  statusMessage: '',
  isFallbackActive: false,
  currentOutfit: null,
  generatedImageUrl: null,
  errorMessage: null,
};

// ============================================
// SLICE CREATOR
// ============================================

export const createOutfitGeneratorSlice: StateCreator<
  WardrobeState,
  [],
  [],
  OutfitGeneratorSlice
> = (set, get) => ({
  ...initialState,

  setAnchorItem: (item) => {
    set({ anchorItem: item });
  },

  /**
   * Transitions to a new status, enforcing valid transition rules.
   * If the transition is invalid, the call is a no-op (Requirement 6.1 —
   * prevents duplicate requests from resetting in-progress state).
   *
   * @param status - The target status to transition to.
   * @param message - Optional override for the status message. Falls back to the default message for the status.
   */
  setStatus: (status, message) => {
    const currentStatus = get().status;
    const allowed = VALID_TRANSITIONS[currentStatus];

    if (!allowed.includes(status)) {
      // Invalid transition — silently ignore to prevent state corruption.
      // This is the guard that prevents duplicate requests from overwriting
      // an in-progress status (Requirement 6.1).
      return;
    }

    set({
      status,
      statusMessage: message ?? DEFAULT_STATUS_MESSAGES[status],
    });
  },

  setCurrentOutfit: (outfit) => {
    set({ currentOutfit: outfit });
  },

  setGeneratedImageUrl: (url) => {
    set({ generatedImageUrl: url });
  },

  setFallbackActive: (active) => {
    set({ isFallbackActive: active });
  },

  /**
   * Sets the error state. Transitions status to 'error' regardless of current
   * status to ensure errors are always surfaced. Uses direct set to bypass
   * transition guard since errors can occur from any state.
   */
  setError: (message) => {
    set({
      status: 'error',
      statusMessage: '',
      errorMessage: message,
    });
  },

  /**
   * Cancels an in-progress generation and resets all state.
   * The AbortController abort is handled by the hook; this action
   * only resets the slice state.
   */
  cancelGeneration: () => {
    set({ ...initialState });
  },

  /**
   * Resets all generation state to initial values.
   * Explicitly nullifies `generatedImageUrl` and `currentOutfit` to release
   * in-memory image data and prevent memory leaks (Requirement 7.5).
   */
  reset: () => {
    set({ ...initialState });
  },
});
