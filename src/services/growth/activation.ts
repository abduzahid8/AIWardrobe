/**
 * Phase 6A–6E — Activation, aha moments, progressive onboarding,
 * first-session intent routing, smart closet onboarding.
 *
 * Activation = meaningful value, never account creation or onboarding
 * completion. No arbitrary single score is the only truth.
 */

import type {
    ActivationState,
    AhaMomentEvent,
    AhaMomentKind,
    FirstSessionIntent,
    OnboardingStage,
} from '../../types/growth';

export const ACTIVATION_MILESTONES = [
    'wardrobeStarted',
    'enoughItemsForOutfit',
    'firstOutfitGenerated',
    'firstOutfitSaved',
    'firstWearConfirmed',
    'firstPurchaseCheck',
    'firstPersonalizationSignal',
    'firstAhaMoment',
] as const;

export type ActivationMilestone = (typeof ACTIVATION_MILESTONES)[number];

export function emptyActivationState(): ActivationState {
    return {
        wardrobeStarted: false,
        enoughItemsForOutfit: false,
        firstOutfitGenerated: false,
        firstOutfitSaved: false,
        firstWearConfirmed: false,
        firstPurchaseCheck: false,
        firstPersonalizationSignal: false,
        firstAhaMoment: false,
        lastMilestoneAt: null,
    };
}

/** Records a milestone; idempotent. Returns true when newly achieved. */
export function recordMilestone(state: ActivationState, milestone: ActivationMilestone, nowIso?: string): boolean {
    if (state[milestone]) return false;
    state[milestone] = true;
    state.lastMilestoneAt = nowIso ?? new Date().toISOString();
    return true;
}

export function milestonesCompleted(state: ActivationState): number {
    return ACTIVATION_MILESTONES.filter((m) => state[m]).length;
}

export function isActivated(state: ActivationState): boolean {
    // Activated = at least one high-value wardrobe action completed.
    return state.firstOutfitSaved || state.firstWearConfirmed || state.firstPurchaseCheck || state.firstAhaMoment;
}

// ── Aha moments (6B) ─────────────────────────────────────────────

const AHA_COPY: Record<AhaMomentKind, string> = {
    forgotten_wardrobe_outfit: 'It made a good outfit from clothes I forgot I owned.',
    no_buy_saved_purchase: 'It told me not to buy something because I already had a substitute.',
    recreated_from_own_wardrobe: 'It recreated a look from my own wardrobe.',
    correct_size_recommendation: 'It correctly recommended my size.',
    missing_piece_unlocked_outfits: 'It found one missing piece that unlocked many outfits.',
    first_value_success: 'First useful recommendation completed.',
};

export function describeAhaMoment(kind: AhaMomentKind): string {
    return AHA_COPY[kind];
}

export function createAhaMoment(kind: AhaMomentKind, contextSummary?: string, requestId?: string): AhaMomentEvent {
    return { kind, occurredAt: new Date().toISOString(), requestId, contextSummary };
}

// ── Progressive onboarding (6C) ──────────────────────────────────

export interface OnboardingProgress {
    stage: OnboardingStage;
    styleIntentCaptured: boolean;
    climateCaptured: boolean;
    garmentCount: number;
    firstValueDelivered: boolean;
    skipped: boolean;
}

/**
 * Advances onboarding. Skip paths always allowed; no profile field is
 * ever required before using the app.
 */
export function nextOnboardingStage(progress: OnboardingProgress): OnboardingStage {
    if (progress.skipped || progress.firstValueDelivered) return 'deepen_personalization';
    if (progress.garmentCount >= 1 && (progress.styleIntentCaptured || progress.garmentCount >= 3)) {
        return 'first_value';
    }
    return 'minimum_setup';
}

/** Minimum items before complete outfit generation is attempted. */
export const MIN_GARMENTS_FOR_FIRST_OUTFIT = 3;

export function firstValueReady(garmentCount: number): { ready: boolean; hint: string | null } {
    if (garmentCount >= MIN_GARMENTS_FOR_FIRST_OUTFIT) return { ready: true, hint: null };
    const needed = MIN_GARMENTS_FOR_FIRST_OUTFIT - garmentCount;
    return { ready: false, hint: `Add ${needed} more item${needed === 1 ? '' : 's'} to unlock complete outfit generation.` };
}

// ── First-session intent routing (6E) ────────────────────────────

const INTENT_SURFACE_PRIORITY: Record<FirstSessionIntent, string[]> = {
    decide_what_to_wear: ['today', 'wardrobe', 'stylist'],
    check_before_buy: ['shop_check', 'wardrobe', 'today'],
    organize_wardrobe: ['wardrobe', 'plans', 'today'],
    recreate_look: ['stylist', 'wardrobe', 'today'],
    improve_style: ['stylist', 'plans', 'wardrobe'],
    plan_travel: ['plans', 'wardrobe', 'today'],
    unknown: ['today', 'wardrobe', 'stylist'],
};

export function surfacesForIntent(intent: FirstSessionIntent): string[] {
    return INTENT_SURFACE_PRIORITY[intent] ?? INTENT_SURFACE_PRIORITY.unknown;
}

// ── Smart closet onboarding (6D) ────────────────────────────────

export type ClosetImportMethod = 'camera_scan' | 'batch_import' | 'photo_library' | 'product_url' | 'manual' | 'skipped';

export interface ClosetOnboardingState {
    method: ClosetImportMethod | null;
    importedCount: number;
    dismissed: boolean;
}

export function closetProgressHint(importedCount: number): string | null {
    if (importedCount >= MIN_GARMENTS_FOR_FIRST_OUTFIT) return null;
    // Honest: only shown when the engine genuinely cannot generate yet.
    return `Add ${MIN_GARMENTS_FOR_FIRST_OUTFIT - importedCount} more to start building outfits.`;
}
