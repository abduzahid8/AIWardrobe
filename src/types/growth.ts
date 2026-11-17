/**
 * Phase 6 — Growth & Monetization domain types.
 *
 * Wraps (never redefines) Phase 1–5 semantics: FitEvidence, StyleEvidence,
 * Fit/Style DNA, BuyCheck, provenance, decision events, entitlement tiers
 * (`free | lite | premium | vip` from `store/subscriptionStore`).
 *
 * Ethics (Section 1): no fake urgency/scarcity, no shame, no dark patterns.
 * Monetization principle: charge for leverage, automation, depth, scale,
 * convenience — never for fixing problems the product creates.
 */

export type SubscriptionTierName = 'free' | 'lite' | 'premium' | 'vip';

/** Phase 6Z — explicit subscription lifecycle (never boolean-only isPremium). */
export type SubscriptionLifecycleState =
    | 'unknown'
    | 'free'
    | 'trial'
    | 'premium_active'
    | 'grace_period'
    | 'billing_issue'
    | 'expired'
    | 'restoring'
    | 'unavailable';

/** Phase 6E — declared first-session job (user's words, not inferred traits). */
export type FirstSessionIntent =
    | 'decide_what_to_wear'
    | 'check_before_buy'
    | 'organize_wardrobe'
    | 'recreate_look'
    | 'improve_style'
    | 'plan_travel'
    | 'unknown';

/** Phase 6C — progressive onboarding stages. Value before deep profiling. */
export type OnboardingStage = 'minimum_setup' | 'first_value' | 'deepen_personalization' | 'complete';

/** Phase 6A — activation as meaningful value, not account creation. */
export interface ActivationState {
    wardrobeStarted: boolean;
    enoughItemsForOutfit: boolean;
    firstOutfitGenerated: boolean;
    firstOutfitSaved: boolean;
    firstWearConfirmed: boolean;
    firstPurchaseCheck: boolean;
    firstPersonalizationSignal: boolean;
    firstAhaMoment: boolean;
    lastMilestoneAt: string | null;
}

/** Phase 6B — aha moments are observed value events, never onboarding completion. */
export type AhaMomentKind =
    | 'forgotten_wardrobe_outfit'
    | 'no_buy_saved_purchase'
    | 'recreated_from_own_wardrobe'
    | 'correct_size_recommendation'
    | 'missing_piece_unlocked_outfits'
    | 'first_value_success';

export interface AhaMomentEvent {
    kind: AhaMomentKind;
    occurredAt: string;
    requestId?: string;
    /** Sanitized context only — no wardrobe contents, images, or body data. */
    contextSummary?: string;
}

/** Phase 6F — home action cards prioritized by urgency × confidence × value. */
export type HomeActionKind =
    | 'todays_outfit'
    | 'upcoming_event'
    | 'wardrobe_gap'
    | 'underused_item'
    | 'price_change'
    | 'fit_feedback_request'
    | 'weekly_plan'
    | 'no_buy_reminder'
    | 'continue_closet_scan'
    | 'seasonal_review';

export interface HomeActionCandidate {
    kind: HomeActionKind;
    urgency: number; // 0..1
    confidence: number; // 0..1
    expectedValue: number; // 0..1
    /** Hours since last shown; enforces cooldown. */
    hoursSinceLastShown: number;
    /** Hours until the action expires (event passes, price window closes). */
    expiresInHours: number | null;
    goalAligned: boolean;
}

/** Phase 6J — optional, non-manipulative streaks. Never shame. */
export interface StreakState {
    id: string;
    count: number;
    lastActiveWeek: string | null;
    /** Streaks are always optional; broken streaks are never punished. */
    optedIn: boolean;
}

/** Phase 6K — notification policy domain. No remote sending in Phase 6. */
export type NotificationType =
    | 'price_target_reached'
    | 'outfit_weather_conflict'
    | 'event_missing_piece'
    | 'planned_item_unavailable'
    | 'weekly_plan_ready'
    | 'fit_feedback_request'
    | 'seasonal_review'
    | 'trial_ending'
    | 'renewal_reminder'
    | 'billing_issue';

export type NotificationUrgency = 'low' | 'normal' | 'high' | 'critical';

export interface NotificationCandidate {
    type: NotificationType;
    urgency: NotificationUrgency;
    expectedValue: number; // 0..1
    confidence: number; // 0..1
    cooldownHours: number;
    hoursSinceLastSameType: number;
    /** Hours until irrelevant; expired candidates must never send. */
    expiresInHours: number | null;
    actionable: boolean;
    quietHoursCompatible: boolean;
}

export type NotificationChannel = 'outfit_reminders' | 'event_reminders' | 'price_alerts' | 'fit_feedback' | 'weekly_review' | 'travel_packing' | 'shopping_reminders';

export type NotificationPreference = 'off' | 'important_only' | 'normal';

/** Phase 6P — single entitlement matrix. No scattered `if (isPremium)`. */
export type CostClass = 'free_compute' | 'cheap_api' | 'expensive_vision' | 'expensive_generation' | 'monitored_service';

export interface FeatureDefinition {
    featureId: string;
    freeAvailability: boolean;
    /** Uses per period; null = unlimited. */
    freeLimit: number | null;
    premiumAvailability: boolean;
    premiumLimit: number | null;
    costClass: CostClass;
    /** Safe fallback when quota/entitlement denies access. */
    fallback: string;
    /** User-comprehensible reason shown at limit ("why limit exists"). */
    upgradeReason: string;
    requiredEntitlement: SubscriptionTierName;
}

/** Phase 6Q — usage metering in user-meaningful units, never "AI credits". */
export interface QuotaState {
    featureId: string;
    used: number;
    limit: number | null; // null = unlimited
    /** ISO timestamp of next reset; null when no rolling period. */
    resetsAt: string | null;
    /** Explicit period label, e.g. "monthly". No hidden rolling limits. */
    period: string | null;
}

/** Phase 6V — plan architecture. Prices come from the store, never hardcoded. */
export interface PlanDefinition {
    planId: string;
    billingPeriod: 'monthly' | 'annual';
    /** Localized display price from the store; null until store provides it. */
    localizedPrice: string | null;
    trialDays: number | null;
    eligible: boolean;
}

/** Phase 6T — paywall component model (reusable, not per-screen hardcoded). */
export interface PaywallContext {
    /** Blocked value that triggered the paywall (6S, 6AT, 6AR). */
    triggerFeatureId: string;
    triggerDescription: string;
    /** What remains available for free — always shown (6R). */
    freeFallback: string;
    /** When the quota resets, if applicable. */
    resetsAt: string | null;
    benefits: string[];
    plans: PlanDefinition[];
    trialDisclosure: string | null;
}

/** Phase 6AG — optional cancellation feedback. Never blocks cancellation. */
export type CancellationReason =
    | 'too_expensive'
    | 'not_using_enough'
    | 'ai_quality'
    | 'missing_feature'
    | 'technical_issue'
    | 'temporary'
    | 'privacy_concern'
    | 'other';

/** Phase 6AI — referral domain (architecture only; no remote rewards). */
export interface ReferralState {
    inviterUserId: string | null;
    inviteToken: string | null;
    status: 'none' | 'invited' | 'activated' | 'reward_eligible' | 'rewarded' | 'flagged_abuse';
    rewardIdempotencyKey: string | null;
}

/** Phase 6AJ — privacy-safe share cards. Explicit opt-in per share. */
export type ShareableKind = 'outfit' | 'recreated_look' | 'capsule' | 'weekly_plan' | 'wardrobe_insight';

export interface ShareCard {
    kind: ShareableKind;
    title: string;
    summary: string;
    /** Revocable, expiring public reference when link sharing is enabled. */
    shareToken: string | null;
    expiresAt: string | null;
}

/** Phase 6AN — conservative savings: verified only, with provenance. */
export interface VerifiedSaving {
    kind: 'skipped_duplicate' | 'cheaper_equivalent' | 'price_target_hit' | 'repair_over_replace';
    amountCents: number;
    currency: string;
    verifiedAt: string;
    /** Provenance: which request/observation verified this saving. */
    provenance: string;
}

/** Phase 6BJ — sponsored placements can never masquerade as organic. */
export interface RankedRecommendation {
    productId: string;
    score: number;
    sponsored: boolean;
}

/** Phase 6BP — versioned KPI definitions. Never changed silently. */
export interface KpiDefinition {
    kpiId: string;
    version: number;
    definition: string;
}

/** Phase 6BR — low-friction feedback with cooldown. */
export type FeedbackReason = 'fit' | 'style' | 'price' | 'duplicate' | 'irrelevant' | 'technical_issue';

export interface FeedbackEvent {
    helpful: boolean;
    reason: FeedbackReason | null;
    feature: string;
    at: string;
}
