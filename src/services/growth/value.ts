/**
 * Phase 6AN–6AO, 6AP–6AQ, 6AR, 6AS, 6AT, 6BH — Wardrobe value
 * analytics: conservative savings, utilization, cohorts, intent
 * routing, contextual education, contextual upsell, premium fallback,
 * no-purchase premium value.
 */

import type { FirstSessionIntent, VerifiedSaving } from '../../types/growth';

// ── Savings model (6AN) — verified only ──────────────────────────
// Counted: skipped duplicate, cheaper equivalent, price target hit,
// repair-over-replace with provenance. NEVER hypothetical retail
// prices or un-acted-upon recommendations.

export type SavingVerificationKind = 'skipped_duplicate' | 'cheaper_equivalent' | 'price_target_hit' | 'repair_over_replace';

export function verifySaving(input: {
    kind: SavingVerificationKind;
    amountCents: number;
    currency: string;
    provenance: string;
    actedUpon: boolean;
}): VerifiedSaving | null {
    if (!input.actedUpon) return null;
    if (!Number.isFinite(input.amountCents) || input.amountCents <= 0) return null;
    if (!input.provenance) return null;
    return {
        kind: input.kind,
        amountCents: Math.round(input.amountCents),
        currency: input.currency,
        verifiedAt: new Date().toISOString(),
        provenance: input.provenance,
    };
}

// ── Utilization metric (6AO) — insight, never shame ─────────────

export function utilizationSummary(activeItems: number, wornItems: number): string {
    if (activeItems <= 0) return 'Add items to start tracking wardrobe use.';
    const pct = Math.round((wornItems / activeItems) * 100);
    return `You used ${wornItems} of ${activeItems} active items (${pct}%).`;
}

// ── Cohorts (6AP) + persona/intent routing (6AQ) ─────────────────

export type GrowthCohort = 'closet_first' | 'shopping_first' | 'fit_first' | 'inspiration_first' | 'planner_first';

export function cohortForIntent(intent: FirstSessionIntent): GrowthCohort {
    switch (intent) {
        case 'check_before_buy': return 'shopping_first';
        case 'organize_wardrobe': return 'closet_first';
        case 'recreate_look': return 'inspiration_first';
        case 'improve_style':
        case 'decide_what_to_wear': return 'planner_first';
        default: return 'closet_first';
    }
}

// ── Contextual education (6AR) — emerges from user's wardrobe ────

export type EducationTopic = 'why_this_outfit' | 'reuse_garment' | 'evaluate_fit' | 'avoid_duplicate';

export function educationCopy(topic: EducationTopic): { title: string; body: string } {
    const copy: Record<EducationTopic, { title: string; body: string }> = {
        why_this_outfit: { title: 'Why this outfit works', body: 'Colors share a temperature, proportions balance, and the anchor leads.' },
        reuse_garment: { title: 'One garment, three ways', body: 'Small swaps — shoes, layer, tuck — reuse what you own.' },
        evaluate_fit: { title: 'How to judge fit', body: 'Shoulders, drape, and comfort after sitting matter more than size labels.' },
        avoid_duplicate: { title: 'Spot duplicates early', body: 'Compare against your wardrobe before buying a near-twin.' },
    };
    return copy[topic];
}

// ── Contextual upsell (6AS) — tied to blocked value ──────────────

export function contextualUpsell(featureId: string, fallback: string, upgradeReason: string): string {
    return `You can ${fallback.replace(/_/g, ' ')} for free. ${upgradeReason}`;
}

// ── Premium fallback (6AT) — downgrade preserves data ────────────

export interface DowngradeOutcome {
    userDataPreserved: boolean;
    historyReadable: boolean;
    expensiveRegenerationDisabled: string[];
}

export function applyDowngrade(premiumFeaturesUsed: string[]): DowngradeOutcome {
    return {
        userDataPreserved: true, // never delete on downgrade
        historyReadable: true, // read-only historical outputs preserved
        expensiveRegenerationDisabled: premiumFeaturesUsed,
    };
}

// ── No-purchase premium value (6BH) ───────────────────────────────

export const NO_PURCHASE_PREMIUM_VALUE = [
    'weekly_planning',
    'outfit_memory',
    'travel_packing',
    'deep_style_dna',
    'advanced_wardrobe_analytics',
    'visual_recreation',
    'organization',
    'fit_intelligence',
] as const;
