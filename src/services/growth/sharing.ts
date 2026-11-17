/**
 * Phase 6AI–6AM — Referral architecture, privacy-safe share cards,
 * virality without leakage, social-proof safety, challenges.
 *
 * Privacy: explicit opt-in per share; revocable; expiring; no private
 * metadata (wardrobe contents, price history, body data, notes).
 */

import type { ReferralState, ShareableKind, ShareCard } from '../../types/growth';

export function createInviteToken(randomPart: string): string {
    return `inv_${randomPart}`;
}

export function initialReferralState(): ReferralState {
    return { inviterUserId: null, inviteToken: null, status: 'none', rewardIdempotencyKey: null };
}

// ── Shareable outputs (6AJ) ──────────────────────────────────────
// Fields that must NEVER appear on a share card.
export const SHARE_FORBIDDEN_FIELDS = [
    'wardrobe_contents',
    'price_history',
    'body_measurements',
    'home_location',
    'private_notes',
    'precise_location',
    'raw_images',
] as const;

export function buildShareCard(input: {
    kind: ShareableKind;
    title: string;
    summary: string;
    /** Caller must strip forbidden fields before calling; verified here. */
    payloadKeys: string[];
    expiresAt: string | null;
}): ShareCard | { error: string } {
    const leaked = input.payloadKeys.filter((k) =>
        (SHARE_FORBIDDEN_FIELDS as readonly string[]).includes(k),
    );
    if (leaked.length > 0) return { error: `share_blocked_private_fields: ${leaked.join(',')}` };
    return {
        kind: input.kind,
        title: input.title,
        summary: input.summary,
        shareToken: null, // Issued only when link sharing is enabled + user opts in.
        expiresAt: input.expiresAt,
    };
}

// ── Social proof safety (6AL) ────────────────────────────────────

export const MIN_SOCIAL_PROOF_COHORT = 50;

export function socialProofCopy(input: { label: string; cohortSize: number; valid: boolean }): string | null {
    if (!input.valid || input.cohortSize < MIN_SOCIAL_PROOF_COHORT) return null; // never fabricate
    return `${input.label} (based on ${input.cohortSize} similar savers)`;
}

// ── Challenges (6AM) — optional, positive, non-shaming ──────────

export type ChallengeKind = 'underused_month' | 'no_duplicate_buy' | 'capsule_week' | 'repeat_best_outfit';

export interface Challenge {
    kind: ChallengeKind;
    title: string;
    goal: string;
    optedIn: boolean;
}

export function challengeFor(kind: ChallengeKind): Challenge {
    const defs: Record<ChallengeKind, { title: string; goal: string }> = {
        underused_month: { title: 'Wear 10 underused items', goal: 'Give forgotten pieces a second chance this month.' },
        no_duplicate_buy: { title: 'No-duplicate-buy month', goal: 'A month without buying something you already own.' },
        capsule_week: { title: 'Capsule week', goal: 'Seven days from a small capsule wardrobe.' },
        repeat_best_outfit: { title: 'Repeat your best outfit', goal: 'Rewear a favorite outfit with confidence.' },
    };
    return { kind, ...defs[kind], optedIn: false };
}
