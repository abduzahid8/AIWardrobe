/**
 * Phase 6AV–6AX — Regional pricing, student-offer abstraction,
 * promo eligibility. Local models only; no remote price changes.
 * Prices are always store-localized; never hardcoded USD.
 */

export interface RegionalPrice {
    planId: string;
    region: string;
    currency: string;
    /** Localized display string from the store; null until provided. */
    displayPrice: string | null;
    taxIncluded: boolean;
}

export function formatRegionalPrice(price: RegionalPrice): string {
    if (!price.displayPrice) return 'Price unavailable in this region';
    return price.taxIncluded ? `${price.displayPrice} (incl. tax)` : price.displayPrice;
}

export type CurrencyCode = string; // ISO-4217; never assume USD.

// ── Student / education offer abstraction (6AW) ──────────────────

export interface StudentEligibility {
    eligible: boolean;
    verificationProvider: string | null;
    expiresAt: string | null;
    reverificationRequired: boolean;
}

/** No fake verification: eligibility is unknown until a real provider confirms. */
export function unknownStudentEligibility(): StudentEligibility {
    return { eligible: false, verificationProvider: null, expiresAt: null, reverificationRequired: false };
}

// ── Promo eligibility (6AX) ──────────────────────────────────────

export interface PromoContext {
    isNewUser: boolean;
    isReturningUser: boolean;
    region: string;
    campaign: string | null;
    hadPriorTrial: boolean;
    hadPriorSubscription: boolean;
    storeOfferAvailable: boolean;
}

export function promoEligible(ctx: PromoContext): { eligible: boolean; reason: string } {
    // Never show offers the store does not support.
    if (!ctx.storeOfferAvailable) return { eligible: false, reason: 'no_store_offer' };
    if (!ctx.campaign) return { eligible: false, reason: 'no_campaign' };
    if (ctx.hadPriorSubscription && !ctx.isReturningUser) return { eligible: false, reason: 'prior_subscriber' };
    return { eligible: true, reason: 'ok' };
}
