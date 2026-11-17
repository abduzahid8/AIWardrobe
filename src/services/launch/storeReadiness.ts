/**
 * Phase 7BE–7BH — Store readiness: listing checklist, claim safety,
 * app-review risks, external purchase flow safety.
 */

export const STORE_LISTING_CHECKLIST = [
    'app_name', 'subtitle', 'long_description', 'screenshots', 'privacy_policy_url',
    'support_url', 'terms_url', 'subscription_disclosure', 'feature_claims_reviewed', 'ai_disclosure',
] as const;

const FORBIDDEN_CLAIMS = [
    'perfect fit guaranteed',
    'always finds cheapest price',
    'ai knows your body perfectly',
    'best stylist',
] as const;

export function claimSafe(copy: string): { safe: boolean; reason: string } {
    const lower = copy.toLowerCase();
    for (const claim of FORBIDDEN_CLAIMS) {
        if (lower.includes(claim)) return { safe: false, reason: `forbidden_claim:${claim}` };
    }
    if (/saves? \$?\d/i.test(copy)) return { safe: false, reason: 'unsubstantiated_savings' };
    return { safe: true, reason: 'ok' };
}

export function preferableClaim(claim: string): string {
    const map: Record<string, string> = {
        'perfect fit guaranteed': 'helps compare fit evidence',
        'always finds cheapest price': 'can compare available prices where data exists',
        'best stylist': 'checks wardrobe compatibility',
    };
    return map[claim] ?? claim;
}

export const APP_REVIEW_RISKS = [
    { risk: 'subscription_disclosure', note: 'Trial/billing/restore copy must be explicit' },
    { risk: 'account_deletion', note: 'In-app deletion path required where applicable' },
    { risk: 'restore_purchases', note: 'Restore button reachable and functional' },
    { risk: 'camera_photo_usage', note: 'Permission strings explain wardrobe use' },
    { risk: 'external_purchase_links', note: 'Merchant destination labeled; affiliate disclosed' },
    { risk: 'tracking_privacy', note: 'No tracking without disclosure where required' },
    { risk: 'ai_generated_content', note: 'VTON labeled as visualization, not fit proof' },
] as const;

export function externalPurchaseDisclosure(merchant: string): string {
    return `Opens ${merchant} in your browser. Your purchase happens with ${merchant}, not AIWardrobe. Selected options carry over when the merchant supports it.`;
}
