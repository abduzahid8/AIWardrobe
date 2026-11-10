// Read only trusted, reviewed private manifests. These checks enforce the
// documented scope and revision; they do not determine legal rights from URLs.
const SHA256 = /^[0-9a-f]{64}$/;
const HTTPS = (url) => {
  try {
    const value = new URL(url);
    return value.protocol === 'https:' && !value.username && !value.password;
  } catch {
    return false;
  }
};
const supplied = (value) => typeof value === 'string' && value.trim().length > 0;
const millis = (value) => typeof value === 'string' ? Date.parse(value) : NaN;

export function trialImageDecision(manifest, asset, now = Date.now()) {
  if (manifest?.version !== 1 || !Array.isArray(manifest.assets)) return 'missing_manifest';
  const matches = manifest.assets.filter((row) =>
    row?.kind === asset.kind && row.id === asset.id && row.sha256 === asset.sha256);
  if (matches.length !== 1) return matches.length ? 'ambiguous_grant' : 'missing_exact_asset_grant';
  const grant = matches[0];
  if (!SHA256.test(asset.sha256) || !HTTPS(grant.sourceUrl) ||
      (asset.sourceUrl && grant.sourceUrl !== asset.sourceUrl)) return 'source_mismatch';
  if (!supplied(grant.rightsHolder) || !supplied(grant.reviewedBy) ||
      !supplied(grant.evidence?.reference) || !SHA256.test(grant.evidence?.sha256 || '')) return 'missing_grant_evidence';
  if (grant.status !== 'active' || !(millis(grant.validFrom) <= now) ||
      !(grant.expiresAt === null || millis(grant.expiresAt) > now)) return 'inactive_grant';
  const trial = grant.nvidiaTrial;
  if (trial?.submit !== true || trial?.territory !== 'global' ||
      trial?.providerServiceImprovement !== true || trial?.terms !== 'nvidia-api-trial' ||
      !Array.isArray(trial?.purposes) || !trial.purposes.includes('internal_outfit_evaluation')) {
    return 'trial_scope_denied';
  }
  return 'cleared';
}

export function purchaseEvidence(item, offer, requirement, now = Date.now()) {
  if (item?.is_active === false || item?.isActive === false) return 'inactive_catalog';
  if (!offer) return 'missing_offer';
  if (offer.sku !== item.id || offer.url !== (item.shopUrl || item.source_url) ||
      !HTTPS(offer.url) || !supplied(offer.variantId)) return 'offer_identity_mismatch';
  const observed = millis(offer.observedAt);
  if (!Number.isFinite(observed) || observed > now || now - observed > 24 * 60 * 60_000) return 'stale_offer';
  if (!['merchant_feed', 'merchant_api', 'verified_checkout'].includes(offer.source) ||
      !supplied(offer.evidenceRef) || !SHA256.test(offer.evidenceSha256 || '')) return 'missing_offer_provenance';
  if (!supplied(requirement?.size) || !supplied(requirement?.sizeSystem) ||
      offer.size !== requirement.size || offer.sizeSystem !== requirement.sizeSystem ||
      offer.stock !== 'in_stock') return 'size_or_stock_unverified';
  if (!/^[A-Z]{2}$/.test(requirement?.country || '') || offer.country !== requirement.country ||
      offer.shippingConfirmed !== true) return 'destination_unverified';
  if (!/^[A-Z]{3}$/.test(requirement?.currency || '') || offer.currency !== requirement.currency ||
      !Number.isFinite(offer.price) || offer.price <= 0 ||
      !Number.isFinite(offer.landedPrice) || offer.landedPrice < offer.price) return 'price_unverified';
  return 'verified';
}

export function outfitPurchaseEvidence(items, offers, requirement, now = Date.now()) {
  const products = items.map((item) => {
    const { size, sizeSystem } = requirement.sizes?.[item.macroCategory] || {};
    const matches = offers.filter((row) => row?.sku === item.id && row.size === size &&
      row.sizeSystem === sizeSystem && row.country === requirement.country);
    const offer = matches.length === 1 ? matches[0] : null;
    const status = matches.length > 1 ? 'ambiguous_offer' : purchaseEvidence(item, offer, { ...requirement, size, sizeSystem }, now);
    return { sku: item.id, status, landedPrice: status === 'verified' ? offer.landedPrice : null };
  });
  const roles = new Set(items.map(({ macroCategory }) => macroCategory));
  if (items.length < 3 || items.length > 4 || roles.size !== items.length ||
      !['top', 'bottom', 'shoes'].every((role) => roles.has(role)) ||
      products.some(({ status }) => status !== 'verified')) return { status: 'unverified', products };
  const total = products.reduce((sum, product) => sum + product.landedPrice, 0);
  return { status: requirement.budget !== undefined && (!(requirement.budget >= total) || !Number.isFinite(requirement.budget))
    ? 'over_budget_or_invalid_budget' : 'verified', total, products };
}
