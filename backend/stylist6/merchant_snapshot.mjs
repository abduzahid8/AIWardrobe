// Normalize an already authorized, private merchant export. No network or
// catalog writes. The caller supplies independent, exact SKU/variant bindings.
import { createHash } from 'node:crypto';
import { purchaseEvidence, reviewerImageDecision, trialImageDecision } from './retail_evidence.mjs';

const MAX_BYTES = 2 * 1024 * 1024;
const SHA256 = /^[a-f0-9]{64}$/;
const IDENTIFIER = /^[\w.:/-]{1,120}$/;
const keyFor = ({ sku, size, sizeSystem }) => JSON.stringify([sku, size, sizeSystem]);

export function importMerchantSnapshot({ bytes, evidenceSha256, evidenceRef, authority, catalog, bindings, rights, now = Date.now() }) {
  if (!Buffer.isBuffer(bytes) || bytes.length > MAX_BYTES || !SHA256.test(evidenceSha256 || '') ||
      createHash('sha256').update(bytes).digest('hex') !== evidenceSha256) throw new Error('merchant_snapshot_digest_mismatch');
  if (!IDENTIFIER.test(authority?.merchantId || '') || !/^[A-Z]{2}$/.test(authority?.country || '') ||
      !/^[A-Z]{3}$/.test(authority?.currency || '') || ![0, 1, 2, 3].includes(authority?.priceDecimals) ||
      !['merchant_feed', 'merchant_api'].includes(authority?.source) ||
      !['synthetic', 'authorized_merchant'].includes(authority?.scope) ||
      typeof evidenceRef !== 'string' || !evidenceRef.trim() ||
      !(catalog instanceof Map) || !Array.isArray(bindings)) throw new Error('merchant_snapshot_authority_required');

  const snapshot = JSON.parse(bytes.toString('utf8'));
  if (snapshot.version !== 1 || snapshot.merchantId !== authority.merchantId || snapshot.country !== authority.country ||
      snapshot.currency !== authority.currency || !Array.isArray(snapshot.variants)) throw new Error('merchant_snapshot_scope_mismatch');
  const bound = new Map(), variantOwners = new Set();
  for (const item of bindings) {
    const key = keyFor(item);
    if (!IDENTIFIER.test(item?.sku || '') || !IDENTIFIER.test(item?.variantId || '') ||
        typeof item.size !== 'string' || !item.size.trim() || typeof item.sizeSystem !== 'string' || !item.sizeSystem.trim() ||
        bound.has(key) || variantOwners.has(item.variantId)) throw new Error('merchant_variant_binding_invalid');
    bound.set(key, item.variantId);
    variantOwners.add(item.variantId);
  }

  const occurrences = new Map();
  for (const row of snapshot.variants) {
    if (!row || typeof row !== 'object') throw new Error('merchant_variant_row_invalid');
    const key = keyFor(row);
    occurrences.set(key, (occurrences.get(key) || 0) + 1);
  }

  const offers = [], provenance = [], excluded = {};
  const factor = 10 ** authority.priceDecimals;
  for (const row of snapshot.variants) {
    const key = keyFor(row), item = catalog.get(row.sku);
    let status = 'verified', offer;
    if (occurrences.get(key) > 1) status = 'duplicate_variant_offer';
    else if (!item || item.id !== row.sku || item.is_active !== true ||
        !bound.has(key) || bound.get(key) !== row.variantId) status = 'variant_identity_unverified';
    else if (![row.priceMinor, row.shippingMinor, row.landedMinor].every((value) => Number.isSafeInteger(value) && value >= 0) ||
        row.priceMinor < 1 || !Number.isSafeInteger(row.priceMinor + row.shippingMinor) ||
        row.landedMinor < row.priceMinor + row.shippingMinor) status = 'invalid_merchant_price';
    else {
      offer = {
        sku: row.sku, variantId: row.variantId, url: row.url, size: row.size, sizeSystem: row.sizeSystem,
        country: row.country, currency: row.currency, stock: row.stock, shippingConfirmed: row.shippingConfirmed,
        observedAt: row.observedAt, price: row.priceMinor / factor, shippingPrice: row.shippingMinor / factor,
        landedPrice: row.landedMinor / factor, source: authority.source, evidenceRef, evidenceSha256,
      };
      status = purchaseEvidence(item, offer, {
        country: authority.country, currency: authority.currency, size: row.size,
        sizeSystem: row.sizeSystem, requireCatalogPriceMatch: true,
      }, now);
    }
    if (status !== 'verified') {
      excluded[status] = (excluded[status] || 0) + 1;
      continue;
    }
    const photo = row.photo;
    const asset = photo?.sourceUrl && photo.sourceUrl === item.image_url && SHA256.test(photo.sha256 || '')
      ? { kind: 'product', id: item.id, sourceUrl: photo.sourceUrl, sha256: photo.sha256 } : null;
    const grant = asset ? trialImageDecision(rights, asset, now) : 'photo_provenance_unverified';
    const review = asset ? reviewerImageDecision(rights, asset, now) : 'photo_provenance_unverified';
    const exactGrant = grant === 'cleared' || review === 'cleared' ? rights.assets.find((entry) =>
      entry.kind === 'product' && entry.id === item.id && entry.sha256 === photo.sha256) : null;
    offers.push(offer);
    provenance.push({
      sku: row.sku, variantId: row.variantId,
      photoSha256: asset?.sha256 ?? null, inference: grant,
      rightsHolder: exactGrant?.rightsHolder ?? null, rightsEvidence: exactGrant?.evidence ?? null,
      reviewerDistribution: review === 'cleared',
    });
  }
  return { scope: authority.scope, merchantId: authority.merchantId, country: authority.country, offers, provenance, excluded };
}
