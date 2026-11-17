/**
 * Read-only FR Group product-variant observations.
 *
 * Product HTML exposes a numeric modelId; their public /ru/catalog/
 * get-product-info returns size-specific variant IDs, availability flags
 * and site-reported counts. This is merchant-reported evidence, NOT a
 * guaranteed checkout/reservation or permission to use product imagery.
 */
import { createHash } from 'node:crypto';

const hash = s => createHash('sha256').update(s).digest('hex');
export const FR_PRODUCT_INFO_ENDPOINT = 'https://frgroup.uz/ru/catalog/get-product-info';
export const FR_VARIANT_SNAPSHOT_MAX_AGE_MS = 30 * 60 * 1000;

export function frModelIdFromHtml(html) {
  if (typeof html !== 'string') return null;
  const match = html.match(/\bvar\s+productId\s*=\s*["']([1-9][0-9]{0,11})["']\s*;/);
  return match ? Number(match[1]) : null;
}

export function parseFrSizeResponse(data, {
  modelId, expectedPrice, collectedAt, responseSha256,
}) {
  if (!data || typeof data !== 'object' || Array.isArray(data) ||
    !Number.isSafeInteger(modelId) || modelId <= 0 ||
    data.modelId !== modelId || !Number.isSafeInteger(expectedPrice) ||
    data.salePrice !== expectedPrice ||
    !/^[a-f0-9]{64}$/.test(responseSha256 || '') ||
    !Number.isFinite(Date.parse(collectedAt)) ||
    data.canBePurchased !== true || data.isOutOfStock !== false ||
    !Array.isArray(data.sizes) || !data.sizes.length || data.sizes.length > 55 ||
    !Array.isArray(data.skuSortedList)) {
    return null;
  }
  const sortedIds = new Set();
  for (const x of data.skuSortedList) {
    if (!Number.isSafeInteger(x?.id) || x.id < 1 || sortedIds.has(x.id)) return null;
    sortedIds.add(x.id);
  }
  const seenVariants = new Set();
  const seenSizes = new Set();
  const sizes = [];
  for (const entry of data.sizes) {
    const size = String(entry?.size || '').trim();
    const variantId = entry?.value;
    const quantity = entry?.quantity;
    if (!/^(3[8-9]|4[0-9]|50)(?:[.]5)?$/.test(size) ||
      !Number.isSafeInteger(variantId) || variantId <= 0 ||
      !sortedIds.has(variantId) || seenVariants.has(variantId) ||
      seenSizes.has(size) || entry.available !== 1 ||
      !Number.isSafeInteger(quantity) || quantity <= 0 || quantity > 99) {
      return null;
    }
    seenVariants.add(variantId);
    seenSizes.add(size);
    sizes.push({ euSize: size, variantId, merchantReportedQty: quantity });
  }
  if (seenVariants.size !== sortedIds.size) return null;
  return {
    status: 'merchant_reported_by_size',
    merchant: 'FR Group',
    modelId,
    checkedAt: new Date(collectedAt).toISOString(),
    refreshBefore: new Date(Date.parse(collectedAt) + FR_VARIANT_SNAPSHOT_MAX_AGE_MS).toISOString(),
    priceMatched: true,
    publicEndpoint: FR_PRODUCT_INFO_ENDPOINT,
    responseSha256,
    availableSizes: sizes,
    totalMerchantReportedUnits: sizes.reduce((sum, s) => sum + s.merchantReportedQty, 0),
    exactCheckoutAvailability: 'unverified',
    deliveryCharge: 'unverified',
  };
}

/**
 * Never use a committed screenshot/snapshot as "in stock" indefinitely.
 * Only a fresh, model-bound seller claim may inform a size-aware review.
 * Even "reported_available" requires a *new* check at checkout and does
 * not authorize activation or end-user purchase guarantees.
 */
export function assessFrVariantSize(row, euSize, now = Date.now()) {
  const evidence = row?.variantEvidence;
  if (row?.brand !== 'Tommy Hilfiger' ||
      row?.sizeStockStatus !== 'merchant_reported_by_size' ||
      evidence?.status !== 'merchant_reported_by_size' ||
      evidence?.merchant !== 'FR Group' ||
      !Number.isFinite(now) || !Number.isFinite(Date.parse(evidence.checkedAt)) ||
      Date.parse(evidence.checkedAt) > now ||
      evidence.refreshBefore !== new Date(Date.parse(evidence.checkedAt) +
        FR_VARIANT_SNAPSHOT_MAX_AGE_MS).toISOString() ||
      now >= Date.parse(evidence.refreshBefore)) {
    return { status: 'unknown_or_expired', checkoutVerified: false };
  }
  const found = evidence.availableSizes?.find(s => s.euSize === String(euSize));
  return found
    ? { status: 'merchant_reported_available_not_reserved',
        merchantVariantId: found.variantId,
        merchantReportedQty: found.merchantReportedQty,
        checkedAt: evidence.checkedAt,
        checkoutVerified: false }
    : { status: 'size_not_listed_in_snapshot', checkoutVerified: false };
}

export async function fetchFrSizeEvidence({
  html, expectedPrice, fetchImpl = fetch,
  collectedAt = new Date().toISOString(), timeoutMs = 8000,
}) {
  const modelId = frModelIdFromHtml(html);
  if (!modelId) return null;
  const url = new URL(FR_PRODUCT_INFO_ENDPOINT);
  url.searchParams.set('productId', String(modelId));
  const response = await fetchImpl(url.toString(), {
    method: 'GET', redirect: 'error', cache: 'no-store',
    headers: { Accept: 'application/json' },
    signal: AbortSignal.timeout(timeoutMs),
  });
  if (!response.ok || !String(response.headers.get('content-type') || '').includes('application/json')) return null;
  const raw = await response.text();
  if (raw.length > 150000) return null;
  let data;
  try { data = JSON.parse(raw); } catch { return null; }
  return parseFrSizeResponse(data, {
    modelId, expectedPrice, collectedAt, responseSha256: hash(raw),
  });
}
