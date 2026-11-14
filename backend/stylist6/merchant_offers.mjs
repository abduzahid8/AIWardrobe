import { priceFreshness } from './catalog_adapter.mjs';
import { HOST_ALLOWLIST, isInferenceAuthorized } from '../../scripts/stylist6/image_access.mjs';

export const MERCHANT_OFFER_VERSION = 's6-merchant-offers-v1';

function hostOf(url) {
  try {
    const u = new URL(String(url || ''));
    return u.protocol === 'https:' && !u.username && !u.password && !u.port
      ? u.hostname.toLowerCase()
      : null;
  } catch {
    return null;
  }
}

export function classifyItemImagePermissions(imageUrl) {
  if (!imageUrl || typeof imageUrl !== 'string') {
    return { status: 'missing', displayAllowed: false, externalInferenceAllowed: false };
  }
  const host = hostOf(imageUrl);
  if (!host) {
    return { status: 'unsafe_url', displayAllowed: false, externalInferenceAllowed: false };
  }
  if (isInferenceAuthorized(imageUrl)) {
    // A network allowlist authorizes technical transport in staging, NOT
    // commercial image rights. Never label merchant photos as licensed for
    // external inference without independently verified rights evidence.
    return {
      status: 'technical_inference_allowlisted_rights_unverified',
      displayAllowed: true,
      technicalInferenceAllowlisted: true,
      externalInferenceAllowed: false,
      rightsEvidenceStatus: 'unverified',
    };
  }
  if (HOST_ALLOWLIST.has(host)) {
    return { status: 'allowlisted_display_only', displayAllowed: true, externalInferenceAllowed: false };
  }
  return { status: 'unlicensed_external_host', displayAllowed: false, externalInferenceAllowed: false };
}

/**
 * Verify merchant offer evidence for a single garment without fabricating
 * size stock, checkout confirmation, price timestamps, or delivery eligibility.
 */
export function verifyMerchantOffer(item, {
  offerRegistry = null,
  requestedSize = null,
  targetMarket = 'UZ',
  now = Date.now(),
} = {}) {
  if (!item || item.isShopItem !== true) {
    return {
      isShopItem: false,
      offerStatus: 'owned_wardrobe',
      checkoutAvailability: 'not_applicable',
    };
  }
  const purchaseUrl = typeof item.shopUrl === 'string' ? item.shopUrl : (typeof item.source_url === 'string' ? item.source_url : null);
  const purchaseHost = hostOf(purchaseUrl);
  const freshness = priceFreshness(item, now);
  const imagePerms = classifyItemImagePermissions(item.imageUrl || item.image_url);
  const observedAt = item.priceObservedAt ?? item.last_seen_at ?? null;
  const priceValid = Number.isFinite(Number(item.price)) && Number(item.price) > 0 && /^[A-Z]{3}$/.test(String(item.currency || ''));

  const offer = offerRegistry instanceof Map
    ? (offerRegistry.get(String(item.id)) || (purchaseUrl ? offerRegistry.get(purchaseUrl) : null) || null)
    : null;

  const offerFresh = Boolean(
    offer &&
    Number.isFinite(Date.parse(offer.observedAt)) &&
    Date.parse(offer.observedAt) <= now &&
    (!offer.refreshBefore || now < Date.parse(offer.refreshBefore)) &&
    offer.targetMarket === targetMarket &&
    (!offer.currency || offer.currency === item.currency) &&
    (!Number.isFinite(offer.price) || Math.abs(offer.price - Number(item.price)) <= Math.max(1, Number(item.price) * 0.01))
  );

  const availableSizes = offerFresh && Array.isArray(offer.availableSizes)
    ? offer.availableSizes.map((s) => ({
        size: String(s.euSize ?? s.size ?? ''),
        variantId: s.variantId ?? null,
        merchantReportedQty: Number.isSafeInteger(s.merchantReportedQty) ? s.merchantReportedQty : null,
      })).filter((s) => s.size)
    : [];

  let requestedSizeStatus = 'unverified';
  if (offerFresh && availableSizes.length > 0 && requestedSize != null) {
    const hit = availableSizes.find((s) => s.size === String(requestedSize));
    requestedSizeStatus = hit ? 'merchant_reported_available_not_reserved' : 'size_not_in_merchant_snapshot';
  }

  const deliveryEligible = offerFresh && typeof offer.deliveryEligible === 'boolean'
    ? offer.deliveryEligible
    : 'unverified';

  const offerStatus = !purchaseHost || !priceValid
    ? 'invalid_listing_evidence'
    : offerFresh && offer.sizeStockStatus === 'merchant_reported_by_size'
      ? 'merchant_reported_live'
      : 'unverified_catalog_listing';

  return {
    version: MERCHANT_OFFER_VERSION,
    isShopItem: true,
    merchant: offer?.merchant || item.brand || null,
    canonicalProductId: offer?.canonicalProductId || String(item.id),
    variantId: offerFresh && offer.variantId ? offer.variantId : null,
    targetMarket,
    purchaseUrl: purchaseHost ? purchaseUrl : null,
    purchaseUrlValid: Boolean(purchaseHost),
    price: priceValid ? Number(item.price) : null,
    currency: priceValid ? String(item.currency) : null,
    priceFreshness: item.priceFreshness || freshness,
    observedAt: offerFresh ? offer.observedAt : (observedAt ? String(observedAt) : null),
    refreshBefore: offerFresh ? (offer.refreshBefore || null) : null,
    sizeStockStatus: offerFresh ? (offer.sizeStockStatus || 'unverified') : 'unverified',
    availableSizesCount: availableSizes.length,
    ...(requestedSize != null ? { requestedSize: String(requestedSize), requestedSizeStatus } : {}),
    deliveryEligible,
    // Checkout reservation / final cart stock is never verified solely from catalog/HTML GET.
    checkoutAvailability: 'unverified',
    imagePermissions: imagePerms,
    offerStatus,
  };
}

export function summarizeDeckOffers(outfits) {
  let shopItems = 0, merchantReportedLive = 0, unverifiedListings = 0;
  let freshOrRecentPrices = 0, staleOrUnknownPrices = 0;
  let uzDeliveryVerified = 0, imageInferencePermitted = 0;
  const seen = new Set();
  for (const outfit of outfits || []) {
    for (const item of outfit?.items || []) {
      if (!item || item.isShopItem !== true) continue;
      const key = String(item.id || '');
      if (seen.has(key)) continue;
      seen.add(key);
      shopItems += 1;
      const ov = item.offerVerification || verifyMerchantOffer(item);
      if (ov.offerStatus === 'merchant_reported_live') merchantReportedLive += 1;
      else unverifiedListings += 1;
      if (ov.priceFreshness === 'fresh' || ov.priceFreshness === 'recent') freshOrRecentPrices += 1;
      else staleOrUnknownPrices += 1;
      if (ov.deliveryEligible === true && ov.targetMarket === 'UZ') uzDeliveryVerified += 1;
      if (ov.imagePermissions?.externalInferenceAllowed) imageInferencePermitted += 1;
    }
  }
  return {
    version: MERCHANT_OFFER_VERSION,
    uniqueShopItems: shopItems,
    merchantReportedLive,
    unverifiedListings,
    freshOrRecentPrices,
    staleOrUnknownPrices,
    uzDeliveryVerified,
    imageInferencePermitted,
    checkoutGuaranteed: false,
  };
}
