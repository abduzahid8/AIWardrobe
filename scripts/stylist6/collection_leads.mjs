/**
 * Pure validation for a third-party public collection index.
 *
 * Merchant category pages can carry variant references and advertised prices
 * without a stable single-product URL, real stock, seller-side size/checkout
 * validation, or photo rights. These are SOURCING LEADS, not SKU offers.
 */
import { createHash } from 'node:crypto';

const SHA = x => createHash('sha256').update(String(x)).digest('hex');

export function validateCollectionLeads(batch) {
  const rejects = [];
  if (batch?.version !== 1 ||
    batch.brand !== 'MANGO' || batch.country !== 'UZ' || batch.currency !== 'UZS' ||
    batch.evidenceLevel !== 'indexed_official_collection_unverified_live' ||
    batch.sourceCollection !== 'https://shop.mango.com/uz/en/c/men/shoes/formal-shoes/ffd4d43e' ||
    !Array.isArray(batch.leads) || batch.leads.length > 20 || !batch.leads.length) {
    throw new Error('collection_scope_untrusted');
  }
  const allowedFamilies = new Set(['derby', 'oxford', 'monk-strap', 'formal-shoes', 'loafer', 'chelsea']);
  const allowedColors = new Set(['Black', 'Brown', 'Cognac']);
  const ids = new Set();
  const normalized = [];
  for (const [index, x] of batch.leads.entries()) {
    const key = x.merchantArticle + '|' + x.color;
    if (!/^[0-9]{8}-[0-9]{2}$/.test(x.merchantArticle || '') ||
      typeof x.name !== 'string' || !x.name.trim() || x.name.length > 130 ||
      !allowedFamilies.has(x.family) || !allowedColors.has(x.color) ||
      !Number.isSafeInteger(x.indexedPriceUZS) || x.indexedPriceUZS < 100000 ||
      x.indexedPriceUZS > 10000000 ||
      !Array.isArray(x.displayedSizeRangeEU) ||
      x.displayedSizeRangeEU.length > 15 || x.displayedSizeRangeEU.length < 1 ||
      new Set(x.displayedSizeRangeEU).size !== x.displayedSizeRangeEU.length ||
      x.displayedSizeRangeEU.some(size => !/^(3[8-9]|4[0-9]|50)$/.test(size)) ||
      ids.has(key)) {
      rejects.push('invalid_collection_reference:' + index);
      continue;
    }
    ids.add(key);
    normalized.push({
      leadId: 'mango-uz-collection-' + SHA(key).slice(0,20),
      brand: 'MANGO', merchantArticle: x.merchantArticle, color: x.color,
      name: x.name, family: x.family,
      indexedPriceUZS: x.indexedPriceUZS,
      displayedSizeRangeEU: x.displayedSizeRangeEU,
      sourceCollection: batch.sourceCollection,
      stage: 'collection_reference_only',
      productUrl: null,
      imageUrl: null,
      merchantVariantId: null,
      observedAt: null, // feed has no per-SKU observation; don't fabricate one
      sizeStockStatus: 'not_checked',
      checkoutStockStatus: 'unverified',
      deliveryTotalStatus: 'unverified',
      imageDisplayRights: 'unverified',
      imageNvidiaRights: 'not_authorized',
      isActive: false,
    });
  }
  if (rejects.length) throw new Error('invalid_collection_leads:' + rejects.join(','));
  return {
    leads: normalized, listedVariantReferences: normalized.length,
    uniqueMerchantArticles: new Set(normalized.map(x => x.merchantArticle)).size,
    readyToPublish: 0,
  };
}
