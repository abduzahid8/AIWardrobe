/**
 * Curate a review-only Uzbekistan footwear batch.
 *
 * No Supabase credentials, DB writes, image downloads, affiliate assumptions,
 * paid AI calls, or inference dispatch. Source observations are tracked
 * separately from confirmed, size-specific shoppable merchant offers.
 */
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { fetchFrSizeEvidence, FR_PRODUCT_INFO_ENDPOINT, FR_VARIANT_SNAPSHOT_MAX_AGE_MS } from './fr_variant_stock.mjs';

export const ALLOWED_BRANDS = Object.freeze(['Tommy Hilfiger', 'Zara', 'MANGO', 'Massimo Dutti']);
export const DISALLOWED_BRANDS = Object.freeze(['ECCO', 'Cabani', 'Basconi']);
export const FR_MENS_SHOES_URL = 'https://frgroup.uz/ru/store/man/obuv/tommy-hilfiger';
export const MAX_FR_DETAIL_PAGES = 12;
const SAFE_URL = (x, allowed) => {
  try {
    const u = new URL(x);
    return u.protocol === 'https:' && !u.username && !u.password && !u.port &&
      allowed.includes(u.hostname.toLowerCase());
  } catch { return false; }
};
const sha = value => createHash('sha256').update(String(value)).digest('hex');
const isValidMoney = value => Number.isSafeInteger(value) && value > 0 && value < 50_000_000;
const b = x => String(x || '').trim();

export function parseFrStructuredProducts(html) {
  if (typeof html !== 'string' || html.length < 100) return [];
  const entries = [];
  // The store emits both product-list ItemList and individual Product JSON-LD.
  for (const m of html.matchAll(/<script[^>]+type=["']application\/ld\+json["'][^>]*>([\s\S]*?)<\/script>/gi)) {
    let parsed;
    try { parsed = JSON.parse(m[1]); } catch { continue; }
    if (parsed?.['@type'] === 'Product') entries.push(parsed);
    else if (Array.isArray(parsed?.itemListElement)) {
      for (const e of parsed.itemListElement) {
        if (e?.item?.['@type'] === 'Product') entries.push(e.item);
      }
    }
  }
  return entries;
}

export function frProductFamily(name) {
  const s = b(name).toLowerCase();
  // A runner with chukka branding is not a dress boot; espadrilles are not
  // a priority in the leather dress-shoe shortage we are addressing.
  if (/runner|high.?top|хайтоп|espadrille|эспадриль/.test(s)) return null;
  if (s.includes('лофер') || s.includes('loafer')) {
    return /espadrille|эспадриль/.test(s) ? 'espadrille' : 'loafer';
  }
  if (/chelsea|челси/.test(s)) return 'chelsea';
  if (/chukka|чакка/.test(s)) return 'chukka';
  if (/derby|дерби/.test(s)) return 'derby';
  if (/оксфорд|oxford/.test(s)) return 'oxford';
  if (/топсайдер|boat shoe/.test(s)) return 'boat-shoe';
  return null;
}

export function projectFrCandidate(product, { observedAt, evidenceHash }) {
  const name = b(product?.name);
  const url = b(product?.offers?.url || product?.url);
  const image = Array.isArray(product?.image) ? product.image[0] : product?.image;
  const sku = b(product?.sku || product?.offers?.sku);
  const brand = b(typeof product?.brand === 'object' ? product.brand.name : product?.brand);
  const family = frProductFamily(name);
  const price = Number(product?.offers?.price);
  if (brand !== 'Tommy Hilfiger' || !family || !sku || !/^[A-Z0-9-]{5,40}$/.test(sku) ||
      !SAFE_URL(url,['frgroup.uz']) || !url.includes('/ru/store/') ||
      !SAFE_URL(image,['media.frgroup.uz']) || !isValidMoney(price) ||
      product?.offers?.priceCurrency !== 'UZS') return null;
  // Product colors can share a vendor SKU: URL fingerprint is the unique
  // candidate identity; SKU is retained separately for merchant reconciliation.
  const id = 'fruz-th-' + sha(url).slice(0,20);
  const observed = new Date(observedAt);
  if (!Number.isFinite(+observed)) throw new Error('invalid_observation_time');
  const listingInStock = String(product?.offers?.availability || '').endsWith('/InStock');
  const shippingToUz = product?.offers?.shippingDetails?.shippingDestination?.addressCountry === 'UZ';
  return {
    id, brand, name, family, price, currency: 'UZS',
    productUrl: url, imageUrl: String(image), merchant: 'FR Group',
    merchantSku: sku,
    source: 'frgroup-uz-mens-shoes',
    observedAt: observed.toISOString(),
    evidenceSha256: evidenceHash,
    priceEvidence: 'direct_merchant_product_jsonld',
    listingInStock, shippingToUz,
    sizeStockStatus: 'unverified', deliveredTotalStatus: 'unverified',
    imageDisplayRights: 'unverified', imageNvidiaRights: 'not_authorized',
    stage: 'review_required',
  };
}

export async function collectFrCandidates({ fetchImpl = fetch, observedAt = new Date().toISOString() } = {}) {
  const get = async (url) => {
    const response = await fetchImpl(url, {
      headers: { 'User-Agent': 'AIWardrobeCatalogReview/1.0' },
      signal: AbortSignal.timeout(8000), redirect: 'error',
    });
    if (!response.ok) throw new Error('merchant_http_' + response.status);
    const html = await response.text();
    if (!html.includes('application/ld+json')) throw new Error('merchant_missing_product_evidence');
    return { html, evidenceHash: sha(html) };
  };
  const root = await get(FR_MENS_SHOES_URL);
  const candidates = parseFrStructuredProducts(root.html).filter(x =>
    b(x.brand) === 'Tommy Hilfiger' && frProductFamily(x.name) &&
    SAFE_URL(x.url, ['frgroup.uz']) && String(x.url).includes('/ru/store/'));
  const urls = [...new Set(candidates.map(x => x.url))].slice(0, MAX_FR_DETAIL_PAGES);
  const rows = [];
  // Small serial read; do not hammer merchant infrastructure.
  for (const url of urls) {
    const page = await get(url);
    const match = parseFrStructuredProducts(page.html).find(x =>
      b(x.offers?.url || x.url) === url && b(x.sku || x.offers?.sku));
    const candidate = projectFrCandidate(match, {
      observedAt, evidenceHash: page.evidenceHash,
    });
    if (candidate) {
      // The listing's JSON-LD InStock alone does not establish any usable EU
      // size. Only the public, model-ID-bound merchant variant endpoint can
      // provide a size-specific *claim*, and it must match the page price.
      let variantEvidence = null;
      try {
        variantEvidence = await fetchFrSizeEvidence({
          html: page.html, expectedPrice: candidate.price,
          fetchImpl, collectedAt: observedAt,
        });
      } catch {
        // Fail safely: preserve the product for review, but do not assert
        // size availability or change the last-observed merchant price.
      }
      rows.push(variantEvidence ? {
        ...candidate, sizeStockStatus: 'merchant_reported_by_size',
        variantEvidence,
      } : candidate);
    }
  }
  return rows;
}

const MANUAL_REVIEW = [
  // These are official Uzbekistan *product pages* verified in public search
  // on 2026-11-13. Direct merchant details (photography, price, size-specific
  // stock) were not read via this tool: do not turn search snippets into a
  // current merchant offer or a dated shop_catalog observation.
  { brand: 'Zara', family: 'loafer', name: 'Leather strap loafers · black',
    price: 1499900, productUrl: 'https://www.zara.com/uz/ru/кожаные-лоферы-с-планкои-p12686720.html?v1=545446972',
    merchantSku: '2686/720/800', evidence: 'turn652910search3' },
  { brand: 'Zara', family: 'loafer', name: 'Leather textured strap loafers · burgundy',
    price: 999900, productUrl: 'https://www.zara.com/uz/ru/кожаные-лоферы-с-планкои-p12685824.html',
    merchantSku: '2685/824/624', evidence: 'turn652910search5' },
  { brand: 'Zara', family: 'loafer', name: 'Leather lug sole loafers · black',
    price: 1099900, productUrl: 'https://www.zara.com/uz/ru/кожаные-лоферы-на-рифленои-подошве-с-планкои-p12666820.html',
    merchantSku: '2666/820/800', evidence: 'turn652910search4' },
  { brand: 'Zara', family: 'oxford', name: 'Leather Oxford shoes · black',
    price: 2099900, productUrl: 'https://www.zara.com/uz/ru/кожаные-туфли-p12435821.html',
    merchantSku: '2435/821/800', evidence: 'turn652910search8' },
  { brand: 'Zara', family: 'derby', name: 'Leather Derby shoes · brown',
    price: 1499900, productUrl: 'https://www.zara.com/uz/ru/кожаные-туфли-p12492820.html',
    merchantSku: '2492/820/700', evidence: 'turn652910search11' },
  { brand: 'MANGO', family: 'loafer', name: 'Gathered leather loafers · black',
    price: 1299000, productUrl: 'https://shop.mango.com/uz/en/p/men/shoes/leather/gathered-leather-loafers/47091370/99/00',
    merchantSku: '47091370-99-00', evidence: 'turn834633search6' },
  { brand: 'MANGO', family: 'derby', name: 'Leather suit shoes · cognac suede',
    price: 1799000, productUrl: 'https://shop.mango.com/uz/en/p/men/shoes/formal-shoes/leather-suit-shoes/27094408/CG/00',
    merchantSku: '27094408-CG-00', evidence: 'turn834633search7' },
  { brand: 'MANGO', family: 'chelsea', name: 'Chelsea leather ankle boots with track sole · black',
    price: 2299000, productUrl: 'https://shop.mango.com/uz/en/p/men/shoes/boots-and-ankle-boots/chelsea-leather-ankle-boots-with-track-sole/47051490/99/00',
    merchantSku: '47051490-99-00', evidence: 'turn471971search0' },
  { brand: 'Massimo Dutti', family: 'oxford', name: 'Formal leather Oxford shoes · brown',
    price: 3000000, productUrl: 'https://www.massimodutti.com/uz/rasmiy-oksford-poyabzali-l12701451',
    merchantSku: '2701/451', evidence: 'turn834633search4' },
];

export function manualReviewCandidates() {
  return MANUAL_REVIEW.map(row => ({
    ...row, id: 'uzreview-' + sha(row.brand + ':' + row.merchantSku + ':' + row.productUrl).slice(0,20),
    merchant: row.brand === 'MANGO' ? 'Mango Uzbekistan' : row.brand + ' Uzbekistan',
    currency: 'UZS', source: 'uz-official-product-search-review',
    observedAt: null, evidenceSha256: null,
    priceEvidence: 'official_search_result_needs_direct_recheck',
    listingInStock: null, shippingToUz: null,
    sizeStockStatus: 'unverified', deliveredTotalStatus: 'unverified',
    imageUrl: null, imageDisplayRights: 'unverified',
    imageNvidiaRights: 'not_authorized', stage: 'needs_direct_merchant_recheck',
  }));
}

export function reconcileExistingCatalog(items, existingRows) {
  const byUrl = new Map();
  for (const row of existingRows || []) {
    if (!row || !SAFE_URL(row.source_url, ['frgroup.uz']) ||
        typeof row.id !== 'string' || byUrl.has(row.source_url)) {
      throw new Error('invalid_existing_catalog_snapshot');
    }
    byUrl.set(row.source_url, row);
  }
  return items.map(item => {
    const exists = byUrl.get(item.productUrl);
    if (!exists) return { ...item, inventoryAction: 'new_review_only', existingCatalogId: null };
    if (exists.brand !== item.brand || exists.is_active !== false || !isValidMoney(exists.old_price_uzs)) {
      throw new Error('existing_catalog_conflict_requires_live_recheck');
    }
    return {
      ...item, inventoryAction: 'review_archived_existing_row',
      existingCatalogId: exists.id, oldPriceUZS: exists.old_price_uzs,
      // Re-enabling a historical inactive row is not implied by an HTTP 200
      // or Schema.org InStock: exact sizes, rights and merchant delivery
      // remain unverified.
    };
  });
}

export function projectInactiveCatalogPreview(items) {
  return items.filter(x => x.stage === 'review_required' &&
      x.inventoryAction === 'new_review_only').map((item, i) => ({
    id: item.id, brand: item.brand, name: item.name, price: item.price,
    currency: 'UZS', category: 'shoes', garment_type: 'shoes',
    image_url: item.imageUrl, source_url: item.productUrl,
    description: 'Merchant product record for offline review only. Site-reported size quantities may expire; checkout, delivery charge, image display and inference rights remain unverified.',
    source: 'frgroup-uz-review-not-active', external_id: item.merchantSku,
    is_active: false, sort_order: i, last_seen_at: item.observedAt,
  }));
}

export function validateReviewBatch(items) {
  if (!Array.isArray(items) || !items.length || items.length > 100) throw new Error('invalid_batch_size');
  const seenIds = new Set(), seenUrls = new Set();
  const errors = [];
  for (const row of items) {
    if (!ALLOWED_BRANDS.includes(row.brand) || DISALLOWED_BRANDS.includes(row.brand)) errors.push('brand_not_allowed:' + row.id);
    if (!/^(fruz-th-|uzreview-)[a-f0-9]{20}$/.test(row.id)) errors.push('invalid_id');
    if (seenIds.has(row.id) || seenUrls.has(row.productUrl)) errors.push('duplicate_id_or_url:' + row.id);
    seenIds.add(row.id); seenUrls.add(row.productUrl);
    if (!isValidMoney(row.price) || row.currency !== 'UZS') errors.push('invalid_money:' + row.id);
    const hosts = row.brand === 'Tommy Hilfiger' ? ['frgroup.uz'] :
      row.brand === 'MANGO' ? ['shop.mango.com'] :
        row.brand === 'Zara' ? ['www.zara.com'] : ['www.massimodutti.com'];
    if (!SAFE_URL(row.productUrl, hosts)) errors.push('untrusted_url:' + row.id);
    if (row.imageUrl && !SAFE_URL(row.imageUrl, ['media.frgroup.uz'])) errors.push('untrusted_image_url:' + row.id);
    if (row.stage === 'review_required') {
      if (!row.observedAt || !/^[a-f0-9]{64}$/.test(row.evidenceSha256 || '') ||
          !row.imageUrl || row.priceEvidence !== 'direct_merchant_product_jsonld') errors.push('missing_live_evidence:' + row.id);
    } else if (row.stage === 'needs_direct_merchant_recheck') {
      if (row.observedAt !== null || row.imageUrl !== null || row.listingInStock !== null ||
          row.shippingToUz !== null || row.priceEvidence !== 'official_search_result_needs_direct_recheck')
        errors.push('nonlive_row_claiming_live:' + row.id);
    } else errors.push('unknown_stage:' + row.id);
    if (row.sizeStockStatus === 'merchant_reported_by_size') {
      const v = row.variantEvidence;
      if (row.stage !== 'review_required' ||
        v?.status !== 'merchant_reported_by_size' ||
        v?.exactCheckoutAvailability !== 'unverified' ||
        v?.deliveryCharge !== 'unverified' ||
        v?.priceMatched !== true ||
        !Number.isSafeInteger(v.modelId) || v.modelId <= 0 ||
        v.merchant !== 'FR Group' || v.publicEndpoint !== FR_PRODUCT_INFO_ENDPOINT ||
        !/^[a-f0-9]{64}$/.test(v.responseSha256 || '') ||
        v.checkedAt !== row.observedAt ||
        !Number.isFinite(Date.parse(v.checkedAt)) ||
        v.refreshBefore !== new Date(Date.parse(v.checkedAt) + FR_VARIANT_SNAPSHOT_MAX_AGE_MS).toISOString() ||
        !Array.isArray(v.availableSizes) || !v.availableSizes.length ||
        v.availableSizes.length > 55 ||
        new Set(v.availableSizes.map(s => s.variantId)).size !== v.availableSizes.length ||
        new Set(v.availableSizes.map(s => s.euSize)).size !== v.availableSizes.length ||
        v.availableSizes.some(s => !/^(3[8-9]|4[0-9]|50)(?:[.]5)?$/.test(s.euSize) ||
          !Number.isSafeInteger(s.variantId) || s.variantId <= 0 ||
          !Number.isSafeInteger(s.merchantReportedQty) || s.merchantReportedQty < 1 || s.merchantReportedQty > 99) ||
        v.totalMerchantReportedUnits !== v.availableSizes.reduce((total, s) => total + s.merchantReportedQty, 0)) {
        errors.push('invalid_size_evidence:' + row.id);
      }
    } else if (row.sizeStockStatus !== 'unverified' || row.variantEvidence) {
      errors.push('untrusted_size_status:' + row.id);
    }
    if (row.deliveredTotalStatus !== 'unverified' ||
      row.imageNvidiaRights !== 'not_authorized' || row.imageDisplayRights !== 'unverified')
      errors.push('inaccurate_purchase_or_image_rights:' + row.id);
  }
  if (errors.length) throw new Error('merchant_batch_rejected: ' + errors.slice(0,20).join(','));
  return { checked: items.length, byBrand: Object.fromEntries(ALLOWED_BRANDS.map(name =>
    [name, items.filter(x => x.brand === name).length])), readyToPublish: 0,
    withDirectMerchantEvidence: items.filter(x => x.stage === 'review_required').length,
    merchantReportedSizeEvidence: items.filter(x =>
      x.sizeStockStatus === 'merchant_reported_by_size').length,
    needDirectRecheck: items.filter(x => x.stage === 'needs_direct_merchant_recheck').length,
    requiresOfferAndImageRights: items.length };
}

async function main() {
  const fr = await collectFrCandidates();
  const dataDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../benchmarks/stylist6');
  const existing = JSON.parse(fs.readFileSync(path.join(dataDir, 'FOOTWEAR_UZ_EXISTING_INACTIVE_20261113.json'), 'utf8'));
  const all = reconcileExistingCatalog([...fr, ...manualReviewCandidates()], existing.matches)
    .sort((a,b) => a.brand.localeCompare(b.brand) || a.family.localeCompare(b.family) || a.id.localeCompare(b.id));
  const report = validateReviewBatch(all);
  const inactiveShopRows = projectInactiveCatalogPreview(all);
  const artifact = {
    version: 1, generatedAt: new Date().toISOString(),
    note: 'Review-only. Some site-reported, time-limited EU size quantities; no confirmed checkout, landed delivery price or image display/inference rights. NEVER upsert directly to shop_catalog.',
    dataSources: [FR_MENS_SHOES_URL, 'official brand UZ product search pages'],
    ...report,
    previouslyArchived: all.filter(row => row.inventoryAction === 'review_archived_existing_row').length,
    netNewCandidates: all.filter(row => row.inventoryAction === 'new_review_only').length,
    eligibleInactiveShopPreview: inactiveShopRows.length,
    reportedSizeVariantCount: all.reduce((n, row) => n + (row.variantEvidence?.availableSizes?.length || 0), 0),
    reportedUnitsInSnapshot: all.reduce((n, row) => n + (row.variantEvidence?.totalMerchantReportedUnits || 0), 0),
    items: all,
  };
  const output = path.join(dataDir,'FOOTWEAR_UZ_CANDIDATES_20261113.json');
  const previewPath = path.join(dataDir,'FOOTWEAR_UZ_INACTIVE_PREVIEW_20261113.json');
  fs.writeFileSync(output, JSON.stringify(artifact, null, 2) + '\n');
  fs.writeFileSync(previewPath, JSON.stringify({
    version: 1, note: 'INACTIVE PREVIEW ONLY; never run an unrestricted upsert or bulk reactivate.',
    rows: inactiveShopRows,
  }, null, 2) + '\n');
  console.log(JSON.stringify({file: output, previewPath,
    byBrand: report.byBrand, checked: report.checked,
    previouslyArchived: artifact.previouslyArchived, netNewCandidates: artifact.netNewCandidates,
    eligibleInactiveShopPreview: artifact.eligibleInactiveShopPreview,
    readyToPublish: 0}, null, 2));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  main().catch(e => { console.error('candidate_collection_failed:' + e.message); process.exitCode = 1; });
}
