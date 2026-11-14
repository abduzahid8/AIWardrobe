/**
 * Collects live, verified Uzbekistan men's footwear supply evidence across FR Group
 * categories (/tufli, /botinki-i-sapogi, /krossovki) and cross-references the
 * existing active Supabase shoes catalog without mutating any DB table.
 *
 * Output (.private/uz-footwear-verified-supply-20261114.json) keeps all unverified
 * fields honest: exact checkout, delivery fee, and third-party image/inference
 * rights are NEVER fabricated.
 */
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { DISALLOWED_BRANDS, parseFrStructuredProducts } from './curate_uz_footwear.mjs';
import { fetchFrSizeEvidence } from './fr_variant_stock.mjs';
import { verifyMerchantOffer } from '../../backend/stylist6/merchant_offers.mjs';
import { isInferenceAuthorized } from './image_access.mjs';

export const FR_CATEGORY_SLUGS = Object.freeze(['tufli', 'botinki-i-sapogi', 'krossovki']);
const FR_BASE = 'https://frgroup.uz/ru/store/man/obuv';
const MAX_PAGES_PER_CATEGORY = 15;
const DETAIL_CONCURRENCY = 4;

const sha = (v) => createHash('sha256').update(String(v)).digest('hex');
const text = (v) => String(v || '').trim();

export const SAFE_HTTPS_HOST = (raw, hosts) => {
  try {
    const u = new URL(raw);
    return u.protocol === 'https:' && !u.username && !u.password && !u.port
      && hosts.includes(u.hostname.toLowerCase());
  } catch {
    return false;
  }
};

export function classifyFrFootwearFamily(name, categorySlug) {
  const s = text(name).toLowerCase();
  if (/espadrille|эспадриль/.test(s)) return 'espadrille';
  if (/loafer|лофер/.test(s)) return 'loafer';
  if (/oxford|оксфорд/.test(s)) return 'oxford';
  if (/derby|дерби/.test(s)) return 'derby';
  if (/monk|монк/.test(s)) return 'monk';
  if (/chelsea|челси/.test(s)) return 'chelsea';
  if (/chukka|чакка/.test(s)) return 'chukka';
  if (/runner|sneaker|кроссов|кед|trainer/.test(s)) return 'sneaker';
  if (categorySlug === 'tufli') return 'dress_shoe';
  if (categorySlug === 'botinki-i-sapogi') return 'boot';
  return 'sneaker';
}

export function normalizeFrShoeOffer(product, { categorySlug, observedAt, evidenceHash, variantEvidence }) {
  const name = text(product?.name);
  const url = text(product?.offers?.url || product?.url);
  const imageUrl = text(Array.isArray(product?.image) ? product.image[0] : product?.image);
  const sku = text(product?.sku || product?.offers?.sku);
  const brand = text(typeof product?.brand === 'object' ? product.brand.name : product?.brand);
  const price = Number(product?.offers?.price);
  const currency = text(product?.offers?.priceCurrency);
  if (!name || !brand || DISALLOWED_BRANDS.includes(brand)) return null;
  if (!sku || !SAFE_HTTPS_HOST(url, ['frgroup.uz']) || !url.includes('/ru/store/')) return null;
  if (!SAFE_HTTPS_HOST(imageUrl, ['media.frgroup.uz'])) return null;
  if (!Number.isFinite(price) || price <= 0 || currency !== 'UZS') return null;

  const listingInStock = String(product?.offers?.availability || '').endsWith('/InStock');
  const deliveryUz = product?.offers?.shippingDetails?.shippingDestination?.addressCountry === 'UZ';
  const availableSizes = variantEvidence?.availableSizes || [];
  const totalUnits = Number(variantEvidence?.totalMerchantReportedUnits || 0);
  const id = `fruz-${sha(url).slice(0, 20)}`;
  const family = classifyFrFootwearFamily(name, categorySlug);

  const observedMs = Date.parse(observedAt);
  const registryEntry = variantEvidence
    ? {
        merchant: 'FR Group',
        canonicalProductId: sku,
        variantId: availableSizes[0]?.variantId ? String(availableSizes[0].variantId) : sku,
        availableSizes,
        observedAt,
        refreshBefore: variantEvidence.refreshBefore || new Date(observedMs + 30 * 60_000).toISOString(),
        sizeStockStatus: 'merchant_reported_by_size',
        price,
        currency,
        targetMarket: deliveryUz ? 'UZ' : 'UNVERIFIED',
        deliveryEligible: deliveryUz,
      }
    : null;
  const offer = verifyMerchantOffer(
    {
      id,
      isShopItem: true,
      brand,
      name,
      price,
      currency,
      shopUrl: url,
      imageUrl,
      lastSeenAt: observedAt,
    },
    {
      offerRegistry: registryEntry ? new Map([[id, registryEntry]]) : null,
      targetMarket: 'UZ',
      now: Number.isFinite(observedMs) ? observedMs : Date.now(),
    },
  );

  return {
    id,
    brand,
    name,
    categorySlug,
    family,
    price,
    currency: 'UZS',
    productUrl: url,
    imageUrl,
    merchant: 'FR Group',
    merchantSku: sku,
    observedAt,
    evidenceSha256: evidenceHash,
    listingInStock,
    deliveryUz,
    sizeStockStatus: variantEvidence ? 'merchant_reported_by_size' : 'unverified',
    availableEuSizes: availableSizes.map((s) => s.euSize),
    reportedUnits: totalUnits,
    imageNvidiaRights: 'not_authorized',
    imageDisplayRights: 'unverified',
    deliveredTotalStatus: 'unverified',
    exactCheckoutAvailability: 'unverified',
    is_active: false,
    offerVerification: offer,
  };
}

function loadEnv() {
  const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
  const envFile = path.join(repoRoot, '.env');
  const out = { ...process.env };
  if (fs.existsSync(envFile)) {
    for (const line of fs.readFileSync(envFile, 'utf8').split(/\r?\n/)) {
      const m = /^([A-Z0-9_]+)=(.*)$/.exec(line.trim());
      if (m && !out[m[1]]) out[m[1]] = m[2].replace(/^['"]|['"]$/g, '');
    }
  }
  return out;
}

async function fetchActiveShoesSummary(repoRoot, env) {
  let rows = [];
  let source = 'none';
  if (env.EXPO_PUBLIC_SUPABASE_URL && env.EXPO_PUBLIC_SUPABASE_ANON_KEY) {
    const u = `${env.EXPO_PUBLIC_SUPABASE_URL.replace(/\/+$/, '')}/rest/v1/shop_catalog?select=id,name,brand,price,currency,image_url,source_url,is_active&is_active=eq.true&garment_type=eq.shoes&limit=500`;
    const res = await fetch(u, {
      headers: {
        apikey: env.EXPO_PUBLIC_SUPABASE_ANON_KEY,
        Authorization: `Bearer ${env.EXPO_PUBLIC_SUPABASE_ANON_KEY}`,
      },
      signal: AbortSignal.timeout(10_000),
    });
    if (res.ok) {
      rows = await res.json();
      source = 'supabase_rest_readonly';
    }
  }
  if (!rows.length) {
    const snapPath = path.join(repoRoot, '.private/catalog-uz-20261113.json');
    if (fs.existsSync(snapPath)) {
      const snap = JSON.parse(fs.readFileSync(snapPath, 'utf8'));
      const list = Array.isArray(snap) ? snap : snap.items || snap.rows || [];
      rows = list.filter((r) => (r.garment_type === 'shoes' || r.macroCategory === 'shoes') && r.is_active !== false);
      source = 'local_catalog_snapshot';
    }
  }
  const byCurrency = {};
  const byFamily = {};
  let allowlistedImages = 0;
  for (const r of rows) {
    const ccy = r.currency || 'UNKNOWN';
    byCurrency[ccy] = (byCurrency[ccy] || 0) + 1;
    const fam = classifyFrFootwearFamily(r.name, '');
    byFamily[fam] = (byFamily[fam] || 0) + 1;
    if (isInferenceAuthorized(r.image_url || r.imageUrl)) allowlistedImages += 1;
  }
  return {
    source,
    totalActive: rows.length,
    allowlistedImages,
    byCurrency,
    byFamily,
    sourceUrls: new Set(rows.map((r) => text(r.source_url || r.shopUrl)).filter(Boolean)),
  };
}

export async function runCollector({ fetchImpl = fetch, writeOutput = true } = {}) {
  const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
  const env = loadEnv();
  const observedAt = new Date().toISOString();

  const getHtml = async (url) => {
    const res = await fetchImpl(url, {
      headers: { 'User-Agent': 'AIWardrobeCatalogReview/1.0' },
      signal: AbortSignal.timeout(10_000),
      redirect: 'error',
    });
    if (!res.ok) throw new Error(`http_${res.status}:${url}`);
    const html = await res.text();
    return { html, evidenceHash: sha(html) };
  };

  // 1. Discover unique URLs across all 3 categories.
  const discovered = new Map();
  const categoryStats = {};
  for (const slug of FR_CATEGORY_SLUGS) {
    const seenInCat = new Set();
    for (let page = 1; page <= MAX_PAGES_PER_CATEGORY; page += 1) {
      const listUrl = `${FR_BASE}/${slug}${page > 1 ? `?page=${page}` : ''}`;
      let html;
      try {
        ({ html } = await getHtml(listUrl));
      } catch (err) {
        if (page > 1 && String(err?.message || '').startsWith('http_404:')) break;
        throw err;
      }
      const products = parseFrStructuredProducts(html);
      let added = 0;
      for (const p of products) {
        const purl = text(p?.offers?.url || p?.url);
        if (!SAFE_HTTPS_HOST(purl, ['frgroup.uz']) || !purl.includes('/ru/store/')) continue;
        if (!seenInCat.has(purl)) {
          seenInCat.add(purl);
          added += 1;
        }
        if (!discovered.has(purl)) discovered.set(purl, { url: purl, categorySlug: slug });
      }
      if (added === 0) break;
    }
    categoryStats[slug] = seenInCat.size;
  }

  // 2. Polite bounded-concurrency detail + variant stock verification.
  const entries = [...discovered.values()];
  const items = [];
  const errors = [];
  for (let i = 0; i < entries.length; i += DETAIL_CONCURRENCY) {
    const batch = entries.slice(i, i + DETAIL_CONCURRENCY);
    const batchResults = await Promise.all(batch.map(async ({ url, categorySlug }) => {
      try {
        const page = await getHtml(url);
        const match = parseFrStructuredProducts(page.html).find(
          (x) => text(x?.offers?.url || x?.url) === url && text(x?.sku || x?.offers?.sku),
        );
        if (!match) return null;
        const price = Number(match?.offers?.price);
        let variantEvidence = null;
        if (Number.isFinite(price) && price > 0) {
          try {
            variantEvidence = await fetchFrSizeEvidence({
              html: page.html,
              expectedPrice: price,
              fetchImpl,
              collectedAt: observedAt,
            });
          } catch {
            variantEvidence = null;
          }
        }
        return normalizeFrShoeOffer(match, {
          categorySlug,
          observedAt,
          evidenceHash: page.evidenceHash,
          variantEvidence,
        });
      } catch (err) {
        errors.push({ url, error: err instanceof Error ? err.message : String(err) });
        return null;
      }
    }));
    for (const item of batchResults) if (item) items.push(item);
  }

  const activeCatalog = await fetchActiveShoesSummary(repoRoot, env);
  const archivedPath = path.join(repoRoot, 'benchmarks/stylist6/FOOTWEAR_UZ_EXISTING_INACTIVE_20261113.json');
  const archivedUrls = fs.existsSync(archivedPath)
    ? new Set((JSON.parse(fs.readFileSync(archivedPath, 'utf8')).matches || []).map((m) => text(m.source_url)).filter(Boolean))
    : new Set();

  let overlapActive = 0;
  let overlapArchived = 0;
  for (const item of items) {
    if (activeCatalog.sourceUrls.has(item.productUrl)) overlapActive += 1;
    if (archivedUrls.has(item.productUrl)) overlapArchived += 1;
  }

  const netNewFr = items.filter((item) => !activeCatalog.sourceUrls.has(item.productUrl));
  const withSizeStock = netNewFr.filter((item) => item.sizeStockStatus === 'merchant_reported_by_size');
  const inStockWithSizes = withSizeStock.filter((item) => item.availableEuSizes.length > 0 && item.deliveryUz);

  const byCategory = {};
  const byFamily = {};
  const byBrand = {};
  for (const item of items) {
    byCategory[item.categorySlug] = (byCategory[item.categorySlug] || 0) + 1;
    byFamily[item.family] = (byFamily[item.family] || 0) + 1;
    byBrand[item.brand] = (byBrand[item.brand] || 0) + 1;
  }

  const summary = {
    version: 1,
    generatedAt: observedAt,
    note: 'Read-only live merchant observation artifact. Staging rows remain is_active=false; external inference, checkout fee, and display rights remain unverified.',
    existingActiveCatalogShoes: {
      source: activeCatalog.source,
      totalActive: activeCatalog.totalActive,
      allowlistedImages: activeCatalog.allowlistedImages,
      byCurrency: activeCatalog.byCurrency,
      byFamily: activeCatalog.byFamily,
    },
    liveFrGroupSupply: {
      categoryListCounts: categoryStats,
      verifiedProductPages: items.length,
      fetchErrors: errors.length,
      overlapWithActiveCatalog: overlapActive,
      overlapWithArchivedRows: overlapArchived,
      netNewDistinctOptions: netNewFr.length,
      withSizeStockEvidence: withSizeStock.length,
      inStockWithUzDeliveryConfirmed: inStockWithSizes.length,
      totalReportedUnits: items.reduce((sum, x) => sum + x.reportedUnits, 0),
      byCategory,
      byFamily,
      byBrand,
    },
    combinedSupplyTotals: {
      existingActiveShoes: activeCatalog.totalActive,
      netNewLiveFrGroupShoes: netNewFr.length,
      totalDistinctVerifiedFootwearLeads: activeCatalog.totalActive + netNewFr.length,
      totalWithVerifiedActiveOrSizeStock: activeCatalog.totalActive + inStockWithSizes.length,
      meets200InitialSupplyObjective: (activeCatalog.totalActive + netNewFr.length) >= 200,
    },
    limitations: [
      'imageNvidiaRights=not_authorized for newly collected FR Group imagery pending owner media-rights review',
      'deliveredTotalStatus=unverified (merchant product endpoint reports item price, not shipping surcharge)',
      'exactCheckoutAvailability=unverified (FR Group public get-product-info reports live catalog quantities without cart reservation)',
    ],
  };

  if (writeOutput) {
    const privDir = path.join(repoRoot, '.private');
    fs.mkdirSync(privDir, { recursive: true });
    fs.writeFileSync(
      path.join(privDir, 'uz-footwear-verified-supply-20261114.json'),
      JSON.stringify({ ...summary, items }, null, 2) + '\n',
    );
    fs.writeFileSync(
      path.join(privDir, 'uz-footwear-staging-preview-20261114.json'),
      JSON.stringify({
        version: 1,
        note: 'INACTIVE PREVIEW ONLY (is_active=false). Requires owner rights clearance and DB migration authorization.',
        count: netNewFr.length,
        rows: netNewFr.map((item, idx) => ({
          id: item.id,
          brand: item.brand,
          name: item.name,
          price: item.price,
          currency: item.currency,
          category: 'shoes',
          garment_type: 'shoes',
          image_url: item.imageUrl,
          source_url: item.productUrl,
          source: 'frgroup-uz-supply-preview',
          external_id: item.merchantSku,
          is_active: false,
          sort_order: idx,
          last_seen_at: item.observedAt,
        })),
      }, null, 2) + '\n',
    );
  }

  return summary;
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  runCollector()
    .then((summary) => {
      console.log(JSON.stringify(summary, null, 2));
    })
    .catch((err) => {
      console.error(err);
      process.exitCode = 1;
    });
}
