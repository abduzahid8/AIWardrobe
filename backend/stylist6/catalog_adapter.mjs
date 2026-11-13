/**
 * Committed-snapshot catalog adapter for the Stylist 6 offline pilot.
 *
 * Maps `data/shop_catalog_*.json` rows (real SKUs, prices, shop links and
 * photos) to the item shape consumed by `backend/stylist6/prototype.mjs`.
 *
 * Why this file exists: the pilot runs without live Supabase credentials, so
 * the `catalog_garment_intelligence` slot labels are unavailable. The slot
 * and subtype projection below is explicit, auditable and fully tested — it
 * is NOT a silent reclassification of the production catalog.
 *
 * Wearing-role projection (knitwear as a base layer): ~27 of the 46
 * reference looks wear a fine-knit mockneck/turtleneck/roll-neck/crewneck as
 * the BASE top under a second layer. Those garments belong in the `top`
 * slot with subtype `fine-knit` even when a text search would file a
 * "sweater" next to heavy outerwear. Cardigans stay `outerwear`. Every
 * projected item carries `projection` provenance so the pilot can count
 * native vs projected fills per role.
 */

import { createHash } from 'node:crypto';
import { assessCatalogCategory } from './catalog_category_integrity.mjs';

export const CATALOG_ADAPTER_VERSION = 'catalog-adapter-v2-category-evidence';

const has = (text, re) => re.test(text);

const COLOR_TABLE = [
  [/ivory/, 'ivory'], [/cream/, 'cream'], [/beige/, 'beige'],
  [/stone/, 'stone'], [/sand/, 'sand'], [/taupe/, 'taupe'],
  [/camel/, 'camel'], [/cognac/, 'cognac'], [/tan/, 'tan'],
  [/chocolate/, 'chocolate'], [/espresso/, 'espresso'],
  [/charcoal/, 'charcoal'], [/grey|gray/, 'grey'],
  [/navy/, 'navy'], [/sky|light blue|pale blue/, 'sky'],
  [/olive/, 'olive'], [/sage/, 'sage'], [/khaki/, 'khaki'],
  [/green/, 'green'], [/burgundy|bordeaux|wine/, 'burgundy'],
  [/brown/, 'brown'], [/black/, 'black'], [/white/, 'white'],
  [/blue/, 'blue'],
];

const TEXTURE_TABLE = [
  [/suede/, 'suede'], [/leather/, 'leather'], [/denim/, 'denim'],
  [/corduroy/, 'corduroy'], [/linen/, 'linen'], [/cashmere/, 'cashmere'],
  [/cable|ribbed|knit|sweater|jumper/, 'knit'], [/quilted|padded/, 'quilted'],
  [/wool/, 'wool'], [/cotton/, 'cotton'],
];

const OUTERWEAR_TABLE = [
  [/\bblazers?\b/, 'blazer'],
  [/\bcardigans?\b/, 'cardigan'],
  [/\bbombers?\b/, 'bomber'],
  [/\bcoats?\b|\bovercoats?\b|\btopcoats?\b|\btrenchs?\b/, 'coat'],
  [/\bovershirts?\b|\bshackets?\b/, 'overshirt'],
  [/\bpuffers?\b|\bquilted\b|\bgilets?\b|\bvests?\b|\bfield jackets?\b|\bbarn jackets?\b|\bchore\b|\btruckers?\b|\bblousons?\b/, 'jacket'],
  [/\bjackets?\b|\bparkas?\b|\bharrington\b|\bsafari\b/, 'jacket'],
];

const TOP_TABLE = [
  [/\bcardigans?\b/, null], // handled in outerwear; never a base top
  [/\bt-shirts?\b|\btees?\b|\bt shirts?\b/, 't-shirt'],
  [/\bpolos?\b|\bpoloshirts?\b/, 'polo'],
  [/\bknits?\b|\bsweaters?\b|\bturtlenecks?\b|\bmocknecks?\b|\bmock necks?\b|\broll[\s-]?necks?\b|\bhigh[\s-]?necks?\b|\bcrewnecks?\b|\bcrew necks?\b|\bjumpers?\b|\bpullovers?\b|\bfair[\s-]?isles?\b/, 'fine-knit'],
  [/\bdress shirts?\b|\bdress-shirts?\b/, 'dress-shirt'],
  [/\bshirts?\b|\boxfords?\b|\bbutton[\s-]?downs?\b|\bflannels?\b|\bpoplin\b|\blinen shirt\b/, 'shirt'],
];

const BOTTOM_TABLE = [
  [/\bjeans\b|\bdenim\b/, 'jeans'],
  [/\bchinos?\b/, 'chino'],
  [/\btailored\b|\bpleated\b|\bwool trousers\b|\bdress trousers?\b/, 'tailored-trouser'],
  [/\btrousers?\b|\bpants\b|\bslacks\b/, 'trouser'],
];

const SHOE_TABLE = [
  [/\bloafers?\b|\bpenny\b|\btassels?\b|\bhorsebits?\b|\bslippers?\b|\bmoccasins?\b|\bboat\b|\bdeck\b/, 'loafer'],
  [/\bchelsea\b/, 'chelsea'],
  [/\bchukkas?\b|\bboots?\b|\bankle\b/, 'boot'],
  [/\boxfords?\b|\bderby\b|\bderbies\b|\bbrogues?\b|\bwingtips?\b|\bmonks?\b|\bstraps?\b/, 'derby'],
  [/\bminimal\b|\bcourt\b|\blow[\s-]?tops?\b/, 'minimal-sneaker'],
  [/\bsneakers?\b|\btrainers?\b|\brunners?\b|\btennis\b/, 'sneaker'],
];

function firstMatch(table, text) {
  for (const [re, subtype] of table) {
    if (has(text, re)) return subtype;
  }
  return undefined;
}

export function inferColor(name) {
  const text = String(name || '').toLowerCase();
  for (const [re, color] of COLOR_TABLE) {
    if (has(text, re)) return color;
  }
  return 'unknown';
}

export function inferTexture(name) {
  const text = String(name || '').toLowerCase();
  for (const [re, texture] of TEXTURE_TABLE) {
    if (has(text, re)) return texture;
  }
  return 'smooth';
}

const isHttps = (value) => {
  try {
    const url = new URL(String(value));
    return url.protocol === 'https:' && !!url.hostname;
  } catch {
    return false;
  }
};

// Authoritative live taxonomy (`catalog_garment_intelligence` slot/subtype).
// Used ONLY when a row carries an explicit, valid live hint (read-only live
// export). Anything unrecognized falls through to the transparent projection
// below — never silently remapped.
const LIVE_SUBTYPES = {
  top: new Set(['t-shirt', 'polo', 'shirt', 'dress-shirt', 'fine-knit']),
  outerwear: new Set(['blazer', 'coat', 'jacket', 'overshirt', 'cardigan', 'bomber', 'pullover']),
  bottom: new Set(['jeans', 'chino', 'tailored-trouser', 'trouser']),
  shoes: new Set(['loafer', 'derby', 'boot', 'minimal-sneaker', 'sneaker', 'chelsea']),
};

export function liveHint(row) {
  const slot = String(row?.fashion_slot || '').toLowerCase();
  const subtype = String(row?.fashion_subtype || '').toLowerCase().replace(/_/g, '-');
  if (LIVE_SUBTYPES[slot]?.has(subtype)) return { macroCategory: slot, subtype };
  return null;
}

// A knit pullover the intelligence table files under outerwear is still worn
// as a BASE top in ~27/46 reference looks (verified on photo_3/photo_5).
// Without this wearing-role projection the 268 live outerwear/pullover items
// would be invisible to every S6 top role on a live replay. Cardigans never
// take this path.
// NOTE: the bare word "pullover" alone is NOT knit evidence (sport anoraks
// exist); require genuine knit/material wording.
const PULLOVER_BASE_LAYER = /\bknits?\b|\bsweaters?\b|\bturtlenecks?\b|\bmocknecks?\b|\bmock\snecks?\b|\broll[\s-]?necks?\b|\bhigh[\s-]?necks?\b|\bcrewnecks?\b|\bcrew\snecks?\b|\bjumpers?\b|\bfair[\s-]?isles?\b|\bcashmere\b|\bmerino\b|\bwool\b/i;
const NOT_BASE_LAYER = /\bcardigans?\b/i;

export function pulloverBaseLayer(row, hint) {
  if (!hint || hint.macroCategory !== 'outerwear' || hint.subtype !== 'pullover') return null;
  const text = `${row?.name || ''} ${row?.description || ''}`;
  if (NOT_BASE_LAYER.test(text) || !PULLOVER_BASE_LAYER.test(text)) return null;
  return {
    macroCategory: 'top',
    subtype: 'fine-knit',
    via: 'live_intelligence_role_projection',
    evidence: ['intelligence:outerwear/pullover worn as base top', 'knit base-layer wording in name/description'],
  };
}

const finitePrice = (value) =>
  typeof value === 'number' && Number.isFinite(value) && value > 0;

/**
 * UZS price-plausibility bands per outfit slot (merchant-offer sanity gate).
 *
 * Provenance: computed from the frozen UZS snapshot (sha256 b4bb7ae0…,
 * built 2026-11-13, 384 rows) — per-slot percentiles over admitted prices:
 *   top:       p0.5=228,564   p50=649,900    p95=9,013,729  p99.5=17,799,356  (min 219,900, max 24,252,000)
 *   outerwear: p0.5=323,978   p50=888,092    p95=5,035,875  p99.5=21,185,987  (min 258,000, max 28,545,957)
 *   bottom:    p0.5=396,902   p50=849,900    p95=8,641,261  p99.5=11,098,060  (min 396,902, max 13,051,283)
 *   shoes:     p0.5=549,900   p50=1,768,340  p95=2,732,889  p99.5=2,893,648   (min 549,900, max 2,893,648)
 *
 * Construction: the fixed floors sit far below each slot's observed cheap
 * end and the fixed ceilings sit just above each slot's observed legit
 * premium end (a 2.5M UZS or even 28.5M Canali cashmere outerwear passes),
 * so the quota-sampled snapshot can never become a rejection wall for
 * live-catalog items outside its sampled range. What the bands catch is
 * offer corruption: x100 minor-unit magnification (e.g. a GANT polo stored
 * at 24,252,000 UZS where the brand's entire segment is ~2M) and currency/
 * unit misparses that land an adult garment below any plausible retail
 * floor. Non-UZS rows have no committed evidence base, so the gate abstains
 * for them — no bounds are invented from FX. Excluded rows are counted
 * truthfully as `price_implausible`, never silently altered.
 */
export const PRICE_PLAUSIBILITY_UZS = Object.freeze({
  top: Object.freeze({ floor: 100_000, ceiling: 20_000_000 }),
  outerwear: Object.freeze({ floor: 100_000, ceiling: 30_000_000 }),
  bottom: Object.freeze({ floor: 100_000, ceiling: 20_000_000 }),
  shoes: Object.freeze({ floor: 150_000, ceiling: 15_000_000 }),
});

/**
 * Truthful price-plausibility verdict for one projected offer. Applies only
 * where committed evidence exists (native UZS rows in a known outfit slot);
 * everything else abstains. Returns the diagnostic reason or null.
 */
export function priceImplausibilityReason(macroCategory, currency, price) {
  if (currency !== 'UZS') return null;
  const band = PRICE_PLAUSIBILITY_UZS[macroCategory];
  if (!band) return null;
  return price >= band.floor && price <= band.ceiling ? null : 'price_implausible';
}

/**
 * Merchant-offer price freshness. `priceAgeDays` derives from
 * `last_seen_at` when the row carries one and is null otherwise (~4.4k live
 * rows lack timestamps — absence stays honest `unknown`, never guessed).
 */
export const PRICE_FRESHNESS_FRESH_MAX_DAYS = 7;
export const PRICE_FRESHNESS_RECENT_MAX_DAYS = 45;

export function priceAgeDays(row, nowMs = Date.now()) {
  const seen = Date.parse(String(row?.last_seen_at || ''));
  if (!Number.isFinite(seen)) return null;
  return Math.max(0, Math.floor((nowMs - seen) / 86_400_000));
}

/** Bounded staleness tier: fresh ≤7d, recent ≤45d, stale >45d, unknown. */
export function priceFreshness(ageDays) {
  if (!Number.isFinite(ageDays)) return 'unknown';
  if (ageDays <= PRICE_FRESHNESS_FRESH_MAX_DAYS) return 'fresh';
  if (ageDays <= PRICE_FRESHNESS_RECENT_MAX_DAYS) return 'recent';
  return 'stale';
}

/**
 * Project one committed snapshot row to a prototype item, or null when the
 * row cannot serve as a shoppable pilot candidate. Never throws for data
 * rows; unusable rows are counted, not hidden.
 */
export function projectCatalogItem(row) {
  if (!row || typeof row !== 'object') return null;
  const id = String(row.id || '');
  if (!id) return null;
  if (row.is_active !== true) return { skipped: 'inactive', id };
  if (!finitePrice(row.price)) return { skipped: 'unpriced', id };
  if (!isHttps(row.image_url) || !isHttps(row.source_url)) {
    return { skipped: 'non_https_media_or_shop', id };
  }
  if (typeof row.currency !== 'string' || !/^[A-Z]{3}$/.test(row.currency)) {
    return { skipped: 'bad_currency', id };
  }
  const integrity = assessCatalogCategory(row);
  if (integrity.status === 'confirmed_category_conflict' || integrity.status === 'native_role_conflict') {
    return { skipped: 'catalog_category_conflict', id };
  }
  const name = String(row.name || id);
  const ageDays = priceAgeDays(row);
  const priceFreshnessTier = priceFreshness(ageDays);
  // The broad source category is authoritative for unmistakable physical
  // roles. A colorway such as "Trench Coat Khaki" on *chino trousers* can
  // poison the text-derived intelligence slot with outerwear/coat; never
  // put trousers in an outfit's coat slot (or footwear in another slot).
  // Tops remain flexible: the source labels blazers/cardigans as "tops".
  const category = String(row.category || '').toLowerCase();
  const garmentType = String(row.garment_type || '').toLowerCase();
  const nativeLockedSlot = category === 'bottoms' || garmentType === 'lower_body'
    ? 'bottom'
    : category === 'shoes' || garmentType === 'shoes' || garmentType === 'footwear'
      ? 'shoes'
      : null;
  const live = liveHint(row);
  const hint = nativeLockedSlot && live && live.macroCategory !== nativeLockedSlot
    ? null
    : live;
  const roleProjected = pulloverBaseLayer(row, hint);
  if (roleProjected) {
    if (priceImplausibilityReason(roleProjected.macroCategory, row.currency, row.price)) {
      return { skipped: 'price_implausible', id };
    }
    return Object.freeze({
      id,
      name,
      macroCategory: roleProjected.macroCategory,
      subtype: roleProjected.subtype,
      color: inferColor(name),
      texture: inferTexture(name),
      imageUrl: row.image_url,
      shopUrl: row.source_url,
      price: row.price,
      currency: row.currency,
      brand: row.brand || null,
      isShopItem: true,
      visualEvidenceVerified: false,
      priceAgeDays: ageDays,
      priceFreshness: priceFreshnessTier,
      projection: { kind: roleProjected.via, evidence: roleProjected.evidence },
    });
  }
  if (hint) {
    if (priceImplausibilityReason(hint.macroCategory, row.currency, row.price)) {
      return { skipped: 'price_implausible', id };
    }
    return Object.freeze({
      id,
      name,
      macroCategory: hint.macroCategory,
      subtype: hint.subtype,
      color: inferColor(name),
      texture: inferTexture(name),
      imageUrl: row.image_url,
      shopUrl: row.source_url,
      price: row.price,
      currency: row.currency,
      brand: row.brand || null,
      isShopItem: true,
      visualEvidenceVerified: false,
      priceAgeDays: ageDays,
      priceFreshness: priceFreshnessTier,
      projection: { kind: 'live_intelligence', evidence: [`intelligence:${hint.macroCategory}/${hint.subtype}`] },
    });
  }
  const text = `${name} ${row.description || ''}`.toLowerCase();
  let macroCategory = null;
  let subtype = null;
  let projection = null;

  if (category === 'bottoms' || garmentType === 'lower_body') {
    macroCategory = 'bottom';
    subtype = firstMatch(BOTTOM_TABLE, text) || 'unknown';
    projection = { kind: 'native_category', evidence: [`category:${category || garmentType}`] };
  } else if (category === 'shoes' || garmentType === 'shoes' || garmentType === 'footwear') {
    macroCategory = 'shoes';
    subtype = firstMatch(SHOE_TABLE, text) || 'unknown';
    projection = { kind: 'native_category', evidence: [`category:${category || garmentType}`] };
  } else if (category === 'tops' || garmentType === 'upper_body') {
    // Tailored outerwear lives in `tops` in this snapshot; an explicit
    // family table (not per-garment special cases) decides the slot.
    const outer = firstMatch(OUTERWEAR_TABLE, text);
    if (outer) {
      macroCategory = 'outerwear';
      subtype = outer;
      projection = { kind: 'outerwear_family', evidence: [`outerwear family:${outer}`] };
    } else {
      macroCategory = 'top';
      subtype = firstMatch(TOP_TABLE, text) || 'unknown';
      if (subtype === 'fine-knit') {
        // Wearing-role projection: a lightweight knit mockneck/turtleneck/
        // crewneck is worn as the BASE top in the reference looks. It fills
        // the top role here instead of being stranded in outerwear next to
        // heavy coats. Cardigans never take this path (see TOP_TABLE).
        projection = {
          kind: 'knit_base_layer',
          evidence: ['knit base-layer wording in name/description', `category:${category || garmentType}`],
        };
      } else {
        projection = { kind: 'native_category', evidence: [`category:${category || garmentType}`] };
      }
    }
  } else {
    return { skipped: 'non_outfit_category', id };
  }
  if (priceImplausibilityReason(macroCategory, row.currency, row.price)) {
    return { skipped: 'price_implausible', id };
  }

  return Object.freeze({
    id,
    name,
    macroCategory,
    subtype,
    color: inferColor(name),
    texture: inferTexture(name),
    imageUrl: row.image_url,
    shopUrl: row.source_url,
    price: row.price,
    currency: row.currency,
    brand: row.brand || null,
    isShopItem: true,
    visualEvidenceVerified: false,
    priceAgeDays: ageDays,
    priceFreshness: priceFreshnessTier,
    projection,
  });
}

/** Base product id with per-shop variant suffixes removed for diversity counts. */
export function baseProductId(id) {
  return String(id).replace(/-v\d+$/i, '');
}

/**
 * Exact source-hash predicate for cached visual profiles. Mirrors the
 * `catalog_visual_fashion.source_hash` contract: md5 over the PostgreSQL
 * `jsonb_build_array(name, description, brand, category, garment_type,
 * image_url)` text rendering (arrays joined with ", "). A profile is only
 * attachable when the row bytes are unchanged since profiling — never
 * invented, never transferred across SKUs or images.
 */
export function sourceHash(row) {
  const parts = [row?.name, row?.description, row?.brand, row?.category, row?.garment_type, row?.image_url]
    .map((value) => JSON.stringify(value ?? null));
  return createHash('md5').update(`[${parts.join(', ')}]`).digest('hex');
}

/**
 * What a passing dev feasibility gate actually proves. A valid merchant URL
 * plus price/currency/SKU identity is listing evidence only: live stock,
 * size and delivery are NEVER confirmed by this gate and must never be
 * presented as purchase-verified.
 */
export const DEV_PURCHASE_EVIDENCE = Object.freeze({
  level: 'listing-only',
  listing: 'Product URL, price, currency and SKU identity validated per look.',
  stock: 'Live stock, size and delivery are NOT confirmed in dev testing.',
});

/**
 * Development-safe per-look feasibility gate. Reuses the same admission
 * predicates as `projectCatalogItem`/`safeCatalogItem` (known slot,
 * reachable photo, merchant URL present, finite price, 3-letter currency,
 * non-empty identity, UZS price plausibility) so the dev bridge no longer
 * needs an unconditional stub. Returns `{ ok, reason?, evidence, stock }`;
 * never throws for data rows. Passing means the listing is well-formed — see
 * DEV_PURCHASE_EVIDENCE: it is not a stock, size or delivery confirmation.
 */
const KNOWN_SLOTS = new Set(['top', 'outerwear', 'bottom', 'shoes']);
export function devLookFeasibility(items) {
  const fail = (reason) => ({ ok: false, reason, evidence: DEV_PURCHASE_EVIDENCE.level, stock: 'not_checked' });
  for (const item of items || []) {
    if (!item || typeof item.id !== 'string' || !item.id.trim()) return fail('invalid_product_identity');
    if (!KNOWN_SLOTS.has(item.macroCategory)) return fail('unknown_slot');
    if (!isHttps(item.imageUrl || item.image_url)) return fail('unrenderable_photo');
    if (item.isShopItem === true) {
      if (!isHttps(item.shopUrl || item.sourceUrl || item.source_url)) return fail('missing_merchant_offer');
      if (!finitePrice(item.price) || !/^[A-Z]{3}$/.test(item.currency || '')) return fail('unpriced_or_currency');
      // Price sanity: a UZS offer far outside its slot's committed plausibility
      // band is offer corruption (unit/currency misparse), not a bargain. The
      // look is excluded truthfully; the offer is never silently re-priced.
      if (priceImplausibilityReason(item.macroCategory, item.currency, item.price)) {
        return fail('price_implausible');
      }
    }
  }
  return { ok: true, evidence: DEV_PURCHASE_EVIDENCE.level, stock: 'not_checked' };
}

/** Keep one row per base product (first wins; input order is the priority). */
export function dedupeVariants(items) {
  const seen = new Set();
  const kept = [];
  for (const item of items) {
    const base = baseProductId(item.id);
    if (seen.has(base)) continue;
    seen.add(base);
    kept.push(item);
  }
  return { kept, duplicates: items.length - kept.length };
}

export function adaptCatalog(rows) {
  const items = [];
  const skipped = {};
  const bump = (reason) => {
    skipped[reason] = (skipped[reason] || 0) + 1;
  };
  for (const row of rows || []) {
    const projected = projectCatalogItem(row);
    if (!projected) {
      bump('unreadable_row');
      continue;
    }
    if (projected.skipped) {
      bump(projected.skipped);
      continue;
    }
    items.push(Object.freeze({
      ...projected, description: row.description || '', type: row.garment_type || projected.macroCategory,
      ...(row.visual_fashion && row.visual_source_hash ? {
        visualFashion: row.visual_fashion, visualSourceHash: row.visual_source_hash,
      } : {}),
    }));
  }
  const bySlotSubtype = {};
  const byProjection = {};
  const unknownSubtype = [];
  for (const item of items) {
    const key = `${item.macroCategory}/${item.subtype}`;
    bySlotSubtype[key] = (bySlotSubtype[key] || 0) + 1;
    const kind = item.projection?.kind || 'none';
    byProjection[kind] = (byProjection[kind] || 0) + 1;
    if (item.subtype === 'unknown') unknownSubtype.push(item.id);
  }
  return {
    adapterVersion: CATALOG_ADAPTER_VERSION,
    inputRows: (rows || []).length,
    items,
    skipped,
    bySlotSubtype,
    byProjection,
    unknownSubtypeSample: unknownSubtype.slice(0, 20),
    unknownSubtypeCount: unknownSubtype.length,
  };
}
