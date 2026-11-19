/**
 * Stylist 6 — Structural Catalog Data-Truth Hygiene Gate.
 *
 * Enforces structural provenance, media integrity, and slot/occasion truth
 * across raw Supabase `shop_catalog` rows and adapted/prepared garment items.
 *
 * Design principles (see docs/STYLIST6_NORTH_STAR.md):
 * - Structural checks over fragile one-off brand-name blacklists.
 * - Never mutate input item objects; return explicit auditable verdicts.
 * - Never fabricate stock, prices, media rights, or missing visual attributes.
 * - Unknown visual metadata remains UNKNOWN (warning, not silent fabrication).
 */
import { assessCatalogCategory } from './catalog_category_integrity.mjs';
import { baseProductId } from './catalog_adapter.mjs';

export const CATALOG_HYGIENE_VERSION = 's6-catalog-hygiene-v1';

const SINGLE_GARMENT_SLOTS = new Set(['top', 'outerwear', 'bottom', 'shoes', 'dress', 'accessory']);
const MIN_IMAGE_DIMENSION_PX = 400;

// Structural locale / generic landing path segments that never identify a purchasable PDP.
const LOCALE_SEGMENT_RE = /^[a-z]{2}(?:[-_][a-z]{2,4})?$/i;
const GENERIC_LANDING_SEGMENT_RE = /^(?:index(?:\.html?)?|home|homepage|shop|store|storefront|catalog|catalogue|men|women|mens|womens|man|woman|menswear|womenswear|unisex|clothing|apparel|fashion|new|new-arrivals|arrivals|sale|sales|outlet|search|cart|bag|checkout|account|login|signin|sign-in|collection|collections|category|categories|all|featured|bestsellers|best-sellers)$/i;
const PRODUCT_QUERY_KEY_RE = /^(?:pid|id|sku|product|product_id|productid|item|item_id|article|style|style_id|variant|code)$/i;

// Non-product / unusable visual imageKind values.
const UNUSABLE_IMAGE_KINDS = new Set([
  'diagram', 'promotion', 'unusable', 'logo', 'placeholder',
  'size_chart', 'size-chart', 'chart', 'banner', 'collage', 'text',
]);
const UNUSABLE_IMAGE_URL_RE = /(?:\.(?:svg|ico)(?:[?#]|$)|\/(?:placeholder|no[-_]?image|default[-_]?avatar|coming[-_]?soon|missing[-_]?image)\b)/i;

// Structural swimwear / beachwear & underwear/sleepwear markers.
const SWIMWEAR_RE = /\b(?:swim(?:wear|suits?|shorts?|trunks?|briefs?)?|swimming(?:\s+(?:shorts?|trunks?|briefs?))?|board[- ]?shorts?|beach(?:wear|side)(?:\s+(?:shorts?|trunks?|swim|tank|t-?shirt|shirt))?|beach[- ]?shorts?|bathing[- ]?(?:suits?|trunks?)|surf[- ]?trunks?|short\s+de\s+bain|maillot\s+de\s+bain|badehose|costume\s+da\s+bagno|speedo)\b|плавк|купальн/i;
const UNDERWEAR_SLEEPWEAR_RE = /\b(?:boxer[- ]?briefs?|underwear|underpants|briefs|pyjamas?|pajamas?|bathrobes?|nightgowns?|lingerie|undershirts?)\b/i;
const BEACH_OCCASION_RE = /\b(?:beach|pool|poolside|swim|swimming|resort|waterpark|seaside|coastal\s+swim)\b|пляж|бассейн/i;
const EVENING_TUXEDO_SHIRT_RE = /\b(?:tuxedo|evening\s+studs?|stud[- ]?front|marcella\s+dinner|wing[- ]?collar\s+dinner|dinner\s+shirts?)\b/i;
const DAYTIME_OCCASION_RE = /\b(?:work|office|business|meeting|conference|casual|weekend|everyday|travel|brunch|commute)\b/i;
const FORMAL_EVENING_OCCASION_RE = /\b(?:formal|black[- ]?tie|tuxedo|gala|evening|cocktail|wedding|opera|ball)\b/i;

// Dress / one-piece classification — carefully excludes "dress shirt", "dress shoes", etc.
const DRESS_MODIFIER_FOLLOWED_BY_GARMENT_RE = /\bdress(?:es)?[- ]+(?:shirts?|shoes?|trousers?|pants?|slacks?|boots?|loafers?|oxfords?|derb(?:y|ies)|coats?|jackets?|socks?|belts?|watch(?:es)?|code|attire|uniform)\b/i;
const DRESS_OR_JUMPSUIT_RE = /\b(?:jumpsuits?|rompers?|playsuits?|sundress(?:es)?|gowns?|kaftans?|caftans?|shirt[- ]?dress(?:es)?|slip[- ]?dress(?:es)?|wrap[- ]?dress(?:es)?|midi[- ]?dress(?:es)?|mini[- ]?dress(?:es)?|maxi[- ]?dress(?:es)?|cocktail[- ]?dress(?:es)?|evening[- ]?dress(?:es)?|sweater[- ]?dress(?:es)?|knit[- ]?dress(?:es)?|linen[- ]?dress(?:es)?|silk[- ]?dress(?:es)?|sheath[- ]?dress(?:es)?|shift[- ]?dress(?:es)?|column[- ]?dress(?:es)?|a[- ]line[- ]?dress(?:es)?|bodycon[- ]?dress(?:es)?|tea[- ]?dress(?:es)?|day[- ]?dress(?:es)?|summer[- ]?dress(?:es)?)\b/i;
const TRAILING_DRESS_NOUN_RE = /(?:^|[\s|/,-])dress(?:es)?$/i;

// Complete multi-piece suit/tracksuit markers.
const COMPLETE_OUTFIT_SLOT_RE = /^(?:outfits?|complete[-_]?suit|full[-_]?suit|suit[-_]?set|two[-_]?piece)$/i;

const RAW_CATEGORY_TO_FAMILY = {
  tops: 'upper',
  top: 'upper',
  outerwear: 'upper',
  upper_body: 'upper',
  bottoms: 'bottom',
  bottom: 'bottom',
  lower_body: 'bottom',
  shoes: 'shoes',
  footwear: 'shoes',
  dress: 'dress',
  dresses: 'dress',
  one_piece: 'dress',
  accessory: 'accessory',
  accessories: 'accessory',
  bags: 'accessory',
  bag: 'accessory',
  outfits: 'outfits',
};

const SLOT_TO_FAMILY = {
  top: 'upper',
  outerwear: 'upper',
  bottom: 'bottom',
  shoes: 'shoes',
  dress: 'dress',
  accessory: 'accessory',
  outfits: 'outfits',
};

const PRESENTATIONS = new Set(['menswear', 'womenswear', 'unisex']);

function isHttpsUrl(raw) {
  if (typeof raw !== 'string') return false;
  const trimmed = raw.trim();
  if (!trimmed || /\s/.test(trimmed)) return false;
  try {
    const u = new URL(trimmed);
    return u.protocol === 'https:' && typeof u.hostname === 'string' && u.hostname.includes('.');
  } catch {
    return false;
  }
}

/**
 * Structural PDP URL validator.
 * Rejects non-HTTPS URLs, homepages (`/`), bare locale roots (`/en-us/`),
 * and generic storefront/category landing roots (`/en-us/men`, `/shop/`)
 * that carry no product slug, SKU token, or product query parameter.
 */
export function isValidProductPdpUrl(rawUrl) {
  if (typeof rawUrl !== 'string' || !rawUrl.trim()) {
    return { ok: false, reason: 'missing_shop_url', pathSegments: [] };
  }
  const trimmed = rawUrl.trim();
  if (/\s/.test(trimmed)) {
    return { ok: false, reason: 'invalid_shop_url', pathSegments: [] };
  }
  let parsed;
  try {
    parsed = new URL(trimmed);
  } catch {
    return { ok: false, reason: 'invalid_shop_url', pathSegments: [] };
  }
  if (parsed.protocol !== 'https:' || !parsed.hostname || !parsed.hostname.includes('.')) {
    return { ok: false, reason: 'non_https_shop_url', pathSegments: [] };
  }
  const segments = parsed.pathname
    .split('/')
    .map(s => {
      try { return decodeURIComponent(s).trim(); } catch { return s.trim(); }
    })
    .filter(Boolean);

  let hasProductQuery = false;
  for (const [k, v] of parsed.searchParams.entries()) {
    if (PRODUCT_QUERY_KEY_RE.test(k) && String(v || '').trim().length > 0) {
      hasProductQuery = true;
      break;
    }
  }
  if (hasProductQuery) {
    return { ok: true, pathSegments: segments };
  }
  if (segments.length === 0) {
    return { ok: false, reason: 'homepage_only_pdp_url', pathSegments: segments };
  }
  if (segments.length === 1) {
    if (LOCALE_SEGMENT_RE.test(segments[0]) || GENERIC_LANDING_SEGMENT_RE.test(segments[0])) {
      return { ok: false, reason: 'homepage_only_pdp_url', pathSegments: segments };
    }
    return { ok: true, pathSegments: segments };
  }
  if (segments.length === 2) {
    const firstGeneric = LOCALE_SEGMENT_RE.test(segments[0]) || GENERIC_LANDING_SEGMENT_RE.test(segments[0]);
    const secondGeneric = LOCALE_SEGMENT_RE.test(segments[1]) || GENERIC_LANDING_SEGMENT_RE.test(segments[1]);
    if (firstGeneric && secondGeneric) {
      return { ok: false, reason: 'generic_category_url_not_pdp', pathSegments: segments };
    }
  }
  // Reject 3-segment paths where every single segment is purely locale + generic root (e.g. /en-us/men/shop)
  if (segments.every(s => LOCALE_SEGMENT_RE.test(s) || GENERIC_LANDING_SEGMENT_RE.test(s))) {
    return { ok: false, reason: 'generic_category_url_not_pdp', pathSegments: segments };
  }
  return { ok: true, pathSegments: segments };
}

function getVisualProfile(item) {
  const vf = item?.visualFashion || item?.visual_fashion;
  if (!vf || typeof vf !== 'object') return null;
  // Respect source-hash freshness when both hashes are present.
  const expectedHash = item?.visualSourceHash || item?.visual_source_hash;
  if (expectedHash && vf.sourceHash && vf.sourceHash !== expectedHash) {
    return null;
  }
  return vf;
}

/**
 * Extract known image pixel dimensions from visual metadata, top-level fields,
 * or explicit CDN resize directives in the image URL.
 */
export function extractKnownImageDimensions(item) {
  const vf = getVisualProfile(item);
  const iq = vf?.imageQuality || item?.imageQuality || item?.image_quality;
  if (iq && typeof iq === 'object') {
    const w = Number(iq.width);
    const h = Number(iq.height);
    if ((Number.isFinite(w) && w > 0) || (Number.isFinite(h) && h > 0)) {
      return {
        width: Number.isFinite(w) && w > 0 ? w : null,
        height: Number.isFinite(h) && h > 0 ? h : null,
        source: 'image_quality_metadata',
      };
    }
  }
  const topW = Number(item?.imageWidth ?? item?.image_width ?? vf?.width);
  const topH = Number(item?.imageHeight ?? item?.image_height ?? vf?.height);
  if ((Number.isFinite(topW) && topW > 0) || (Number.isFinite(topH) && topH > 0)) {
    return {
      width: Number.isFinite(topW) && topW > 0 ? topW : null,
      height: Number.isFinite(topH) && topH > 0 ? topH : null,
      source: 'item_dimensions',
    };
  }
  const url = String(item?.imageUrl || item?.image_url || '');
  if (url) {
    const widthMatch = url.match(/(?:[,/_?&]|^)w_(\d{2,4})(?:[,/_&.?]|$)/i)
      || url.match(/[?&](?:w|width|wid|imwidth)=(\d{2,4})(?:&|$)/i);
    const heightMatch = url.match(/(?:[,/_?&]|^)h_(\d{2,4})(?:[,/_&.?]|$)/i)
      || url.match(/[?&](?:h|height|hei)=(\d{2,4})(?:&|$)/i);
    const w = widthMatch ? Number(widthMatch[1]) : null;
    const h = heightMatch ? Number(heightMatch[1]) : null;
    if ((Number.isFinite(w) && w > 0) || (Number.isFinite(h) && h > 0)) {
      return { width: w, height: h, source: 'url_resize_param' };
    }
  }
  return null;
}

/**
 * Structural swimwear / beachwear detector.
 */
export function isSwimwearItem(item) {
  const subtype = String(item?.subtype || item?.fashion_subtype || '').toLowerCase();
  if (/swim|boardshort|beach/.test(subtype)) return true;
  const name = String(item?.name || item?.title || '');
  const desc = String(item?.description || '');
  const shopUrl = String(item?.shopUrl || item?.source_url || '');
  let pathText = '';
  try {
    if (shopUrl) pathText = decodeURIComponent(new URL(shopUrl).pathname).replace(/[-_/]+/g, ' ');
  } catch {
    pathText = '';
  }
  return SWIMWEAR_RE.test(`${name} ${pathText}`) || /\b(?:swimwear|swim[- ]?shorts?|board[- ]?shorts?|bathing\s+suit)\b/i.test(desc);
}

/**
 * Structural underwear / sleepwear detector (non-streetwear).
 */
export function isUnderwearOrSleepwearItem(item) {
  const name = String(item?.name || item?.title || '');
  const shopUrl = String(item?.shopUrl || item?.source_url || '');
  let pathText = '';
  try {
    if (shopUrl) pathText = decodeURIComponent(new URL(shopUrl).pathname).replace(/[-_/]+/g, ' ');
  } catch {
    pathText = '';
  }
  // Careful: "briefcase" is a bag, not underwear; \bbriefs\b is plural only.
  return UNDERWEAR_SLEEPWEAR_RE.test(`${name} ${pathText}`);
}

/**
 * Structural evening / black-tie tuxedo & dinner stud shirt detector.
 * Inspects product title, description, and canonical merchant URL slug
 * (e.g. M&S `regular-fit-pure-cotton-dinner-shirt`) so formal stud-front
 * evening shirts stay available for formal/black-tie while being excluded
 * from daytime work/office/casual styling.
 */
export function isEveningTuxedoShirtItem(item) {
  const name = String(item?.name || item?.title || '');
  const desc = String(item?.description || '');
  const shopUrl = String(item?.shopUrl || item?.source_url || '');
  let pathText = '';
  try {
    if (shopUrl) pathText = decodeURIComponent(new URL(shopUrl).pathname).replace(/[-_/]+/g, ' ');
  } catch {
    pathText = '';
  }
  return EVENING_TUXEDO_SHIRT_RE.test(`${name} ${pathText} ${desc}`);
}

function isDressOrJumpsuitText(item) {
  const name = String(item?.name || item?.title || '').trim();
  const subtype = String(item?.subtype || item?.fashion_subtype || '').toLowerCase().trim();
  if (['dress', 'gown', 'jumpsuit', 'romper', 'playsuit', 'sundress', 'kaftan'].includes(subtype)) {
    return true;
  }
  if (!name) return false;
  if (DRESS_MODIFIER_FOLLOWED_BY_GARMENT_RE.test(name)) return false;
  if (DRESS_OR_JUMPSUIT_RE.test(name)) return true;
  const cleaned = name.replace(/\s*\([^)]*\)\s*$/g, '').trim();
  if (TRAILING_DRESS_NOUN_RE.test(cleaned)) {
    // Double-check against compound garment titles like "wool dress" vs "casual dress"
    return !/\b(?:fancy\s+dress\s+costume)\b/i.test(cleaned);
  }
  return false;
}

/**
 * Structural garment slot classifier & cross-source contradiction detector.
 * - Cleanly classifies dresses/jumpsuits into `'dress'` (never `'top'`).
 * - Flags multi-piece complete outfits (`'outfits'`).
 * - Detects raw category vs intelligence slot contradictions.
 * - Detects high-confidence `catalog_visual_fashion` vs `catalog_garment_intelligence`
 *   cross-family slot contradictions (`confidence >= 0.85, margin >= 0.15`).
 */
export function classifyGarmentSlot(item) {
  const rawCategory = String(item?.category || '').trim().toLowerCase();
  const rawGarmentType = String(item?.garment_type || '').trim().toLowerCase();
  const intelSlot = String(
    item?.fashion_slot ||
    item?.garment_intelligence?.descriptor?.slot ||
    item?.macroCategory ||
    '',
  ).trim().toLowerCase();
  const productKind = String(item?.garment_intelligence?.descriptor?.productKind || '').trim().toLowerCase();

  const vf = getVisualProfile(item);
  const visualSlotObs = vf?.attributes?.slot;
  const visualSlotConfident = !!(
    visualSlotObs &&
    typeof visualSlotObs.value === 'string' &&
    Number.isFinite(visualSlotObs.confidence) &&
    visualSlotObs.confidence >= 0.85 &&
    Number.isFinite(visualSlotObs.margin) &&
    visualSlotObs.margin >= 0.15
  );
  const visualSlot = visualSlotConfident ? String(visualSlotObs.value).trim().toLowerCase() : null;

  // 1. Dresses / Jumpsuits cleanly map to 'dress' (never 'top').
  if (
    rawCategory === 'dress' || rawCategory === 'dresses' || rawCategory === 'one_piece' ||
    rawGarmentType === 'dress' || rawGarmentType === 'one_piece' ||
    intelSlot === 'dress' ||
    isDressOrJumpsuitText(item)
  ) {
    // Still flag if a dress was stuffed into shoes or bottoms with a visual contradiction.
    const rawFam = RAW_CATEGORY_TO_FAMILY[rawCategory] || null;
    const contradiction = rawFam === 'shoes' || rawFam === 'bottom'
      ? 'dress_in_incompatible_raw_category'
      : null;
    return {
      slot: 'dress',
      source: 'dress_classification',
      contradiction,
      rawCategory: rawCategory || null,
      intelSlot: intelSlot || null,
      visualSlot,
    };
  }

  // 2. Accessories cleanly map to 'accessory'.
  if (
    rawCategory === 'accessory' || rawCategory === 'accessories' ||
    rawCategory === 'bag' || rawCategory === 'bags' ||
    intelSlot === 'accessory'
  ) {
    return {
      slot: 'accessory',
      source: 'accessory_classification',
      contradiction: null,
      rawCategory: rawCategory || null,
      intelSlot: intelSlot || null,
      visualSlot,
    };
  }

  // 3. Complete multi-piece outfits / suits / tracksuits.
  if (COMPLETE_OUTFIT_SLOT_RE.test(intelSlot) || productKind === 'complete-suit') {
    return {
      slot: 'outfits',
      source: 'multi_piece_outfit',
      contradiction: 'multi_piece_outfit_sku',
      rawCategory: rawCategory || null,
      intelSlot: intelSlot || null,
      visualSlot,
    };
  }

  // 4. Resolve canonical single-garment slot.
  let slot = SINGLE_GARMENT_SLOTS.has(intelSlot) ? intelSlot : 'unknown';
  let source = SINGLE_GARMENT_SLOTS.has(intelSlot) ? 'garment_intelligence' : 'unknown';
  if (slot === 'unknown') {
    if (rawCategory === 'bottoms' || rawCategory === 'bottom' || rawGarmentType === 'lower_body') {
      slot = 'bottom';
      source = 'raw_category';
    } else if (rawCategory === 'shoes' || rawGarmentType === 'shoes' || rawGarmentType === 'footwear') {
      slot = 'shoes';
      source = 'raw_category';
    } else if (rawCategory === 'outerwear') {
      slot = 'outerwear';
      source = 'raw_category';
    } else if (rawCategory === 'tops' || rawCategory === 'top' || rawGarmentType === 'upper_body') {
      slot = 'top';
      source = 'raw_category';
    }
  }

  // 5. Structural slot contradiction checks (record all independent contradictions).
  const contradictions = [];

  // 5a. Existing category/path integrity check (title + URL agree on opposite role).
  if (rawCategory || rawGarmentType) {
    const catCheck = assessCatalogCategory({
      category: item?.category,
      garment_type: item?.garment_type,
      name: item?.name || item?.title,
      source_url: item?.source_url || item?.shopUrl,
    });
    if (catCheck.status === 'confirmed_category_conflict' || catCheck.status === 'native_role_conflict') {
      contradictions.push(`category_integrity_${catCheck.status}`);
    }
  }

  // 5b. Raw category vs garment intelligence cross-family conflict (e.g., category='tops' while intel='bottom').
  if (rawCategory && intelSlot) {
    const rawFam = RAW_CATEGORY_TO_FAMILY[rawCategory];
    const intelFam = SLOT_TO_FAMILY[intelSlot];
    if (rawFam && intelFam && rawFam !== intelFam) {
      contradictions.push(`raw_category_vs_intel_conflict:${rawCategory}->${intelSlot}`);
    }
  }

  // 5c. High-confidence visual_fashion slot vs garment intelligence slot contradiction.
  if (visualSlot && slot !== 'unknown') {
    const slotFam = SLOT_TO_FAMILY[slot];
    const visualFam = SLOT_TO_FAMILY[visualSlot];
    if (visualSlot === 'outfits') {
      // Blazers photographed on full suit models can register visual 'outfits' at ~0.85;
      // require higher confidence (>=0.90 / 0.93 for blazers) before flagging as multi-piece outfits.
      const conf = Number(visualSlotObs.confidence);
      const isBlazer = String(item?.subtype || item?.fashion_subtype || '').toLowerCase() === 'blazer';
      if (conf >= (isBlazer ? 0.93 : 0.90)) {
        contradictions.push(`visual_vs_intel_slot_conflict:${slot}->outfits`);
      }
    } else if (slotFam && visualFam && slotFam !== visualFam) {
      contradictions.push(`visual_vs_intel_slot_conflict:${slot}->${visualSlot}`);
    }
  }

  return {
    slot,
    source,
    contradiction: contradictions[0] || null,
    contradictions,
    rawCategory: rawCategory || null,
    intelSlot: intelSlot || null,
    visualSlot,
  };
}

/**
 * Convenience helper for checking high-confidence visual vs intelligence slot contradictions.
 */
export function hasVisualSlotContradiction(item) {
  const res = classifyGarmentSlot(item);
  const all = Array.isArray(res.contradictions) ? res.contradictions : (res.contradiction ? [res.contradiction] : []);
  return all.some(c => typeof c === 'string' && c.startsWith('visual_vs_intel_slot_conflict'));
}

/**
 * Look up per-SKU visual quality flywheel entry (from `catalog_visual_quality`).
 * Supports both `Map` and plain Object keyed by exact `id` or `baseProductId(id)`.
 */
export function lookupQualityEntry(item, qualityMap) {
  const neutral = { quarantined: false, qualityScore: 0.65, quarantineReason: null, raw: null };
  if (!qualityMap || !item?.id) return neutral;
  const id = String(item.id);
  const baseId = baseProductId(id);
  const entry = qualityMap instanceof Map
    ? (qualityMap.get(id) ?? qualityMap.get(baseId) ?? null)
    : (typeof qualityMap === 'object' ? (qualityMap[id] ?? qualityMap[baseId] ?? null) : null);
  if (!entry || typeof entry !== 'object') return neutral;

  const approvals = Number(entry.approvals || 0);
  const rejections = Number(entry.rejections || 0);
  const photoDefects = Number(entry.photoDefects ?? entry.photo_defects ?? 0);
  const categoryMismatches = Number(entry.categoryMismatches ?? entry.category_mismatches ?? 0);
  const rawScore = Number(entry.qualityScore ?? entry.quality_score);
  const qualityScore = Number.isFinite(rawScore) ? Math.max(0, Math.min(1, rawScore)) : 0.65;

  let quarantined = entry.quarantined === true || entry.isQuarantined === true;
  let quarantineReason = entry.quarantineReason ?? entry.quarantine_reason ?? null;
  if (!quarantined) {
    if (photoDefects >= 3) {
      quarantined = true;
      quarantineReason = 'chronic_photo_defects';
    } else if (categoryMismatches >= 2) {
      quarantined = true;
      quarantineReason = 'chronic_category_mismatches';
    } else if (approvals + rejections >= 5 && qualityScore < 0.25) {
      quarantined = true;
      quarantineReason = 'low_wilson_quality_score';
    }
  }
  return { quarantined, qualityScore, quarantineReason, raw: entry };
}

function normalizePresentation(value) {
  const raw = String(value || '').trim().toLowerCase();
  if (raw === 'male' || raw === 'man' || raw === 'men' || raw === 'mens' || raw === 'menswear') return 'menswear';
  if (raw === 'female' || raw === 'woman' || raw === 'women' || raw === 'womens' || raw === 'womenswear') return 'womenswear';
  if (raw === 'unisex' || raw === 'all' || raw === 'non-binary' || raw === 'gender-neutral') return 'unisex';
  return null;
}

/**
 * Evaluate one catalog (or wardrobe) item against all structural hygiene gates.
 *
 * @param {object} item - Raw Supabase `shop_catalog` row OR adapted/prepared item.
 * @param {Map|Record<string,object>|null} [qualityMap=null] - Per-SKU `catalog_visual_quality` lookup.
 * @param {object|null} [context=null] - Optional request context (`occasion`, `presentation`, `minImageDimension`, `allowSwimwear`).
 */
export function evaluateCatalogItemHygiene(item, qualityMap = null, context = null) {
  if (!item || typeof item !== 'object') {
    return {
      eligible: false,
      itemId: '',
      resolvedSlot: 'unknown',
      reasons: ['unreadable_item'],
      warnings: [],
      qualityScore: 0,
      quarantined: false,
      pdpValid: false,
      mediaValid: false,
      dimensions: null,
      isSwimwear: false,
    };
  }

  const itemId = String(item.id || '').trim();
  const isShopItem = item.isShopItem !== false; // Raw shop_catalog rows omit isShopItem -> true.
  const reasons = [];
  const warnings = [];
  const minDim = Number.isFinite(context?.minImageDimension) ? context.minImageDimension : MIN_IMAGE_DIMENSION_PX;

  if (!itemId) reasons.push('missing_item_id');
  if (item.is_active === false || item.isActive === false) reasons.push('inactive_item');

  // 1. Structural shop offer & PDP URL validation (shop items only).
  let pdpValid = true;
  if (isShopItem) {
    const shopUrl = item.shopUrl || item.source_url || item.sourceUrl;
    const pdp = isValidProductPdpUrl(shopUrl);
    pdpValid = pdp.ok;
    if (!pdp.ok) reasons.push(pdp.reason);

    const price = Number(item.price);
    if (!Number.isFinite(price) || price <= 0) reasons.push('invalid_price');
    const currency = String(item.currency || '').trim();
    if (!/^[A-Z]{3}$/.test(currency)) reasons.push('invalid_currency');
  }

  // 2. Structural media integrity.
  const imageUrl = String(item.imageUrl || item.image_url || '').trim();
  let mediaValid = true;
  if (!isHttpsUrl(imageUrl)) {
    mediaValid = false;
    reasons.push('missing_or_invalid_image_url');
  } else if (UNUSABLE_IMAGE_URL_RE.test(imageUrl)) {
    mediaValid = false;
    reasons.push('unusable_image_url_format');
  }

  const vf = getVisualProfile(item);
  if (!vf) {
    warnings.push('missing_visual_profile');
  } else {
    const ik = vf.imageKind;
    const kindValue = typeof ik === 'string'
      ? ik.trim().toLowerCase()
      : (ik && typeof ik === 'object' && typeof ik.value === 'string' ? ik.value.trim().toLowerCase() : null);
    const kindConfidence = typeof ik === 'string'
      ? 1
      : (Number.isFinite(ik?.confidence) ? ik.confidence : 0);
    const kindEvidence = typeof ik === 'object' ? String(ik?.evidence || '') : 'explicit';
    if (kindValue && kindValue !== 'product') {
      if (UNUSABLE_IMAGE_KINDS.has(kindValue) || kindConfidence >= 0.8 || kindEvidence === 'human') {
        mediaValid = false;
        reasons.push(`non_product_image_kind:${kindValue}`);
      } else {
        warnings.push(`uncertain_image_kind:${kindValue}`);
      }
    }
  }

  const dimensions = extractKnownImageDimensions(item);
  if (dimensions) {
    const tooSmallW = Number.isFinite(dimensions.width) && dimensions.width < minDim;
    const tooSmallH = Number.isFinite(dimensions.height) && dimensions.height < minDim;
    if (tooSmallW || tooSmallH) {
      mediaValid = false;
      reasons.push('low_res_image');
    }
  }

  // 3. Visual quality flywheel quarantine check.
  const quality = lookupQualityEntry(item, qualityMap);
  if (quality.quarantined) {
    reasons.push(quality.quarantineReason ? `quality_quarantined:${quality.quarantineReason}` : 'quality_quarantined');
  }

  // 4. Structural slot & occasion truth.
  const slotInfo = classifyGarmentSlot(item);
  const resolvedSlot = slotInfo.slot;
  if (resolvedSlot === 'unknown') {
    reasons.push('unknown_garment_slot');
  }
  if (Array.isArray(slotInfo.contradictions) && slotInfo.contradictions.length) {
    reasons.push(...slotInfo.contradictions);
  } else if (slotInfo.contradiction) {
    reasons.push(slotInfo.contradiction);
  }

  // 5. Swimwear / beachwear, underwear/sleepwear, and evening-shirt daytime occasion gates.
  const swimwear = isSwimwearItem(item);
  const occasionText = `${context?.occasion || ''} ${context?.brief || ''} ${context?.request || ''}`;
  const isBeachOccasion = context?.allowSwimwear === true || BEACH_OCCASION_RE.test(occasionText);
  if (swimwear && !isBeachOccasion) {
    reasons.push('swimwear_in_non_beach_occasion');
  }
  if (isUnderwearOrSleepwearItem(item)) {
    reasons.push('underwear_or_sleepwear_item');
  }
  const isDaytimeNonBlackTie = DAYTIME_OCCASION_RE.test(occasionText) && !FORMAL_EVENING_OCCASION_RE.test(occasionText);
  if (resolvedSlot === 'top' && isDaytimeNonBlackTie && isEveningTuxedoShirtItem(item)) {
    reasons.push('evening_tuxedo_shirt_in_daytime_occasion');
  }

  // 6. Optional presentation / gender alignment (only when item explicitly carries a gender).
  const requestedPresentation = normalizePresentation(context?.presentation || context?.bodyProfile?.gender);
  const itemGender = normalizePresentation(item.gender);
  if (
    requestedPresentation && requestedPresentation !== 'unisex' &&
    itemGender && itemGender !== 'unisex' &&
    itemGender !== requestedPresentation
  ) {
    reasons.push(`presentation_mismatch:${itemGender}!=${requestedPresentation}`);
  }

  const uniqueReasons = [...new Set(reasons)];
  return {
    eligible: uniqueReasons.length === 0,
    itemId,
    resolvedSlot,
    reasons: uniqueReasons,
    warnings,
    qualityScore: quality.qualityScore,
    quarantined: quality.quarantined,
    pdpValid,
    mediaValid,
    dimensions,
    isSwimwear: swimwear,
  };
}

/**
 * Filter a full catalog array through the structural hygiene gate.
 * Returns hygienic items alongside transparent exclusion & slot diagnostics.
 * Never mutates input objects.
 *
 * @param {Array<object>} items
 * @param {object} [context={}]
 * @param {Map|Record<string,object>|null} [qualityMap=null]
 */
export function filterHygienicCatalog(items, context = {}, qualityMap = null) {
  const input = Array.isArray(items) ? items : [];
  const requiredSet = new Set(
    (Array.isArray(context?.requiredIds) ? context.requiredIds : []).map(id => String(id)),
  );
  const eligibleItems = [];
  const excluded = [];
  const byReason = {};
  const bySlot = {};
  const warningsByType = {};

  for (const item of input) {
    const verdict = evaluateCatalogItemHygiene(item, qualityMap, context);
    for (const w of verdict.warnings) {
      warningsByType[w] = (warningsByType[w] || 0) + 1;
    }
    const isPinnedAnchor = verdict.itemId && requiredSet.has(verdict.itemId);
    if (verdict.eligible || (isPinnedAnchor && item?.isShopItem === false)) {
      // Preserve input immutability; attach clean resolvedSlot when absent or dress-corrected.
      const needsSlotPatch = !item.macroCategory || (verdict.resolvedSlot === 'dress' && item.macroCategory !== 'dress');
      const outItem = needsSlotPatch
        ? Object.freeze({ ...item, macroCategory: verdict.resolvedSlot })
        : item;
      eligibleItems.push(outItem);
      bySlot[verdict.resolvedSlot] = (bySlot[verdict.resolvedSlot] || 0) + 1;
    } else {
      for (const reason of verdict.reasons) {
        const bucket = reason.split(':')[0];
        byReason[bucket] = (byReason[bucket] || 0) + 1;
      }
      excluded.push({
        id: verdict.itemId,
        reasons: verdict.reasons,
        resolvedSlot: verdict.resolvedSlot,
      });
    }
  }

  return {
    version: CATALOG_HYGIENE_VERSION,
    items: eligibleItems,
    excluded,
    stats: {
      inputCount: input.length,
      eligibleCount: eligibleItems.length,
      excludedCount: excluded.length,
      byReason,
      bySlot,
      warningsByType,
    },
  };
}
