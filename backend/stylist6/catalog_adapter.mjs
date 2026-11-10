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

export const CATALOG_ADAPTER_VERSION = 'catalog-adapter-v1';

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

const finitePrice = (value) =>
  typeof value === 'number' && Number.isFinite(value) && value > 0;

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
  const name = String(row.name || id);
  const hint = liveHint(row);
  if (hint) {
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
      projection: { kind: 'live_intelligence', evidence: [`intelligence:${hint.macroCategory}/${hint.subtype}`] },
    });
  }
  const text = `${name} ${row.description || ''}`.toLowerCase();
  const category = String(row.category || '').toLowerCase();
  const garmentType = String(row.garment_type || '').toLowerCase();

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
    projection,
  });
}

/** Base product id with per-shop variant suffixes removed for diversity counts. */
export function baseProductId(id) {
  return String(id).replace(/-v\d+$/i, '');
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
    items.push(projected);
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
