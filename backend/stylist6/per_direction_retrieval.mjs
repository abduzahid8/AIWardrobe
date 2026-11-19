/**
 * Stylist 6 — Per-Direction Orthogonal Candidate Pool Retrieval.
 *
 * Solves the 12-SKU global bottleneck (0.56% of 8,554 SKUs) and cardigan+chino
 * monotony by retrieving a dedicated, direction-aligned, price-balanced,
 * cross-direction-diversified candidate menu for EACH of the 3 Creative Director
 * blueprints across the full hygienic catalog.
 *
 * Hard principles (see docs/STYLIST6_NORTH_STAR.md):
 * - Real verified SKUs only; never invents items, prices, stock, or FX rates.
 * - Budget headroom uses ONLY matching currency or caller-supplied verified FX.
 * - Unbudgeted price-tier balancing uses honest within-currency catalog percentiles.
 * - Provider-facing `public` records strip all SKU IDs, shop URLs, image URLs,
 *   and private user identifiers.
 */
import { filterHygienicCatalog, lookupQualityEntry } from './catalog_hygiene.mjs';
import { baseProductId, inferCatalogColor, inferTexture } from './catalog_adapter.mjs';
import { currentVisual, confidentVisual } from '../../src/domain/visualFashion.ts';
import { classifyOuterwearLayerRole } from '../../src/domain/outfitGrammar.ts';

export const PER_DIRECTION_RETRIEVAL_VERSION = 's6-per-direction-retrieval-v1';

const DEFAULT_SLOTS = ['top', 'outerwear', 'bottom', 'shoes'];
const ALLOWED_SLOTS = new Set(['top', 'outerwear', 'bottom', 'shoes', 'dress', 'accessory']);

const PHOTO_FIELDS = [
  'color', 'subtype', 'formality', 'texture', 'pattern', 'volume',
  'length', 'bulk', 'hemShape', 'closure', 'accentHue',
  'soleProfile', 'footwearPurpose', 'collar',
];

const COLOR_ALIAS_MAP = new Map([
  ['#0a1931', 'navy'], ['0a1931', 'navy'],
  ['gray', 'grey'], ['ivory', 'cream'], ['off-white', 'cream'], ['ecru', 'cream'],
  ['taupe', 'beige'], ['sand', 'beige'], ['slate', 'grey'],
  ['chocolate', 'brown'], ['espresso', 'brown'], ['cognac', 'brown'],
  ['oxblood', 'burgundy'], ['bordeaux', 'burgundy'], ['wine', 'burgundy'],
  ['sage', 'olive'], ['forest', 'green'], ['sky', 'blue'],
]);

const WARM_COLORS = new Set(['cream', 'beige', 'camel', 'tan', 'brown', 'olive', 'khaki', 'rust', 'burgundy', 'stone', 'yellow', 'orange']);
const COOL_COLORS = new Set(['navy', 'blue', 'grey', 'charcoal', 'black', 'white', 'green', 'purple', 'pink']);
const NEUTRAL_COLORS = new Set(['black', 'white', 'cream', 'beige', 'stone', 'grey', 'charcoal', 'navy', 'brown', 'camel', 'tan', 'khaki', 'olive']);

const FORMALITY_RANK = { athletic: 0, casual: 1, smart: 2, formal: 3 };

// Structural archetype-to-subtype/text matchers so Creative Director concepts map cleanly
// onto real catalog garments without fragile per-product rules.
const OUTERWEAR_ARCHETYPES = {
  structured_tailoring: { subtypes: new Set(['blazer', 'sport-coat']), text: /\b(blazer|suit jacket|sport[- ]?coat|tailored jacket|double[- ]breasted|unstructured jacket)\b/i },
  soft_draped_knit: { subtypes: new Set(['cardigan', 'knit-jacket']), text: /\b(cardigan|shawl[- ]collar|knit[- ]?jacket|sweater[- ]?jacket|knitted jacket)\b/i },
  relaxed_utility_overshirt: { subtypes: new Set(['overshirt', 'chore-jacket', 'shacket', 'jacket']), text: /\b(overshirt|shacket|chore|field jacket|safari|utility|barn jacket|work jacket|shirt[- ]?jacket)\b/i },
  longline_coat: { subtypes: new Set(['coat']), text: /\b(coat|overcoat|topcoat|trench|mac|peacoat|parka|duster)\b/i },
  cropped_blouson_jacket: { subtypes: new Set(['bomber', 'suede-jacket', 'jacket']), text: /\b(bomber|blouson|harrington|trucker|biker|leather jacket|suede jacket|varsity)\b/i },
  leather_statement: { subtypes: new Set(['jacket', 'bomber', 'suede-jacket']), text: /\b(leather|suede|shearling|biker|moto)\b/i },
  suede_jacket: { subtypes: new Set(['suede-jacket', 'jacket', 'bomber', 'overshirt']), text: /\b(suede\s+(?:jacket|blouson|bomber|overshirt|trucker|shirt[- ]?jacket))\b/i },
  breathable_light_layer: { subtypes: new Set(['overshirt', 'cardigan', 'blazer']), text: /\b(linen|cotton[- ]?linen|cotton|unlined|lightweight|seersucker|open[- ]weave|chambray|gauze|lyocell)\b/i },
};

const TROUSER_ARCHETYPES = {
  tailored_pleated: { subtypes: new Set(['tailored-trouser', 'trouser']), text: /\b(pleat|pleated|tailored|flannel|wool trouser|dress trouser|crease)\b/i },
  clean_straight_trouser: { subtypes: new Set(['trouser', 'tailored-trouser']), text: /\b(straight|tailored|trouser|slack|gabardine|twill)\b/i },
  relaxed_chino: { subtypes: new Set(['chino']), text: /\b(chino|cotton twill|relaxed chino)\b/i },
  refined_dark_denim: { subtypes: new Set(['jeans']), text: /\b(jeans|denim|selvedge|indigo|raw denim)\b/i },
  fluid_wide_leg: { subtypes: new Set(['trouser', 'tailored-trouser', 'knit-trouser']), text: /\b(wide|relaxed|fluid|drape|linen trouser|drawstring)\b/i },
  tapered_smart_pant: { subtypes: new Set(['chino', 'trouser', 'tailored-trouser']), text: /\b(tapered|slim|cropped|smart)\b/i },
  athletic_leisure: { subtypes: new Set(['athletic-trouser', 'knit-trouser']), text: /\b(jogger|track|sweatpant|athletic|knit trouser)\b/i },
};

const FOOTWEAR_ARCHETYPES = {
  sleek_lace_up_dress: { subtypes: new Set(['oxford', 'derby', 'monk']), text: /\b(oxford|wholecut|cap[- ]toe|balmoral|monk|dress shoe|derby)\b/i },
  soft_suede_loafer: { subtypes: new Set(['loafer', 'driver']), text: /\b(loafer|penny|tassel|belgian|suede|horsebit|moccasin|driver)\b/i },
  sculptural_chelsea_boot: { subtypes: new Set(['chelsea', 'dress-boot', 'boot']), text: /\b(chelsea|chukka|jodhpur|dress boot|ankle boot|boot)\b/i },
  clean_minimal_sneaker: { subtypes: new Set(['minimal-sneaker', 'sneaker']), text: /\b(minimal|court|low[- ]top|leather sneaker|sneaker|trainer)\b/i },
  refined_derby_or_monk: { subtypes: new Set(['derby', 'monk', 'brogue']), text: /\b(derby|blucher|monk|brogue|wingtip|split[- ]toe)\b/i },
  warm_weather_driver_espadrille: { subtypes: new Set(['driver', 'espadrille', 'loafer', 'sandal']), text: /\b(espadrille|driver|driving|unlined loafer|boat|deck|sandal)\b/i },
  athletic_trainer: { subtypes: new Set(['sport-sneaker', 'sneaker']), text: /\b(running|training|sport|athletic|runner)\b/i },
};

const TEXTURE_KEYWORDS = {
  suede: /\bsuede\b/i,
  leather: /\b(leather|calfskin|nappa|cordovan|grain leather)\b/i,
  denim: /\b(denim|selvedge|chambray)\b/i,
  linen: /\b(linen|flax)\b/i,
  wool: /\b(wool|flannel|worsted|merino|virgin wool|lambswool)\b/i,
  cashmere: /\b(cashmere|cashwool)\b/i,
  knit: /\b(knit|knitted|ribbed|cable|jersey|boucle|crochet|waffle|pique)\b/i,
  tweed: /\b(tweed|herringbone|houndstooth|donegal|boucle)\b/i,
  corduroy: /\b(corduroy|cord)\b/i,
  cotton: /\b(cotton|poplin|oxford cloth|twill|gabardine|canvas)\b/i,
  silk: /\b(silk|satin|velvet)\b/i,
  technical: /\b(technical|nylon|shell|ripstop|gore[- ]?tex|waterproof|mesh)\b/i,
  smooth: /\b(smooth|poplin|fine|worsted|calfskin|polished)\b/i,
};

const PULLOVER_KNIT_BASE_RE = /\b(?:knits?|sweaters?|turtlenecks?|mocknecks?|mock\s+necks?|roll[- ]?necks?|high[- ]?necks?|crewnecks?|crew\s+necks?|jumpers?|fair[- ]?isles?|cashmere|merino|wool)\b/i;

const safe = (x, n = 65) =>
  typeof x === 'string'
    ? x.replace(/https?:\/\/[^\s]+|\b[\w.%+-]+@[\w.-]+\.[a-z]{2,}\b/gi, '[redacted]')
      .replace(/[\u0000-\u001f<>{}]/g, ' ').trim().slice(0, n)
    : '';

export function canonicalColor(raw) {
  if (typeof raw !== 'string') return null;
  const cleaned = raw.trim().toLowerCase().replace(/[\s_]+/g, '-');
  if (!cleaned || ['unknown', 'neutral', 'missing', 'other', 'multi', 'multicolor'].includes(cleaned)) return null;
  if (COLOR_ALIAS_MAP.has(cleaned)) return COLOR_ALIAS_MAP.get(cleaned);
  const stripped = cleaned.replace(/^(?:light|dark|deep|pale|muted|dusty|soft|mid|warm|cool|rich|bright|washed)-/, '');
  if (COLOR_ALIAS_MAP.has(stripped)) return COLOR_ALIAS_MAP.get(stripped);
  if (WARM_COLORS.has(stripped) || COOL_COLORS.has(stripped) || NEUTRAL_COLORS.has(stripped)) return stripped;
  return /^[a-z]{3,16}$/.test(stripped) ? stripped : null;
}

function getVisualAttrs(item) {
  const vf = currentVisual({
    ...item,
    visualFashion: item?.visualFashion || item?.visual_fashion,
    visualSourceHash: item?.visualSourceHash || item?.visual_source_hash,
    imageUrl: item?.imageUrl || item?.image_url,
    image_url: item?.image_url || item?.imageUrl,
  });
  if (!vf?.attributes) return {};
  return Object.fromEntries(PHOTO_FIELDS.flatMap(field => {
    const obs = vf.attributes[field];
    return confidentVisual(obs) && typeof obs.value === 'string' &&
      /^[a-z0-9 _-]{1,45}$/i.test(obs.value)
      ? [[field, obs.value.toLowerCase()]]
      : [];
  }));
}

function resolveEnrichedItem(item, qualityMap) {
  const rawSlot = String(
    item?.macroCategory ||
    item?.fashion_slot ||
    item?.garment_intelligence?.descriptor?.slot ||
    '',
  ).trim().toLowerCase();
  const rawSubtype = String(
    item?.subtype ||
    item?.fashion_subtype ||
    item?.garment_intelligence?.descriptor?.subtype ||
    'unknown',
  ).trim().toLowerCase().replace(/_/g, '-');

  // Preserve wearing-role projection for raw catalog knit pullovers filed under outerwear.
  let slot = rawSlot;
  let subtype = rawSubtype;
  const nameText = `${item?.name || item?.title || ''} ${item?.description || ''}`;
  if (!item?.macroCategory && slot === 'outerwear' && subtype === 'pullover' &&
      !/\bcardigans?\b/i.test(nameText) && PULLOVER_KNIT_BASE_RE.test(nameText)) {
    slot = 'top';
    subtype = 'fine-knit';
  }

  const photo = getVisualAttrs(item);
  const intel = item?.garment_intelligence?.descriptor || {};
  const textColor = canonicalColor(item?.color) ||
    canonicalColor(intel.color) ||
    canonicalColor(inferCatalogColor(item));
  const color = canonicalColor(photo.color) || textColor || 'unknown';
  const colorVerifiedByPhoto = !!canonicalColor(photo.color);

  const inferredTex = inferTexture(item?.name || item?.title || '');
  const texture = (photo.texture || item?.texture || (intel.texture && intel.texture !== 'unknown' ? intel.texture : null) || inferredTex || 'unknown').toLowerCase();
  const formality = (photo.formality || item?.formality || item?.fashion_formality || intel.formality || 'unknown').toLowerCase();
  const volume = (photo.volume || item?.volume || item?.fit || intel.volume || 'unknown').toLowerCase();
  const pattern = (photo.pattern || item?.pattern || intel.pattern || 'unknown').toLowerCase();
  const bulk = (photo.bulk || intel.bulk || 'unknown').toLowerCase();
  const length = (photo.length || intel.length || 'unknown').toLowerCase();
  const hemShape = (photo.hemShape || intel.hemShape || 'unknown').toLowerCase();
  const insulated = Boolean(intel.insulated) || /\b(puffer|padded|quilted|down[- ]filled|primaloft|thinsulate|shearling|insulated)\b/i.test(nameText);
  const materials = Array.isArray(intel.materials) ? intel.materials.map(m => String(m).toLowerCase()) : [];

  const quality = lookupQualityEntry(item, qualityMap);
  const resolvedSubtype = photo.subtype && subtype === 'unknown' ? photo.subtype : subtype;
  const searchText = `${item?.name || item?.title || ''} ${item?.description || ''} ${materials.join(' ')} ${resolvedSubtype} ${texture}`.toLowerCase();
  const layerRole = classifyOuterwearLayerRole({
    macroCategory: slot,
    subtype: resolvedSubtype,
    name: item?.name || item?.title || '',
    description: item?.description || '',
    materials,
    insulated,
    bulk,
  });
  const isPeacoat = slot === 'outerwear' && /\bpeacoats?\b/i.test(searchText);
  const isCommuterRainOrParka = slot === 'outerwear' && (
    insulated ||
    /\b(parkas?|puffers?|raincoats?|anoraks?|waterproof|commuter\s+(?:trench|coat)|down\s+(?:coat|jacket|parka)|primaloft|thinsulate|ski\s+jacket|windbreaker)\b/i.test(searchText)
  );
  const isHeavyWarmWeatherHazard = (
    (slot === 'outerwear' && (
      layerRole === 'outdoor_heavy_coat' ||
      insulated ||
      bulk === 'heavy' ||
      ['coat', 'pullover', 'hoodie'].includes(resolvedSubtype) ||
      /\b(wool|cashmere|merino|alpaca|mohair|lambswool|flannel|tweed|corduroy|velvet|fleece|waxed|chunky|shawl[- ]?(?:collar|neck)|heavyweight|leather|suede|biker|moto|shearling|padded|quilted|primaloft)\b/i.test(searchText)
    )) ||
    (slot === 'top' && (
      bulk === 'heavy' ||
      /\b(turtlenecks?|mock[- ]?necks?|roll[- ]?necks?|chunky\s+knit|cable[- ]?knit|heavy\s+wool|cashmere\s+sweater|wool\s+sweater|alpaca|mohair|fleece|heavy\s+flannel|sherpa|hoodie|sweatshirt)\b/i.test(searchText)
    )) ||
    (slot === 'bottom' && (
      bulk === 'heavy' ||
      /\b(heavy\s+wool|flannel|tweed|corduroy|fleece|thermal|insulated|snow)\b/i.test(searchText)
    )) ||
    (slot === 'shoes' && (
      ['boot', 'chelsea', 'dress-boot', 'hiking-boot'].includes(resolvedSubtype) ||
      /\b(boots?|chelsea|chukka|shearling|fleece[- ]?lined|winter\s+boot|snow\s+boot)\b/i.test(searchText)
    ))
  );

  return {
    item,
    id: String(item?.id || ''),
    baseId: baseProductId(String(item?.id || '')),
    slot,
    subtype: resolvedSubtype,
    name: String(item?.name || item?.title || ''),
    searchText,
    brand: String(item?.brand || '').trim().toLowerCase(),
    color,
    colorVerifiedByPhoto,
    texture,
    formality,
    volume,
    pattern,
    bulk,
    length,
    hemShape,
    insulated,
    materials,
    layerRole,
    isPeacoat,
    isCommuterRainOrParka,
    isHeavyWarmWeatherHazard,
    price: Number(item?.price),
    currency: String(item?.currency || '').trim().toUpperCase(),
    imageUrl: String(item?.imageUrl || item?.image_url || ''),
    photo,
    photoFactCount: Object.keys(photo).length,
    qualityScore: quality.qualityScore,
    quarantined: quality.quarantined,
  };
}

function resolveNumericTemp(context) {
  const raw = context?.weather?.temp ?? context?.weatherCelsius;
  if (raw === null || raw === undefined || raw === '') return null;
  const temp = Number(raw);
  return Number.isFinite(temp) ? temp : null;
}

function isIndoorEditorialOccasion(context) {
  const text = `${context?.occasion || ''} ${context?.dressCode || ''} ${context?.brief || ''} ${context?.request || ''}`.toLowerCase();
  if (/\b(outdoor|hike|hiking|ski|snow|storm|rainy\s+commute|winter\s+walk|camping)\b/.test(text)) return false;
  return /\b(work|office|business|meeting|conference|date|dinner|cocktail|smart[-_ ]?casual|interview|presentation|gallery|brunch|evening)\b/.test(text);
}

function isDateOrDinnerOccasion(context) {
  const text = `${context?.occasion || ''} ${context?.dressCode || ''}`.toLowerCase();
  return /\b(date|dinner|cocktail|evening|anniversary|night\s+out)\b/.test(text);
}

function resolveThermalBand(direction, context) {
  const explicit = String(direction?.thermalBand || '').trim().toLowerCase();
  if (['hot', 'warm', 'mild', 'transitional', 'cool', 'cold'].includes(explicit)) {
    if (explicit === 'mild') return 'transitional';
    return explicit;
  }
  const temp = resolveNumericTemp(context);
  if (temp === null) return 'transitional';
  if (temp >= 26) return 'hot';
  if (temp >= 24) return 'warm';
  if (temp > 13) return 'transitional';
  if (temp > 10) return 'cool';
  return 'cold';
}

function resolveTargetFormality(direction, context) {
  const dirF = String(direction?.formality || '').trim().toLowerCase();
  if (Object.hasOwn(FORMALITY_RANK, dirF)) return dirF;
  const occasion = `${context?.occasion || ''} ${context?.dressCode || ''}`.toLowerCase();
  if (/\b(workout|gym|training|sport|athletic|run)\b/.test(occasion)) return 'athletic';
  if (/\b(formal|black tie|gala|ceremony|wedding)\b/.test(occasion)) return 'formal';
  if (/\b(work|office|business|date|dinner|evening|smart)\b/.test(occasion)) return 'smart';
  return 'casual';
}

function inferArchetypeKey(slot, direction) {
  const raw = slot === 'outerwear'
    ? (direction?.outerwearArchetype || direction?.outerwearFamily || '')
    : slot === 'bottom'
      ? (direction?.trouserArchetype || direction?.bottomFamily || '')
      : slot === 'shoes'
        ? (direction?.footwearArchetype || direction?.footwearFamily || '')
        : '';
  const norm = String(raw).trim().toLowerCase().replace(/[\s-]+/g, '_');
  if (norm) return norm;
  // Infer gracefully from roles or artDirection when callers supply minimal direction specs.
  const roleSubtypes = Array.isArray(direction?.roles?.[slot]) ? direction.roles[slot].map(s => String(s).toLowerCase()) : [];
  const art = String(direction?.artDirection || direction?.name || '').toLowerCase();
  if (slot === 'outerwear') {
    if (roleSubtypes.includes('blazer') || /\bblazer|tailored jacket|sport[- ]?coat\b/.test(art)) return 'structured_tailoring';
    if (roleSubtypes.includes('coat') || /\bcoat|trench|overcoat|long\b/.test(art)) return 'longline_coat';
    if (roleSubtypes.includes('overshirt') || /\bovershirt|chore|utility|shacket\b/.test(art)) return 'relaxed_utility_overshirt';
    if (roleSubtypes.includes('cardigan') || /\bcardigan|knit layer|knit jacket\b/.test(art)) return 'soft_draped_knit';
    if (roleSubtypes.includes('bomber') || roleSubtypes.includes('jacket') || /\bbomber|leather|blouson|suede\b/.test(art)) return 'cropped_blouson_jacket';
  }
  if (slot === 'bottom') {
    if (roleSubtypes.includes('tailored-trouser') || /\bpleat|tailored trouser|flannel\b/.test(art)) return 'tailored_pleated';
    if (roleSubtypes.includes('jeans') || /\bdenim|jeans\b/.test(art)) return 'refined_dark_denim';
    if (roleSubtypes.includes('chino') || /\bchino\b/.test(art)) return 'relaxed_chino';
    if (/\bwide|fluid|drape\b/.test(art)) return 'fluid_wide_leg';
  }
  if (slot === 'shoes') {
    if (roleSubtypes.includes('oxford') || /\boxford|lace[- ]up\b/.test(art)) return 'sleek_lace_up_dress';
    if (roleSubtypes.includes('loafer') || /\bloafer|suede shoe\b/.test(art)) return 'soft_suede_loafer';
    if (roleSubtypes.includes('chelsea') || roleSubtypes.includes('boot') || /\bchelsea|boot\b/.test(art)) return 'sculptural_chelsea_boot';
    if (roleSubtypes.includes('minimal-sneaker') || roleSubtypes.includes('sneaker') || /\bsneaker|trainer\b/.test(art)) return 'clean_minimal_sneaker';
    if (roleSubtypes.includes('derby') || roleSubtypes.includes('monk') || /\bderby|monk\b/.test(art)) return 'refined_derby_or_monk';
  }
  return '';
}

function matchesArchetype(enriched, slot, archetypeKey) {
  if (!archetypeKey) return false;
  const table = slot === 'outerwear' ? OUTERWEAR_ARCHETYPES
    : slot === 'bottom' ? TROUSER_ARCHETYPES
      : slot === 'shoes' ? FOOTWEAR_ARCHETYPES : null;
  if (!table) return false;
  const spec = table[archetypeKey];
  if (spec) {
    return spec.subtypes.has(enriched.subtype) || spec.text.test(enriched.searchText);
  }
  // Fuzzy keyword match for custom creative archetype labels (e.g. "suede_bomber", "pleated_wool").
  const tokens = archetypeKey.split('_').filter(t => t.length >= 3);
  return tokens.some(tok => enriched.subtype.includes(tok) || enriched.searchText.includes(tok));
}

function extractTargetTextures(direction) {
  const raw = [
    direction?.materialTexture,
    direction?.textureContrast,
    direction?.artDirection,
  ].filter(Boolean).join(' ').toLowerCase();
  const matched = [];
  for (const [tex, re] of Object.entries(TEXTURE_KEYWORDS)) {
    if (re.test(raw)) matched.push(tex);
  }
  return matched;
}

function buildWithinCurrencyPriceBands(enrichedItems) {
  const buckets = new Map();
  for (const e of enrichedItems) {
    if (!Number.isFinite(e.price) || e.price <= 0 || !e.currency) continue;
    const key = `${e.slot}|${e.currency}`;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(e.price);
  }
  const bands = new Map();
  for (const [key, prices] of buckets) {
    prices.sort((a, b) => a - b);
    const p35 = prices[Math.floor((prices.length - 1) * 0.35)];
    const p70 = prices[Math.floor((prices.length - 1) * 0.70)];
    bands.set(key, { p35, p70, min: prices[0], max: prices[prices.length - 1], count: prices.length });
  }
  return bands;
}

const SLOT_BUDGET_HEADROOM = {
  top: 0.50,
  outerwear: 0.68,
  bottom: 0.55,
  shoes: 0.65,
  dress: 0.75,
  accessory: 0.35,
};

const TIGHT_SLOT_BUDGET_HEADROOM = {
  top: 0.22,
  outerwear: 0.38,
  bottom: 0.28,
  shoes: 0.44,
  dress: 0.55,
  accessory: 0.22,
};

const TIGHT_SLOT_IDEAL_SHARE = {
  top: 0.17,
  outerwear: 0.30,
  bottom: 0.22,
  shoes: 0.35,
  dress: 0.45,
  accessory: 0.16,
};

const TIGHT_BUDGET_USD_THRESHOLD = 500;

function resolveVerifiedFxRates(context, extraFx = null) {
  const candidates = [
    extraFx?.uzsPerUnit,
    extraFx,
    context?.fxSnapshot?.uzsPerUnit,
    context?.budgetFx?.snapshot?.uzsPerUnit,
    context?.budgetFx?.uzsPerUnit,
    context?.budget?.fxSnapshot?.uzsPerUnit,
    context?.valueFxRates,
  ];
  for (const rates of candidates) {
    if (!rates || typeof rates !== 'object') continue;
    const cleaned = {};
    let validCount = 0;
    for (const [cur, val] of Object.entries(rates)) {
      const code = String(cur || '').trim().toUpperCase();
      const num = Number(val);
      if (/^[A-Z]{3}$/.test(code) && Number.isFinite(num) && num > 0) {
        cleaned[code] = num;
        validCount++;
      }
    }
    if (validCount >= 2) return cleaned;
  }
  return null;
}

function convertItemPriceToCurrency(enriched, targetCurrency, context, extraConverter = null, fxRates = null) {
  const target = String(targetCurrency || '').trim().toUpperCase();
  if (!target) return null;
  const fn = typeof extraConverter === 'function'
    ? extraConverter
    : (typeof context?.priceToBudgetCurrency === 'function' ? context.priceToBudgetCurrency : null);
  if (fn) {
    try {
      const val = fn(enriched.item, target, context);
      if (Number.isFinite(val) && val > 0) return val;
    } catch { /* fall through */ }
  }
  if (!Number.isFinite(enriched.price) || enriched.price <= 0 || !enriched.currency) return null;
  if (enriched.currency === target) return enriched.price;
  if (fxRates && Number.isFinite(fxRates[enriched.currency]) && Number.isFinite(fxRates[target])) {
    const converted = (enriched.price * fxRates[enriched.currency]) / fxRates[target];
    return Number.isFinite(converted) && converted > 0 ? converted : null;
  }
  return null;
}

function computeDynamicBudgetPlan({ bySlot, activeSlots, context, limit, directionCount, priceToBudgetCurrency = null, fxSnapshot = null }) {
  const budget = context?.budget;
  const budgetAmount = Number(budget?.amount);
  const budgetCurrency = String(budget?.currency || '').trim().toUpperCase();
  if (!Number.isFinite(budgetAmount) || budgetAmount <= 0 || !budgetCurrency) {
    return null;
  }
  const fxRates = resolveVerifiedFxRates(context, fxSnapshot);

  let budgetUsdEquivalent = null;
  if (budgetCurrency === 'USD') {
    budgetUsdEquivalent = budgetAmount;
  } else if (fxRates && Number.isFinite(fxRates[budgetCurrency]) && Number.isFinite(fxRates.USD)) {
    budgetUsdEquivalent = (budgetAmount * fxRates[budgetCurrency]) / fxRates.USD;
  }

  // Compute honest converted price distributions per active slot.
  const convertedBySlot = new Map();
  const minFloorBySlot = new Map();
  const p25BySlot = new Map();
  for (const slot of activeSlots) {
    const prices = [];
    for (const e of bySlot.get(slot) || []) {
      const c = convertItemPriceToCurrency(e, budgetCurrency, context, priceToBudgetCurrency, fxRates);
      if (c !== null) prices.push(c);
    }
    prices.sort((a, b) => a - b);
    convertedBySlot.set(slot, prices);
    if (prices.length > 0) {
      // Reserve realistic multi-look floor (3rd cheapest or 5th percentile) so 3 disjoint looks stay feasible.
      const floorIdx = Math.min(prices.length - 1, Math.min(2, Math.floor(prices.length * 0.05)));
      minFloorBySlot.set(slot, prices[floorIdx]);
      p25BySlot.set(slot, prices[Math.floor((prices.length - 1) * 0.25)]);
    } else {
      minFloorBySlot.set(slot, 0);
      p25BySlot.set(slot, 0);
    }
  }

  const p25FourSlotSum = activeSlots.slice(0, 4).reduce((sum, s) => sum + (p25BySlot.get(s) || 0), 0);
  const isTight = (
    (budgetUsdEquivalent !== null && budgetUsdEquivalent <= TIGHT_BUDGET_USD_THRESHOLD + 1e-6) ||
    (budgetUsdEquivalent === null && p25FourSlotSum > 0 && budgetAmount <= p25FourSlotSum * 1.15)
  );

  const slotCaps = {};
  const coreSlots = activeSlots.slice(0, 4);
  for (const slot of activeSlots) {
    const standardCap = budgetAmount * (SLOT_BUDGET_HEADROOM[slot] ?? 0.60);
    const otherMinReserve = coreSlots
      .filter(o => o !== slot)
      .reduce((sum, o) => sum + (minFloorBySlot.get(o) || 0), 0);
    const dynamicReserveCap = otherMinReserve > 0 && budgetAmount > otherMinReserve
      ? budgetAmount - otherMinReserve
      : standardCap;

    if (!isTight) {
      slotCaps[slot] = Math.min(standardCap, dynamicReserveCap > 0 ? Math.max(dynamicReserveCap, standardCap * 0.85) : standardCap);
      continue;
    }

    const tightShareCap = budgetAmount * (TIGHT_SLOT_BUDGET_HEADROOM[slot] ?? 0.35);
    const preferredTightCap = Math.min(
      tightShareCap,
      dynamicReserveCap > 0 ? dynamicReserveCap : tightShareCap,
    );
    const prices = convertedBySlot.get(slot) || [];
    const neededHealthy = Math.min(limit * Math.max(1, directionCount), 18);
    const countAtPreferred = prices.filter(p => p <= preferredTightCap + 1e-6).length;
    const countAtShare = prices.filter(p => p <= tightShareCap + 1e-6).length;

    if (countAtPreferred >= neededHealthy || countAtPreferred >= Math.min(limit, 12)) {
      slotCaps[slot] = preferredTightCap;
    } else if (countAtShare >= Math.min(limit, 12)) {
      slotCaps[slot] = tightShareCap;
    } else {
      // Smooth fallback on tiny synthetic catalogs while still respecting dynamic 3-slot headroom.
      const relaxed = Math.min(standardCap, Math.max(tightShareCap, dynamicReserveCap));
      slotCaps[slot] = relaxed;
    }
  }

  return {
    budgetAmount,
    budgetCurrency,
    budgetUsdEquivalent,
    isTight,
    slotCaps,
    fxRates,
    priceToBudgetCurrency,
  };
}

function evaluatePriceTierAndBudget(enriched, bands, context, budgetPlan = null) {
  const budget = context?.budget;
  const budgetAmount = budgetPlan?.budgetAmount ?? Number(budget?.amount);
  const budgetCurrency = budgetPlan?.budgetCurrency ?? String(budget?.currency || '').trim().toUpperCase();

  if (Number.isFinite(budgetAmount) && budgetAmount > 0 && budgetCurrency) {
    const converted = convertItemPriceToCurrency(
      enriched,
      budgetCurrency,
      context,
      budgetPlan?.priceToBudgetCurrency || null,
      budgetPlan?.fxRates || null,
    );
    if (converted !== null) {
      const cap = budgetPlan?.slotCaps?.[enriched.slot] ?? (budgetAmount * (SLOT_BUDGET_HEADROOM[enriched.slot] ?? 0.60));
      const withinHeadroom = converted <= cap + 1e-6;
      const share = converted / budgetAmount;
      const ideal = TIGHT_SLOT_IDEAL_SHARE[enriched.slot] ?? 0.25;
      const tier = budgetPlan?.isTight
        ? (share <= ideal ? 'accessible' : share <= (TIGHT_SLOT_BUDGET_HEADROOM[enriched.slot] ?? 0.35) * 0.90 ? 'mid' : 'elevated')
        : (share <= 0.24 ? 'accessible' : share <= 0.40 ? 'mid' : 'elevated');
      return { priceTier: tier, withinHeadroom, budgetShare: share, convertedPrice: converted, budgetChecked: true };
    }
    // Fail closed on unconvertible shop currencies when an explicit spending cap is active
    // (never fabricate FX or admit INR/unknown-currency SKUs into a USD/EUR/GBP/UZS budgeted pool).
    if (enriched.item?.isShopItem !== false) {
      return {
        priceTier: 'elevated',
        withinHeadroom: false,
        budgetShare: null,
        convertedPrice: null,
        budgetChecked: true,
        unconvertibleCurrency: true,
      };
    }
  }

  const band = bands.get(`${enriched.slot}|${enriched.currency}`);
  if (!band || !Number.isFinite(enriched.price)) {
    return { priceTier: 'mid', withinHeadroom: true, budgetShare: null, convertedPrice: null, budgetChecked: false };
  }
  const tier = enriched.price <= band.p35 ? 'accessible'
    : enriched.price <= band.p70 ? 'mid' : 'elevated';
  return { priceTier: tier, withinHeadroom: true, budgetShare: null, convertedPrice: null, budgetChecked: false };
}

function collectExclusions(context, personalStyleDna, tasteProfile) {
  const avoidColors = new Set();
  const dislikedPatterns = new Set();
  const brandExclusions = new Set();

  const addColors = list => {
    for (const c of Array.isArray(list) ? list : []) {
      const canon = canonicalColor(c);
      if (canon) avoidColors.add(canon);
    }
  };
  const addPatterns = list => {
    for (const p of Array.isArray(list) ? list : []) {
      const raw = String(p || '').trim().toLowerCase();
      if (!raw) continue;
      if (/stripe|pinstripe/.test(raw)) dislikedPatterns.add('stripe');
      else if (/check|plaid|gingham|tartan/.test(raw)) dislikedPatterns.add('check');
      else if (/print|graphic|logo|floral|camo/.test(raw)) dislikedPatterns.add('print');
      else if (/solid|plain/.test(raw)) dislikedPatterns.add('solid');
    }
  };
  const addBrands = list => {
    for (const b of Array.isArray(list) ? list : []) {
      const clean = String(b || '').trim().toLowerCase();
      if (clean) brandExclusions.add(clean);
    }
  };

  addColors(context?.preferences?.avoidColors);
  addColors(context?.preferences?.avoidedColors);
  addColors(personalStyleDna?.avoidColors);
  addColors(personalStyleDna?.preferences?.avoidColors);
  addColors(personalStyleDna?.preferences?.avoidedColors);

  addPatterns(context?.preferences?.dislikedPatterns);
  addPatterns(context?.dislikedPatterns);
  addPatterns(personalStyleDna?.dislikedPatterns);
  addPatterns(personalStyleDna?.preferences?.dislikedPatterns);

  addBrands(context?.preferences?.avoidBrands);
  addBrands(personalStyleDna?.brandExclusions);
  addBrands(personalStyleDna?.preferences?.brandExclusions);
  addBrands(tasteProfile?.excludedBrands);

  return { avoidColors, dislikedPatterns, brandExclusions };
}

function computeTasteAdjustment(enriched, tasteProfile, personalStyleDna) {
  let adj = 0;

  // 1. Legacy feedback_taste.mjs itemTally Map
  if (tasteProfile?.active && tasteProfile.itemTally instanceof Map) {
    const score = tasteProfile.itemTally.get(enriched.id);
    if (Number.isFinite(score)) adj += Math.max(-0.08, Math.min(0.08, score * 0.08));
  }

  // 2. Slot-aware taste_learning.mjs profile (optional peer module contract)
  const slotPref = tasteProfile?.slotPreferences?.[enriched.slot];
  if (slotPref && typeof slotPref === 'object') {
    const subBoost = Number(slotPref.subtypes?.[enriched.subtype] ?? slotPref.subtypeWeights?.[enriched.subtype]);
    if (Number.isFinite(subBoost)) adj += Math.max(-0.06, Math.min(0.06, subBoost * 0.06));
    const colBoost = Number(slotPref.colors?.[enriched.color] ?? slotPref.colorWeights?.[enriched.color]);
    if (Number.isFinite(colBoost)) adj += Math.max(-0.05, Math.min(0.05, colBoost * 0.05));
  }
  if (tasteProfile?.brandAffinity && typeof tasteProfile.brandAffinity === 'object' && enriched.brand) {
    const bVal = tasteProfile.brandAffinity instanceof Map
      ? tasteProfile.brandAffinity.get(enriched.brand)
      : tasteProfile.brandAffinity[enriched.brand];
    if (Number.isFinite(bVal)) adj += Math.max(-0.04, Math.min(0.04, bVal * 0.04));
  }
  if (tasteProfile?.colorAffinity && typeof tasteProfile.colorAffinity === 'object' && enriched.color !== 'unknown') {
    const cVal = tasteProfile.colorAffinity instanceof Map
      ? tasteProfile.colorAffinity.get(enriched.color)
      : tasteProfile.colorAffinity[enriched.color];
    if (Number.isFinite(cVal)) adj += Math.max(-0.04, Math.min(0.04, cVal * 0.04));
  }

  // 3. Owned wardrobe DNA continuity (soft preference, never a rigid rule)
  const wardrobeDna = personalStyleDna?.ownedWardrobeDna || personalStyleDna?.ownedWardrobeSummary;
  if (wardrobeDna && typeof wardrobeDna === 'object') {
    const domColors = new Set((wardrobeDna.dominantColors || []).map(canonicalColor).filter(Boolean));
    if (enriched.color !== 'unknown' && domColors.has(enriched.color)) adj += 0.02;
    const topBrands = new Set((wardrobeDna.topBrands || []).map(b => String(b).toLowerCase()));
    if (enriched.brand && topBrands.has(enriched.brand)) adj += 0.015;
  }

  return Math.max(-0.12, Math.min(0.12, adj));
}

/**
 * Score one enriched garment item against a specific Creative Director blueprint,
 * user Personal Style DNA, learned taste profile, thermal band, and catalog quality.
 */
export function scoreItemForDirection(enriched, direction, scoringCtx = {}) {
  const {
    context = {},
    personalStyleDna = null,
    tasteProfile = null,
    competingArchetypes = new Set(),
    directionAllowsTailoredCoat = false,
    budgetPlan = null,
    priceInfo = null,
    globalBrandCounts = null,
  } = scoringCtx;

  const slot = enriched.slot;
  const targetSubtypes = new Set(
    (Array.isArray(direction?.roles?.[slot]) ? direction.roles[slot] : [])
      .map(s => String(s || '').trim().toLowerCase()),
  );
  const archetypeKey = inferArchetypeKey(slot, direction);
  const targetPalette = new Set(
    (Array.isArray(direction?.palette) ? direction.palette : [])
      .map(canonicalColor).filter(Boolean),
  );
  const targetTextures = extractTargetTextures(direction);
  const targetFormality = resolveTargetFormality(direction, context);
  const thermalBand = resolveThermalBand(direction, context);
  const numericTemp = resolveNumericTemp(context);
  const indoorOccasion = isIndoorEditorialOccasion(context);
  const isWarmOrHot = thermalBand === 'hot' || thermalBand === 'warm' || (numericTemp !== null && numericTemp >= 24);
  const targetSilhouette = String(direction?.silhouette || 'balanced').toLowerCase();
  const layerRole = enriched.layerRole || classifyOuterwearLayerRole({
    macroCategory: slot,
    subtype: enriched.subtype,
    name: enriched.name || enriched.searchText || '',
    insulated: enriched.insulated,
    bulk: enriched.bulk,
  });
  const isPeacoat = enriched.isPeacoat === true || (slot === 'outerwear' && /\bpeacoats?\b/i.test(enriched.searchText || ''));
  const isCommuterRainOrParka = enriched.isCommuterRainOrParka === true || (
    slot === 'outerwear' && (
      enriched.insulated === true ||
      /\b(parkas?|puffers?|raincoats?|anoraks?|waterproof|commuter\s+(?:trench|coat)|down\s+(?:coat|jacket|parka)|primaloft|thinsulate|windbreaker)\b/i.test(enriched.searchText || '')
    )
  );
  const isHeavyWarmHazard = enriched.isHeavyWarmWeatherHazard === true || (
    (slot === 'outerwear' && (
      layerRole === 'outdoor_heavy_coat' ||
      enriched.insulated === true ||
      enriched.bulk === 'heavy' ||
      ['coat', 'pullover', 'hoodie'].includes(enriched.subtype) ||
      /\b(wool|cashmere|merino|alpaca|mohair|lambswool|flannel|tweed|corduroy|velvet|fleece|waxed|chunky|shawl[- ]?(?:collar|neck)|heavyweight|leather|suede|biker|moto|shearling|padded|quilted|primaloft)\b/i.test(enriched.searchText || '')
    )) ||
    (slot === 'top' && /\b(turtlenecks?|mock[- ]?necks?|roll[- ]?necks?|chunky\s+knit|cable[- ]?knit|heavy\s+wool|cashmere\s+sweater|wool\s+sweater|alpaca|mohair|fleece|hoodie|sweatshirt)\b/i.test(enriched.searchText || '')) ||
    (slot === 'bottom' && /\b(heavy\s+wool|flannel|tweed|corduroy|fleece|thermal|insulated|snow)\b/i.test(enriched.searchText || '')) ||
    (slot === 'shoes' && (['boot', 'chelsea', 'dress-boot', 'hiking-boot'].includes(enriched.subtype) || /\b(boots?|chelsea|chukka|shearling|winter\s+boot|snow\s+boot)\b/i.test(enriched.searchText || '')))
  );

  // 1. Role & Archetype Alignment (-0.05..0.30)
  let archetypeScore = 0;
  if (targetSubtypes.has(enriched.subtype)) archetypeScore += 0.22;
  if (archetypeKey && matchesArchetype(enriched, slot, archetypeKey)) {
    archetypeScore += targetSubtypes.has(enriched.subtype) ? 0.08 : 0.20;
  }
  // Differentiate away from another direction's claimed hero archetype when this direction asks for something else
  if (archetypeScore < 0.15 && competingArchetypes.size > 0) {
    for (const comp of competingArchetypes) {
      if (comp && comp !== archetypeKey && matchesArchetype(enriched, slot, comp)) {
        archetypeScore -= 0.05;
        break;
      }
    }
  }
  archetypeScore = Math.max(-0.05, Math.min(0.30, archetypeScore));

  // 2. Palette & Undertone Alignment (-0.25..0.20)
  let paletteScore = 0;
  if (enriched.color !== 'unknown') {
    if (targetPalette.has(enriched.color)) {
      paletteScore += enriched.colorVerifiedByPhoto ? 0.13 : 0.095;
    } else if (NEUTRAL_COLORS.has(enriched.color)) {
      paletteScore += 0.03;
    }
    const harmony = String(direction?.colorHarmonyMode || '').toLowerCase();
    if (harmony.includes('warm') && WARM_COLORS.has(enriched.color)) paletteScore += 0.025;
    if (harmony.includes('cool') && COOL_COLORS.has(enriched.color)) paletteScore += 0.025;

    const flattering = new Set([
      ...(personalStyleDna?.appearance?.flatteringColors || []),
      ...(personalStyleDna?.appearance?.powerNeutrals || []),
      ...(personalStyleDna?.favoriteColors || []),
      ...(context?.preferences?.favoriteColors || []),
    ].map(canonicalColor).filter(Boolean));
    const appearanceAvoid = new Set(
      (personalStyleDna?.appearance?.avoidColors || []).map(canonicalColor).filter(Boolean),
    );
    if (flattering.has(enriched.color)) paletteScore += 0.045;
    if (appearanceAvoid.has(enriched.color)) paletteScore -= 0.25;
  }
  paletteScore = Math.max(-0.25, Math.min(0.20, paletteScore));

  // 3. Material & Texture Alignment (0..0.12)
  let textureScore = 0;
  if (targetTextures.length > 0) {
    const matchedTex = targetTextures.some(t =>
      enriched.texture === t ||
      enriched.materials.includes(t) ||
      TEXTURE_KEYWORDS[t]?.test(enriched.searchText),
    );
    if (matchedTex) textureScore = 0.12;
  } else if (enriched.texture !== 'unknown') {
    textureScore = 0.03;
  }

  // 4. Silhouette & Body Proportion Fit (-0.04..0.12)
  let silhouetteScore = 0.03;
  if (targetSilhouette === 'tailored') {
    if (enriched.volume === 'slim' || enriched.volume === 'regular') silhouetteScore += 0.05;
    if (enriched.hemShape === 'tapered' || enriched.hemShape === 'straight') silhouetteScore += 0.02;
    if (enriched.volume === 'wide') silhouetteScore -= 0.03;
  } else if (targetSilhouette === 'relaxed') {
    if (enriched.volume === 'wide' || enriched.volume === 'regular') silhouetteScore += 0.05;
    if (enriched.volume === 'slim') silhouetteScore -= 0.02;
  } else {
    if (enriched.volume === 'regular') silhouetteScore += 0.05;
  }
  const bodyDirectives = personalStyleDna?.body?.directives || personalStyleDna?.bodyDirectives || null;
  const fitPref = String(
    personalStyleDna?.body?.fitPreference ||
    personalStyleDna?.fitPreference ||
    context?.preferences?.fitPreference ||
    '',
  ).toLowerCase();
  if (fitPref === 'fitted' && enriched.volume === 'slim') silhouetteScore += 0.03;
  if (fitPref === 'loose' && enriched.volume === 'wide') silhouetteScore += 0.03;
  if (fitPref === 'balanced' && enriched.volume === 'regular') silhouetteScore += 0.03;
  if (bodyDirectives) {
    const recCuts = Array.isArray(bodyDirectives.recommendedCuts) ? bodyDirectives.recommendedCuts : [];
    const avoidCuts = Array.isArray(bodyDirectives.avoidCuts) ? bodyDirectives.avoidCuts : [];
    if (recCuts.some(c => typeof c === 'string' && c.length >= 4 && enriched.searchText.includes(c.toLowerCase().split(' ')[0]))) {
      silhouetteScore += 0.03;
    }
    if (avoidCuts.some(c => typeof c === 'string' && c.length >= 4 && enriched.searchText.includes(c.toLowerCase()))) {
      silhouetteScore -= 0.05;
    }
  }
  silhouetteScore = Math.max(-0.04, Math.min(0.12, silhouetteScore));

  // 5. Formality & Shoe-to-Trouser Integrity Alignment (-0.22..0.14)
  let formalityScore = 0.04;
  if (Object.hasOwn(FORMALITY_RANK, enriched.formality) && Object.hasOwn(FORMALITY_RANK, targetFormality)) {
    const dist = Math.abs(FORMALITY_RANK[enriched.formality] - FORMALITY_RANK[targetFormality]);
    formalityScore = dist === 0 ? 0.10 : dist === 1 ? 0.06 : dist === 2 ? -0.04 : -0.10;
  }
  if (slot === 'shoes') {
    const trouserArch = inferArchetypeKey('bottom', direction);
    const outerArch = inferArchetypeKey('outerwear', direction);
    const casualTrouserDirection =
      trouserArch === 'relaxed_chino' ||
      trouserArch === 'refined_dark_denim' ||
      outerArch === 'relaxed_utility_overshirt' ||
      targetFormality === 'casual';
    const isHyperFormalBlackOxford =
      (enriched.subtype === 'oxford' && (enriched.color === 'black' || /\b(whole[- ]?cut|patent|balmoral)\b/i.test(enriched.searchText))) ||
      /\b(whole[- ]?cut\s+oxford|patent\s+oxford|black\s+patent)\b/i.test(enriched.searchText);
    if (casualTrouserDirection && isHyperFormalBlackOxford) {
      formalityScore -= 0.14;
    } else if (casualTrouserDirection && ['loafer', 'driver', 'derby', 'minimal-sneaker', 'espadrille'].includes(enriched.subtype)) {
      formalityScore += 0.04;
    }
    if ((targetFormality === 'formal' || /\b(work|office|business|date|dinner|cocktail|evening)\b/i.test(String(context?.occasion || ''))) &&
        ['sport-sneaker', 'hiking-boot'].includes(enriched.subtype)) {
      formalityScore -= 0.22;
    }
  }
  // Minimalist / Quiet Luxury restraint: penalize loud retro/busy patterns (argyle, fair isle, paisley, camo)
  const styleKeyLower = String(context?.style || '').trim().toLowerCase().replace(/[\s-]+/g, '_');
  if ((styleKeyLower === 'minimalist' || styleKeyLower === 'quiet_luxury') &&
      /\b(argyle|fair[- ]?isle|paisley|camouflage|camo|tie[- ]?dye|hawaiian|floral|tartan|checkerboard|color[- ]?block)\b/i.test(enriched.searchText || '')) {
    formalityScore -= 0.28;
  }

  // 6. Thermal Band & Indoor-Layer-First Alignment (-0.42..0.18)
  let thermalScore = 0.04;
  if (isWarmOrHot) {
    const trulyAiry = /\b(linen|cotton[- ]?linen|seersucker|unlined|gauze|chambray|open[- ]weave|breathable|lightweight|lyocell|tencel|summer|resort)\b/i.test(enriched.searchText);
    const lightSummerPiece = enriched.bulk === 'light' || trulyAiry ||
      (slot === 'top' && /\b(poplin|short[- ]sleeve|camp[- ]?collar|polo|t-shirt|tee|cotton)\b/i.test(enriched.searchText)) ||
      (slot === 'outerwear' && /\b(overshirt|shacket|chore|shirt[- ]?jacket|cardigan|blazer)\b/i.test(enriched.searchText) && /\b(cotton|linen|lyocell|viscose|seersucker|unlined|lightweight)\b/i.test(enriched.searchText)) ||
      (slot === 'bottom' && /\b(linen|chino|cotton|seersucker|lyocell|lightweight)\b/i.test(enriched.searchText)) ||
      (slot === 'shoes' && ['loafer', 'driver', 'espadrille', 'minimal-sneaker', 'sneaker'].includes(enriched.subtype));

    if (isHeavyWarmHazard) {
      thermalScore = -0.42;
    } else if (trulyAiry) {
      thermalScore = 0.18;
    } else if (lightSummerPiece) {
      thermalScore = 0.14;
    } else if (slot === 'outerwear' && ['overshirt', 'cardigan', 'blazer'].includes(enriched.subtype)) {
      thermalScore = 0.08;
    }
  } else if (thermalBand === 'cold' || (numericTemp !== null && numericTemp <= 10)) {
    const insulating = enriched.insulated || enriched.bulk === 'heavy' ||
      layerRole === 'outdoor_heavy_coat' ||
      ['coat', 'jacket', 'cardigan', 'blazer', 'chelsea', 'boot', 'dress-boot', 'fine-knit', 'sweater'].includes(enriched.subtype) ||
      /\b(wool|cashmere|flannel|tweed|corduroy|merino|lined|shearling|leather)\b/i.test(enriched.searchText);
    const tooSummer = /\b(linen|seersucker|espadrille|sandal|shorts|tank[- ]top|sleeveless|short[- ]sleeve)\b/i.test(enriched.searchText);
    if (tooSummer) thermalScore = -0.14;
    else if (insulating) thermalScore = 0.11;
  } else {
    // Cool / Mild / Transitional (>10C and <24C): Indoor-Layer-First editorial priority!
    const above10 = numericTemp === null ? true : numericTemp > 10;
    const above13 = numericTemp === null ? true : numericTemp > 13;
    if (slot === 'outerwear' && above10) {
      if (isCommuterRainOrParka) {
        thermalScore = -0.42;
      } else if (layerRole === 'outdoor_heavy_coat') {
        if (indoorOccasion) {
          if (above13 || !directionAllowsTailoredCoat) {
            thermalScore = isPeacoat ? -0.38 : -0.34;
          } else {
            // Optional single tailored-coat direction at 10-13C Work: refined tailored overcoat ok, never peacoat monopoly
            thermalScore = isPeacoat ? -0.22 : 0.04;
          }
        } else {
          thermalScore = above13 ? -0.20 : 0.04;
        }
      } else if (layerRole === 'indoor_refined_layer') {
        const refinedIndoorBoost = ['blazer', 'sport-coat', 'cardigan', 'overshirt', 'knit-jacket', 'chore-jacket', 'suede-jacket', 'shacket'].includes(enriched.subtype) ||
          /\b(blazer|sport[- ]?coat|cardigan|overshirt|knit[- ]?jacket|chore[- ]?(?:jacket|coat)|suede|shacket)\b/i.test(enriched.searchText);
        thermalScore = indoorOccasion
          ? (refinedIndoorBoost ? 0.17 : 0.13)
          : (refinedIndoorBoost ? 0.12 : 0.09);
      } else {
        thermalScore = 0.06;
      }
    } else if (enriched.insulated && slot === 'outerwear') {
      thermalScore = -0.24;
    } else if (['blazer', 'overshirt', 'cardigan', 'jacket', 'loafer', 'derby', 'minimal-sneaker', 'chino', 'tailored-trouser', 'fine-knit', 'shirt', 'dress-shirt', 'polo'].includes(enriched.subtype)) {
      thermalScore = 0.09;
    }
  }

  // 7. Personal Taste + Quality Flywheel + Tight-Budget Comfort + Anti-Monopoly + Photo Evidence
  const tasteAdj = computeTasteAdjustment(enriched, tasteProfile, personalStyleDna);
  const qualityAdj = (enriched.qualityScore - 0.65) * 0.14;
  const visualEvidenceBoost = Math.min(0.03, enriched.photoFactCount * 0.005);

  let budgetComfortAdj = 0;
  if (budgetPlan?.isTight && priceInfo?.budgetChecked && Number.isFinite(priceInfo.budgetShare)) {
    const ideal = TIGHT_SLOT_IDEAL_SHARE[slot] ?? 0.25;
    const capShare = TIGHT_SLOT_BUDGET_HEADROOM[slot] ?? 0.35;
    if (priceInfo.budgetShare <= ideal) budgetComfortAdj = 0.07;
    else if (priceInfo.budgetShare <= capShare * 0.92) budgetComfortAdj = 0.035;
  }

  let brandAntiMonopolyAdj = 0;
  if (globalBrandCounts instanceof Map && enriched.brand) {
    const priorUses = globalBrandCounts.get(enriched.brand) || 0;
    if (priorUses >= 3) brandAntiMonopolyAdj = -Math.min(0.06, (priorUses - 2) * 0.015);
  }

  // Calibrated sum so tasteProfile, indoor-layer priority, thermal safety, and DNA
  // decisively separate world-class matches without saturating at 1.0.
  const rawTotal = 0.08 +
    archetypeScore * 0.85 +
    paletteScore * 0.80 +
    textureScore * 0.80 +
    silhouetteScore * 0.80 +
    formalityScore * 0.80 +
    thermalScore * 0.80 +
    tasteAdj +
    qualityAdj +
    visualEvidenceBoost +
    budgetComfortAdj +
    brandAntiMonopolyAdj;
  const score = Math.round(Math.max(0, Math.min(1, rawTotal)) * 10000) / 10000;

  return {
    score,
    breakdown: {
      archetype: Math.round(archetypeScore * 1000) / 1000,
      palette: Math.round(paletteScore * 1000) / 1000,
      texture: Math.round(textureScore * 1000) / 1000,
      silhouette: Math.round(silhouetteScore * 1000) / 1000,
      formality: Math.round(formalityScore * 1000) / 1000,
      thermal: Math.round(thermalScore * 1000) / 1000,
      taste: Math.round(tasteAdj * 1000) / 1000,
      quality: Math.round(qualityAdj * 1000) / 1000,
      visualEvidence: Math.round(visualEvidenceBoost * 1000) / 1000,
      budgetComfort: Math.round(budgetComfortAdj * 1000) / 1000,
    },
  };
}

function buildPublicEntry(token, enriched, priceTier, tasteProfile) {
  const catalogTextClues = {};
  if (!Object.hasOwn(enriched.photo, 'color') && enriched.color !== 'unknown') {
    catalogTextClues.color = enriched.color;
  }
  if (!Object.hasOwn(enriched.photo, 'texture') && enriched.texture !== 'unknown') {
    catalogTextClues.texture = enriched.texture;
  }
  if (!Object.hasOwn(enriched.photo, 'formality') && enriched.formality !== 'unknown') {
    catalogTextClues.formality = enriched.formality;
  }
  if (!Object.hasOwn(enriched.photo, 'volume') && enriched.volume !== 'unknown') {
    catalogTextClues.volume = enriched.volume;
  }

  let explicitOwnerTaste = null;
  if (tasteProfile?.active && tasteProfile.itemTally instanceof Map) {
    const score = tasteProfile.itemTally.get(enriched.id);
    if (Number.isFinite(score) && Math.abs(score) >= 0.15) {
      explicitOwnerTaste = {
        ownerRated: score > 0 ? 'liked' : 'disliked',
        confidence: Math.round(Math.min(1, Math.abs(score)) * 100) / 100,
      };
    }
  }

  return {
    token,
    slot: enriched.slot,
    subtype: safe(enriched.subtype, 35),
    title: safe(enriched.name, 80),
    photo: enriched.photo,
    catalogTextClues,
    priceTier,
    ...(explicitOwnerTaste ? { explicitOwnerTaste } : {}),
    ...(Number.isFinite(enriched.price) && enriched.price > 0 && /^[A-Z]{3}$/.test(enriched.currency)
      ? { price: Math.round(enriched.price * 100) / 100, currency: enriched.currency }
      : {}),
  };
}

function defaultOrthogonalDirections(context) {
  const thermalBand = resolveThermalBand({}, context);
  const numericTemp = resolveNumericTemp(context);
  const formality = resolveTargetFormality({}, context);
  const indoorOccasion = isIndoorEditorialOccasion(context);
  const dateOrDinner = isDateOrDinnerOccasion(context);
  const isWarmOrHot = thermalBand === 'hot' || thermalBand === 'warm' || (numericTemp !== null && numericTemp >= 24);
  const isTrulyCold = thermalBand === 'cold' || (numericTemp !== null && numericTemp <= 10);
  const allowOptionalCoolCoatDir3 = !isWarmOrHot && !isTrulyCold && !dateOrDinner &&
    numericTemp !== null && numericTemp > 10 && numericTemp <= 13;

  if (isWarmOrHot) {
    return [
      {
        id: 'ai-director-1',
        name: 'Riviera Tailored Ease',
        directionType: 'signature_mastery',
        silhouette: 'tailored',
        outerwearArchetype: 'structured_tailoring',
        trouserArchetype: 'tailored_pleated',
        footwearArchetype: 'soft_suede_loafer',
        palette: ['stone', 'white', 'navy', 'tan'],
        colorHarmonyMode: 'cool_architectural_contrast',
        materialTexture: 'linen cotton poplin',
        formality: formality === 'formal' ? 'smart' : formality,
        thermalBand,
        roles: {
          top: ['shirt', 'dress-shirt', 'polo'],
          outerwear: ['blazer'],
          bottom: ['trouser', 'tailored-trouser', 'chino'],
          shoes: ['loafer', 'driver'],
        },
      },
      {
        id: 'ai-director-2',
        name: 'Breezy Utility Overshirt',
        directionType: 'silhouette_texture_shift',
        silhouette: 'relaxed',
        outerwearArchetype: 'relaxed_utility_overshirt',
        trouserArchetype: 'relaxed_chino',
        footwearArchetype: 'warm_weather_driver_espadrille',
        palette: ['olive', 'cream', 'beige', 'brown'],
        colorHarmonyMode: 'warm_earth_analogous',
        materialTexture: 'linen cotton twill',
        formality: 'smart',
        thermalBand,
        roles: {
          top: ['polo', 't-shirt', 'shirt'],
          outerwear: ['overshirt', 'jacket'],
          bottom: ['chino', 'trouser'],
          shoes: ['driver', 'loafer', 'espadrille'],
        },
      },
      {
        id: 'ai-director-3',
        name: 'Airy Open-Weave Knit',
        directionType: 'editorial_statement',
        silhouette: 'balanced',
        outerwearArchetype: 'soft_draped_knit',
        trouserArchetype: 'fluid_wide_leg',
        footwearArchetype: 'clean_minimal_sneaker',
        palette: ['cream', 'blue', 'stone', 'white'],
        colorHarmonyMode: 'neutral_anchor_plus_accent',
        materialTexture: 'cotton knit linen',
        formality: 'casual',
        thermalBand,
        roles: {
          top: ['t-shirt', 'polo', 'shirt'],
          outerwear: ['cardigan'],
          bottom: ['trouser', 'chino'],
          shoes: ['minimal-sneaker', 'sneaker', 'loafer'],
        },
      },
    ];
  }

  return [
    {
      id: 'ai-director-1',
      name: 'Signature Mastery',
      directionType: 'signature_mastery',
      silhouette: 'tailored',
      outerwearArchetype: 'structured_tailoring',
      trouserArchetype: 'tailored_pleated',
      footwearArchetype: formality === 'casual' || dateOrDinner ? 'soft_suede_loafer' : 'sleek_lace_up_dress',
      palette: ['navy', 'white', 'grey', 'cream'],
      colorHarmonyMode: 'cool_architectural_contrast',
      materialTexture: 'wool cotton leather',
      formality,
      thermalBand,
      roles: {
        top: ['dress-shirt', 'shirt', 'fine-knit'],
        outerwear: ['blazer'],
        bottom: ['tailored-trouser', 'trouser'],
        shoes: formality === 'casual' || dateOrDinner ? ['loafer', 'derby'] : ['derby', 'oxford', 'loafer'],
      },
    },
    {
      id: 'ai-director-2',
      name: 'Silhouette & Texture Shift',
      directionType: 'silhouette_texture_shift',
      silhouette: 'relaxed',
      outerwearArchetype: 'relaxed_utility_overshirt',
      trouserArchetype: 'relaxed_chino',
      footwearArchetype: 'soft_suede_loafer',
      palette: ['olive', 'cream', 'stone', 'brown'],
      colorHarmonyMode: 'warm_earth_analogous',
      materialTexture: 'suede knit cotton',
      formality: 'smart',
      thermalBand,
      roles: {
        top: ['fine-knit', 'polo', 'shirt'],
        outerwear: ['overshirt', 'jacket'],
        bottom: ['chino', 'trouser'],
        shoes: ['loafer', 'derby', 'driver'],
      },
    },
    {
      id: 'ai-director-3',
      name: isTrulyCold ? 'Editorial Coat Statement' : 'Editorial Knit & Suede Statement',
      directionType: 'editorial_statement',
      silhouette: 'balanced',
      outerwearArchetype: isTrulyCold
        ? 'longline_coat'
        : (dateOrDinner ? 'suede_jacket' : 'soft_draped_knit'),
      trouserArchetype: formality === 'formal' || (indoorOccasion && !dateOrDinner)
        ? 'clean_straight_trouser'
        : 'refined_dark_denim',
      footwearArchetype: isTrulyCold ? 'sculptural_chelsea_boot' : 'refined_derby_or_monk',
      palette: ['charcoal', 'camel', 'cream', 'brown'],
      colorHarmonyMode: 'neutral_anchor_plus_accent',
      materialTexture: isTrulyCold ? 'wool flannel leather' : 'knit suede wool',
      formality,
      thermalBand,
      roles: {
        top: ['fine-knit', 'polo', 'shirt'],
        outerwear: isTrulyCold
          ? ['coat', 'jacket', 'bomber']
          : allowOptionalCoolCoatDir3
            ? ['cardigan', 'jacket', 'coat']
            : ['cardigan', 'bomber', 'jacket'],
        bottom: formality === 'formal' || (indoorOccasion && !dateOrDinner)
          ? ['trouser', 'tailored-trouser', 'chino']
          : ['jeans', 'trouser'],
        shoes: isTrulyCold
          ? ['chelsea', 'boot', 'derby']
          : ['derby', 'monk', 'chelsea', 'loafer'],
      },
    },
  ];
}

function selectSlotCandidatesForDirection(scoredCandidates, {
  limit,
  globallyAssignedBaseIds,
  globallyAssignedImageUrls,
  explicitLuxury,
  requiredIds,
  isTightBudget = false,
  maxHeavyOutdoorCoats = Infinity,
}) {
  const chosen = [];
  const chosenBaseIds = new Set();
  const chosenImages = new Set();
  const familyCounts = new Map();
  const subtypeCounts = new Map();
  const brandCounts = new Map();
  const colorCounts = new Map();
  let heavyCoatCount = 0;

  const maxPerFamily = Math.max(2, Math.ceil(limit / 5));
  const maxPerSubtype = Math.max(4, Math.ceil(limit * 0.45));
  const maxPerBrand = Math.max(3, Math.ceil(limit * 0.30));
  const maxPerColor = Math.max(4, Math.ceil(limit * 0.50));

  const familyKey = c => `${c.enriched.subtype}|${c.enriched.color}|${c.enriched.texture}`;
  const canAdd = (c, { allowShared = false, enforceDiversity = true, enforceHeavyCoatCap = true } = {}) => {
    if (chosen.length >= limit) return false;
    if (chosenBaseIds.has(c.enriched.baseId)) return false;
    if (c.enriched.imageUrl && chosenImages.has(c.enriched.imageUrl)) return false;
    const isPinned = requiredIds.has(c.enriched.id);
    if (!allowShared && !isPinned) {
      if (globallyAssignedBaseIds.has(c.enriched.baseId)) return false;
      if (c.enriched.imageUrl && globallyAssignedImageUrls.has(c.enriched.imageUrl)) return false;
    }
    if (enforceHeavyCoatCap && !isPinned && c.enriched.layerRole === 'outdoor_heavy_coat') {
      if (heavyCoatCount >= maxHeavyOutdoorCoats) return false;
    }
    if (enforceDiversity && !isPinned) {
      const fKey = familyKey(c);
      if ((familyCounts.get(fKey) || 0) >= maxPerFamily) return false;
      if ((subtypeCounts.get(c.enriched.subtype) || 0) >= maxPerSubtype) return false;
      if (c.enriched.brand && (brandCounts.get(c.enriched.brand) || 0) >= maxPerBrand) return false;
      if (c.enriched.color && c.enriched.color !== 'unknown' && (colorCounts.get(c.enriched.color) || 0) >= maxPerColor) return false;
    }
    return true;
  };
  const take = c => {
    chosen.push(c);
    chosenBaseIds.add(c.enriched.baseId);
    if (c.enriched.imageUrl) chosenImages.add(c.enriched.imageUrl);
    if (c.enriched.layerRole === 'outdoor_heavy_coat') heavyCoatCount++;
    const fKey = familyKey(c);
    familyCounts.set(fKey, (familyCounts.get(fKey) || 0) + 1);
    subtypeCounts.set(c.enriched.subtype, (subtypeCounts.get(c.enriched.subtype) || 0) + 1);
    if (c.enriched.brand) brandCounts.set(c.enriched.brand, (brandCounts.get(c.enriched.brand) || 0) + 1);
    if (c.enriched.color && c.enriched.color !== 'unknown') {
      colorCounts.set(c.enriched.color, (colorCounts.get(c.enriched.color) || 0) + 1);
    }
  };

  // Pass 0: Always admit pinned required anchors first if they belong to this slot.
  for (const c of scoredCandidates) {
    if (requiredIds.has(c.enriched.id) && canAdd(c, { allowShared: true, enforceDiversity: false, enforceHeavyCoatCap: false })) {
      take(c);
    }
  }

  // Pass 1: Price-tier balanced disjoint selection (stronger accessible quota under tight budgets).
  if (!explicitLuxury && limit >= 6) {
    const accessibleQuota = isTightBudget
      ? Math.max(3, Math.floor(limit * 0.50))
      : Math.max(2, Math.floor(limit * 0.32));
    const midQuota = isTightBudget
      ? Math.max(2, Math.floor(limit * 0.35))
      : Math.max(2, Math.floor(limit * 0.35));
    let accTaken = 0;
    let midTaken = 0;

    for (const c of scoredCandidates) {
      if (accTaken >= accessibleQuota) break;
      if (c.priceInfo.priceTier === 'accessible' && canAdd(c, { allowShared: false, enforceDiversity: true })) {
        take(c);
        accTaken++;
      }
    }
    for (const c of scoredCandidates) {
      if (midTaken >= midQuota) break;
      if (c.priceInfo.priceTier === 'mid' && canAdd(c, { allowShared: false, enforceDiversity: true })) {
        take(c);
        midTaken++;
      }
    }
  }

  // Pass 2: Best disjoint & subtype/brand/color/family-diverse candidates across all tiers.
  for (const c of scoredCandidates) {
    if (chosen.length >= limit) break;
    if (canAdd(c, { allowShared: false, enforceDiversity: true })) take(c);
  }

  // Pass 3: Relax intra-pool diversity caps while staying disjoint across directions and honoring coat cap.
  for (const c of scoredCandidates) {
    if (chosen.length >= limit) break;
    if (canAdd(c, { allowShared: false, enforceDiversity: false, enforceHeavyCoatCap: true })) take(c);
  }

  // Pass 3b: Only if non-coat candidates were exhausted on a tiny catalog, relax heavy-coat cap while staying disjoint.
  for (const c of scoredCandidates) {
    if (chosen.length >= limit) break;
    if (canAdd(c, { allowShared: false, enforceDiversity: false, enforceHeavyCoatCap: false })) take(c);
  }

  // Pass 4: Only on small catalogs where disjoint items are exhausted, allow sharing across directions.
  for (const c of scoredCandidates) {
    if (chosen.length >= limit) break;
    if (canAdd(c, { allowShared: true, enforceDiversity: true })) take(c);
  }
  for (const c of scoredCandidates) {
    if (chosen.length >= limit) break;
    if (canAdd(c, { allowShared: true, enforceDiversity: false, enforceHeavyCoatCap: false })) take(c);
  }

  // Re-order final balanced pool so pinned anchors lead, followed by highest direction-fit score.
  chosen.sort((a, b) =>
    (Number(requiredIds.has(b.enriched.id)) - Number(requiredIds.has(a.enriched.id))) ||
    (b.score - a.score) ||
    (b.enriched.qualityScore - a.enriched.qualityScore) ||
    a.enriched.id.localeCompare(b.enriched.id),
  );
  return chosen;
}

/**
 * Retrieve dedicated, direction-aligned, cross-direction-diversified,
 * price-tier-balanced candidate pools for each Creative Director blueprint.
 *
 * @param {object} params
 * @param {Array<object>} params.directions - Up to 3 Creative Director blueprints.
 * @param {Array<object>} params.catalog - Full shop/wardrobe catalog (raw DB rows or adapted items).
 * @param {object} [params.context={}] - Request context (`occasion`, `style`, `weather`, `budget`, `preferences`, `requiredIds`, `presentation`, `fxSnapshot`, `priceToBudgetCurrency`).
 * @param {object|null} [params.personalStyleDna=null] - Normalized Personal Style DNA.
 * @param {object|null} [params.tasteProfile=null] - Learned taste / feedback model.
 * @param {Map|Record<string,object>|null} [params.qualityMap=null] - Per-SKU `catalog_visual_quality` map.
 * @param {number} [params.perSlotLimit=14] - Target candidates per slot per direction (4..24).
 * @param {Function|null} [params.priceToBudgetCurrency=null] - Optional verified price converter.
 * @param {object|null} [params.fxSnapshot=null] - Optional verified CBU FX snapshot.
 */
export function buildPerDirectionCandidatePools({
  directions = [],
  catalog = [],
  context = {},
  personalStyleDna = null,
  tasteProfile = null,
  qualityMap = null,
  perSlotLimit = 14,
  priceToBudgetCurrency = null,
  fxSnapshot = null,
} = {}) {
  const limit = Math.max(3, Math.min(24, Math.floor(Number(perSlotLimit) || 14)));
  const activeDirections = (Array.isArray(directions) && directions.length > 0
    ? directions.slice(0, 3)
    : defaultOrthogonalDirections(context));

  // 1. Structural hygiene filtering across the full catalog.
  const hygiene = context?.hygieneAlreadyApplied === true
    ? {
        items: (Array.isArray(catalog) ? catalog : []).filter(item => !lookupQualityEntry(item, qualityMap).quarantined),
        stats: {
          inputCount: Array.isArray(catalog) ? catalog.length : 0,
          eligibleCount: 0,
          excludedCount: 0,
          byReason: {},
          bySlot: {},
          warningsByType: {},
        },
      }
    : filterHygienicCatalog(catalog, context, qualityMap);
  if (context?.hygieneAlreadyApplied === true) {
    hygiene.stats.eligibleCount = hygiene.items.length;
    hygiene.stats.excludedCount = hygiene.stats.inputCount - hygiene.items.length;
  }

  const requiredIds = new Set(
    (Array.isArray(context?.requiredIds) ? context.requiredIds : []).map(id => String(id)),
  );
  const { avoidColors, dislikedPatterns, brandExclusions } = collectExclusions(
    context,
    personalStyleDna,
    tasteProfile,
  );
  const explicitLuxury = /\b(?:luxury|premium|designer|high[- ]?end|couture)\b|люкс|премиальн/i.test(
    `${context?.brief || ''} ${context?.request || ''}`,
  );

  // 2. Enrich all hygienic items & enforce explicit user exclusions (never blocking pinned anchors).
  const enrichedAll = [];
  for (const raw of hygiene.items) {
    const e = resolveEnrichedItem(raw, qualityMap);
    if (!ALLOWED_SLOTS.has(e.slot)) continue;
    const isPinned = requiredIds.has(e.id);
    if (!isPinned) {
      if (e.color !== 'unknown' && avoidColors.has(e.color)) continue;
      if (e.pattern !== 'unknown' && dislikedPatterns.has(e.pattern)) continue;
      if (e.brand && brandExclusions.has(e.brand)) continue;
    }
    enrichedAll.push(e);
  }

  // 3. Honest within-currency price quantile bands (zero fabricated FX).
  const priceBands = buildWithinCurrencyPriceBands(enrichedAll);

  // 4. Determine active slots (always top/outerwear/bottom/shoes + dress/accessory if requested and stocked).
  const bySlot = new Map();
  for (const e of enrichedAll) {
    if (!bySlot.has(e.slot)) bySlot.set(e.slot, []);
    bySlot.get(e.slot).push(e);
  }
  const activeSlots = [...DEFAULT_SLOTS];
  for (const extraSlot of ['dress', 'accessory']) {
    const requestedByDir = activeDirections.some(d => Array.isArray(d?.roles?.[extraSlot]) && d.roles[extraSlot].length > 0);
    if (requestedByDir && (bySlot.get(extraSlot)?.length || 0) > 0) {
      activeSlots.push(extraSlot);
    }
  }

  // 4b. Compute dynamic budget-headroom plan BEFORE sorting & slicing each direction.
  const budgetPlan = computeDynamicBudgetPlan({
    bySlot,
    activeSlots,
    context,
    limit,
    directionCount: activeDirections.length,
    priceToBudgetCurrency,
    fxSnapshot,
  });

  // Precompute all directions' archetypes per slot for cross-direction contrast scoring.
  const archetypesBySlot = new Map(activeSlots.map(slot => [
    slot,
    activeDirections.map(d => inferArchetypeKey(slot, d)),
  ]));

  const numericTemp = resolveNumericTemp(context);
  const indoorOccasion = isIndoorEditorialOccasion(context);
  const dateOrDinner = isDateOrDinnerOccasion(context);

  // 5. Select per-direction candidate pools with cross-direction & cross-slot overlap minimization,
  //    indoor-layer-first discipline (>10C), warm-weather breathability (>=24C), and anti-monopoly diversity.
  const globallyAssignedBaseIds = new Set();
  const globallyAssignedImageUrls = new Set();
  const globalBrandCountsBySlot = new Map(activeSlots.map(slot => [slot, new Map()]));
  let tailoredCoatDirectionsUsed = 0;
  const rawDirectionSelections = [];

  for (let dIdx = 0; dIdx < activeDirections.length; dIdx++) {
    const direction = activeDirections[dIdx];
    const dirThermal = resolveThermalBand(direction, context);
    const isWarmOrHotDir = dirThermal === 'hot' || dirThermal === 'warm' || (numericTemp !== null && numericTemp >= 24);
    const outerArch = inferArchetypeKey('outerwear', direction);
    const dirAsksForCoat = outerArch === 'longline_coat' ||
      (Array.isArray(direction?.roles?.outerwear) && direction.roles.outerwear.includes('coat'));

    // Indoor >10C discipline: NEVER inject heavy outdoor coats/overcoats/peacoats into indoor
    // Work, Date Night, Evening, or Smart Casual pools at >10C; reserve outdoor heavy coats for <=10C.
    const indoorTempGuard = numericTemp !== null && numericTemp > 10 && indoorOccasion;
    const directionAllowsTailoredCoat = !isWarmOrHotDir && !indoorTempGuard;
    const maxHeavyOutdoorCoats = isWarmOrHotDir
      ? 0
      : !indoorTempGuard
        ? Infinity
        : directionAllowsTailoredCoat
          ? Math.min(3, Math.max(1, Math.floor(limit * 0.25)))
          : 0;

    const slotSelections = {};

    for (const slot of activeSlots) {
      const slotItems = bySlot.get(slot) || [];
      const competing = new Set(
        (archetypesBySlot.get(slot) || []).filter((arch, idx) => idx !== dIdx && arch),
      );
      const scored = [];
      for (const enriched of slotItems) {
        const priceInfo = evaluatePriceTierAndBudget(enriched, priceBands, context, budgetPlan);
        if (!priceInfo.withinHeadroom && !requiredIds.has(enriched.id)) continue;
        const { score, breakdown } = scoreItemForDirection(enriched, direction, {
          context,
          personalStyleDna,
          tasteProfile,
          competingArchetypes: competing,
          directionAllowsTailoredCoat,
          budgetPlan,
          priceInfo,
          globalBrandCounts: globalBrandCountsBySlot.get(slot),
        });
        scored.push({ enriched, score, breakdown, priceInfo });
      }

      // Warm/Hot (>=24C) breathability gate: exclude heavy winter/waxed/leather/wool hazards
      // whenever enough unassigned breathable candidates exist for this direction's slot.
      let effectiveScored = scored;
      const countUnassigned = list => list.filter(c => !globallyAssignedBaseIds.has(c.enriched.baseId)).length;
      if (isWarmOrHotDir) {
        const breathableOnly = effectiveScored.filter(c => !c.enriched.isHeavyWarmWeatherHazard || requiredIds.has(c.enriched.id));
        if (countUnassigned(breathableOnly) >= limit) {
          effectiveScored = breathableOnly;
        }
      }

      // Indoor (>10C) outerwear gate: exclude peacoats, parkas, puffers, commuter rain trenches,
      // and heavy outdoor coats whenever enough unassigned indoor layers exist.
      if (slot === 'outerwear' && indoorTempGuard) {
        const noCommuterOrPeacoat = effectiveScored.filter(c =>
          (!c.enriched.isCommuterRainOrParka && !c.enriched.isPeacoat) || requiredIds.has(c.enriched.id),
        );
        if (countUnassigned(noCommuterOrPeacoat) >= limit) {
          effectiveScored = noCommuterOrPeacoat;
        }
        if (maxHeavyOutdoorCoats === 0) {
          const indoorOnly = effectiveScored.filter(c =>
            c.enriched.layerRole !== 'outdoor_heavy_coat' || requiredIds.has(c.enriched.id),
          );
          if (countUnassigned(indoorOnly) >= limit) {
            effectiveScored = indoorOnly;
          }
        }
      }

      // Deterministic sort: highest score, then higher qualityScore, then stable ID tie-breaker.
      effectiveScored.sort((a, b) =>
        (b.score - a.score) ||
        (b.enriched.qualityScore - a.enriched.qualityScore) ||
        a.enriched.id.localeCompare(b.enriched.id),
      );

      const picked = selectSlotCandidatesForDirection(effectiveScored, {
        limit,
        globallyAssignedBaseIds,
        globallyAssignedImageUrls,
        explicitLuxury,
        requiredIds,
        isTightBudget: budgetPlan?.isTight === true,
        maxHeavyOutdoorCoats: slot === 'outerwear' ? maxHeavyOutdoorCoats : Infinity,
      });

      const slotBrandMap = globalBrandCountsBySlot.get(slot);
      for (const c of picked) {
        globallyAssignedBaseIds.add(c.enriched.baseId);
        if (c.enriched.imageUrl) globallyAssignedImageUrls.add(c.enriched.imageUrl);
        if (c.enriched.brand) slotBrandMap.set(c.enriched.brand, (slotBrandMap.get(c.enriched.brand) || 0) + 1);
      }
      if (slot === 'outerwear' && picked.some(c => c.enriched.layerRole === 'outdoor_heavy_coat')) {
        tailoredCoatDirectionsUsed++;
      }
      slotSelections[slot] = picked;
    }

    rawDirectionSelections.push({ direction, dIdx, slotSelections });
  }

  // 6. Assign canonical unique opaque P-tokens across the unified pool so every physical SKU
  //    has exactly ONE token even if shared across directions on tiny catalogs.
  const tokenByBaseId = new Map();
  const unifiedPool = [];
  let tokenSeq = 0;

  const getOrCreateTokenEntry = (candidate, dIdx) => {
    const existing = tokenByBaseId.get(candidate.enriched.baseId);
    if (existing) {
      if (!existing.directionIndices.includes(dIdx)) existing.directionIndices.push(dIdx);
      if (candidate.score > existing.strength) existing.strength = candidate.score;
      return existing;
    }
    const token = 'P' + String(tokenSeq++).padStart(3, '0');
    const entry = {
      token,
      slot: candidate.enriched.slot,
      item: candidate.enriched.item,
      strength: candidate.score,
      priceTier: candidate.priceInfo.priceTier,
      directionIndices: [dIdx],
      public: buildPublicEntry(token, candidate.enriched, candidate.priceInfo.priceTier, tasteProfile),
    };
    tokenByBaseId.set(candidate.enriched.baseId, entry);
    unifiedPool.push(entry);
    return entry;
  };

  const directionPools = rawDirectionSelections.map(({ direction, dIdx, slotSelections }) => {
    const slotsOut = {};
    const flatPool = [];
    for (const slot of activeSlots) {
      const entries = (slotSelections[slot] || []).map(c => {
        const canonical = getOrCreateTokenEntry(c, dIdx);
        return {
          token: canonical.token,
          slot,
          item: canonical.item,
          score: c.score,
          strength: c.score,
          breakdown: c.breakdown,
          priceTier: c.priceInfo.priceTier,
          public: canonical.public,
        };
      });
      slotsOut[slot] = entries;
      flatPool.push(...entries);
    }
    return {
      directionId: direction?.id || `ai-director-${dIdx + 1}`,
      directionIndex: dIdx,
      directionName: direction?.name || `Direction ${dIdx + 1}`,
      directionType: direction?.directionType || ['signature_mastery', 'silhouette_texture_shift', 'editorial_statement'][dIdx] || 'editorial_statement',
      archetype: {
        outerwearArchetype: inferArchetypeKey('outerwear', direction),
        trouserArchetype: inferArchetypeKey('bottom', direction),
        footwearArchetype: inferArchetypeKey('shoes', direction),
        silhouette: direction?.silhouette || 'balanced',
        palette: Array.isArray(direction?.palette) ? [...direction.palette] : [],
        formality: resolveTargetFormality(direction, context),
        thermalBand: resolveThermalBand(direction, context),
      },
      slots: slotsOut,
      pool: flatPool,
    };
  });

  // 7. Transparent pool-diversity and coverage metrics.
  const totalAssignments = directionPools.reduce((sum, d) => sum + d.pool.length, 0);
  const sharedEntries = unifiedPool.filter(e => e.directionIndices.length > 1).length;
  const crossDirectionOverlapRate = unifiedPool.length > 0
    ? Math.round((sharedEntries / unifiedPool.length) * 10000) / 10000
    : 0;

  const perSlotCounts = Object.fromEntries(activeSlots.map(slot => [
    slot,
    unifiedPool.filter(e => e.slot === slot).length,
  ]));
  const priceTierDistribution = { accessible: 0, mid: 0, elevated: 0 };
  for (const e of unifiedPool) {
    if (Object.hasOwn(priceTierDistribution, e.priceTier)) priceTierDistribution[e.priceTier]++;
  }
  const outerwearSubtypes = new Set(unifiedPool.filter(e => e.slot === 'outerwear').map(e => e.public.subtype));
  const footwearSubtypes = new Set(unifiedPool.filter(e => e.slot === 'shoes').map(e => e.public.subtype));

  return {
    version: PER_DIRECTION_RETRIEVAL_VERSION,
    hygieneStats: hygiene.stats,
    directions: directionPools,
    unifiedPool,
    metrics: {
      totalCatalogInput: Array.isArray(catalog) ? catalog.length : 0,
      hygienicCatalogCount: hygiene.items.length,
      uniqueSkusExposed: unifiedPool.length,
      totalDirectionAssignments: totalAssignments,
      catalogCoverageRatio: hygiene.items.length > 0
        ? Math.round((unifiedPool.length / hygiene.items.length) * 10000) / 10000
        : 0,
      perSlotCounts,
      crossDirectionOverlapRate,
      outerwearSubtypeBreadth: outerwearSubtypes.size,
      footwearSubtypeBreadth: footwearSubtypes.size,
      priceTierDistribution,
      tightBudgetApplied: budgetPlan?.isTight === true,
      ...(budgetPlan ? { slotBudgetCaps: budgetPlan.slotCaps } : {}),
      tailoredCoatDirectionsUsed,
    },
  };
}
