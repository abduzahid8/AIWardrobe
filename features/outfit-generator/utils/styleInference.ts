/**
 * Style Inference — derive aesthetic tags, materials, and patterns from an item's
 * name, description, brand, and color, without requiring any extra DB columns.
 *
 * Used to:
 *   1. Score & pre-filter catalog items per vibe (old_money, streetwear, …)
 *      before sending them to the LLM, so the model only picks from genuinely
 *      on-style candidates.
 *   2. Enrich the prompt bullets with structured tags the LLM can reason over.
 *   3. Guide sanitize / substitution so fallback picks respect the aesthetic.
 *
 * Everything here is pure, deterministic, and inexpensive. It's designed to
 * run both on the client (React Native) and in Deno edge functions.
 */

export type StyleId =
  | 'old_money'
  | 'streetwear'
  | 'minimalist'
  | 'y2k'
  | 'business_casual'
  | 'casual'
  | 'classic';

export interface InferredItemAttributes {
  /** Ranked list of matching aesthetics, most-likely first. */
  styleTags: StyleId[];
  /** Lowercased fabric words found (linen, wool, cashmere, denim, leather, …). */
  materials: string[];
  /** Patterns / finishes (striped, checked, graphic, washed, satin, …). */
  patterns: string[];
  /** Formality score 0 (athletic/loungewear) … 1 (suiting/black tie). */
  formality: number;
  /** Whether the item reads as neutral / earth-tone palette. */
  isNeutralPalette: boolean;
}

export interface ItemForInference {
  name?: string;
  description?: string;
  brand?: string;
  color?: string;
  type?: string;
  category?: string;
  macroCategory?: string;
}

/* ────────────────────────────────────────────────────────────────────
 * Keyword lexicons
 *
 * These are intentionally broad and tuned against the kind of product
 * titles we see on Zara/COS/Massimo Dutti/Arket feeds (e.g. "100% LINEN
 * RELAXED-FIT SHORTS", "BASIC HEAVYWEIGHT T-SHIRT", "OVERSIZED HOODIE").
 * Add to these as new product patterns appear.
 * ──────────────────────────────────────────────────────────────────── */

const MATERIAL_WORDS = [
  'linen', 'wool', 'cashmere', 'merino', 'silk', 'satin', 'cotton', 'poplin',
  'oxford', 'denim', 'leather', 'suede', 'corduroy', 'tweed', 'velvet',
  'cashmere blend', 'wool blend', 'canvas', 'nylon', 'polyester', 'mesh',
  'fleece', 'terry', 'jersey', 'knit', 'cashmere-blend', 'wool-blend',
];

const PATTERN_WORDS = [
  'striped', 'stripes', 'checked', 'check', 'plaid', 'houndstooth',
  'pinstripe', 'graphic', 'logo', 'print', 'printed', 'tie-dye', 'washed',
  'distressed', 'ripped', 'embroidered', 'floral', 'camo', 'animal print',
  'monogram', 'colorblock', 'color-block',
];

const NEUTRAL_COLOR_WORDS = [
  'beige', 'cream', 'ivory', 'ecru', 'stone', 'sand', 'camel', 'tan',
  'khaki', 'brown', 'chocolate', 'taupe', 'white', 'off-white', 'oatmeal',
  'oat', 'black', 'charcoal', 'grey', 'gray', 'navy', 'forest', 'olive',
  'burgundy',
];

const BOLD_COLOR_WORDS = [
  'neon', 'pink', 'fuchsia', 'magenta', 'lime', 'yellow', 'orange',
  'turquoise', 'aqua', 'metallic', 'silver', 'gold', 'iridescent', 'holographic',
  'rhinestone', 'glitter', 'sequin',
];

/* ────────────────────────────────────────────────────────────────────
 * Per-style signal lexicons
 *
 * Each entry is a list of [regex | keyword, weight] pairs. Positive
 * weights push the score up, negative weights push it down.
 * ──────────────────────────────────────────────────────────────────── */

type Signal = [RegExp, number];

function kw(word: string, weight: number): Signal {
  return [new RegExp(`\\b${word.replace(/[-/\\^$*+?.()|[\]{}]/g, '\\$&')}\\b`, 'i'), weight];
}

const STYLE_SIGNALS: Record<StyleId, Signal[]> = {
  old_money: [
    kw('cashmere', 4), kw('merino', 3), kw('wool', 3), kw('linen', 3),
    kw('silk', 2), kw('cotton', 1),
    kw('blazer', 4), kw('loafer', 5), kw('loafers', 5), kw('oxford', 3),
    kw('knit polo', 4), kw('polo', 2),
    kw('cardigan', 3), kw('chino', 3), kw('chinos', 3), kw('trouser', 3),
    kw('trousers', 3), kw('tailored', 3), kw('pleated', 3), kw('pinstripe', 2),
    kw('houndstooth', 3), kw('herringbone', 3), kw('tweed', 3), kw('flannel', 3),
    kw('suit', 3), kw('suiting', 3), kw('knit sweater', 3), kw('pullover', 2),
    kw('turtleneck', 3),
    kw('relaxed-fit', 1), kw('classic', 2), kw('regular-fit', 1),
    kw('cream', 2), kw('camel', 3), kw('navy', 2), kw('beige', 2),
    kw('burgundy', 2), kw('forest', 2), kw('tan', 2),
    kw('ralph lauren', 5), kw('brunello', 5),
    kw('loro piana', 5), kw('massimo dutti', 3), kw('arket', 2), kw('brooks', 3),
    kw('brooks brothers', 5),
    // Shoes-specific positives (loafers, penny loafers, bit loafers, boat shoes,
    // derbies, dress shoes, leather shoes are all classic Old Money footwear)
    kw('penny loafer', 5), kw('penny loafers', 5), kw('bit loafer', 5),
    kw('bit loafers', 5), kw('metal bit', 4), kw('boat shoe', 4), kw('boat shoes', 4),
    kw('derby', 4), kw('dress shoes', 4), kw('dress shoe', 4), kw('leather shoes', 3),
    kw('leather loafer', 5), kw('leather loafers', 5), kw('moc toe', 2),
    kw('double buckle', 3), kw('raised seam', 1), kw('suede', 3),
    // Negatives
    kw('hoodie', -4), kw('graphic', -3), kw('logo', -2), kw('oversized', -2),
    kw('cargo', -3), kw('track pants', -4), kw('sweatpants', -3),
    kw('ripped', -3), kw('distressed', -3),
    kw('neon', -4), kw('sequin', -4), kw('rhinestone', -4),
    kw('square-toe', -4), kw('square toe', -4), kw('low-rise', -3), kw('low rise', -3),
    kw('skinny', -2), kw('backpack', -2),
    // Footwear to AVOID for old_money
    kw('chunky sneaker', -4), kw('chunky sneakers', -4), kw('basketball', -5),
    kw('skate sneaker', -4), kw('skate sneakers', -4), kw('thick-soled', -3),
    kw('thick soled', -3), kw('retro sneaker', -2), kw('retro sneakers', -2),
    kw('rope lace', -2),
    // IMPORTANT: plain "sneaker" is mildly negative (clean whites are ok) and
    // "chunky" alone is bad regardless of what it qualifies.
    kw('sneaker', -1), kw('sneakers', -1), kw('chunky', -3),
  ],
  streetwear: [
    kw('hoodie', 5), kw('oversized', 4), kw('baggy', 4), kw('cargo', 5),
    kw('graphic', 4), kw('logo', 3), kw('printed', 3), kw('print', 2),
    kw('sneaker', 3), kw('sneakers', 3), kw('chunky', 3), kw('puffer', 4),
    kw('bomber', 3), kw('track', 3), kw('sweatpants', 3), kw('joggers', 4),
    kw('streetwear', 5), kw('utility', 3), kw('workwear', 3), kw('parka', 3),
    kw('ripped', 2), kw('distressed', 2), kw('washed', 1),
    kw('stüssy', 5), kw('stussy', 5), kw('off-white', 4), kw('nike', 3),
    kw('adidas', 3), kw('supreme', 5), kw('palace', 4),
    // Negatives
    kw('blazer', -3), kw('loafer', -3), kw('tailored', -3), kw('cashmere', -2),
    kw('oxford shirt', -2), kw('pinstripe', -3),
  ],
  minimalist: [
    kw('minimal', 5), kw('clean', 2), kw('seamless', 3), kw('structured', 2),
    kw('essential', 3), kw('basic', 2), kw('plain', 3), kw('solid', 2),
    kw('cos', 5), kw('uniqlo', 4), kw('acne', 4), kw('jil sander', 5),
    kw('the row', 5), kw('everlane', 4), kw('arket', 3),
    kw('crew neck', 2), kw('turtleneck', 2), kw('mock neck', 2), kw('knit polo', 3),
    kw('merino', 3), kw('wool', 2), kw('cotton', 1),
    kw('black', 2), kw('white', 2), kw('grey', 2), kw('gray', 2),
    kw('beige', 2), kw('navy', 1), kw('stone', 2), kw('cream', 2), kw('sand', 2),
    kw('suede', 2), kw('poplin', 2), kw('linen', 2),
    // Minimalist footwear: clean leather sneakers, minimal loafers, plain boots
    kw('minimal sneaker', 5), kw('minimal sneakers', 5), kw('leather sneaker', 3),
    kw('leather sneakers', 3), kw('leather loafer', 3), kw('leather loafers', 3),
    kw('loafer', 2), kw('loafers', 2),
    // Negatives
    kw('graphic', -4), kw('logo', -3), kw('print', -3), kw('neon', -5),
    kw('sequin', -5), kw('rhinestone', -5), kw('floral', -3),
    kw('colorblock', -3), kw('color-block', -3),
    kw('chunky', -3), kw('basketball', -4), kw('retro sneaker', -2),
  ],
  y2k: [
    kw('y2k', 5), kw('low-rise', 4), kw('low rise', 4), kw('crop', 3),
    kw('cropped', 3), kw('bedazzled', 5), kw('rhinestone', 5), kw('sequin', 4),
    kw('metallic', 4), kw('satin', 2), kw('velour', 4), kw('butterfly', 3),
    kw('baby tee', 4), kw('tube top', 5), kw('halter', 3), kw('mini skirt', 3),
    kw('platform', 4), kw('iridescent', 4), kw('holographic', 4),
    kw('pink', 2), kw('fuchsia', 3), kw('neon', 3),
    kw('juicy couture', 5), kw('ed hardy', 5),
    // Negatives
    kw('blazer', -3), kw('tailored', -3), kw('wool', -2), kw('oxford', -3),
  ],
  business_casual: [
    kw('blazer', 4), kw('chino', 4), kw('chinos', 4), kw('trouser', 4),
    kw('trousers', 4), kw('oxford', 4), kw('button-down', 3), kw('button down', 3),
    kw('loafer', 4), kw('loafers', 4), kw('dress shirt', 4), kw('dress shoes', 4),
    kw('dress shoe', 4), kw('derby', 4), kw('poplin', 3), kw('tailored', 4),
    kw('knit polo', 3), kw('turtleneck', 3), kw('cardigan', 3),
    kw('flannel', 3), kw('herringbone', 3), kw('tweed', 3), kw('suede', 3),
    kw('slim-fit', 2), kw('slim fit', 2), kw('straight-fit', 1),
    kw('shirt', 2), kw('polo', 2),
    kw('penny loafer', 4), kw('penny loafers', 4), kw('bit loafer', 4),
    kw('bit loafers', 4), kw('leather shoes', 3), kw('leather loafer', 4),
    kw('leather loafers', 4),
    kw('navy', 2), kw('charcoal', 2), kw('tan', 2), kw('white', 1),
    kw('cream', 2), kw('brown', 2), kw('burgundy', 2), kw('light blue', 2),
    kw('hugo boss', 4), kw('theory', 4), kw('everlane', 2),
    // Negatives
    kw('hoodie', -4), kw('graphic', -3), kw('cargo', -4), kw('ripped', -4),
    kw('sweatpants', -4), kw('sequin', -5), kw('neon', -5), kw('crop', -2),
    kw('chunky', -3), kw('basketball', -5), kw('skate', -3),
    kw('low-rise', -3), kw('low rise', -3), kw('skinny', -2),
    kw('square-toe', -4), kw('square toe', -4),
  ],
  casual: [
    kw('t-shirt', 2), kw('tee', 2), kw('jeans', 2), kw('jean', 1),
    kw('sweater', 2), kw('cardigan', 1), kw('sneaker', 2), kw('sneakers', 2),
    kw('polo', 1), kw('shorts', 1), kw('denim', 2), kw('regular-fit', 1),
    kw('knit', 2), kw('slack', 2), kw('slacks', 2), kw('loafer', 2), kw('loafers', 2),
    kw('chino', 2), kw('chinos', 2), kw('canvas', 1), kw('suede', 2),
    kw('basketball', -3), kw('graphic hoodie', -3), kw('sweatpants', -2),
  ],
  classic: [
    // Core tailored pieces
    kw('suit', 5), kw('blazer', 4), kw('sport coat', 4), kw('sport jacket', 4),
    kw('tailored', 4), kw('trouser', 4), kw('trousers', 4), kw('pleated', 3),
    kw('slack', 3), kw('slacks', 3),
    kw('oxford', 4), kw('button-down', 3), kw('button down', 3), kw('dress shirt', 5),
    kw('cardigan', 3), kw('turtleneck', 4), kw('knit polo', 4),
    // Fabrics
    kw('flannel', 4), kw('tweed', 4), kw('herringbone', 4), kw('cashmere', 4),
    kw('merino', 3), kw('wool', 3), kw('linen', 2), kw('silk', 2),
    kw('poplin', 3), kw('oxford cloth', 3), kw('suede', 3),
    // Colors
    kw('navy', 3), kw('charcoal', 3), kw('cream', 2), kw('burgundy', 3),
    kw('camel', 3), kw('beige', 2), kw('tan', 2), kw('forest', 2),
    // Shoes
    kw('loafer', 4), kw('loafers', 4), kw('oxford shoe', 5), kw('oxford shoes', 5),
    kw('derby', 4), kw('dress shoes', 5), kw('dress shoe', 5),
    kw('leather shoes', 3), kw('suede shoes', 3),
    kw('penny loafer', 5), kw('penny loafers', 5), kw('bit loafer', 5), kw('bit loafers', 5),
    kw('brogue', 4), kw('brogues', 4), kw('monk strap', 4), kw('chelsea boot', 3), kw('chelsea boots', 3),
    // Brands
    kw('brooks brothers', 5), kw('ralph lauren', 4), kw('brunello', 5),
    kw('loro piana', 5), kw('massimo dutti', 3), kw('arket', 2),
    // Accessories
    kw('tie', 3), kw('pocket square', 3), kw('scarf', 2),
    // Negatives
    kw('hoodie', -4), kw('graphic', -3), kw('logo', -2),
    kw('cargo', -4), kw('sweatpants', -4), kw('joggers', -4),
    kw('track pants', -4), kw('shorts', -3), kw('bermuda', -3),
    kw('puffer', -3), kw('bomber', -3), kw('denim jacket', -2),
    kw('chunky sneaker', -4), kw('chunky sneakers', -4), kw('athletic', -3),
    kw('square-toe', -4), kw('square toe', -4),
    kw('neon', -4), kw('sequin', -4), kw('ripped', -3),
    kw('low-rise', -3), kw('low rise', -3), kw('skinny', -2), kw('oversized', -2),
    kw('backpack', -2),
  ],
};

/* ────────────────────────────────────────────────────────────────────
 * Core scoring
 * ──────────────────────────────────────────────────────────────────── */

function buildSearchBlob(item: ItemForInference): string {
  return [
    item.name,
    item.description,
    item.brand,
    item.color,
    item.type,
    item.category,
    item.macroCategory,
  ]
    .filter(Boolean)
    .join(' ')
    .toLowerCase();
}

function extractMatchingWords(blob: string, dictionary: string[]): string[] {
  const out: string[] = [];
  for (const word of dictionary) {
    if (blob.includes(word)) out.push(word);
  }
  return out;
}

function scoreStyle(blob: string, signals: Signal[]): number {
  let score = 0;
  for (const [re, weight] of signals) {
    if (re.test(blob)) score += weight;
  }
  return score;
}

function clamp01(n: number): number {
  if (n < 0) return 0;
  if (n > 1) return 1;
  return n;
}

/**
 * Estimate how formal an item reads.
 *   0.0 → sweatpants, hoodies, athletic wear
 *   0.5 → jeans, t-shirts, casual
 *   1.0 → wool blazer, oxford shirt, pinstripe suit
 */
function estimateFormality(blob: string): number {
  let score = 0.5;
  const bumps: [RegExp, number][] = [
    [/\b(suit|blazer|tailored|trouser|trousers|oxford|dress shirt|loafer|pinstripe|tweed|herringbone|cashmere|wool|silk)\b/, +0.1],
    [/\b(chino|chinos|polo|cardigan|linen|button-down|button down|poplin|merino)\b/, +0.06],
    [/\b(jeans|t-shirt|tee|sneaker|sneakers|hoodie|sweatpants|joggers|track|cargo|graphic|printed|ripped)\b/, -0.1],
    [/\b(baggy|oversized|distressed|athletic|workout|running|fleece|terry)\b/, -0.06],
  ];
  for (const [re, delta] of bumps) {
    if (re.test(blob)) score += delta;
  }
  return clamp01(score);
}

/**
 * Main entry point: infer attributes from an item's unstructured text.
 */
export function inferItemAttributes(item: ItemForInference): InferredItemAttributes {
  const blob = buildSearchBlob(item);

  const materials = extractMatchingWords(blob, MATERIAL_WORDS);
  const patterns = extractMatchingWords(blob, PATTERN_WORDS);

  const styleScores: { style: StyleId; score: number }[] = (
    Object.keys(STYLE_SIGNALS) as StyleId[]
  ).map((style) => ({
    style,
    score: scoreStyle(blob, STYLE_SIGNALS[style]),
  }));

  styleScores.sort((a, b) => b.score - a.score);

  const styleTags = styleScores
    .filter((s) => s.score > 0)
    .map((s) => s.style);

  // Every item is at least "casual" if nothing else matched.
  if (styleTags.length === 0) styleTags.push('casual');

  const neutralHits = extractMatchingWords(blob, NEUTRAL_COLOR_WORDS).length;
  const boldHits = extractMatchingWords(blob, BOLD_COLOR_WORDS).length;
  const isNeutralPalette = neutralHits > 0 && neutralHits >= boldHits;

  return {
    styleTags,
    materials,
    patterns,
    formality: estimateFormality(blob),
    isNeutralPalette,
  };
}

/**
 * Score how well an item matches a requested style on a 0–1 scale.
 * Used to pre-filter and rank items before sending to the LLM.
 */
export function scoreItemForStyle(item: ItemForInference, style: StyleId): number {
  const blob = buildSearchBlob(item);
  const rawScore = scoreStyle(blob, STYLE_SIGNALS[style] || []);
  // Map raw scores (typically -6 … +12) to roughly 0..1
  const normalized = (rawScore + 4) / 16;
  return clamp01(normalized);
}

/**
 * Filter + rank a list of items for a requested style. Items with strongly
 * negative scores are dropped entirely (they'd contaminate the candidate set);
 * the rest are sorted best-first. If filtering would leave too few items,
 * we fall back to returning all items sorted by score.
 *
 * IMPORTANT: a complete outfit needs a top, a bottom, AND shoes. If the raw
 * filter drops every shoe candidate (e.g. the catalog has only chunky sneakers
 * and the user asked for Old Money), we'd end up with outfits that can never
 * have shoes. To prevent that, we ALWAYS keep the top-`perCategoryFloor`
 * items per macroCategory, even if they scored below the drop threshold.
 */
export function rankItemsForStyle<T extends ItemForInference>(
  items: T[],
  style: StyleId,
  opts: { minKeep?: number; dropThreshold?: number; perCategoryFloor?: number } = {},
): T[] {
  const { minKeep = 8, dropThreshold = -3, perCategoryFloor = 3 } = opts;

  const scored = items.map((item) => ({
    item,
    raw: scoreStyle(buildSearchBlob(item), STYLE_SIGNALS[style] || []),
    macro: (item.macroCategory || '').toLowerCase(),
  }));

  scored.sort((a, b) => b.raw - a.raw);

  const passesThreshold = new Set<number>();
  scored.forEach((s, idx) => {
    if (s.raw > dropThreshold) passesThreshold.add(idx);
  });

  // Always keep the top `perCategoryFloor` per macroCategory, regardless of
  // global score. This guarantees each category (top / bottom / outerwear /
  // shoes) has at least some representation in the candidate pool.
  if (perCategoryFloor > 0) {
    const perCatCount = new Map<string, number>();
    scored.forEach((s, idx) => {
      if (!s.macro) return;
      const count = perCatCount.get(s.macro) || 0;
      if (count < perCategoryFloor) {
        passesThreshold.add(idx);
        perCatCount.set(s.macro, count + 1);
      }
    });
  }

  const filtered = scored.filter((_, idx) => passesThreshold.has(idx));
  const final = filtered.length >= minKeep ? filtered : scored;

  return final.map((s) => s.item);
}

/**
 * Decide whether an outfit should force a two-top layered composition
 * (base top + outerwear/main top + bottom + shoes). Layering is the
 * backbone of old_money / business_casual / streetwear looks, and is
 * always required when the weather is cool. A user prompt that explicitly
 * asks for a summer / no-jacket look disables layering.
 */
export interface LayeringWeather {
  temp?: number | null;
  condition?: string | null;
}

export function needsLayering(
  style: string | null | undefined,
  weather?: LayeringWeather | null,
  prompt?: string | null,
): boolean {
  const normalized = normalizeStyleId(style || '');
  const promptBlob = (prompt || '').toLowerCase();
  if (/\b(summer|hot|heatwave|tee[-\s]?only|no jacket|no outerwear|beach)\b/.test(promptBlob)) {
    return false;
  }

  const condition = (weather?.condition || '').toLowerCase();
  const temp = typeof weather?.temp === 'number' ? weather!.temp! : null;
  const coldTemp = temp != null && temp < 18;
  const coldCondition = /\b(cold|chilly|freezing|snow|rain|drizzle|wind|storm)\b/.test(condition);
  if (coldTemp || coldCondition) return true;

  if (normalized === 'old_money' || normalized === 'business_casual' || normalized === 'classic') return true;
  if (normalized === 'streetwear') return true; // hoodie-over-tee is canonical
  if (normalized === 'y2k') return false;
  return false;
}

/**
 * Normalize any free-form style string (e.g. "Old Money", "old-money",
 * " OLD_MONEY ") to our canonical StyleId. Falls back to 'casual'.
 */
export function normalizeStyleId(raw: string | null | undefined): StyleId {
  if (!raw) return 'casual';
  const key = raw.toLowerCase().replace(/[\s-]+/g, '_').trim();
  if (
    key === 'old_money' ||
    key === 'streetwear' ||
    key === 'minimalist' ||
    key === 'y2k' ||
    key === 'business_casual' ||
    key === 'casual' ||
    key === 'classic'
  ) {
    return key as StyleId;
  }
  return 'casual';
}
