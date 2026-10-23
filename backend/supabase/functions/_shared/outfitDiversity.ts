/**
 * outfitDiversity — product-uniqueness + soft visual-variety layer for
 * AI outfit decks.
 *
 * The hard rule is PRODUCT uniqueness: the same non-anchor actual product
 * (id / normalized name / normalized image) must not appear in more than
 * one look when valid alternatives exist. Same formula is allowed; same
 * clothing is not. A deck of
 *
 *   navy blazer  + white oxford  + grey tailored trousers + brown loafers
 *   charcoal blazer + blue poplin + black trousers       + penny loafers
 *   beige blazer + striped shirt + navy trousers         + tassel loafers
 *
 * is VALID: every actual product is different, even though the formula
 * (blazer + shirt + trousers + loafers) repeats.
 *
 * Three layers, in order:
 *
 *   1. PRODUCT IDENTITY — id / normalized name / normalized image.
 *      Garment family is NOT identity.
 *   2. DECK SELECTION — pick the COMBINATION of looks with zero
 *      non-anchor product reuse, quality first. Relaxation is allowed
 *      only for a genuinely thin inventory
 *      (`insufficient_unique_products`).
 *   3. SOFT VISUAL VARIETY — fingerprint + similarity scoring used only
 *      as a tie-breaker among equally good unique-product decks.
 *      visualSimilarity / sameArchetype describe a deck; they never
 *      reject an otherwise valid look.
 *
 * Pure + deterministic, no imports: safe in the Expo client and inside the
 * Deno edge function. Covered by __tests__/features/outfitVisualDiversity.test.ts.
 */

// ── Slots ────────────────────────────────────────────────────────────────

export type DiversitySlot = 'outerwear' | 'top' | 'bottom' | 'shoes';

export const DIVERSITY_SLOTS: readonly DiversitySlot[] = ['outerwear', 'top', 'bottom', 'shoes'] as const;

/**
 * Slots that define the outfit FORMULA.
 *
 * Footwear is a finishing detail, not a styling direction: swapping loafers
 * for Chelsea boots does not turn the same blazer + shirt + trousers look
 * into a different outfit. It still contributes to the similarity SCORE, but
 * it describes the shared formula; it is not a rejection rule.
 */
export const FORMULA_SLOTS: readonly DiversitySlot[] = ['outerwear', 'top', 'bottom'] as const;

/**
 * Hard ceiling for "these two looks look the same to a human".
 *
 * Chosen from the score distribution produced by this module (see
 * __tests__/features/outfitVisualDiversity.test.ts for the live numbers):
 *   identical formula, different SKUs   -> 0.80 .. 0.93
 *   same formula, one slot swapped      -> 0.55 .. 0.70
 *   clearly different concepts           -> 0.10 .. 0.35
 *
 * 0.65 sits inside the "one slot swapped" band, which is why the archetype
 * rule (sameArchetype) is a SEPARATE, hard gate: two looks that share the
 * outfit FORMULA are rejected at every rung, no matter how well their
 * colours happen to differ.
 */
export const MAX_VISUAL_SIMILARITY = 0.65;

/** Max number of garment families two looks may share (anchor slot excluded). */
export const MAX_SHARED_VISUAL_FAMILIES = 1;

/**
 * Similarity weights. Composition dominates on purpose: two looks built on
 * the same formula read as "the same outfit" even when every colour, fabric
 * and SKU differs. Exact colours are the weakest signal.
 *
 * archetype + families + silhouette = 60% of the score: the SHAPE of the
 * outfit decides whether two looks are the same idea. Colour (10%) and
 * pattern (5%) together are worth less than the silhouette alone.
 */
export const DIVERSITY_WEIGHTS = {
    archetype: 0.32,
    families: 0.18,
    silhouette: 0.10,
    color: 0.10,
    footwear: 0.08,
    material: 0.07,
    pattern: 0.05,
    formality: 0.10,
} as const;

/**
 * Escape hatch for the shared-family rule: a pair that shares two garment
 * families is still allowed when the looks are clearly different CONCEPTS
 * (similarity below this value), e.g. a coat + knit + denim look next to a
 * blazer + shirt + tailored look.
 *
 * 0.50 sits below every "same concept" pair measured by the score
 * distribution (0.52 .. 0.67) and above every "clearly different concept"
 * pair (0.09 .. 0.29).
 */
export const FAMILY_ESCAPE_SIMILARITY = 0.50;

/** Deck-score tuning: quality is normalised per look (0..1). */
export const DIVERSITY_BONUS_PER_ARCHETYPE = 0.35;
export const DIVERSITY_SIMILARITY_PENALTY = 1.0;

// ── Item text helpers ───────────────────────────────────────────────────

function text(value: unknown): string {
    return typeof value === 'string' ? value : value == null ? '' : String(value);
}

/** Name-ish fields only — used for garment identity (a description may lie). */
function nameBlob(item: any): string {
    if (!item) return '';
    return `${text(item.name)} ${text(item.type)} ${text(item.subCategory)} ${text(item.category)}`.toLowerCase();
}

/**
 * Garment-signal blobs.
 *
 * The product NAME is the reliable signal; the catalog metadata is coarse
 * ("blazer" for EVERY outerwear row) and must never override what the name
 * actually says. A cardigan catalogued under subCategory "blazer" is still a
 * cardigan. The metadata is only used when the name carries no garment word
 * at all ("Zara Slim Fit" + subCategory "blazer").
 */
function garmentBlobs(item: any): { name: string; meta: string } {
    return {
        name: text(item?.name).toLowerCase(),
        meta: `${text(item?.type)} ${text(item?.subCategory)} ${text(item?.category)}`.toLowerCase(),
    };
}

/** Every text field — used for material / colour evidence. */
function fullBlob(item: any): string {
    if (!item) return '';
    return `${text(item.name)} ${text(item.type)} ${text(item.subCategory)} ${text(item.category)} ${text(item.description)} ${text(item.material)} ${text(item.brand)}`.toLowerCase();
}

export function normalizeNameKey(name: unknown): string {
    return text(name).toLowerCase().replace(/[^a-z0-9]+/g, ' ').replace(/\s+/g, ' ').trim();
}

export function imageKey(item: any): string {
    const raw = text(item?.imageUrl || item?.image).trim().toLowerCase();
    if (!raw) return '';
    return raw
        .split('#')[0]
        .split('?')[0]
        .replace(/^https?:\/\//, '')
        .replace(/^\/\//, '')
        .replace(/\/+$/, '');
}

export interface ProductIdentity {
    /** Stable id key (`id:...`). */
    idKey: string;
    /** Normalized catalog name key (`n:...`). */
    nameKey: string;
    /** Normalized image URL key (`i:...`). */
    imageKey: string;
    /**
     * Best single display/compare key for the product: normalized name when
     * available, otherwise the image URL, otherwise the id. Two rows that
     * differ only by SKU id or thumbnail transform share this key.
     */
    canonicalKey: string;
    /** All equivalence keys. A shared key means "same product". */
    keys: string[];
}

/**
 * Product identity for deck uniqueness.
 *
 * A product is the same product when ANY of these match:
 *   - catalog id,
 *   - normalized name (case / punctuation / ingestion suffix),
 *   - normalized image URL (thumbnail query params stripped).
 *
 * Garment family is deliberately NOT part of identity. A black loafer and a
 * brown loafer are two valid products of the same family; using the family
 * as identity would wrongly collapse them and force repeated products.
 */
export function productIdentityOf(item: any): ProductIdentity {
    const id = text(item?.id).trim();
    const idKey = id ? `id:${id}` : '';
    const name = normalizeNameKey(item?.name);
    const nameKey = name ? `n:${name}` : '';
    const image = imageKey(item);
    const imageKeyValue = image ? `i:${image}` : '';
    const canonicalKey = nameKey || imageKeyValue || idKey;
    const keys = [idKey, nameKey, imageKeyValue].filter(Boolean);
    return { idKey, nameKey, imageKey: imageKeyValue, canonicalKey, keys };
}

/** All equivalence keys for one product (id + normalized name + image). */
export function productIdentityKeys(item: any): string[] {
    return productIdentityOf(item).keys;
}

/** Best single key for logs / reports. */
export function canonicalProductKey(item: any): string {
    return productIdentityOf(item).canonicalKey;
}

/**
 * Count distinct products, collapsing rows that share an id, name or image.
 * Used by the inventory assessment (family is not identity).
 */
export function countDistinctProducts(items: any[]): number {
    const groups: Array<Set<string>> = [];
    for (const item of Array.isArray(items) ? items : []) {
        if (!item) continue;
        const keys = productIdentityKeys(item);
        if (keys.length === 0) continue;
        const matches = groups.filter((group) => keys.some((key) => group.has(key)));
        if (matches.length === 0) {
            groups.push(new Set(keys));
            continue;
        }
        const [first, ...rest] = matches;
        for (const key of keys) first.add(key);
        for (const group of rest) {
            for (const key of group) first.add(key);
            groups.splice(groups.indexOf(group), 1);
        }
    }
    return groups.length;
}

// ── Slot resolution ─────────────────────────────────────────────────────

const SLOT_BY_MACRO: Record<string, DiversitySlot> = {
    outerwear: 'outerwear',
    top: 'top',
    bottom: 'bottom',
    shoes: 'shoes',
};

/** Which slot of the 2x2 card an item belongs to. */
export function diversitySlotOf(item: any): DiversitySlot | null {
    const macro = text(item?.macroCategory).toLowerCase().trim();
    if (SLOT_BY_MACRO[macro]) return SLOT_BY_MACRO[macro];
    const blob = nameBlob(item);
    if (/\b(shoe|sneaker|boot|loafer|sandal|trainer|derby|moccasin|oxford)\b/.test(blob)) return 'shoes';
    if (/\b(pant|trouser|jean|chino|short|skirt|jogger)\b/.test(blob)) return 'bottom';
    if (/\b(jacket|coat|blazer|hoodie|cardigan|sweater|pullover|overshirt|vest|puffer|bomber|trench|peacoat)\b/.test(blob)) return 'outerwear';
    if (/\b(shirt|tee|t-?shirt|polo|blouse|henley|tank|top)\b/.test(blob)) return 'top';
    return null;
}

// ── Garment families (finer than the coarse "variety family") ───────────
//
// The coarse families used by the product dedupe (outer:blazer,
// top:shirt, bottom:tailored, shoe:loafer) cannot tell a blazer from a
// sport coat, nor a cardigan from an overshirt. These tables are ordered:
// the FIRST match wins.

const OUTERWEAR_FAMILIES: Array<[RegExp, string]> = [
    [/\b(tuxedos?|dinner\s*suit|black[\s-]?tie)\b/, 'tuxedo'],
    // A dinner jacket is a different garment from a tuxedo: a shawl-collar
    // midnight dinner jacket reads as a different look than a classic black
    // tuxedo, and black-tie decks must be able to show both.
    [/\b(dinner\s*jackets?|tuxedo\s*jackets?)\b/, 'dinner-jacket'],
    [/\bsuede\b/, 'suede-jacket'],
    [/\b(blazers?|suit\s*jackets?|sport\s*coats?|blousons?)\b/, 'blazer'],
    [/\b(overcoats?|top\s*coats?|peacoats?|trench|duffle|greatcoats?|mac\s*coats?|winter\s*coats?|parkas?|puffer|down\s*jackets?|capes?)\b/, 'coat'],
    [/\b(overshirts?|shackets?|safari\s*jackets?|field\s*jackets?|chore\s*(coats?|jackets?)|casual\s*jackets?|work\s*jackets?|coach\s*jackets?|ruckers?)\b/, 'overshirt'],
    [/\bcardigans?\b/, 'cardigan'],
    [/\b(hoodies?|sweatshirts?|sweat[\s-]shirts?)\b/, 'hoodie'],
    [/\bbombers?\b/, 'bomber'],
    [/\b(gilets?|waistcoats?|vests?)\b/, 'vest'],
    [/\b(technical|softshell|windbreakers?|anoraks?|rain\s*jackets?|shell\s*jackets?|3-?in-?1)\b/, 'utility'],
    [/\b(sweaters?|pullovers?|jumpers?|crewnecks?|roll[\s-]?necks?|turtlenecks?|fisherman|cardigan\s*jackets?)\b/, 'knitwear'],
    // A plain coat and a peacoat are the same ROLE; a plain jacket is not.
    [/\b(coats?)\b/, 'coat'],
    [/\b(jackets?)\b/, 'jacket-generic'],
];

const TOP_FAMILIES: Array<[RegExp, string]> = [
    [/\b(tuxedos?|dress\s*shirts?|formal\s*shirts?|wing[\s-]?collar|french\s*cuff|pleated\s*shirts?)\b/, 'dress-shirt'],
    [/\b(oxfords?|poplin)\b/, 'dress-shirt'],
    [/\b(polos?|piqu[eé])\b/, 'polo'],
    [/\b(sweaters?|crewnecks?|roll[\s-]?necks?|turtlenecks?|merino|cashmere|knitwear|fine[\s-]?gauge\s*knit)\b/, 'knit'],
    [/\bhenleys?\b/, 'henley'],
    [/\bblouses?\b/, 'blouse'],
    [/\b(athletic|training|performance|dri-?fit|gym|running)\b/, 'athletic'],
    [/\b(t-?shirts?|tees?)\b/, 'tee'],
    [/\b(tanks?|sleeveless)\b/, 'tank'],
    [/\b(shirts?|button[\s-]?downs?|flannel|chambray|linen)\b/, 'shirt'],
];

const BOTTOM_FAMILIES: Array<[RegExp, string]> = [
    [/\b(shorts?|bermudas?)\b/, 'short'],
    [/\b(jeans?|denim)\b/, 'jean'],
    [/\b(corduroy|cords?)\b/, 'corduroy'],
    [/\b(joggers?|sweatpants?|sweat[\s-]pants?|track\s*pants?|leisure)\b/, 'jogger'],
    [/\bleather\s*(pants?|trousers?)/, 'leather'],
    [/\b(chinos?|khakis?|cotton\s*(pants?|trousers?)|casual\s*(pants?|trousers?)|twill)\b/, 'chino'],
    [/\b(tailored?|dress\s*(pants?|trousers?)|suit|pleated|wool\s*(pants?|trousers?)|slacks?|formal\s*(pants?|trousers?)|tux|trousers)\b/, 'tailored'],
    [/\b(pants?)\b/, 'trousers-generic'],
];

const SHOE_FAMILIES: Array<[RegExp, string]> = [
    [/\b(patent|opera\s*pumps?|tuxedo\s*shoes?)\b/, 'tuxedo-shoe'],
    [/\b(moccasins?|driving|driver|gommino)\b/, 'moccasin'],
    [/\b(loafers?|penny|tassel)\b/, 'loafer'],
    [/\b(oxfords?|derbies|derby|brogues?|monk|cap-?toe|wingtip)\b/, 'dress-shoe'],
    [/\b(boots?|chelsea|chukka)\b/, 'boot'],
    [/\b(sneakers?|trainers?|running\s*shoes?|court)\b/, 'sneaker'],
    [/\b(sandals?|slides?|mules?|espadrilles?|flip[\s-]?flops?)\b/, 'sandal'],
    [/\bshoes?\b/, 'dress-shoe'],
];

const FAMILY_TABLES: Record<DiversitySlot, Array<[RegExp, string]>> = {
    outerwear: OUTERWEAR_FAMILIES,
    top: TOP_FAMILIES,
    bottom: BOTTOM_FAMILIES,
    shoes: SHOE_FAMILIES,
};

/** Slot prefix used by the family keys ('outer:', 'top:', 'bottom:', 'shoe:'). */
const SLOT_PREFIX: Record<DiversitySlot, string> = {
    outerwear: 'outer',
    top: 'top',
    bottom: 'bottom',
    shoes: 'shoe',
};

function familyFor(slot: DiversitySlot, item: any): string {
    const { name, meta } = garmentBlobs(item);
    if (name) {
        for (const [pattern, family] of FAMILY_TABLES[slot]) {
            if (pattern.test(name)) return family;
        }
    }
    for (const [pattern, family] of FAMILY_TABLES[slot]) {
        if (meta && pattern.test(meta)) return family;
    }
    return 'other';
}

/**
 * Fine-grained garment family key, e.g. 'outer:blazer', 'top:dress-shirt',
 * 'bottom:tailored', 'shoe:loafer'. Deliberately finer than the coarse
 * `garmentVarietyFamily` used by the product dedupe — that one answers
 * "same SKU family?", this one answers "same visual garment type?".
 */
export function garmentVisualFamily(item: any): string {
    const slot = diversitySlotOf(item);
    if (!slot) return '';
    return `${SLOT_PREFIX[slot]}:${familyFor(slot, item)}`;
}

/**
 * Coarse archetype token per fine family.
 *
 * The ARCHETYPE is the outfit FORMULA ("blazer + shirt + tailored trousers +
 * loafers"). It must be coarse enough that an oxford shirt and a poplin
 * shirt are the same slot in the formula, and fine enough that a blazer and
 * a cardigan are not.
 */
const ARCHETYPE_TOKEN_BY_FAMILY: Record<string, string> = {
    // outerwear
    'outer:tuxedo': 'tuxedo',
    'outer:dinner-jacket': 'dinner-jacket',
    'outer:blazer': 'blazer',
    'outer:coat': 'coat',
    'outer:suede-jacket': 'jacket',
    'outer:jacket-generic': 'jacket',
    'outer:overshirt': 'overshirt',
    'outer:cardigan': 'cardigan',
    'outer:knitwear': 'knitwear',
    'outer:hoodie': 'hoodie',
    'outer:bomber': 'bomber',
    'outer:vest': 'vest',
    'outer:utility': 'utility',
    // top
    'top:dress-shirt': 'shirt',
    'top:shirt': 'shirt',
    'top:blouse': 'shirt',
    'top:henley': 'tee',
    'top:tee': 'tee',
    'top:tank': 'tee',
    'top:knit': 'knit',
    'top:polo': 'polo',
    'top:athletic': 'athletic',
    // bottom
    'bottom:tailored': 'tailored',
    'bottom:trousers-generic': 'tailored',
    'bottom:chino': 'chino',
    'bottom:corduroy': 'chino',
    'bottom:jean': 'jean',
    'bottom:jogger': 'jogger',
    'bottom:short': 'shorts',
    'bottom:leather': 'leather',
    // shoes
    'shoe:tuxedo-shoe': 'dress-shoe',
    'shoe:dress-shoe': 'dress-shoe',
    'shoe:loafer': 'loafer',
    'shoe:moccasin': 'moccasin',
    'shoe:sneaker': 'sneaker',
    'shoe:boot': 'boot',
    'shoe:sandal': 'sandal',
};

/** The archetype token of an item — the family without its slot prefix. */
export function garmentArchetypeToken(item: any): string {
    const slot = diversitySlotOf(item);
    if (!slot) return 'none';
    const family = garmentVisualFamily(item);
    return ARCHETYPE_TOKEN_BY_FAMILY[family] ?? familyFor(slot, item);
}

/**
 * Structural detail tokens that change how a garment READS even when the
 * family, colour family and material are identical — the "structure" half of
 * the §12 rule (structured double-breasted blazer vs unstructured linen
 * blazer).
 */
const VARIANT_PATTERNS: Array<[RegExp, string]> = [
    [/double[\s-]?breasted/, 'double-breasted'],
    [/single[\s-]?breasted/, 'single-breasted'],
    [/shawl/, 'shawl-collar'],
    [/peak/, 'peak-lapel'],
    [/notch/, 'notch-lapel'],
    [/unstructured|soft[\s-]?shoulder/, 'unstructured'],
    [/structured/, 'structured'],
    [/quilted/, 'quilted'],
    [/pleated/, 'pleated'],
    [/double[\s-]?faced|reversible/, 'double-faced'],
    [/patch[\s-]?pocket/, 'patch-pocket'],
    [/hooded|hood/, 'hooded'],
    [/zip/, 'zipped'],
    [/buttoned|button[\s-]?up/, 'buttoned'],
];

/** Structural variant tokens of an item. */
export function variantTokensOf(item: any): string[] {
    const { name, meta } = garmentBlobs(item);
    const blob = name || meta;
    const tokens: string[] = [];
    for (const [pattern, token] of VARIANT_PATTERNS) {
        if (!tokens.includes(token) && pattern.test(blob)) tokens.push(token);
    }
    return tokens;
}

// ── Colour ──────────────────────────────────────────────────────────────

const COLOR_TOKENS: Array<[RegExp, string, string]> = [
    // [pattern, canonical token, coarse family]
    [/\b(navy|midnight|ink)\b/, 'navy', 'cool-dark'],
    [/\b(indigo)\b/, 'indigo', 'cool-dark'],
    [/\b(black|noir)\b/, 'black', 'dark'],
    [/\b(charcoal|graphite)\b/, 'charcoal', 'dark'],
    [/\b(dark\s*grey|dark\s*gray|dark\s*brown)\b/, 'dark-neutral', 'dark'],
    [/\b(grey|gray|greige|slate)\b/, 'grey', 'mid'],
    [/\b(white|off[\s-]?white)\b/, 'white', 'light'],
    [/\b(cream|ivory|ecru|milk|off[\s-]?white)\b/, 'cream', 'light'],
    [/\b(beige|sand|stone|taupe|oat)\b/, 'beige', 'light'],
    [/\b(light\s*blue|sky|ice)\b/, 'light-blue', 'light'],
    [/\b(blue|chambray)\b/, 'blue', 'cool-mid'],
    [/\b(tan|camel|khaki|fawn|cognac)\b/, 'tan', 'earth'],
    [/\b(brown|chocolate|coffee|espresso|walnut)\b/, 'brown', 'earth'],
    [/\b(olive|green|forest|sage|mint|bottle)\b/, 'green', 'earth'],
    [/\b(burgundy|wine|maroon|oxblood)\b/, 'burgundy', 'warm'],
    [/\b(red|scarlet|crimson)\b/, 'red', 'warm'],
    [/\b(rust|terracotta|orange)\b/, 'rust', 'warm'],
    [/\b(pink|lilac|purple|mauve|plum)\b/, 'pink', 'warm'],
    [/\b(yellow|mustard)\b/, 'yellow', 'warm'],
];

/** Canonical colour tokens of an item (colour field + name). */
export function colorTokensOf(item: any): string[] {
    const blob = `${text(item?.color)} ${text(item?.name)} ${text(item?.description)}`.toLowerCase();
    const tokens: string[] = [];
    for (const [pattern, token] of COLOR_TOKENS) {
        if (tokens.includes(token)) continue;
        if (pattern.test(blob)) tokens.push(token);
    }
    return tokens;
}

/** Coarse colour family of an item (dark / light / mid / earth / warm / cool). */
export function colorFamilyOf(item: any): string {
    const blob = `${text(item?.color)} ${text(item?.name)} ${text(item?.description)}`.toLowerCase();
    for (const [pattern, , family] of COLOR_TOKENS) {
        if (pattern.test(blob)) return family;
    }
    return 'neutral';
}

// ── Material / texture / pattern ────────────────────────────────────────

const MATERIAL_PATTERNS: Array<[RegExp, string]> = [
    [/\bsuede\b/, 'suede'],
    [/\bcashmere\b/, 'cashmere'],
    [/\bmerino\b/, 'merino'],
    [/\b(wool|flannel|tweed|herringbone|harris|tropical)\b/, 'wool'],
    [/\blinen\b/, 'linen'],
    [/\bdenim\b/, 'denim'],
    [/\bcorduroy\b/, 'corduroy'],
    [/\bvelvet\b/, 'velvet'],
    [/\bsilk\b/, 'silk'],
    [/\bleather\b/, 'leather'],
    [/\bcotton\b/, 'cotton'],
    [/\b(jersey|knit\w*)\b/, 'jersey'],
    [/\b(synthetic|polyester|nylon|technical|performance|recycled)\b/, 'technical'],
];

/** Material tokens of an item (material field + name + description). */
export function materialsOf(item: any): string[] {
    const blob = fullBlob(item);
    const out: string[] = [];
    for (const [pattern, material] of MATERIAL_PATTERNS) {
        if (!out.includes(material) && pattern.test(blob)) out.push(material);
    }
    return out;
}

const PATTERN_PATTERNS: Array<[RegExp, string]> = [
    [/\b(stripe|striped|pinstripe|striped)\b/, 'stripe'],
    [/\b(check|checked|plaid|tartan|gingham|houndstooth|windowpane)\b/, 'check'],
    [/\b(herringbone|tweed|jacquard|textured|texture)\b/, 'texture'],
    [/\b(print|printed|graphic|floral|logo|camo|patterned)\b/, 'print'],
];

/** Pattern profile of an item: solid / stripe / check / texture / print. */
export function patternOf(item: any): string {
    const declared = text(item?.pattern).toLowerCase().trim();
    const blob = `${declared} ${text(item?.name)} ${text(item?.description)}`.toLowerCase();
    for (const [pattern, profile] of PATTERN_PATTERNS) {
        if (pattern.test(blob)) return profile;
    }
    if (/solid/.test(blob)) return 'solid';
    return 'solid';
}

/** Texture profile of an item: knit / suede / denim / textured / technical / smooth. */
export function textureOf(item: any): string {
    const materials = materialsOf(item);
    if (materials.includes('jersey')) return 'knit';
    if (materials.includes('suede')) return 'suede';
    if (materials.includes('denim')) return 'denim';
    if (materials.includes('technical')) return 'technical';
    if (materials.includes('wool') || materials.includes('corduroy')) return 'textured';
    if (materials.includes('knit')) return 'knit';
    return 'smooth';
}

// ── Formality ───────────────────────────────────────────────────────────

export type FormalityTier = 'athletic' | 'casual' | 'smart-casual' | 'formal' | 'evening';

const FORMALITY_SCORE: Record<FormalityTier, number> = {
    athletic: 0,
    casual: 1,
    'smart-casual': 2,
    formal: 3,
    evening: 4,
};

const FORMALITY_BY_FAMILY: Record<string, FormalityTier> = {
    // outerwear
    'outer:tuxedo': 'evening',
    'outer:suede-jacket': 'smart-casual',
    'outer:blazer': 'formal',
    'outer:coat': 'formal',
    'outer:overshirt': 'casual',
    'outer:cardigan': 'smart-casual',
    'outer:hoodie': 'casual',
    'outer:bomber': 'casual',
    'outer:vest': 'smart-casual',
    'outer:utility': 'casual',
    'outer:knitwear': 'smart-casual',
    'outer:jacket-generic': 'smart-casual',
    // top
    'top:dress-shirt': 'formal',
    'top:polo': 'smart-casual',
    'top:knit': 'smart-casual',
    'top:henley': 'casual',
    'top:blouse': 'formal',
    'top:athletic': 'athletic',
    'top:tee': 'casual',
    'top:tank': 'casual',
    'top:shirt': 'smart-casual',
    // bottom
    'bottom:short': 'casual',
    'bottom:jean': 'casual',
    'bottom:corduroy': 'smart-casual',
    'bottom:jogger': 'athletic',
    'bottom:leather': 'casual',
    'bottom:chino': 'smart-casual',
    'bottom:tailored': 'formal',
    'bottom:trousers-generic': 'smart-casual',
    // shoes
    'shoe:tuxedo-shoe': 'evening',
    'shoe:moccasin': 'smart-casual',
    'shoe:loafer': 'smart-casual',
    'shoe:dress-shoe': 'formal',
    'shoe:boot': 'smart-casual',
    'shoe:sneaker': 'casual',
    'shoe:sandal': 'casual',
};

/** Formality tier of a single garment. */
export function formalityOf(item: any): FormalityTier {
    const slot = diversitySlotOf(item);
    if (!slot) return 'casual';
    const family = garmentVisualFamily(item);
    const tier = FORMALITY_BY_FAMILY[family];
    if (tier) return tier;
    if (slot === 'shoes') return 'casual';
    return 'smart-casual';
}

// ── Fingerprint ─────────────────────────────────────────────────────────

export interface OutfitSlotEntry {
    slot: DiversitySlot;
    /** Fine family key with slot prefix, e.g. 'outer:blazer'. */
    family: string;
    /** Family without the slot prefix — the archetype token. */
    archetypeToken: string;
    itemId: string;
    colors: string[];
    colorFamily: string;
    materials: string[];
    /** Structural detail tokens (shawl collar, double-breasted, …). */
    variants: string[];
    pattern: string;
    texture: string;
    formality: FormalityTier;
    formalityScore: number;
    /** Identity keys (id / normalized name / image) for product overlap. */
    identityKeys: string[];
}

export interface OutfitFingerprint {
    /** Per-slot detail. Missing slots are absent from the map. */
    slots: Partial<Record<DiversitySlot, OutfitSlotEntry>>;
    /** Structural formula, e.g. 'blazer-dress-shirt-tailored-loafer'. */
    archetype: string;
    /** The archetype token per slot. */
    archetypeTokens: Partial<Record<DiversitySlot, string>>;
    /** Overall silhouette of the look. */
    silhouette: string;
    /** Mean formality score of the look (0 athletic .. 4 evening). */
    formalityScore: number;
    /** Formality label of the look. */
    formality: FormalityTier;
    /** All colour tokens in the look. */
    colors: string[];
    /** Dominant (most frequent) colour family. */
    colorFamily: string;
    /** All material tokens in the look. */
    materials: string[];
    /** Overall pattern profile of the look. */
    patternProfile: string;
    /** Overall texture profile of the look. */
    textureProfile: string;
    /** Footwear family ('loafer', 'sneaker', …). */
    footwearFamily: string;
    /** Fine family keys of every slot. */
    families: string[];
    /** Product ids in the look. */
    itemIds: string[];
}

function identityKeysOf(item: any): string[] {
    return productIdentityKeys(item);
}

const SILHOUETTE_BY_FAMILY: Record<string, string> = {
    'outer:blazer': 'tailored',
    'outer:coat': 'tailored',
    'outer:tuxedo': 'evening-formal',
    'outer:suede-jacket': 'smart-casual',
    'outer:overshirt': 'relaxed',
    'outer:bomber': 'relaxed',
    'outer:utility': 'relaxed',
    'outer:hoodie': 'relaxed',
    'outer:cardigan': 'layered-knit',
    'outer:knitwear': 'layered-knit',
    'outer:vest': 'smart-casual',
};

function silhouetteOf(slots: Partial<Record<DiversitySlot, OutfitSlotEntry>>): string {
    const outer = slots.outerwear;
    const bottom = slots.bottom;
    const top = slots.top;
    const shoe = slots.shoes;
    const has = (entry: OutfitSlotEntry | undefined, families: string[]) =>
        Boolean(entry && families.includes(entry.family));

    if (has(outer, ['outer:tuxedo']) || (shoe?.family === 'shoe:tuxedo-shoe')) return 'evening-formal';
    const allAthletic = [outer, top, bottom, shoe]
        .filter(Boolean)
        .every((entry) => (entry as OutfitSlotEntry).formality === 'athletic');
    if (allAthletic && [outer, top, bottom, shoe].some(Boolean)) return 'sporty';
    if (has(outer, ['outer:blazer', 'outer:coat']) && has(bottom, ['bottom:tailored'])) return 'tailored';
    if (has(bottom, ['bottom:jean', 'bottom:short', 'bottom:jogger'])) return 'relaxed';
    if (has(top, ['top:knit']) || has(outer, ['outer:cardigan', 'outer:knitwear'])) return 'layered-knit';
    if (has(outer, ['outer:overshirt', 'outer:bomber', 'outer:utility', 'outer:hoodie', 'outer:vest'])) return 'relaxed';
    return 'smart-casual';
}

/**
 * Build the visual fingerprint of one complete look.
 *
 * `ignoreSlot` marks the slot pinned by the user anchor: it is excluded
 * from archetype / family comparison so a repeated anchor never counts as
 * visual repetition.
 */
export function buildOutfitFingerprint(items: any[]): OutfitFingerprint {
    const slots: Partial<Record<DiversitySlot, OutfitSlotEntry>> = {};
    const list = Array.isArray(items) ? items : [];
    for (const item of list) {
        if (!item) continue;
        const slot = diversitySlotOf(item);
        if (!slot) continue;
        if (slots[slot]) continue; // first item of a slot wins
        const family = garmentVisualFamily(item);
        const archetypeToken = garmentArchetypeToken(item);
        const materials = materialsOf(item);
        slots[slot] = {
            slot,
            family,
            archetypeToken,
            itemId: text(item?.id),
            colors: colorTokensOf(item),
            colorFamily: colorFamilyOf(item),
            materials,
            variants: variantTokensOf(item),
            pattern: patternOf(item),
            texture: textureOf(item),
            formality: formalityOf(item),
            formalityScore: FORMALITY_SCORE[formalityOf(item)],
            identityKeys: identityKeysOf(item),
        };
    }

    const archetypeTokens: Partial<Record<DiversitySlot, string>> = {};
    for (const slot of DIVERSITY_SLOTS) {
        const entry = slots[slot];
        if (entry) archetypeTokens[slot] = entry.archetypeToken;
    }

    const archetype = DIVERSITY_SLOTS
        .filter((slot) => archetypeTokens[slot])
        .map((slot) => archetypeTokens[slot])
        .join('-');

    const allColors: string[] = [];
    const allMaterials: string[] = [];
    let formalitySum = 0;
    let formalityCount = 0;
    for (const slot of DIVERSITY_SLOTS) {
        const entry = slots[slot];
        if (!entry) continue;
        for (const color of entry.colors) if (!allColors.includes(color)) allColors.push(color);
        for (const material of entry.materials) if (!allMaterials.includes(material)) allMaterials.push(material);
        formalitySum += entry.formalityScore;
        formalityCount += 1;
    }
    const formalityScore = formalityCount > 0 ? formalitySum / formalityCount : 2;

    const families = DIVERSITY_SLOTS
        .filter((slot) => slots[slot])
        .map((slot) => (slots[slot] as OutfitSlotEntry).family);

    const colorCounts = new Map<string, number>();
    for (const slot of DIVERSITY_SLOTS) {
        const entry = slots[slot];
        if (!entry) continue;
        colorCounts.set(entry.colorFamily, (colorCounts.get(entry.colorFamily) || 0) + 1);
    }
    let colorFamily = 'mixed';
    let best = 0;
    for (const [family, count] of colorCounts) {
        if (count > best) {
            best = count;
            colorFamily = family;
        }
    }
    if (best <= 1 && colorCounts.size > 1) colorFamily = 'mixed';

    const patternProfile = dominantProfile(DIVERSITY_SLOTS.map((slot) => slots[slot]).filter(Boolean).map((e) => (e as OutfitSlotEntry).pattern));
    const textureProfile = dominantProfile(DIVERSITY_SLOTS.map((slot) => slots[slot]).filter(Boolean).map((e) => (e as OutfitSlotEntry).texture));

    return {
        slots,
        archetype,
        archetypeTokens,
        silhouette: silhouetteOf(slots),
        formalityScore,
        formality: formalityTierFor(formalityScore),
        colors: allColors,
        colorFamily,
        materials: allMaterials,
        patternProfile,
        textureProfile,
        footwearFamily: slots.shoes?.family ?? '',
        families,
        itemIds: DIVERSITY_SLOTS.map((slot) => slots[slot]?.itemId).filter(Boolean) as string[],
    };
}

function dominantProfile(values: string[]): string {
    if (values.length === 0) return 'solid';
    const counts = new Map<string, number>();
    for (const value of values) counts.set(value, (counts.get(value) || 0) + 1);
    let best = '';
    let bestCount = 0;
    for (const [value, count] of counts) {
        if (count > bestCount) {
            best = value;
            bestCount = count;
        }
    }
    return best;
}

function formalityTierFor(score: number): FormalityTier {
    if (score >= 3.5) return 'evening';
    if (score >= 2.5) return 'formal';
    if (score >= 1.5) return 'smart-casual';
    if (score >= 0.5) return 'casual';
    return 'athletic';
}

// ── Similarity ──────────────────────────────────────────────────────────

export interface SimilarityOptions {
    /** Slots excluded from the comparison (the pinned anchor slot). */
    ignoreSlots?: readonly DiversitySlot[];
}

function jaccard(a: string[], b: string[]): number {
    if (a.length === 0 && b.length === 0) return 0;
    const setA = new Set(a);
    const setB = new Set(b);
    if (setA.size === 0 || setB.size === 0) return 0;
    let intersection = 0;
    for (const value of setA) if (setB.has(value)) intersection += 1;
    const union = setA.size + setB.size - intersection;
    return union === 0 ? 0 : intersection / union;
}

const SILHOUETTE_RELATIONS: Record<string, number> = {
    'tailored|smart-casual': 0.5,
    'smart-casual|relaxed': 0.5,
    'tailored|relaxed': 0.25,
    'smart-casual|layered-knit': 0.5,
    'layered-knit|relaxed': 0.35,
    'evening-formal|tailored': 0.25,
    'evening-formal|smart-casual': 0.2,
    'sporty|relaxed': 0.2,
};

function silhouetteRelation(a: string, b: string): number {
    if (a === b) return 1;
    const key = [a, b].sort().join('|');
    return SILHOUETTE_RELATIONS[key] ?? 0;
}

/**
 * True when two garments of the SAME family are far enough apart visually
 * (different colour family AND different material) that they do not read as
 * "the same garment again" — e.g. a structured navy wool blazer vs a soft
 * beige linen blazer.
 */
export function isVisuallyDistinctVariant(a: OutfitSlotEntry, b: OutfitSlotEntry): boolean {
    if (a.family !== b.family) return false;
    const variantsDiffer =
        a.variants.join('|') !== b.variants.join('|')
        && (a.variants.length > 0 || b.variants.length > 0);
    if (variantsDiffer) return true;
    const colorDiffers = a.colorFamily !== b.colorFamily;
    const materialDiffers =
        a.materials.length > 0 && b.materials.length > 0
        && a.materials.join('|') !== b.materials.join('|');
    return colorDiffers && materialDiffers;
}

/**
 * Number of garment families two looks share (anchor slot excluded).
 * Same-family pairs that are visually distinct variants do not count.
 */
export function sharedVisualFamilies(
    a: OutfitFingerprint,
    b: OutfitFingerprint,
    opts: SimilarityOptions = {},
): number {
    const ignore = new Set(opts.ignoreSlots ?? []);
    let shared = 0;
    for (const slot of DIVERSITY_SLOTS) {
        if (ignore.has(slot)) continue;
        const left = a.slots[slot];
        const right = b.slots[slot];
        if (!left || !right) continue;
        if (left.family !== right.family) continue;
        if (isVisuallyDistinctVariant(left, right)) continue;
        shared += 1;
    }
    return shared;
}

/**
 * True when two looks are built on the SAME OUTFIT FORMULA.
 *
 * The archetype token is the coarse ROLE a garment plays in the formula
 * (blazer / shirt / tailored / loafer). Two looks that agree on the role in
 * every shared slot are the same styling formula even when every product
 * id, colour, fabric and photo differs:
 *
 *   navy blazer  + white oxford  + grey tailored trousers + brown loafers
 *   charcoal     + blue poplin   + black trousers       + penny loafers
 *
 * This is the rule that product-id dedupe cannot express, and it is why the
 * reported failure mode ("three SKUs of blazer + shirt + trousers + loafers")
 * is caught even when the numeric similarity score stays under the
 * threshold.
 *
 * ESCAPE: NONE at outfit level. A structured navy double-breasted blazer
 * and a soft beige unstructured linen blazer are different PRODUCTS (item
 * dedupe honours that), but they still fill the same blazer slot, so two
 * looks built on them are variations of one formula, not two directions.
 * §12 asks that a repeated family not decide the ITEM comparison; it must
 * not be read as "a different colourway creates a new outfit".
 *
 * The comparison runs over FORMULA_SLOTS only. Footwear is a finishing
 * detail: two looks that agree on outerwear + top + bottom are the same
 * outfit even when the shoes differ.
 *
 * Only slots present in BOTH looks are compared, so a layered 4-slot look
 * and a 3-slot look without outerwear are still comparable.
 */
export function sameArchetype(
    a: OutfitFingerprint,
    b: OutfitFingerprint,
    opts: SimilarityOptions = {},
): boolean {
    const ignore = new Set(opts.ignoreSlots ?? []);
    const shared = FORMULA_SLOTS.filter(
        (slot) => !ignore.has(slot) && a.slots[slot] && b.slots[slot],
    );
    if (shared.length === 0) return false;
    for (const slot of shared) {
        const left = a.slots[slot];
        const right = b.slots[slot];
        if (!left || !right) continue;
        if (left.archetypeToken !== right.archetypeToken) return false;
    }
    return true;
}

/** Number of identical products (id / name / photo) two looks share. */
export function sharedExactProducts(
    a: OutfitFingerprint,
    b: OutfitFingerprint,
    opts: SimilarityOptions = {},
): number {
    const ignore = new Set(opts.ignoreSlots ?? []);
    let shared = 0;
    for (const slot of DIVERSITY_SLOTS) {
        if (ignore.has(slot)) continue;
        const left = a.slots[slot];
        const right = b.slots[slot];
        if (!left || !right) continue;
        if (left.identityKeys.some((key) => right.identityKeys.includes(key))) shared += 1;
    }
    return shared;
}

export interface DeckProductDiversity {
    /** Distinct non-anchor products that appear in more than one look. */
    nonAnchorProductReuseCount: number;
    /** Canonical key of every repeated non-anchor product. */
    repeatedProductKeys: string[];
    /**
     * Canonical keys that appear in more than one look. Unlike the robust
     * cluster count above, this is a direct name/image-key view for logs.
     */
    repeatedCanonicalKeys: string[];
    /** How many looks contain the anchor product. */
    anchorLookCount: number;
    /** Extra anchor appearances beyond the first (0 for a normal deck). */
    anchorRepetitionCount: number;
    /** Canonical key of every item in every look (telemetry). */
    lookProductKeys: string[][];
}

function clusterProductOccurrences(
    occurrences: Array<{ keys: string[]; canonicalKey: string; look: number; anchor: boolean }>,
): number[][] {
    const parent = occurrences.map((_, index) => index);
    const find = (index: number): number => {
        let current = index;
        while (parent[current] !== current) {
            parent[current] = parent[parent[current]];
            current = parent[current];
        }
        return current;
    };
    const union = (a: number, b: number): void => {
        const rootA = find(a);
        const rootB = find(b);
        if (rootA !== rootB) parent[rootB] = rootA;
    };
    const owner = new Map<string, number>();
    occurrences.forEach((occurrence, index) => {
        for (const key of occurrence.keys) {
            const previous = owner.get(key);
            if (previous === undefined) owner.set(key, index);
            else union(index, previous);
        }
    });
    const groups = new Map<number, number[]>();
    occurrences.forEach((_, index) => {
        const root = find(index);
        if (!groups.has(root)) groups.set(root, []);
        groups.get(root)!.push(index);
    });
    return [...groups.values()];
}

/**
 * Measure product reuse in a final deck.
 *
 * Two rows are the same product when id / normalized name / normalized
 * image match. Garment family is intentionally ignored: different loafers
 * are different products. Items matching `anchorKeys` are reported via the
 * anchor counters and excluded from the non-anchor reuse count.
 */
export function analyzeDeckProductDiversity(
    looksItems: any[][],
    opts: { anchorKeys?: ReadonlySet<string> } = {},
): DeckProductDiversity {
    const occurrences: Array<{ keys: string[]; canonicalKey: string; look: number; anchor: boolean }> = [];
    const lookProductKeys: string[][] = [];
    const anchorKeys = opts.anchorKeys;
    (looksItems || []).forEach((items, look) => {
        const canonicalForLook: string[] = [];
        for (const item of Array.isArray(items) ? items : []) {
            if (!item) continue;
            const identity = productIdentityOf(item);
            const anchor = Boolean(anchorKeys && identity.keys.some((key) => anchorKeys.has(key)));
            occurrences.push({
                keys: identity.keys,
                canonicalKey: identity.canonicalKey,
                look,
                anchor,
            });
            canonicalForLook.push(identity.canonicalKey);
        }
        lookProductKeys.push(canonicalForLook);
    });

    const repeatedProductKeys = new Set<string>();
    let nonAnchorProductReuseCount = 0;
    for (const group of clusterProductOccurrences(occurrences)) {
        const looks = new Set(group.map((index) => occurrences[index].look));
        if (looks.size < 2) continue;
        const nonAnchor = group
            .map((index) => occurrences[index])
            .filter((occurrence) => !occurrence.anchor);
        if (nonAnchor.length === 0) continue;
        nonAnchorProductReuseCount += 1;
        repeatedProductKeys.add(nonAnchor[0].canonicalKey);
    }

    const canonicalLooks = new Map<string, Set<number>>();
    for (const occurrence of occurrences) {
        if (occurrence.anchor || !occurrence.canonicalKey) continue;
        if (!canonicalLooks.has(occurrence.canonicalKey)) canonicalLooks.set(occurrence.canonicalKey, new Set());
        canonicalLooks.get(occurrence.canonicalKey)!.add(occurrence.look);
    }
    const repeatedCanonicalKeys = [...canonicalLooks.entries()]
        .filter(([, looks]) => looks.size > 1)
        .map(([key]) => key);

    const anchorLookCount = new Set(
        occurrences.filter((occurrence) => occurrence.anchor).map((occurrence) => occurrence.look),
    ).size;

    return {
        nonAnchorProductReuseCount,
        repeatedProductKeys: [...repeatedProductKeys],
        repeatedCanonicalKeys,
        anchorLookCount,
        anchorRepetitionCount: Math.max(0, anchorLookCount - 1),
        lookProductKeys,
    };
}

/**
 * visualSimilarity — 0.0 (completely different concepts) .. 1.0 (the same
 * outfit twice). Composition dominates; exact colours barely matter.
 */
export function visualSimilarity(
    a: OutfitFingerprint,
    b: OutfitFingerprint,
    opts: SimilarityOptions = {},
): number {
    const ignore = new Set(opts.ignoreSlots ?? []);
    const slots = DIVERSITY_SLOTS.filter(
        (slot) => !ignore.has(slot) && (a.slots[slot] || b.slots[slot]),
    );
    if (slots.length === 0) return 0;

    // 1. Archetype / composition — the outfit FORMULA.
    //    The archetype token is the COARSE role of the garment in the
    //    formula (blazer / shirt / tailored / loafer), so an oxford shirt and
    //    a poplin shirt share the formula while a blazer and a cardigan do
    //    not. A same-family-but-visually-different garment (navy wool blazer
    //    vs beige linen blazer) does NOT count as a match: it is a
    //    variation of the role, not the same garment in the same role.
    let tokenMatches = 0;
    let comparedSlots = 0;
    for (const slot of slots) {
        const left = a.slots[slot];
        const right = b.slots[slot];
        if (!left || !right) continue;
        comparedSlots += 1;
        if (left.archetypeToken !== right.archetypeToken) continue;
        if (left.family === right.family && isVisuallyDistinctVariant(left, right)) {
            continue;
        }
        tokenMatches += 1;
    }
    const archetypeScore = comparedSlots > 0 ? tokenMatches / comparedSlots : 0;

    // 2. Fine garment families (slot-aware Jaccard).
    const leftFamilies = slots.map((slot) => a.slots[slot]?.family).filter(Boolean) as string[];
    const rightFamilies = slots.map((slot) => b.slots[slot]?.family).filter(Boolean) as string[];
    const familyScore = jaccard(leftFamilies, rightFamilies);

    // 3. Silhouette.
    const silhouetteScore = silhouetteRelation(a.silhouette, b.silhouette);

    // 4. Colour palette.
    const colorScore = jaccard(a.colors, b.colors) * 0.7
        + (a.colorFamily !== 'mixed' && a.colorFamily === b.colorFamily ? 0.3 : 0);

    // 5. Footwear.
    const footwearScore = a.footwearFamily && a.footwearFamily === b.footwearFamily ? 1 : 0;

    // 6. Material / texture.
    const materialScore = jaccard(a.materials, b.materials);

    // 7. Pattern.
    const patternScore = a.patternProfile === b.patternProfile ? 1 : 0;

    // 8. Formality coherence.
    const formalityDelta = Math.abs(a.formalityScore - b.formalityScore);
    const formalityScore = Math.max(0, 1 - formalityDelta / 2);

    const score =
        DIVERSITY_WEIGHTS.archetype * archetypeScore
        + DIVERSITY_WEIGHTS.families * familyScore
        + DIVERSITY_WEIGHTS.silhouette * silhouetteScore
        + DIVERSITY_WEIGHTS.color * colorScore
        + DIVERSITY_WEIGHTS.footwear * footwearScore
        + DIVERSITY_WEIGHTS.material * materialScore
        + DIVERSITY_WEIGHTS.pattern * patternScore
        + DIVERSITY_WEIGHTS.formality * formalityScore;

    return Math.max(0, Math.min(1, score));
}

// ── Inventory assessment ────────────────────────────────────────────────

export interface InventoryDiversity {
    /** Distinct visual families available per slot. */
    familiesPerSlot: Partial<Record<DiversitySlot, number>>;
    /** Distinct archetype TOKENS available per slot (the formula roles). */
    tokensPerSlot: Partial<Record<DiversitySlot, number>>;
    /** Distinct actual PRODUCTS available per slot (id/name/image identity). */
    uniqueProductsPerSlot: Partial<Record<DiversitySlot, number>>;
    /** Smallest per-slot unique-product count (the uniqueness bottleneck). */
    maxDistinctProducts: number;
    /**
     * True when every active slot has at least `target` distinct products.
     * This is the hard uniqueness capacity: when true, repeating a
     * non-anchor product across looks is a bug, not a thin inventory.
     */
    supportsUniqueProducts: boolean;
    /** Slots that actually participate in the deck structure. */
    activeSlots: DiversitySlot[];
    /**
     * Number of active slots where the pool offers exactly ONE garment
     * family. Those slots CANNOT be varied, so any two looks are bound to
     * share them — sharing such a slot is a fact of the inventory, not a
     * styling failure, and must not count against the diversity rule.
     *
     * A black-tie pool with one formal shirt and one tuxedo trouser has
     * forcedSharedFamilies = 2; the three dinner jackets must be allowed to
     * share them.
     */
    forcedSharedFamilies: number;
    /**
     * Upper bound on how many mutually different archetypes the pool can
     * produce (the smallest per-slot token count). 3 means the pool can in
     * principle support three different directions.
     */
    maxDistinctArchetypes: number;
    /**
     * Upper bound on the number of DISTINCT OUTFIT CONCEPTS the pool can
     * produce: the product of the per-slot archetype-token counts.
     *
     * A pool with 1 blazer / 1 shirt / 1 trouser / 1 shoe has a capacity of
     * 1 — one formula exists, repetition is unavoidable. A pool with
     * 8 blazers / 2 coats / 3 overshirts / 5 shirts / 4 polos / 5 knits /
     * 6 tailored / 4 chinos / 3 jeans / 5 loafers / 3 derbies / 4 sneakers
     * has a capacity of 81 — three different directions are easy, and
     * returning the same formula three times is a bug, not a thin inventory.
     */
    conceptCapacity: number;
    /**
     * True when the pool can, in principle, support `target` DIFFERENT
     * styling concepts. Diagnostic only: same formula is allowed, so
     * this never forbids a unique-product deck.
     */
    supportsTargetConcepts: boolean;
    /**
     * True when EVERY active slot offers exactly one archetype token: the
     * formula is fixed, so repeating it is unavoidable.
     */
    homogeneous: boolean;
    /** True when the pool contains more than one possible formula. */
    repetitionAvoidable: boolean;
}

export interface InventoryOptions {
    /** Slots that are pinned by the anchor (excluded from the assessment). */
    ignoreSlots?: readonly DiversitySlot[];
    /** Slots that participate in the deck structure. */
    activeSlots?: readonly DiversitySlot[];
    /** Deck size the pool must support. Default 3. */
    target?: number;
}

/**
 * Detect whether the valid inventory is genuinely thin.
 *
 * A pool with one blazer, one shirt, one trouser and one shoe cannot produce
 * three different concepts — repetition is unavoidable and the deck must be
 * allowed to repeat. A pool with 8 blazers, 2 coats, 3 overshirts, 5 shirts,
 * 4 polos, 5 knits, 6 tailored trousers, 4 chinos, 3 jeans, 5 loafers,
 * 3 derbies and 4 sneakers can — repeating one formula there is a bug.
 */
export function assessInventoryDiversity(
    items: any[],
    opts: InventoryOptions = {},
): InventoryDiversity {
    const ignore = new Set(opts.ignoreSlots ?? []);
    const families = new Map<DiversitySlot, Set<string>>();
    const tokens = new Map<DiversitySlot, Set<string>>();
    const products = new Map<DiversitySlot, any[]>();
    const seenSlots = new Set<DiversitySlot>();
    for (const item of Array.isArray(items) ? items : []) {
        if (!item) continue;
        const slot = diversitySlotOf(item);
        if (!slot || ignore.has(slot)) continue;
        seenSlots.add(slot);
        if (!products.has(slot)) products.set(slot, []);
        products.get(slot)!.push(item);
        const family = garmentVisualFamily(item);
        if (family) {
            if (!families.has(slot)) families.set(slot, new Set<string>());
            families.get(slot)!.add(family);
        }
        const token = garmentArchetypeToken(item);
        if (token && token !== 'none') {
            if (!tokens.has(slot)) tokens.set(slot, new Set<string>());
            tokens.get(slot)!.add(token);
        }
    }
    const activeSlots = (opts.activeSlots
        ? opts.activeSlots.filter((slot) => !ignore.has(slot))
        : DIVERSITY_SLOTS.filter((slot) => seenSlots.has(slot) && !ignore.has(slot))
    ) as DiversitySlot[];

    const familiesPerSlot: Partial<Record<DiversitySlot, number>> = {};
    const tokensPerSlot: Partial<Record<DiversitySlot, number>> = {};
    const uniqueProductsPerSlot: Partial<Record<DiversitySlot, number>> = {};
    for (const slot of activeSlots) {
        familiesPerSlot[slot] = families.get(slot)?.size ?? 0;
        tokensPerSlot[slot] = tokens.get(slot)?.size ?? 0;
        uniqueProductsPerSlot[slot] = countDistinctProducts(products.get(slot) ?? []);
    }
    // Capacity is measured in FORMULAS, so it counts the slots that define a
    // formula. A pool with one blazer, one shirt, one trouser and five pairs
    // of shoes can only ever produce ONE look — varying the shoes is not a
    // new styling direction.
    const formulaSlots = activeSlots.filter((slot) => FORMULA_SLOTS.includes(slot));
    const tokenCounts = formulaSlots.map((slot) => tokensPerSlot[slot] ?? 0);
    const maxDistinctArchetypes = tokenCounts.length ? Math.min(...tokenCounts) : 0;
    // Cap the product so a huge catalogue cannot overflow into Infinity.
    const conceptCapacity = tokenCounts.length
        ? tokenCounts.reduce((product, count) => product * Math.max(count, 1), 1)
        : 0;
    const target = opts.target ?? 3;
    const supportsTargetConcepts = conceptCapacity >= target;
    const productCounts = activeSlots.map((slot) => uniqueProductsPerSlot[slot] ?? 0);
    const maxDistinctProducts = productCounts.length ? Math.min(...productCounts) : 0;
    const supportsUniqueProducts = activeSlots.length > 0 && maxDistinctProducts >= target;
    const homogeneous = formulaSlots.length > 0 && tokenCounts.every((count) => count <= 1);
    const forcedSharedFamilies = activeSlots.filter(
        (slot) => (familiesPerSlot[slot] ?? 0) === 1,
    ).length;

    return {
        familiesPerSlot,
        tokensPerSlot,
        uniqueProductsPerSlot,
        maxDistinctProducts,
        supportsUniqueProducts,
        activeSlots,
        forcedSharedFamilies,
        maxDistinctArchetypes,
        conceptCapacity: Math.min(conceptCapacity, 100000),
        supportsTargetConcepts,
        homogeneous,
        repetitionAvoidable: !homogeneous,
    };
}

// ── Deck selection ──────────────────────────────────────────────────────

export interface DiverseDeckCandidate<O> {
    outfit: O;
    fingerprint: OutfitFingerprint;
    quality: number;
    source: 'edge' | 'fill';
    index: number;
}

export interface AssembleDiverseDeckOptions<O> {
    /** Candidate looks in preference order (edge output first). */
    candidates: O[];
    /** Items of a look. */
    itemsOf: (outfit: O) => any[];
    /** Quality of a look in 0..1 (match score / confidence). */
    qualityOf?: (outfit: O) => number;
    /** Identity keys of an item (id / name / image / family). */
    keysOf?: (item: any) => string[];
    /** Product keys of the locked anchor — never counted as repetition. */
    anchorKeys?: ReadonlySet<string>;
    /** Slot pinned by the anchor (excluded from visual comparison). */
    anchorSlot?: DiversitySlot | null;
    /** Deck size to reach. Default 3. */
    target?: number;
    /** Max non-anchor products two looks may share. Default 0 (hard). */
    maxSharedProducts?: number;
    /**
     * Soft visual-similarity reference. Kept for API compatibility and
     * diagnostics; visual similarity no longer rejects a pair. Same
     * archetype is allowed when the products are unique.
     */
    maxVisualSimilarity?: number;
    /** Soft shared-family reference (diagnostics only). */
    maxSharedVisualFamilies?: number;
    /** Builds one replacement look from the remaining pool. */
    buildFill?: (usedKeys: Set<string>, attempt: number) => O | null;
    /** Fill attempt budget. Default 24. */
    maxFillAttempts?: number;
    /** Product keys already used in the deck (mutated as looks are accepted). */
    usedKeys: Set<string>;
    /** Whole valid pool — used for the thin-inventory assessment. */
    poolItems?: any[];
    /** Debug sink (development builds only). */
    debug?: (message: string, data?: unknown) => void;
}

export interface DiverseDeckSimilarity {
    a: number;
    b: number;
    score: number;
}

export interface AssembleDiverseDeckResult<O> {
    deck: O[];
    /** True when the strict visual rule could not be honoured. */
    relaxed: boolean;
    /**
     * Why the deck had to relax (null in strict mode):
     *   'thin_inventory' — the pool genuinely cannot produce N concepts.
     *   'diversity_unavailable_despite_varied_inventory' — the pool COULD
     *     support N concepts but no qualifying combination was found; the
     *     least repetitive trio the inventory allows is returned instead.
     *   'no_candidates' — there were not enough valid looks at all.
     */
    relaxReason: string | null;
    /** Archetype of every accepted look, in deck order. */
    archetypes: string[];
    /** Pairwise similarity of the accepted looks. */
    similarities: DiverseDeckSimilarity[];
    /** Inventory assessment behind the decision. */
    inventory: InventoryDiversity;
    /** How many replacement looks were generated. */
    fillsUsed: number;
    /** Product-reuse diagnostics for the accepted deck. */
    productDiversity: DeckProductDiversity;
}

function identityKeySets(item: any, keysOf?: (item: any) => string[]): string[] {
    if (keysOf) return keysOf(item).filter(Boolean);
    return identityKeysOf(item);
}

function sharedByKeys(
    left: string[][],
    right: string[][],
    anchorKeys?: ReadonlySet<string>,
): number {
    const rightKeys = new Set<string>();
    for (const keys of right) for (const key of keys) rightKeys.add(key);
    let shared = 0;
    for (const keys of left) {
        if (anchorKeys && keys.some((key) => anchorKeys.has(key))) continue;
        if (keys.some((key) => rightKeys.has(key))) shared += 1;
    }
    return shared;
}

/**
 * assembleDiverseDeck — the deck-level product-uniqueness gate.
 *
 * Priority order (hardest first):
 *   1. the caller's style / occasion validation (looks are already filtered)
 *   2. no non-anchor product may repeat across looks
 *   3. outfit quality
 *   4. soft styling variety (archetype / similarity scoring only)
 *
 * Same archetype is NOT a rejection. Three "blazer + shirt + trousers +
 * loafers" looks are valid when every actual product is different. Garment
 * family is NOT product identity either: two different loafers are two
 * products. Product identity is id / normalized name / normalized image.
 *
 * The only relaxation is a genuinely thin inventory. The inventory
 * assessment counts distinct PRODUCTS per slot; when it cannot supply
 * `target` unique products, repetition is unavoidable and the deck is
 * marked `insufficient_unique_products`. When enough products exist, the
 * uniqueness rule never relaxes.
 *
 * The deck is chosen as a COMBINATION, not accepted look-by-look: the best
 * single look must not occupy every slot of the deck.
 */
export function assembleDiverseDeck<O>(opts: AssembleDiverseDeckOptions<O>): AssembleDiverseDeckResult<O> {
    const target = opts.target ?? 3;
    const maxSharedProducts = Math.max(0, Math.floor(opts.maxSharedProducts ?? 0));
    const maxFillAttempts = opts.maxFillAttempts ?? 48;
    const ignoreSlots: DiversitySlot[] = opts.anchorSlot ? [opts.anchorSlot] : [];
    const debug = opts.debug;

    const pool: DiverseDeckCandidate<O>[] = [];
    const pushCandidate = (outfit: O, source: 'edge' | 'fill'): DiverseDeckCandidate<O> => {
        const items = opts.itemsOf(outfit) || [];
        const quality = opts.qualityOf ? opts.qualityOf(outfit) : 0.9;
        const candidate: DiverseDeckCandidate<O> = {
            outfit,
            fingerprint: buildOutfitFingerprint(items),
            quality: Number.isFinite(quality) ? quality : 0.9,
            source,
            index: pool.length,
        };
        pool.push(candidate);
        return candidate;
    };

    for (const candidate of opts.candidates || []) pushCandidate(candidate, 'edge');

    const keySetsFor = (candidate: DiverseDeckCandidate<O>): string[][] =>
        (opts.itemsOf(candidate.outfit) || []).map((item) => identityKeySets(item, opts.keysOf));

    // Pairwise caches (recomputed lazily as the pool grows).
    const similarityCache = new Map<string, number>();
    const similarityOf = (a: DiverseDeckCandidate<O>, b: DiverseDeckCandidate<O>): number => {
        const key = a.index < b.index ? `${a.index}:${b.index}` : `${b.index}:${a.index}`;
        const cached = similarityCache.get(key);
        if (cached !== undefined) return cached;
        const score = visualSimilarity(a.fingerprint, b.fingerprint, { ignoreSlots });
        similarityCache.set(key, score);
        return score;
    };
    const sharedCache = new Map<string, number>();
    const sharedOf = (a: DiverseDeckCandidate<O>, b: DiverseDeckCandidate<O>): number => {
        const key = a.index < b.index ? `${a.index}:${b.index}` : `${b.index}:${a.index}`;
        const cached = sharedCache.get(key);
        if (cached !== undefined) return cached;
        const shared = sharedByKeys(keySetsFor(a), keySetsFor(b), opts.anchorKeys);
        sharedCache.set(key, shared);
        return shared;
    };

    /**
     * Soft deck score. Product uniqueness is enforced by hard filtering, so
     * this score only decides WHICH valid combination to show. Quality stays
     * dominant; archetype/similarity are small tie-breakers.
     */
    const deckScore = (indices: number[]): number => {
        const members = indices.map((index) => pool[index]);
        const quality = members.reduce((sum, member) => sum + member.quality, 0);
        const archetypes = new Set(members.map((member) => member.fingerprint.archetype));
        const bonus = DIVERSITY_BONUS_PER_ARCHETYPE * 0.35 * (archetypes.size - 1);
        let similarityPenalty = 0;
        let sharedPenalty = 0;
        let pairs = 0;
        for (let i = 0; i < members.length; i++) {
            for (let j = i + 1; j < members.length; j++) {
                similarityPenalty += similarityOf(members[i], members[j]);
                // Only thin inventories can reach a non-zero value here.
                sharedPenalty += sharedOf(members[i], members[j]);
                pairs += 1;
            }
        }
        const meanSimilarity = pairs > 0 ? similarityPenalty / pairs : 0;
        const meanShared = pairs > 0 ? sharedPenalty / pairs : 0;
        return quality
            + bonus
            - DIVERSITY_SIMILARITY_PENALTY * 0.5 * meanSimilarity
            - 1.5 * meanShared;
    };

    // Rejection diagnostics (development builds only). Capped so a big pool
    // cannot flood the console.
    const loggedRejections = new Set<string>();
    const logRejection = (a: DiverseDeckCandidate<O>, b: DiverseDeckCandidate<O>, shared: number) => {
        if (!debug) return;
        if (loggedRejections.size >= 12) return;
        const key = `${Math.min(a.index, b.index)}:${Math.max(a.index, b.index)}:shared_product`;
        if (loggedRejections.has(key)) return;
        loggedRejections.add(key);
        debug('decision: REJECT_SHARED_PRODUCT', {
            pair: [a.index + 1, b.index + 1],
            sharedProducts: shared,
            archetypes: [a.fingerprint.archetype, b.fingerprint.archetype],
        });
    };

    const combinationAllowed = (indices: number[], allowedShared: number): boolean => {
        for (let i = 0; i < indices.length; i++) {
            for (let j = i + 1; j < indices.length; j++) {
                const a = pool[indices[i]];
                const b = pool[indices[j]];
                const shared = sharedOf(a, b);
                if (shared > allowedShared) {
                    logRejection(a, b, shared);
                    return false;
                }
            }
        }
        return true;
    };

    const MAX_COMBINATIONS = 250000;

    const selectCombination = (size: number, allowedShared: number): number[] | null => {
        const n = pool.length;
        if (n < size || size < 1) return null;
        let best: number[] | null = null;
        let bestScore = Number.NEGATIVE_INFINITY;
        const consider = (indices: number[]) => {
            if (!combinationAllowed(indices, allowedShared)) return;
            const score = deckScore(indices);
            if (score > bestScore) {
                bestScore = score;
                best = indices;
            }
        };
        // Number of combinations C(n, size) — exhaustive search is only
        // affordable for small pools.
        let combinations = 1;
        for (let i = 0; i < size; i++) {
            combinations = combinations * (n - i) / (i + 1);
            if (combinations > MAX_COMBINATIONS) break;
        }
        if (combinations <= MAX_COMBINATIONS) {
            const current: number[] = [];
            const walk = (start: number) => {
                if (current.length === size) {
                    consider([...current]);
                    return;
                }
                for (let i = start; i < n; i++) {
                    current.push(i);
                    walk(i + 1);
                    current.pop();
                }
            };
            walk(0);
            return best;
        }
        // Greedy fallback for very large pools: start from the strongest
        // look, then add the most compatible companions.
        const seed = pool
            .map((member, index) => ({ index, quality: member.quality }))
            .sort((a, b) => b.quality - a.quality)[0];
        if (!seed) return null;
        const chosen = [seed.index];
        while (chosen.length < size) {
            let bestAdd = -1;
            let bestAddScore = Number.NEGATIVE_INFINITY;
            for (let i = 0; i < n; i++) {
                if (chosen.includes(i)) continue;
                const trial = [...chosen, i];
                if (!combinationAllowed(trial, allowedShared)) continue;
                const score = deckScore(trial);
                if (score > bestAddScore) {
                    bestAddScore = score;
                    bestAdd = i;
                }
            }
            if (bestAdd < 0) return null;
            chosen.push(bestAdd);
        }
        consider(chosen);
        return best;
    };

    const selectBestUpTo = (maxSize: number, allowedShared: number): number[] | null => {
        for (let size = Math.min(maxSize, pool.length); size >= 1; size--) {
            const selection = selectCombination(size, allowedShared);
            if (selection) return selection;
        }
        return null;
    };

    const inventoryOf = (): InventoryDiversity => assessInventoryDiversity(
        [
            ...pool.flatMap((member) => opts.itemsOf(member.outfit) || []),
            ...(opts.poolItems || []),
        ],
        { ignoreSlots, target },
    );

    const seedUsedKeys = (indices: number[]) => {
        for (const index of indices) {
            const member = pool[index];
            for (const item of opts.itemsOf(member.outfit) || []) {
                for (const key of identityKeySets(item, opts.keysOf)) {
                    if (opts.anchorKeys && opts.anchorKeys.has(key)) continue;
                    opts.usedKeys.add(key);
                }
                // Soft steering only: replacement looks prefer a different
                // garment type, but family never blocks a unique product.
                const slot = diversitySlotOf(item);
                if (slot && ignoreSlots.includes(slot)) continue;
                const family = garmentVisualFamily(item);
                if (family) opts.usedKeys.add(`__vf:${family}`);
            }
        }
    };

    // The inventory assessment decides whether uniqueness is physically
    // possible. It counts distinct products, not archetype tokens.
    const inventory = inventoryOf();
    const uniquenessSupported = inventory.supportsUniqueProducts;
    const sharedLimit = uniquenessSupported ? maxSharedProducts : Number.POSITIVE_INFINITY;

    let selection = selectCombination(target, sharedLimit);
    let fillsUsed = 0;

    if (opts.buildFill) {
        if (selection) seedUsedKeys(selection);
        let bestScore = selection ? deckScore(selection) : Number.NEGATIVE_INFINITY;
        let staleAttempts = 0;
        const patience = 5;
        while (fillsUsed < maxFillAttempts) {
            if (selection && staleAttempts >= patience) break;
            const attempt = fillsUsed;
            fillsUsed += 1;
            const fill = opts.buildFill(opts.usedKeys, attempt);
            // A rejected replacement (style judge / budget) must NOT end the
            // search — the next attempt rotates to different slot picks.
            if (!fill) {
                if (!selection) staleAttempts = 0;
                continue;
            }
            const candidate = pushCandidate(fill, 'fill');
            if (debug) {
                debug(`replacement selected: ${candidate.fingerprint.archetype}`, {
                    attempt,
                    archetype: candidate.fingerprint.archetype,
                    products: (opts.itemsOf(fill) || []).map((item) => canonicalProductKey(item)),
                });
            }
            const next = selectCombination(target, sharedLimit);
            if (!next) {
                // Still searching for the first valid unique combination.
                staleAttempts = 0;
                continue;
            }
            const score = deckScore(next);
            if (!selection || score > bestScore) {
                selection = next;
                bestScore = score;
                staleAttempts = 0;
            } else {
                staleAttempts += 1;
            }
        }
    }

    let relaxed = false;
    let relaxReason: string | null = null;

    if (!uniquenessSupported) {
        // Genuinely thin inventory: uniqueness is impossible in at least one
        // slot. Keep the least repetitive combination and report honestly.
        if (!selection) selection = selectBestUpTo(target, Number.POSITIVE_INFINITY);
        relaxed = Boolean(selection);
        relaxReason = selection ? 'insufficient_unique_products' : 'no_candidates';
    } else if (!selection) {
        // The inventory has enough unique products, but the candidate set /
        // fill builder could not produce a full trio. Never relax the
        // uniqueness rule; show the largest valid unique subset instead.
        selection = selectBestUpTo(target, maxSharedProducts);
        relaxReason = selection ? 'insufficient_unique_candidates' : 'no_candidates';
    }

    if (debug) {
        for (const member of pool) {
            debug(`candidate ${member.index + 1} archetype: ${member.fingerprint.archetype}`, {
                source: member.source,
                quality: member.quality,
                products: (opts.itemsOf(member.outfit) || []).map((item) => canonicalProductKey(item)),
            });
        }
        debug(`inventory: uniqueProductsPerSlot=${JSON.stringify(inventory.uniqueProductsPerSlot)} maxDistinctProducts=${inventory.maxDistinctProducts} supportsUniqueProducts=${inventory.supportsUniqueProducts}`, {
            familiesPerSlot: inventory.familiesPerSlot,
            tokensPerSlot: inventory.tokensPerSlot,
        });
    }

    if (!selection) {
        return {
            deck: [],
            relaxed: true,
            relaxReason: relaxReason ?? 'no_candidates',
            archetypes: [],
            similarities: [],
            inventory,
            fillsUsed,
            productDiversity: analyzeDeckProductDiversity([], { anchorKeys: opts.anchorKeys }),
        };
    }

    const chosen = selection
        .map((index) => pool[index])
        .sort((a, b) => (b.quality - a.quality) || (a.index - b.index));

    const similarities: DiverseDeckSimilarity[] = [];
    for (let i = 0; i < chosen.length; i++) {
        for (let j = i + 1; j < chosen.length; j++) {
            similarities.push({ a: i, b: j, score: similarityOf(chosen[i], chosen[j]) });
        }
    }

    const productDiversity = analyzeDeckProductDiversity(
        chosen.map((member) => opts.itemsOf(member.outfit) || []),
        { anchorKeys: opts.anchorKeys },
    );

    if (debug) {
        for (const entry of similarities) {
            debug(`similarity ${entry.a + 1}<->${entry.b + 1} = ${entry.score.toFixed(2)}`, {
                archetypes: [chosen[entry.a].fingerprint.archetype, chosen[entry.b].fingerprint.archetype],
            });
        }
        debug(`decision: ${relaxed ? `RELAXED (${relaxReason})` : 'UNIQUE_PRODUCTS'}`, {
            archetypes: chosen.map((member) => member.fingerprint.archetype),
            nonAnchorProductReuseCount: productDiversity.nonAnchorProductReuseCount,
            repeatedProductKeys: productDiversity.repeatedProductKeys,
            anchorRepetitionCount: productDiversity.anchorRepetitionCount,
            inventory: {
                homogeneous: inventory.homogeneous,
                maxDistinctProducts: inventory.maxDistinctProducts,
                supportsUniqueProducts: inventory.supportsUniqueProducts,
            },
        });
    }

    // Seed the shared used-keys set with the accepted looks so later
    // consumers (and the next tap) avoid repeating the same products.
    seedUsedKeys(selection);

    return {
        deck: chosen.map((member) => member.outfit),
        relaxed,
        relaxReason,
        archetypes: chosen.map((member) => member.fingerprint.archetype),
        similarities,
        inventory,
        fillsUsed,
        productDiversity,
    };
}

/** Compact human-readable fingerprint summary (debug output). */
export function describeFingerprint(fingerprint: OutfitFingerprint): string {
    return [
        fingerprint.archetype,
        `silhouette=${fingerprint.silhouette}`,
        `formality=${fingerprint.formality}`,
        `colors=${fingerprint.colors.join('/') || '-'}`,
        `materials=${fingerprint.materials.join('/') || '-'}`,
        `pattern=${fingerprint.patternProfile}`,
        `texture=${fingerprint.textureProfile}`,
    ].join(' ');
}
