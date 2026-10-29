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
 * Four layers, in order:
 *
 *   1. PRODUCT IDENTITY — id / normalized name / normalized image.
 *      Garment family is NOT identity. The name normalization is Unicode
 *      safe: Cyrillic / Greek / CJK names keep their identity instead of
 *      collapsing to an empty key.
 *   2. SLOT-AWARE CAPACITY — for every ACTIVE slot of the requested
 *      composition, count the distinct products the valid pool can
 *      supply. A slot with fewer distinct products than the deck target
 *      is `forced` (repetition is unavoidable there); every other active
 *      slot is `strict` (repetition is a defect).
 *   3. DECK SELECTION — pick the COMBINATION of looks with no non-anchor
 *      product reuse in a strict slot, quality first. Visual variety only
 *      breaks ties. Relaxation reasons:
 *        'insufficient_unique_products'    — a slot is genuinely thin.
 *        'insufficient_unique_candidates'  — the pool is rich but no full
 *            deck exists; the largest UNIQUE subset is returned instead
 *            of repeating a product.
 *        'no_candidates'                   — not enough valid looks.
 *   4. SOFT VISUAL VARIETY — fingerprint + similarity scoring used only
 *      as a tie-breaker among equally good unique-product decks.
 *      visualSimilarity / sameArchetype / sharedExactProducts DESCRIBE a
 *      deck; they never reject an otherwise valid look. A repeated outfit
 *      FORMULA is legal — only a repeated actual PRODUCT is not.
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
 * Reference score for "these two looks look the same to a human".
 *
 * Chosen from the score distribution produced by this module (see
 * __tests__/features/outfitVisualDiversity.test.ts for the live numbers):
 *   identical formula, different SKUs   -> 0.80 .. 0.93
 *   same formula, one slot swapped      -> 0.55 .. 0.70
 *   clearly different concepts           -> 0.10 .. 0.35
 *
 * DIAGNOSTIC ONLY. The score is reported (telemetry, report script) and
 * used as a tie-breaker between equally good unique-product decks. It never
 * rejects a look: a repeated outfit FORMULA is legal, and a hard score gate
 * would throw away correct, product-unique decks.
 */
export const MAX_VISUAL_SIMILARITY = 0.65;

/**
 * Reference count of garment families two looks may share (anchor slot
 * excluded). DIAGNOSTIC ONLY — reported, never enforced.
 */
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
 * Reference threshold for the shared-family description: a pair that shares
 * two garment families still reads as one concept when its similarity stays
 * below this value, e.g. a coat + knit + denim look next to a blazer + shirt
 * + tailored look.
 *
 * DIAGNOSTIC ONLY — the shared-family count describes a deck, it never
 * rejects a look.
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

/**
 * Separator between "meaningful" characters of a product name.
 *
 * Built with `new RegExp` (not a literal) so an engine without Unicode
 * property escapes degrades to a broad Latin/Greek/Cyrillic/CJK range instead
 * of throwing a parse-time SyntaxError at module load.
 */
const NON_ALPHANUMERIC: RegExp = (() => {
    try {
        return new RegExp('[^\\p{L}\\p{N}]+', 'gu');
    } catch {
        return /[^a-z0-9\u00c0-\u024f\u0370-\u03ff\u0400-\u052f\u4e00-\u9fff\uac00-\ud7af]+/gi;
    }
})();

/** NFKC normalization when the runtime supports it. */
function unicodeFold(value: string): string {
    try {
        return typeof (value as any).normalize === 'function' ? value.normalize('NFKC') : value;
    } catch {
        return value;
    }
}

/**
 * Normalized product-name key.
 *
 * Unicode-aware on purpose. The catalog contains Cyrillic, accented and CJK
 * names, and an ASCII-only normalizer (`[^a-z0-9]`) collapses every non-Latin
 * product to the SAME empty string — which would make hundreds of unrelated
 * products look like ONE product and wrongly force a "thin inventory"
 * verdict. NFKC + Unicode letter/number classes preserve the meaningful
 * identity of a multilingual name:
 *
 *   'Коричневые кожаные лоферы' → 'коричневые кожаные лоферы'
 */
export function normalizeNameKey(name: unknown): string {
    const raw = unicodeFold(text(name)).toLowerCase();
    return raw.replace(NON_ALPHANUMERIC, ' ').replace(/\s+/g, ' ').trim();
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
    // A dress replaces the top AND the bottom. The structural validator
    // normalizes it to the 'top' slot, so product identity and capacity must
    // use the same mapping or a dress deck would look "bottomless".
    dress: 'top',
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
 * DESCRIPTIVE, NOT A RULE. This was once a hard gate that rejected every
 * repeated formula. That was wrong: three blazer + shirt + trousers +
 * loafer looks are a CORRECT deck when every blazer, shirt, trouser and
 * shoe is a different actual product. The product-uniqueness layer (layer 3
 * of this module) is what decides a deck; this function only describes the
 * formula for telemetry and the report script.
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

/**
 * Number of identical products (id / name / photo) two looks share.
 * DIAGNOSTIC ONLY — the deck selector enforces product uniqueness itself.
 */
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
    /**
     * Every repeated non-anchor product with the slot it occupies. A repeat
     * in a `forced` slot is physically unavoidable; a repeat in a `strict`
     * slot is a defect.
     */
    repeatedProducts: Array<{
        key: string;
        slot: DiversitySlot | null;
        lookCount: number;
        lookIds: number[];
        /**
         * True when the repeat is legitimate: the slot physically cannot
         * supply enough distinct products for the deck (a thin slot), or the
         * repeated product IS the locked anchor.
         */
        legitimate: boolean;
    }>;
    /**
     * Repeats that are DEFECTS: a non-anchor product repeated in a slot that
     * had enough distinct products to avoid it. This is the number that must
     * stay at zero.
     */
    illegalProductReuseCount: number;
    /** Repeats that are unavoidable because the slot is genuinely thin. */
    forcedRepeatProducts: string[];
    /** Repeats of the locked anchor (always allowed). */
    anchorRepeatProducts: string[];
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
    opts: { anchorKeys?: ReadonlySet<string>; forcedSlots?: readonly DiversitySlot[] } = {},
): DeckProductDiversity {
    const occurrences: Array<{ keys: string[]; canonicalKey: string; look: number; anchor: boolean; slot: DiversitySlot | null }> = [];
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
                slot: diversitySlotOf(item),
            });
            canonicalForLook.push(identity.canonicalKey);
        }
        lookProductKeys.push(canonicalForLook);
    });

    const repeatedProductKeys = new Set<string>();
    const repeatedProducts: DeckProductDiversity['repeatedProducts'] = [];
    const forcedSlots = new Set<DiversitySlot>(opts.forcedSlots || []);
    const forcedRepeatProducts = new Set<string>();
    const anchorRepeatProducts = new Set<string>();
    let nonAnchorProductReuseCount = 0;
    let illegalProductReuseCount = 0;
    for (const group of clusterProductOccurrences(occurrences)) {
        const looks = new Set(group.map((index) => occurrences[index].look));
        if (looks.size < 2) continue;
        const nonAnchor = group
            .map((index) => occurrences[index])
            .filter((occurrence) => !occurrence.anchor);
        // A cluster made only of the anchor is an allowed anchor repetition.
        if (nonAnchor.length === 0) {
            for (const occurrence of group.map((index) => occurrences[index])) {
                anchorRepeatProducts.add(occurrence.canonicalKey);
            }
            continue;
        }
        nonAnchorProductReuseCount += 1;
        repeatedProductKeys.add(nonAnchor[0].canonicalKey);
        const lookIds = [...new Set(nonAnchor.map((occurrence) => occurrence.look))].sort((a, b) => a - b);
        const slot = nonAnchor[0].slot;
        // The repeat is legitimate only when the slot is physically too small
        // to fill the deck. Anything else is a defect.
        const legitimate = slot != null && forcedSlots.has(slot);
        if (legitimate) forcedRepeatProducts.add(nonAnchor[0].canonicalKey);
        else illegalProductReuseCount += 1;
        repeatedProducts.push({
            key: nonAnchor[0].canonicalKey,
            slot,
            lookCount: lookIds.length,
            lookIds,
            legitimate,
        });
    }
    repeatedProducts.sort((a, b) => String(a.slot).localeCompare(String(b.slot)) || a.key.localeCompare(b.key));

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
        repeatedProducts,
        illegalProductReuseCount,
        forcedRepeatProducts: [...forcedRepeatProducts],
        anchorRepeatProducts: [...anchorRepeatProducts],
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

/** Per-slot uniqueness capacity for the requested deck size. */
export interface SlotCapacity {
    /** The slot this entry describes. */
    slot: DiversitySlot;
    /** Distinct actual products available for this slot. */
    uniqueProducts: number;
    /**
     * True when the slot cannot supply `target` distinct products, so
     * repetition in THIS slot is physically unavoidable. Only these slots
     * may repeat a product.
     */
    forced: boolean;
    /** True when the slot is pinned by the anchor (repetition allowed). */
    anchor: boolean;
}

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
     * Per-slot capacity detail. A slot is `forced` only when it genuinely
     * cannot supply `target` distinct products — the deck selector uses this
     * to allow repetition in ONE sparse slot without loosening the whole
     * deck.
     */
    slotCapacity: SlotCapacity[];
    /**
     * Active slots where product repetition is PHYSICALLY unavoidable
     * (fewer distinct products than the deck size). These — and only these —
     * may repeat a product.
     */
    forcedRepeatSlots: DiversitySlot[];
    /**
     * Active slots where product repetition is FORBIDDEN because valid
     * alternatives exist.
     */
    strictSlots: DiversitySlot[];
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
    /**
     * Slots that participate in the deck structure. Only these are counted:
     * a single outerwear item must never make a non-layered request look
     * thin. Defaults to the slots the pool actually contains.
     */
    activeSlots?: readonly DiversitySlot[];
    /** Deck size the pool must support. Default 3. */
    target?: number;
}

/**
 * Detect whether the valid inventory is genuinely thin, PER SLOT.
 *
 * The question is not "can the pool produce three different FORMULAS?" — a
 * repeated formula is legal. The question is "can every ACTIVE slot supply
 * `target` DISTINCT PRODUCTS?".
 *
 * A pool with one blazer, one shirt, one trouser and one shoe cannot: the
 * blazer, shirt and trouser slots are `forced`, so repetition there is
 * physically unavoidable and the deck is allowed to repeat them. A pool with
 * 8 blazers, 5 shirts, 6 tailored trousers, 5 loafers and 3 derbies can:
 * every slot is `strict`, so repeating a product there is a defect.
 *
 * The anchor slot is excluded (the anchor is pinned by the user), and only
 * the ACTIVE slots are counted — a single outerwear item must never make a
 * non-layered request look thin.
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

    // Per-slot uniqueness capacity. A slot may repeat a product ONLY when it
    // cannot supply `target` distinct products — that is the difference
    // between "one sparse slot" (shoes) and "the whole deck is thin".
    const slotCapacity: SlotCapacity[] = activeSlots.map((slot) => ({
        slot,
        uniqueProducts: uniqueProductsPerSlot[slot] ?? 0,
        forced: (uniqueProductsPerSlot[slot] ?? 0) < target,
        anchor: ignore.has(slot),
    }));
    const forcedRepeatSlots = slotCapacity.filter((entry) => entry.forced).map((entry) => entry.slot);
    const strictSlots = slotCapacity.filter((entry) => !entry.forced && !entry.anchor).map((entry) => entry.slot);

    return {
        familiesPerSlot,
        tokensPerSlot,
        uniqueProductsPerSlot,
        maxDistinctProducts,
        supportsUniqueProducts,
        activeSlots,
        slotCapacity,
        forcedRepeatSlots,
        strictSlots,
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

/**
 * Capacity verdict produced by the AUTHORITATIVE pool.
 *
 * The edge owns the valid pool (style / occasion / climate filtered, plus the
 * shop supplement), so it owns the "is this slot genuinely thin?" question.
 * The client only sees a truncated view of that pool and must not re-derive
 * the verdict from it — otherwise a thin slot the edge legitimately repeated
 * would be rejected on the client and the user would lose cards.
 *
 * When supplied, the verdict wins for every slot it names. Slots it does not
 * name fall back to the locally computed assessment.
 */
export interface DeckCapacity {
    /** Slots that participate in the deck composition (edge composition). */
    activeSlots?: readonly DiversitySlot[];
    /** Slots rich enough that repetition is a defect, not a necessity. */
    strictSlots?: readonly DiversitySlot[];
    /** Slots physically too small to fill the deck — repetition allowed. */
    forcedRepeatSlots?: readonly DiversitySlot[];
}

export interface AssembleDiverseDeckOptions<O> {
    /** Candidate looks in preference order (edge output first). */
    candidates: O[];
    /** Items of a look. */
    itemsOf: (outfit: O) => any[];
    /** Quality of a look in 0..1 (match score / confidence). */
    qualityOf?: (outfit: O) => number;
    /** Integer priority among already validated decks, e.g. unused physical products. */
    priorityOfDeck?: (outfits: O[]) => number;
    /** Product identity keys of an item (id / normalized name / image). */
    keysOf?: (item: any) => string[];
    /** Product keys of the locked anchor — never counted as repetition. */
    anchorKeys?: ReadonlySet<string>;
    /** Slot pinned by the anchor (excluded from capacity and comparison). */
    anchorSlot?: DiversitySlot | null;
    /** Deck size to reach. Default 3. */
    target?: number;
    /**
     * Slots that participate in the deck composition. Only these count for
     * capacity: a single outerwear item must never make a NON-LAYERED
     * request look thin. Defaults to the slots the candidate looks use.
     */
    activeSlots?: readonly DiversitySlot[];
    /**
     * Capacity verdict from the authoritative pool (the edge). When present
     * it decides which slots may repeat; see DeckCapacity.
     */
    capacity?: DeckCapacity;
    /**
     * Soft visual-similarity reference. Kept for API compatibility and
     * diagnostics; visual similarity never rejects a pair. Same archetype is
     * allowed when the products are unique.
     */
    maxVisualSimilarity?: number;
    /** Soft shared-family reference (diagnostics only). */
    maxSharedVisualFamilies?: number;
    /** Builds one replacement look from the remaining pool. */
    buildFill?: (usedKeys: Set<string>, attempt: number) => O | null;
    /** Fill attempt budget. Default 48. */
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
    /**
     * True when the deck repeats a product or is smaller than the target.
     * Repetition is only ever permitted in a slot whose inventory is
     * physically too small to fill the deck.
     */
    relaxed: boolean;
    /**
     * Why the deck had to relax (null in strict mode):
     *   'insufficient_unique_products' — at least one ACTIVE slot cannot
     *     supply `target` distinct products, so repetition there is forced.
     *   'insufficient_unique_candidates' — the inventory is rich, but the
     *     candidate set could not produce a full deck. The largest valid
     *     UNIQUE subset is returned instead of repeating a product.
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
    /**
     * True when the candidates carried a REAL quality signal (not one
     * constant placeholder value). When false, quality cannot discriminate
     * and the visual tie-breaker decides — a known limitation, not
     * calibrated personalization.
     */
    hasQualitySignal: boolean;
    /** Quality spread across the candidate pool (min / max). */
    qualityRange: { min: number; max: number };
}

function identityKeySets(item: any, keysOf?: (item: any) => string[]): string[] {
    if (keysOf) return keysOf(item).filter(Boolean);
    return identityKeysOf(item);
}

/** One item of a look, with its identity keys and the slot it occupies. */
interface ItemProductRef {
    keys: string[];
    slot: DiversitySlot | null;
}

/** How two looks share actual products. */
interface PairProductReuse {
    /** Shared products in a slot that HAS alternatives — a hard defect. */
    violations: Array<{ slot: DiversitySlot | null; keys: string[] }>;
    /** Shared products allowed because their slot cannot fill the deck. */
    forced: Array<{ slot: DiversitySlot | null; keys: string[] }>;
    /** Shared products allowed because they are the pinned anchor. */
    anchor: Array<{ keys: string[] }>;
}

/**
 * Quality granularity. Quality differences larger than this dominate the
 * soft visual tie-breaker; smaller differences are treated as a tie so the
 * variety signal can decide between otherwise equivalent looks.
 */
const QUALITY_BUCKET = 0.05;
/** Weight applied to the bucketed quality sum so it dominates the tie-breaker. */
const QUALITY_WEIGHT = 1000;

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
 * Relaxation is PER SLOT, never per deck. The inventory assessment counts
 * distinct PRODUCTS per ACTIVE slot:
 *   • a slot that cannot supply `target` distinct products is `forced` —
 *     repetition there is physically unavoidable (`insufficient_unique_products`);
 *   • every other active slot is `strict` — repetition there is a defect.
 * A rich inventory NEVER relaxes: when the candidates cannot form a full
 * deck, the largest UNIQUE subset is returned instead
 * (`insufficient_unique_candidates`).
 *
 * The deck is chosen as a COMBINATION, not accepted look-by-look: the best
 * single look must not occupy every slot of the deck.
 */
export function assembleDiverseDeck<O>(opts: AssembleDiverseDeckOptions<O>): AssembleDiverseDeckResult<O> {
    const target = Math.max(1, Math.floor(opts.target ?? 3));
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
            quality: Number.isFinite(quality) ? Math.max(0, Math.min(1, quality)) : 0.9,
            source,
            index: pool.length,
        };
        pool.push(candidate);
        return candidate;
    };

    for (const candidate of opts.candidates || []) pushCandidate(candidate, 'edge');

    const itemRefsFor = (candidate: DiverseDeckCandidate<O>): ItemProductRef[] =>
        (opts.itemsOf(candidate.outfit) || []).map((item) => ({
            keys: identityKeySets(item, opts.keysOf),
            slot: diversitySlotOf(item),
        }));

    /**
     * Slots that participate in the composition. An explicit `activeSlots`
     * wins (the caller knows the requested structure); otherwise the slots
     * the candidate looks and the valid pool actually use are used.
     */
    const explicitActiveSlots = opts.activeSlots && opts.activeSlots.length
        ? DIVERSITY_SLOTS.filter((slot) => opts.activeSlots!.includes(slot)).filter((slot) => !ignoreSlots.includes(slot))
        : null;
    // The authoritative capacity verdict (the edge owns the valid pool).
    // Its active slots win over the derived ones, and its per-slot verdict
    // wins over the locally computed one for every slot it names.
    const capacity = opts.capacity;
    const capacityActiveSlots = capacity?.activeSlots && capacity.activeSlots.length
        ? DIVERSITY_SLOTS.filter((slot) => capacity!.activeSlots!.includes(slot)).filter((slot) => !ignoreSlots.includes(slot))
        : null;
    const deriveActiveSlots = (): DiversitySlot[] => {
        if (explicitActiveSlots) return explicitActiveSlots;
        if (capacityActiveSlots) return capacityActiveSlots;
        const seen = new Set<DiversitySlot>();
        for (const candidate of pool) {
            for (const ref of itemRefsFor(candidate)) {
                if (ref.slot) seen.add(ref.slot);
            }
        }
        for (const item of opts.poolItems || []) {
            const slot = diversitySlotOf(item);
            if (slot) seen.add(slot);
        }
        return DIVERSITY_SLOTS.filter((slot) => seen.has(slot) && !ignoreSlots.includes(slot));
    };

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

    // The inventory decides WHICH SLOTS may repeat. It is recomputed as the
    // candidate pool grows (fills add products), so the reuse rule stays
    // slot-aware from the start and never relies on a stale snapshot.
    const computeInventory = (): InventoryDiversity => assessInventoryDiversity(
        [
            ...pool.flatMap((member) => opts.itemsOf(member.outfit) || []),
            ...(opts.poolItems || []),
        ],
        { ignoreSlots, activeSlots: deriveActiveSlots(), target },
    );
    const reuseCache = new Map<string, PairProductReuse>();
    /**
     * A slot may repeat only when it is genuinely thin. The authoritative
     * verdict wins for the slots it names; unnamed slots keep the locally
     * computed verdict.
     */
    const resolveForcedSlots = (assessment: InventoryDiversity): Set<DiversitySlot> => {
        const computed = new Set<DiversitySlot>(assessment.forcedRepeatSlots);
        if (!capacity) return computed;
        const forced = new Set<DiversitySlot>(capacity.forcedRepeatSlots || []);
        const strict = new Set<DiversitySlot>(capacity.strictSlots || []);
        for (const slot of [...computed, ...forced, ...strict]) {
            if (forced.has(slot)) computed.add(slot);
            else if (strict.has(slot)) computed.delete(slot);
        }
        return computed;
    };
    /**
     * The assessment as it is EFFECTIVELY applied. When the authoritative
     * verdict is present it replaces the local one, so the reported
     * inventory (telemetry, the response payload, the client's capacity
     * override) always matches the rule the selector actually enforced.
     */
    const applyCapacity = (assessment: InventoryDiversity): InventoryDiversity => {
        const forced = resolveForcedSlots(assessment);
        return {
            ...assessment,
            forcedRepeatSlots: assessment.activeSlots.filter((slot) => forced.has(slot)),
            strictSlots: assessment.activeSlots.filter((slot) => !forced.has(slot)),
        };
    };
    let inventory = applyCapacity(computeInventory());
    let strictSlots = new Set<DiversitySlot>(inventory.strictSlots);
    let forcedSlots = resolveForcedSlots(inventory);

    const refreshInventory = () => {
        inventory = applyCapacity(computeInventory());
        strictSlots = new Set<DiversitySlot>(inventory.strictSlots);
        forcedSlots = resolveForcedSlots(inventory);
        reuseCache.clear();
    };

    if (debug) {
        debug('ACTIVE_DIVERSITY_SLOTS', {
            activeSlots: inventory.activeSlots,
            target,
            uniqueProductsPerSlot: inventory.uniqueProductsPerSlot,
        });
        debug(inventory.supportsUniqueProducts ? 'UNIQUE_INVENTORY_SUPPORTED' : 'UNIQUE_INVENTORY_UNSUPPORTED', {
            strictSlots: inventory.strictSlots,
            forcedRepeatSlots: inventory.forcedRepeatSlots,
            maxDistinctProducts: inventory.maxDistinctProducts,
        });
    }

    /**
     * STRICT identity of an item for the duplicate-LOOK guard: the same
     * photo or the same row id. A shared generic name is deliberately NOT
     * enough here — placeholder rows are all called "item", and treating
     * those as one product would refuse perfectly distinct cards.
     */
    const strictIdentityOf = (item: any): string[] => {
        const identity = productIdentityOf(item);
        const keys: string[] = [];
        if (identity.imageKey) keys.push(identity.imageKey);
        if (identity.idKey) keys.push(identity.idKey);
        return keys;
    };
    /**
     * True when two looks carry the same items in every position (a perfect
     * matching exists and every matched item shares a photo or a row id).
     * The locked anchor is ignored: it is SUPPOSED to appear in every look.
     */
    const looksCarrySameProducts = (a: DiverseDeckCandidate<O>, b: DiverseDeckCandidate<O>): boolean => {
        const itemsOf = (candidate: DiverseDeckCandidate<O>) => (opts.itemsOf(candidate.outfit) || []).filter((item) => {
            if (!opts.anchorKeys) return true;
            return !identityKeySets(item, opts.keysOf).some((key) => opts.anchorKeys!.has(key));
        });
        const left = itemsOf(a);
        const right = itemsOf(b);
        if (left.length === 0 || left.length !== right.length) return false;
        const leftKeys = left.map(strictIdentityOf);
        const rightKeys = right.map(strictIdentityOf);
        const used = new Array(rightKeys.length).fill(false);
        const match = (index: number): boolean => {
            if (index === leftKeys.length) return true;
            for (let j = 0; j < rightKeys.length; j++) {
                if (used[j]) continue;
                const sharesStrictKey = leftKeys[index].length > 0
                    && rightKeys[j].length > 0
                    && leftKeys[index].some((key) => rightKeys[j].includes(key));
                if (!sharesStrictKey) continue;
                used[j] = true;
                if (match(index + 1)) return true;
                used[j] = false;
            }
            return false;
        };
        return match(0);
    };
    const reuseOf = (a: DiverseDeckCandidate<O>, b: DiverseDeckCandidate<O>): PairProductReuse => {
        const key = a.index < b.index ? `${a.index}:${b.index}` : `${b.index}:${a.index}`;
        const cached = reuseCache.get(key);
        if (cached) return cached;
        const result: PairProductReuse = { violations: [], forced: [], anchor: [] };
        const refsA = itemRefsFor(a);
        const refsB = itemRefsFor(b);
        for (const refA of refsA) {
            for (const refB of refsB) {
                const shared = refA.keys.filter((identityKey) => refB.keys.includes(identityKey));
                if (!shared.length) continue;
                const anchorKeys = opts.anchorKeys;
                if (anchorKeys && shared.some((identityKey) => anchorKeys.has(identityKey))) {
                    result.anchor.push({ keys: shared });
                    continue;
                }
                const slot = refA.slot ?? refB.slot;
                // A slot with alternatives forbids the repeat. A slot that
                // CANNOT fill the deck forces it. An unknown slot is treated
                // as strict: we cannot prove alternatives exist, so we do
                // not grant permission to repeat.
                if (slot && forcedSlots.has(slot)) {
                    result.forced.push({ slot, keys: shared });
                } else {
                    result.violations.push({ slot, keys: shared });
                }
            }
        }
        // A look whose product MULTISET equals another look's is the SAME
        // card. Repeating it adds no information to the deck, so the copy is
        // rejected even when every one of its slots is genuinely thin: the
        // honest answer is a SMALLER deck (or a look that differs
        // somewhere), never the same card twice.
        //
        // The comparison matches ITEMS, not slot names: a non-layered
        // request may fill its top slot with an outerwear piece, and the
        // same item must still be recognised wherever it sits.
        if (looksCarrySameProducts(a, b)) {
            result.violations.push({ slot: null, keys: ['duplicate_look'] });
        }
        reuseCache.set(key, result);
        return result;
    };

    /**
     * Soft deck score. Product uniqueness is enforced by hard filtering, so
     * this score only decides WHICH valid combination to show.
     *
     * Quality DOMINATES: the quality sum is bucketed and scaled so any real
     * quality difference outweighs the entire variety term. Variety only
     * breaks ties between combinations of equivalent quality.
     */
    const deckScore = (indices: number[]): number => {
        const members = indices.map((index) => pool[index]);
        const qualitySum = members.reduce((sum, member) => sum + member.quality, 0);
        const qualityBucket = Math.round(qualitySum / QUALITY_BUCKET) * QUALITY_BUCKET;
        const archetypes = new Set(members.map((member) => member.fingerprint.archetype));
        const bonus = DIVERSITY_BONUS_PER_ARCHETYPE * 0.35 * (archetypes.size - 1);
        let similarityPenalty = 0;
        let pairs = 0;
        for (let i = 0; i < members.length; i++) {
            for (let j = i + 1; j < members.length; j++) {
                similarityPenalty += similarityOf(members[i], members[j]);
                pairs += 1;
            }
        }
        const meanSimilarity = pairs > 0 ? similarityPenalty / pairs : 0;
        const variety = bonus - DIVERSITY_SIMILARITY_PENALTY * 0.5 * meanSimilarity;
        const priority = opts.priorityOfDeck?.(members.map(member => member.outfit)) ?? 0;
        return (Number.isFinite(priority) ? Math.max(0, Math.floor(priority)) : 0) * (target * QUALITY_WEIGHT + 10)
            + qualityBucket * QUALITY_WEIGHT + variety;
    };

    // Rejection diagnostics (development builds only). Capped so a big pool
    // cannot flood the console.
    const loggedRejections = new Set<string>();
    const logRejection = (a: DiverseDeckCandidate<O>, b: DiverseDeckCandidate<O>, reuse: PairProductReuse) => {
        if (!debug) return;
        if (loggedRejections.size >= 12) return;
        const key = `${Math.min(a.index, b.index)}:${Math.max(a.index, b.index)}:shared_product`;
        if (loggedRejections.has(key)) return;
        loggedRejections.add(key);
        debug('PRODUCT_REUSE_REJECT', {
            pair: [a.index + 1, b.index + 1],
            slots: reuse.violations.map((entry) => entry.slot),
            products: reuse.violations.map((entry) => entry.keys[0]),
            archetypes: [a.fingerprint.archetype, b.fingerprint.archetype],
        });
    };
    const logAllowedReuse = (a: DiverseDeckCandidate<O>, b: DiverseDeckCandidate<O>, reuse: PairProductReuse) => {
        if (!debug) return;
        for (const entry of reuse.forced) {
            debug('FORCED_SLOT_REUSE', {
                slot: entry.slot,
                product: entry.keys[0],
                pair: [a.index + 1, b.index + 1],
                uniqueProductsInSlot: inventory.uniqueProductsPerSlot[entry.slot ?? 'top'] ?? 0,
            });
        }
        for (const entry of reuse.anchor) {
            debug('ANCHOR_REUSE_ALLOWED', {
                product: entry.keys[0],
                pair: [a.index + 1, b.index + 1],
            });
        }
    };

    const combinationAllowed = (indices: number[]): boolean => {
        for (let i = 0; i < indices.length; i++) {
            for (let j = i + 1; j < indices.length; j++) {
                const a = pool[indices[i]];
                const b = pool[indices[j]];
                const reuse = reuseOf(a, b);
                if (reuse.violations.length > 0) {
                    logRejection(a, b, reuse);
                    return false;
                }
                logAllowedReuse(a, b, reuse);
            }
        }
        return true;
    };

    const MAX_COMBINATIONS = 250000;

    const selectCombination = (size: number): number[] | null => {
        const n = pool.length;
        if (n < size || size < 1) return null;
        let best: number[] | null = null;
        let bestScore = Number.NEGATIVE_INFINITY;
        const consider = (indices: number[]) => {
            if (!combinationAllowed(indices)) return;
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
        // Greedy fallback for very large pools. Every candidate is tried as
        // the seed so one strong-but-conflicting look cannot starve the
        // deck (the greedy trap).
        let bestSeed: number[] | null = null;
        let bestSeedScore = Number.NEGATIVE_INFINITY;
        for (let seed = 0; seed < n; seed++) {
            const chosen = [seed];
            let failed = false;
            while (chosen.length < size) {
                let bestAdd = -1;
                let bestAddScore = Number.NEGATIVE_INFINITY;
                for (let i = 0; i < n; i++) {
                    if (chosen.includes(i)) continue;
                    const trial = [...chosen, i];
                    if (!combinationAllowed(trial)) continue;
                    const score = deckScore(trial);
                    if (score > bestAddScore) {
                        bestAddScore = score;
                        bestAdd = i;
                    }
                }
                if (bestAdd < 0) { failed = true; break; }
                chosen.push(bestAdd);
            }
            if (failed) continue;
            const score = deckScore(chosen);
            if (score > bestSeedScore) {
                bestSeedScore = score;
                bestSeed = chosen;
            }
        }
        if (bestSeed) consider(bestSeed);
        return best;
    };

    /**
     * Largest valid combination up to `maxSize` under the SAME slot-aware
     * uniqueness rule. Used when the target size cannot be reached: the
     * answer is a SMALLER valid deck, never a deck with repeated products.
     */
    const selectBestUpTo = (maxSize: number): number[] | null => {
        for (let size = Math.min(maxSize, pool.length); size >= 1; size--) {
            const selection = selectCombination(size);
            if (selection) return selection;
        }
        return null;
    };

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

    let selection = selectCombination(target);
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
            refreshInventory();
            if (debug) {
                debug(`replacement selected: ${candidate.fingerprint.archetype}`, {
                    attempt,
                    archetype: candidate.fingerprint.archetype,
                    products: (opts.itemsOf(fill) || []).map((item) => canonicalProductKey(item)),
                });
            }
            const next = selectCombination(target);
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

    if (!selection) {
        // The inventory is rich but the candidate set could not produce a
        // full deck. NEVER relax product uniqueness to reach the target —
        // return the largest valid UNIQUE subset instead.
        selection = selectBestUpTo(target);
        if (selection && selection.length < target) {
            relaxed = true;
            // Name the REAL binding constraint: when an active slot cannot
            // supply `target` distinct products the deck was capped by the
            // inventory itself; otherwise the candidates were the limit.
            relaxReason = forcedSlots.size > 0
                ? 'insufficient_unique_products'
                : 'insufficient_unique_candidates';
        }
    }

    if (debug) {
        for (const member of pool) {
            debug(`candidate ${member.index + 1} archetype: ${member.fingerprint.archetype}`, {
                source: member.source,
                quality: member.quality,
                products: (opts.itemsOf(member.outfit) || []).map((item) => canonicalProductKey(item)),
            });
        }
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
            hasQualitySignal: false,
            qualityRange: { min: 0, max: 0 },
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
        { anchorKeys: opts.anchorKeys, forcedSlots: inventory.forcedRepeatSlots },
    );

    // A repeat is only acceptable when it happens in a slot that cannot
    // fill the deck. Any repeat in a slot with alternatives is a defect and
    // must never be reported as a relaxed-but-fine deck.
    const forbiddenRepeats = productDiversity.repeatedProducts.filter(
        (entry) => !(entry.slot && forcedSlots.has(entry.slot)),
    );
    if (forbiddenRepeats.length > 0) {
        relaxed = true;
        relaxReason = 'repeated_products_in_strict_slot';
    } else if (productDiversity.repeatedProducts.length > 0) {
        relaxed = true;
        relaxReason = 'insufficient_unique_products';
    }

    const qualities = pool.map((member) => member.quality);
    const qualityMin = qualities.length ? Math.min(...qualities) : 0;
    const qualityMax = qualities.length ? Math.max(...qualities) : 0;
    const hasQualitySignal = qualities.length > 0 && qualityMax - qualityMin > 1e-9;

    if (debug) {
        for (const entry of similarities) {
            debug(`similarity ${entry.a + 1}<->${entry.b + 1} = ${entry.score.toFixed(2)}`, {
                archetypes: [chosen[entry.a].fingerprint.archetype, chosen[entry.b].fingerprint.archetype],
            });
        }
        debug('DIVERSE_DECK_SELECTED', {
            deckSize: chosen.length,
            target,
            archetypes: chosen.map((member) => member.fingerprint.archetype),
            nonAnchorProductReuseCount: productDiversity.nonAnchorProductReuseCount,
            repeatedProductKeys: productDiversity.repeatedProductKeys,
            repeatedSlots: productDiversity.repeatedProducts.map((entry) => entry.slot),
            anchorRepetitionCount: productDiversity.anchorRepetitionCount,
            activeSlots: inventory.activeSlots,
            forcedRepeatSlots: inventory.forcedRepeatSlots,
            relaxed,
            relaxReason,
            hasQualitySignal,
        });
        if (chosen.length < target) {
            debug('UNIQUE_CANDIDATE_SUBSET_ONLY', {
                deckSize: chosen.length,
                target,
                reason: 'no fully unique combination of the requested size',
            });
        }
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
        hasQualitySignal,
        qualityRange: { min: qualityMin, max: qualityMax },
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
