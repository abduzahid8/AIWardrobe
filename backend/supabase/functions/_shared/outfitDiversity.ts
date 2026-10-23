/**
 * outfitDiversity — VISUAL / STYLISTIC diversity layer for AI outfit decks.
 *
 * The deck-level dedupe that already exists (features/outfit-generator/utils/
 * deckDedupe.ts) guarantees PRODUCT diversity: two looks never share the same
 * SKU / normalized name / photo / product family more than once. That is
 * necessary but NOT sufficient. A deck of
 *
 *   navy blazer  + white oxford  + grey tailored trousers + brown loafers
 *   charcoal blazer + blue poplin + black trousers       + penny loafers
 *   beige blazer + striped shirt + navy trousers         + tassel loafers
 *
 * has three different sets of product ids and still reads as ONE outfit to a
 * human. This module models the thing the user actually perceives: the
 * STYLING CONCEPT of a complete look.
 *
 * Three layers, in order:
 *
 *   1. FINGERPRINT — a structured description of one look:
 *      slot families, outfit archetype, silhouette, formality, palette,
 *      materials, texture, pattern, footwear family.
 *   2. SIMILARITY — a normalized 0..1 distance between two fingerprints.
 *      Composition (archetype / silhouette) dominates; colours are the
 *      smallest term on purpose.
 *   3. DECK SELECTION — pick the COMBINATION of looks that maximises
 *      quality + diversity instead of accepting/rejecting looks one by one.
 *
 * The deck selection NEVER relaxes the visual rule while the inventory could
 * have supported `target` different concepts: a catalogue with 8 blazers,
 * 2 coats and 3 overshirts must produce three different formulas, not three
 * colourways of one.
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
 * it cannot rescue a pair from the "same formula" rule.
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
    const raw = text(item?.imageUrl || item?.image).trim().split('?')[0].toLowerCase();
    return raw;
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
    const keys: string[] = [];
    const id = text(item?.id).trim();
    if (id) keys.push(`id:${id}`);
    const nk = normalizeNameKey(item?.name);
    if (nk) keys.push(`n:${nk}`);
    const img = imageKey(item);
    if (img) keys.push(`i:${img}`);
    return keys;
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
     * styling concepts. When true, relaxing the visual rule is FORBIDDEN.
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
    const seenSlots = new Set<DiversitySlot>();
    for (const item of Array.isArray(items) ? items : []) {
        if (!item) continue;
        const slot = diversitySlotOf(item);
        if (!slot || ignore.has(slot)) continue;
        seenSlots.add(slot);
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
    for (const slot of activeSlots) {
        familiesPerSlot[slot] = families.get(slot)?.size ?? 0;
        tokensPerSlot[slot] = tokens.get(slot)?.size ?? 0;
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
    const homogeneous = formulaSlots.length > 0 && tokenCounts.every((count) => count <= 1);
    const forcedSharedFamilies = activeSlots.filter(
        (slot) => (familiesPerSlot[slot] ?? 0) === 1,
    ).length;

    return {
        familiesPerSlot,
        tokensPerSlot,
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
    /** Max non-anchor products two looks may share. Default 1. */
    maxSharedProducts?: number;
    /** Max pairwise visual similarity. Default 0.65. */
    maxVisualSimilarity?: number;
    /** Max shared garment families. Default 1. */
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
}

interface Rung {
    name: string;
    maxSimilarity: number;
    maxSharedFamilies: number;
    /**
     * True only for the absolute floor: two looks may then share the exact
     * same outfit formula. Every other rung forbids it, so three SKUs of
     * "blazer + shirt + tailored trousers + loafers" can never be presented
     * as three styling directions.
     */
    allowSameArchetype: boolean;
}

/**
 * The strictness ladder.
 *
 * Only the FIRST rung is a legitimate outcome. The middle rungs exist for
 * genuinely thin inventories (one blazer, one shirt, one trouser, one shoe
 * cannot produce three different ideas) and are FORBIDDEN whenever the
 * inventory assessment says the pool could have supported three different
 * concepts.
 */
const RUNGS: Rung[] = [
    { name: 'strict', maxSimilarity: MAX_VISUAL_SIMILARITY, maxSharedFamilies: MAX_SHARED_VISUAL_FAMILIES, allowSameArchetype: false },
    { name: 'relaxed-visual', maxSimilarity: 0.74, maxSharedFamilies: MAX_SHARED_VISUAL_FAMILIES, allowSameArchetype: false },
    { name: 'relaxed-formula', maxSimilarity: 0.86, maxSharedFamilies: 2, allowSameArchetype: false },
    { name: 'product-only', maxSimilarity: 1.01, maxSharedFamilies: DIVERSITY_SLOTS.length, allowSameArchetype: true },
];

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
 * assembleDiverseDeck — the deck-level diversity gate.
 *
 * Order of operations (never skipped):
 *   1. collect candidates (edge looks first, then targeted replacements)
 *   2. score every look
 *   3. search the COMBINATION of looks that maximises quality + diversity
 *   4. if no combination qualifies, generate targeted replacements and
 *      search again
 *   5. only then relax the visual rule (thin inventory)
 *
 * The deck is NOT built by accepting looks one at a time: the best single
 * look must not occupy every slot of the deck.
 */
export function assembleDiverseDeck<O>(opts: AssembleDiverseDeckOptions<O>): AssembleDiverseDeckResult<O> {
    const target = opts.target ?? 3;
    const maxSharedProducts = opts.maxSharedProducts ?? 1;
    const maxVisualSimilarity = opts.maxVisualSimilarity ?? MAX_VISUAL_SIMILARITY;
    const maxSharedVisualFamilies = opts.maxSharedVisualFamilies ?? MAX_SHARED_VISUAL_FAMILIES;
    const maxFillAttempts = opts.maxFillAttempts ?? 24;
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
    const familiesOf = (a: DiverseDeckCandidate<O>, b: DiverseDeckCandidate<O>): number =>
        sharedVisualFamilies(a.fingerprint, b.fingerprint, { ignoreSlots });

    /**
     * How many garment families two looks may share.
     *
     * The base limit is MAX_SHARED_VISUAL_FAMILIES, raised when the POOL
     * itself cannot vary a slot: a black-tie inventory with a single formal
     * shirt and a single tuxedo trouser forces every pair of looks to share
     * two families — punishing that would make a correct deck impossible.
     */
    let forcedSharedFamilies = 0;
    const allowedSharedFamilies = (): number =>
        Math.max(maxSharedVisualFamilies, forcedSharedFamilies);

    const deckScore = (indices: number[]): number => {
        const members = indices.map((index) => pool[index]);
        const quality = members.reduce((sum, member) => sum + member.quality, 0);
        const archetypes = new Set(members.map((member) => member.fingerprint.archetype));
        const bonus = DIVERSITY_BONUS_PER_ARCHETYPE * (archetypes.size - 1);
        let penalty = 0;
        let pairs = 0;
        for (let i = 0; i < members.length; i++) {
            for (let j = i + 1; j < members.length; j++) {
                penalty += DIVERSITY_SIMILARITY_PENALTY * similarityOf(members[i], members[j]);
                pairs += 1;
            }
        }
        // The penalty is the MEAN pairwise similarity, so the score does not
        // change meaning when the deck grows from 3 looks to 15.
        const meanPenalty = pairs > 0 ? penalty / pairs : 0;
        return quality + bonus - meanPenalty;
    };

    // Rejection diagnostics (development builds only). Capped so a big pool
    // cannot flood the console.
    const loggedRejections = new Set<string>();
    const logRejection = (a: DiverseDeckCandidate<O>, b: DiverseDeckCandidate<O>, reason: string, similarity: number) => {
        if (!debug) return;
        if (loggedRejections.size >= 12) return;
        const key = `${Math.min(a.index, b.index)}:${Math.max(a.index, b.index)}:${reason}`;
        if (loggedRejections.has(key)) return;
        loggedRejections.add(key);
        debug(`decision: ${reason}`, {
            pair: [a.index + 1, b.index + 1],
            similarity: Number(similarity.toFixed(2)),
            threshold: MAX_VISUAL_SIMILARITY,
            archetypes: [a.fingerprint.archetype, b.fingerprint.archetype],
        });
    };

    const combinationAllowed = (indices: number[], rung: Rung): boolean => {
        for (let i = 0; i < indices.length; i++) {
            for (let j = i + 1; j < indices.length; j++) {
                const a = pool[indices[i]];
                const b = pool[indices[j]];
                if (sharedOf(a, b) > maxSharedProducts) return false;
                const similarity = similarityOf(a, b);
                // The invariant is `visualSimilarity < MAX_VISUAL_SIMILARITY`,
                // so a pair that lands exactly ON the threshold is rejected.
                if (similarity >= rung.maxSimilarity) {
                    logRejection(a, b, 'REJECT_VISUAL_NEAR_DUPE', similarity);
                    return false;
                }
                // Same outfit FORMULA — the rule product dedupe cannot express.
                if (!rung.allowSameArchetype && sameArchetype(a.fingerprint, b.fingerprint, { ignoreSlots })) {
                    logRejection(a, b, 'REJECT_SAME_ARCHETYPE', similarity);
                    return false;
                }
                // Shared garment families are a structural smell, but two
                // looks can share a garment TYPE and still be different
                // concepts. The escape hatch keeps genuinely different looks
                // (very low similarity) even when a family repeats.
                if (familiesOf(a, b) > allowedSharedFamilies() && similarity >= FAMILY_ESCAPE_SIMILARITY) {
                    logRejection(a, b, 'REJECT_SHARED_FAMILY', similarity);
                    return false;
                }
            }
        }
        return true;
    };

    const MAX_COMBINATIONS = 250000;

    const selectCombination = (rung: Rung): number[] | null => {
        const n = pool.length;
        if (n < target) return null;
        let best: number[] | null = null;
        let bestScore = Number.NEGATIVE_INFINITY;
        const consider = (indices: number[]) => {
            if (!combinationAllowed(indices, rung)) return;
            const score = deckScore(indices);
            if (score > bestScore) {
                bestScore = score;
                best = indices;
            }
        };
        // Number of combinations C(n, target) — exhaustive search is only
        // affordable for small pools.
        let combinations = 1;
        for (let i = 0; i < target; i++) {
            combinations = combinations * (n - i) / (i + 1);
            if (combinations > MAX_COMBINATIONS) break;
        }
        if (combinations <= MAX_COMBINATIONS) {
            const current: number[] = [];
            const walk = (start: number) => {
                if (current.length === target) {
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
        // look, then add the most different companions.
        const seed = pool
            .map((member, index) => ({ index, quality: member.quality }))
            .sort((a, b) => b.quality - a.quality)[0];
        if (!seed) return null;
        const chosen = [seed.index];
        while (chosen.length < target) {
            let bestAdd = -1;
            let bestAddScore = Number.NEGATIVE_INFINITY;
            for (let i = 0; i < n; i++) {
                if (chosen.includes(i)) continue;
                const trial = [...chosen, i];
                if (!combinationAllowed(trial, rung)) continue;
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
            }
        }
    };

    /**
     * Steering: mark the VISUAL FAMILIES of the current best trio as used so
     * the replacement builder explores a DIFFERENT garment type instead of
     * another SKU of the same formula. Without this the first replacement
     * happily re-picks the blazer family and is then rejected by the
     * similarity gate — wasting an attempt.
     */
    const seedVisualFamilies = (indices: number[]) => {
        for (const index of indices) {
            const member = pool[index];
            for (const item of opts.itemsOf(member.outfit) || []) {
                const slot = diversitySlotOf(item);
                if (slot && ignoreSlots.includes(slot)) continue; // the anchor repeats by design
                const family = garmentVisualFamily(item);
                if (family) opts.usedKeys.add(`__vf:${family}`);
            }
        }
    };

    // The strict rung honours the caller's overrides.
    const strictRung: Rung = {
        name: 'strict',
        maxSimilarity: maxVisualSimilarity,
        maxSharedFamilies: maxSharedVisualFamilies,
        allowSameArchetype: false,
    };
    const ladder: Rung[] = [strictRung, ...RUNGS.slice(1)];

    // The inventory assessment drives two decisions: whether relaxing the
    // visual rule is allowed at all, and how many garment families two
    // looks may share (slots the pool cannot vary are not a styling fault).
    const inventory = inventoryOf();
    forcedSharedFamilies = inventory.forcedSharedFamilies;

    let rungIndex = 0;
    let selection = selectCombination(ladder[0]);
    let fillsUsed = 0;

    if (opts.buildFill) {
        // Seed the shared used-keys with the current best combination so the
        // replacement builder starts from pieces the deck is not likely to
        // reuse, and can only build a look from genuinely fresh products.
        const seedFrom = selection ?? selectCombination(ladder[ladder.length - 1]);
        if (seedFrom) seedUsedKeys(seedFrom);
        // Actively search for diverse alternatives instead of deleting the
        // rejected looks: build fresh candidates from the unused pool.
        //
        // The search continues while the deck score IMPROVES — stopping at
        // the first acceptable combination would leave "blazer + shirt +
        // tailored trousers + loafers" next to a second blazer look when a
        // third, genuinely different direction was one replacement away.
        let bestScore = selection ? deckScore(selection) : Number.NEGATIVE_INFINITY;
        let staleAttempts = 0;
        const patience = 3;
        while (fillsUsed < maxFillAttempts && staleAttempts < patience) {
            const attempt = fillsUsed;
            fillsUsed += 1;
            // Steer the replacement toward another FORMULA. Only the SINGLE
            // strongest look is used as the reference: the replacement must
            // explore a different formula than the look most likely to be
            // kept, while the global combination search decides the rest.
            const steer = selectCombination(ladder[ladder.length - 1]);
            if (steer) {
                const reference = steer
                    .map((index) => pool[index])
                    .sort((a, b) => (b.quality - a.quality) || (a.index - b.index))[0];
                if (reference) seedVisualFamilies([reference.index]);
            }
            const fill = opts.buildFill(opts.usedKeys, attempt);
            // A rejected replacement (style judge / budget) must NOT end the
            // search — the next attempt rotates to different slot picks.
            if (!fill) continue;
            const candidate = pushCandidate(fill, 'fill');
            if (debug) {
                debug(`replacement selected: ${candidate.fingerprint.archetype}`, {
                    attempt,
                    archetype: candidate.fingerprint.archetype,
                });
            }
            const previous = selection;
            selection = selectCombination(ladder[0]);
            const score = selection ? deckScore(selection) : Number.NEGATIVE_INFINITY;
            if (score > bestScore) {
                bestScore = score;
                staleAttempts = 0;
            } else {
                // No improvement — keep the better of the two and keep the
                // previous combination when the new candidate did not help.
                selection = previous ?? selection;
                staleAttempts += 1;
            }
        }
    }

    let relaxReason: string | null = null;

    if (!selection) {
        // Relaxation is a LAST resort and is FORBIDDEN when the valid
        // inventory could have supported `target` different concepts. A pool
        // with 8 blazers / 2 coats / 3 overshirts / … must never fall back to
        // "blazer + shirt + tailored trousers + loafers" three times.
        if (!inventory.supportsTargetConcepts) {
            for (let i = 1; i < ladder.length - 1 && !selection; i++) {
                rungIndex = i;
                selection = selectCombination(ladder[i]);
            }
            if (selection) relaxReason = 'thin_inventory';
        } else {
            relaxReason = 'diversity_unavailable_despite_varied_inventory';
        }
    }

    if (!selection) {
        // Absolute floor: product-level diversity only. The deck score still
        // prefers the most different combination, so the user gets the least
        // repetitive trio the inventory allows.
        rungIndex = ladder.length - 1;
        selection = selectCombination(ladder[rungIndex]);
        if (!relaxReason) {
            relaxReason = inventory.supportsTargetConcepts
                ? 'diversity_unavailable_despite_varied_inventory'
                : 'thin_inventory';
        }
    }

    const relaxed = rungIndex > 0;

    if (debug) {
        for (const member of pool) {
            debug(`candidate ${member.index + 1} archetype: ${member.fingerprint.archetype}`, {
                source: member.source,
                quality: member.quality,
            });
        }
        debug(`inventory: conceptCapacity=${inventory.conceptCapacity} supportsTarget=${inventory.supportsTargetConcepts} homogeneous=${inventory.homogeneous} forcedSharedFamilies=${inventory.forcedSharedFamilies}`, {
            familiesPerSlot: inventory.familiesPerSlot,
            tokensPerSlot: inventory.tokensPerSlot,
        });
    }

    if (!selection) {
        // Nothing qualified at all — keep the highest-quality looks so the
        // user still sees a deck instead of an empty screen.
        const best = pool
            .slice()
            .sort((a, b) => b.quality - a.quality)
            .slice(0, target);
        return {
            deck: best.map((member) => member.outfit),
            relaxed: true,
            relaxReason: relaxReason ?? 'no_candidates',
            archetypes: best.map((member) => member.fingerprint.archetype),
            similarities: [],
            inventory,
            fillsUsed,
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

    if (debug) {
        for (const entry of similarities) {
            debug(`similarity ${entry.a + 1}<->${entry.b + 1} = ${entry.score.toFixed(2)}`, {
                archetypes: [chosen[entry.a].fingerprint.archetype, chosen[entry.b].fingerprint.archetype],
            });
        }
        // §17 diagnostics: a replacement look and how far it sits from the
        // looks it now ships next to.
        for (const member of chosen) {
            if (member.source !== 'fill') continue;
            const others = chosen.filter((other) => other.index !== member.index);
            const scores = others.map((other) => similarityOf(member, other));
            debug(`replacement shipped: ${member.fingerprint.archetype}`, {
                similarityWithKeptLooks: scores.map((score) => Number(score.toFixed(2))),
            });
        }
        debug(`decision: ${relaxed ? `RELAXED (${relaxReason})` : 'STRICT'}`, {
            archetypes: chosen.map((member) => member.fingerprint.archetype),
            inventory: {
                homogeneous: inventory.homogeneous,
                maxDistinctArchetypes: inventory.maxDistinctArchetypes,
                conceptCapacity: inventory.conceptCapacity,
                supportsTargetConcepts: inventory.supportsTargetConcepts,
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
