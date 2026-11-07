export interface CatalogOrderItem {
    id: string;
    name: string;
    garmentType?: string | null;
}

const RECENT_SIMILARITY_WINDOW = 2;

const SIMILARITY_REPLACEMENTS: Array<[RegExp, string]> = [
    [/\bt[\s-]?shirts?\b/g, ' tee '],
    [/\btees?\b/g, ' tee '],
    [/\bslim[\s-]?fit\b/g, ' slimfit '],
    [/\brelaxed[\s-]?fit\b/g, ' relaxedfit '],
    [/\bloose[\s-]?fit\b/g, ' loosefit '],
    [/\bregular[\s-]?fit\b/g, ' regularfit '],
    [/\bv[\s-]?neck\b/g, ' vneck '],
    [/\bcrew[\s-]?neck\b/g, ' crewneck '],
    [/\bmock[\s-]?neck\b/g, ' mockneck '],
    [/\bbutton[\s-]?down\b/g, ' buttondown '],
    [/\blong[\s-]?sleeve(?:d)?\b/g, ' longsleeve '],
    [/\bshort[\s-]?sleeve(?:d)?\b/g, ' shortsleeve '],
    [/\bheavy[\s-]?weight\b/g, ' heavyweight '],
    [/\bmedium[\s-]?weight\b/g, ' mediumweight '],
    [/\blight[\s-]?weight\b/g, ' lightweight '],
];

const COLOR_WORDS = new Set([
    'black', 'white', 'offwhite', 'ecru', 'cream', 'beige', 'stone', 'sand',
    'taupe', 'camel', 'khaki', 'olive', 'green', 'sage', 'mint',
    'blue', 'navy', 'indigo', 'brown', 'chocolate', 'grey', 'gray',
    'silver', 'charcoal', 'red', 'burgundy', 'maroon', 'pink',
    'purple', 'lilac', 'yellow', 'mustard', 'orange', 'rust',
]);

const LOW_SIGNAL_WORDS = new Set([
    'basic', 'classic', 'essential', 'premium',
    'soft', 'touch', 'washed', 'faded', 'heavyweight', 'mediumweight', 'lightweight',
    'regularfit', 'relaxedfit', 'slimfit', 'loosefit',
    'textured', 'structured', 'comfort', 'interlock', 'stretch',
    'cotton', 'viscose', 'polyester', 'polyamide', 'elastane',
    'lyocell', 'modal', 'jersey',
    'fit', 'detail', 'details', 'edition',
    'men', 'man', 'zara',
]);

const TOKEN_ALIASES = new Map([
    ['tshirt', 'tee'],
    ['tee', 'tee'],
    ['tees', 'tee'],
    ['trainer', 'sneaker'],
    ['trainers', 'sneaker'],
    ['sweater', 'knit'],
    ['jumper', 'knit'],
]);

function normalizeText(value: string): string {
    return value
        .toLowerCase()
        .replace(/https?:\/\//g, '')
        .replace(/[^a-z0-9]+/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
}

function normalizeSimilaritySource(value: string): string {
    let text = value.toLowerCase();

    for (const [pattern, replacement] of SIMILARITY_REPLACEMENTS) {
        text = text.replace(pattern, replacement);
    }

    return text;
}

function singularizeToken(token: string): string {
    if (token.endsWith('ies') && token.length > 4) {
        return `${token.slice(0, -3)}y`;
    }

    if (token.endsWith('s') && token.length > 3 && !token.endsWith('ss')) {
        return token.slice(0, -1);
    }

    return token;
}

function getSimilarityTokens(item: CatalogOrderItem): string[] {
    const normalized = normalizeText(normalizeSimilaritySource(item.name));
    const seen = new Set<string>();
    const tokens: string[] = [];

    for (const rawToken of normalized.split(' ')) {
        if (!rawToken || /^\d+$/.test(rawToken)) continue;

        const singular = singularizeToken(rawToken);
        const token = TOKEN_ALIASES.get(singular) ?? singular;

        if (token.length < 2) continue;
        if (COLOR_WORDS.has(token) || LOW_SIGNAL_WORDS.has(token)) continue;
        if (seen.has(token)) continue;

        seen.add(token);
        tokens.push(token);
    }

    return tokens;
}

export function buildCatalogSimilarityKey(item: CatalogOrderItem): string {
    const garmentType = String(item.garmentType || 'item').trim().toLowerCase() || 'item';
    const tokens = getSimilarityTokens(item);
    return `${garmentType}:${tokens.join(' ') || garmentType}`;
}

/**
 * Normalize an image URL for duplicate detection.
 * Strips query params (thumbnail transforms like ?width=300), lowercases,
 * and removes trailing slashes so the same product image from different
 * sources / thumbnail sizes collapses to one key.
 */
export function normalizeCatalogImageKey(url: unknown): string {
    if (typeof url !== 'string') return '';
    const trimmed = url.trim().toLowerCase();
    if (!trimmed) return '';
    const withoutQuery = trimmed.split('?')[0].split('#')[0];
    return withoutQuery.replace(/\/+$/, '');
}

/**
 * Normalize a product name for duplicate detection.
 * Lowercase + strip non-alphanumerics + collapse whitespace, so
 * "Leather loafers with p…" variants from different ingestion sources
 * (manual vs apify-massimodutti vs apify-zara-men-uz) collapse even when
 * their catalog `id`s differ.
 */
export function normalizeCatalogNameKey(name: unknown): string {
    if (typeof name !== 'string') return '';
    return name
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, ' ')
        .replace(/\s+/g, ' ')
        .trim();
}

const SHOE_FAMILY_PATTERNS: Array<[RegExp, string]> = [
    [/\bloafer\b/, 'loafer'],
    [/\bloafer/, 'loafer'],
    [/\bpenny\b/, 'loafer'],
    [/\bsneaker\b|\btrainer\b|\btennis\b/, 'sneaker'],
    [/\bboot\b|\bchelsea\b|\bchukka\b/, 'boot'],
    [/\boxford\b|\bderby\b|\bbrogue\b/, 'dress-shoe'],
    [/\bsandal\b|\bslide\b|\bflip\b/, 'sandal'],
    [/\bespadrille\b/, 'espadrille'],
    [/\bmule\b|\bslip[\s-]?on\b|\bmoc\b/, 'slip-on'],
];

/**
 * Coarse shoe family (loafer / sneaker / boot / …) used to guarantee visual
 * variety in the Essentials grid. Without this, a catalog with 8 loafer
 * variants can fill the whole `shoes: 3` quota with near-identical cards
 * (the "same loafers 3x" bug) even when similarity keys differ slightly
 * ("oiled leather penny loafer" vs "leather loafer with penny strap").
 */
export function getCatalogShoeFamily(name: unknown): string {
    const text = String(name ?? '').toLowerCase();
    for (const [pattern, family] of SHOE_FAMILY_PATTERNS) {
        if (pattern.test(text)) return family;
    }
    return 'other-shoe';
}

/**
 * Coarse garment family used so a deck cannot fill every card with
 * near-identical SKUs (penny loafer vs oiled loafer, wool trouser vs
 * darted wool trouser). Slot pick skips a family while unused families
 * still exist, then falls back so thin pools never starve.
 */
export function garmentVarietyFamily(
    item: { name?: string; type?: string; macroCategory?: string } | null | undefined,
): string {
    if (!item) return '';
    const macro = String(item.macroCategory || '').toLowerCase();
    const blob = `${item.type || ''} ${item.name || ''}`.toLowerCase();
    if (macro === 'shoes' || /\b(shoe|loafer|sneaker|boot|oxford|derby|brogue)\b/.test(blob)) {
        return `shoe:${getCatalogShoeFamily(item.name)}`;
    }
    if (macro === 'bottom') {
        if (/\b(jean|denim|jort)\b/.test(blob)) return 'bottom:jean';
        if (/\bchino\b/.test(blob)) return 'bottom:chino';
        if (/\b(shorts?|bermuda)\b/.test(blob)) return 'bottom:short';
        if (/\b(suit|wool|tailored|darted|pleated|trouser)\b/.test(blob)) return 'bottom:tailored';
        return 'bottom:other';
    }
    if (macro === 'outerwear') {
        if (/\b(blazer|suit\s*jacket|sport\s*coat)\b/.test(blob)) return 'outer:blazer';
        if (/\b(trench|overcoat|peacoat|top\s*coat|coat)\b/.test(blob)) return 'outer:coat';
        if (/\bhoodie\b/.test(blob)) return 'outer:hoodie';
        return 'outer:other';
    }
    if (macro === 'top') {
        if (/\bpolo\b/.test(blob)) return 'top:polo';
        if (/\b(oxford|shirt|poplin)\b/.test(blob)) return 'top:shirt';
        if (/\b(tee|t-shirt|tshirt)\b/.test(blob)) return 'top:tee';
        if (/\b(knit|sweater|cardigan)\b/.test(blob)) return 'top:knit';
        return 'top:other';
    }
    return macro || '';
}

export function spreadSimilarCatalogItems<T extends CatalogOrderItem>(
    items: T[],
    recentItems: CatalogOrderItem[] = [],
): T[] {
    if (items.length <= 1) return items;

    const recentKeys = recentItems
        .slice(-RECENT_SIMILARITY_WINDOW)
        .map(buildCatalogSimilarityKey);
    const remaining = items.map((item) => ({
        item,
        similarityKey: buildCatalogSimilarityKey(item),
    }));
    const ordered: typeof remaining = [];

    while (remaining.length > 0) {
        let nextIndex = remaining.findIndex(
            (candidate) => !recentKeys.includes(candidate.similarityKey),
        );

        if (nextIndex === -1) {
            const lastKey = recentKeys[recentKeys.length - 1];
            nextIndex = remaining.findIndex(
                (candidate) => candidate.similarityKey !== lastKey,
            );
        }

        if (nextIndex === -1) {
            nextIndex = 0;
        }

        const [nextItem] = remaining.splice(nextIndex, 1);
        ordered.push(nextItem);
        recentKeys.push(nextItem.similarityKey);

        if (recentKeys.length > RECENT_SIMILARITY_WINDOW) {
            recentKeys.shift();
        }
    }

    return ordered.map(({ item }) => item);
}

/**
 * Merge a fresh catalog slice into the generation pool without duplicates.
 * Primary (already-loaded page) wins order; the random window appends only
 * unseen ids. Used so each Generate tap draws from a different slice of the
 * full catalog instead of the same first-N rows every time.
 */
export function mergeShopSlices<T extends { id: string | number }, U extends { id: string | number }>(primary: T[], extra: U[]): Array<T | U> {
    if (extra.length === 0) return primary;
    const seen = new Set(primary.map((i) => String(i.id)));
    const out: Array<T | U> = [...primary];
    for (const item of extra) {
        const key = String(item.id);
        if (seen.has(key)) continue;
        seen.add(key);
        out.push(item);
    }
    return out;
}

/**
 * Collapse same-product listings to one entry (keeps the FIRST — callers
 * pre-sort best-first, so the winner survives). Feeds ingest the same
 * product under many UUIDs ("Leather loafers with penny strap" ×22); without
 * this, lookalikes occupy N elite trim slots and every deck recycles them.
 * Key = slot + normalized name, so colorways ("Navy" vs "Sand" blazer) stay
 * distinct while true duplicates collapse.
 */
export function collapseSimilarByName<
    T extends { id: string | number; name?: string | null; macroCategory?: string },
>(items: T[]): T[] {
    const seen = new Set<string>();
    const out: T[] = [];
    for (const item of items) {
        const key = `${String(item.macroCategory || '').toLowerCase()}:${normalizeCatalogNameKey(item.name)}`;
        if (seen.has(key)) continue;
        seen.add(key);
        out.push(item);
    }
    return out;
}

const ACCESSORY_TYPE_RE =
    /\b(sunglasses?|eyewear|glasses|watch(?:es)?|belts?|handbags?|backpacks?|wallets?|scar(?:f|ves)|hats?|caps?|beanies?|gloves?|jewel\w*|bracelets?|necklaces?|ties?|cufflinks?)\b/;
const TECH_ACCESSORY_RE = /\b(iphone|phone[\s-]?cases?|airpods?)\b/;
const OUTERWEAR_NAME_RE =
    /\b(jacket|coat|blazer|hoodie|cardigan|sweater|pullover|puffer|bomber|vest|outerwear|trench|peacoat|sweatshirt|knitwear|jumper|fleece)\b/;
const TOP_NAME_RE =
    /\bt-?shirts?\b|\btees?\b|\bpolos?\b|\bblouses?\b|\bshirts?\b|\btank\b|\bhenley\b|\bovershirts?\b/;
const BOTTOM_NAME_RE =
    /\bpants?\b|\btrousers?\b|\bjeans?\b|\bshorts\b|\bskirts?\b|\bchinos?\b|\bbermudas?\b|\btrunks\b/;
const SHOES_NAME_RE =
    /\bshoes?\b|\bsneakers?\b|\bloafers?\b|\bboots?\b|\bsandals?\b|\bboat\b|\bderb\w*\b|\boxfords?\b|\bmules?\b|\bespadrilles?\b|кроссовки|кеды|туфли|ботинки|лоферы|сапоги|мокасины|сандалии/;
const SHORT_SLEEVE_NAME_RE = /\bshort[\s-]?sleeve(?:s|d)?\b/;

/**
 * Slot a shop_catalog row. NAME beats stored garment_type (tees mistagged as
 * lower_body). Descriptions are ignored: "pair with a blazer" on trousers
 * must not put pants in the LAYER tile. When the photo type is lower_body
 * but the title is a blazer, drop the row — the scrape stole a jacket name
 * onto a trousers image (live screenshot: grey pants labeled suit blazer).
 */
export function inferShopRowMacro(garmentType: string, name: string, _description?: string): string {
    const nameBlob = String(name || '').toLowerCase();
    const nameSansSleeves = nameBlob.replace(SHORT_SLEEVE_NAME_RE, ' ');
    if (garmentType === 'accessory') return 'other';
    if (TECH_ACCESSORY_RE.test(nameBlob)) return 'other';
    if (SHORT_SLEEVE_NAME_RE.test(nameBlob) && !BOTTOM_NAME_RE.test(nameSansSleeves)) return 'top';

    const nameIsOuter = OUTERWEAR_NAME_RE.test(nameBlob) && !SHORT_SLEEVE_NAME_RE.test(nameBlob);
    const nameIsBottom = BOTTOM_NAME_RE.test(nameSansSleeves);
    const nameIsTop = TOP_NAME_RE.test(nameBlob);
    const nameIsShoes = SHOES_NAME_RE.test(nameBlob);

    if (garmentType === 'lower_body' && nameIsOuter && !nameIsBottom) return 'other';
    if (garmentType === 'shoes' && nameIsOuter && !nameIsShoes && !nameIsTop) return 'other';

    // Accessory names beat the stored garment_type: a "43Mm Silicone Strap
    // Sport Watch" row ships with garment_type=upper_body and would otherwise
    // slot as a TOP. Clothing words still win (belt-waist pants stay pants).
    if (ACCESSORY_TYPE_RE.test(nameBlob) && !nameIsTop && !nameIsBottom && !nameIsOuter && !nameIsShoes) return 'other';

    // Jeans can be a color/product prefix: the named shirt remains a top.
    if (nameIsTop && !nameIsOuter && /\b(?:t[ -]?shirts?|tees?|shirts?|polos?|blouses?)\b(?!.*\b(?:pants?|trousers?|jeans|chinos?)\b)/.test(nameBlob)) return 'top';
    if (nameIsBottom && !nameIsOuter) return 'bottom';
    if (nameIsOuter && !nameIsBottom) return 'outerwear';
    if (nameIsTop && !nameIsOuter) return 'top';
    if (nameIsShoes) return 'shoes';
    if (nameIsBottom) return 'bottom';

    if (garmentType === 'upper_body') return 'top';
    if (garmentType === 'lower_body') return 'bottom';
    if (garmentType === 'shoes') return 'shoes';
    if (garmentType === 'outerwear') return 'outerwear';
    return 'top';
}

const IMAGE_EXT_RE = /\.(jpe?g|png|webp|avif)(\?|#|$)/i;
const BRAND_CDN_RE = /massimodutti\.net|static\.zara\.net|uniqlo\.com|suitsupply\.com|static\.zara/i;

/**
 * Live shop_catalog rows often share one supabase object (or a URL with no
 * file). Those render as "wool blazer" with a hoodie photo. Brand CDNs and
 * URLs that actually end in an image extension are the only photos we trust
 * to match the name on the card.
 */
export function isTrustedProductImage(url?: string | null): boolean {
    const raw = String(url || '').trim();
    if (!/^https?:\/\//i.test(raw)) return false;
    // AllSaints' gallery selector ends in .json but serves an actual image.
    // Keep this scoped to its official image route, not arbitrary JSON URLs.
    if (/^https:\/\/media\.i\.allsaints\.com\/image\/(?:list|upload)\//i.test(raw)) return true;
    if (IMAGE_EXT_RE.test(raw)) return true;
    if (BRAND_CDN_RE.test(raw)) return true;
    return false;
}

/** Brand product CDNs that actually render in the simulator (not 404 supabase stubs). */
export function isBrandCatalogImage(url?: string | null): boolean {
    const raw = String(url || '').trim();
    return /^https?:\/\//i.test(raw) && BRAND_CDN_RE.test(raw);
}

export function productImageKey(item: { image?: string | null; imageUrl?: string | null } | null | undefined): string {
    const raw = String(item?.imageUrl || item?.image || '').trim().split('?')[0].toLowerCase();
    return raw;
}

/** One photo → one product. Duplicate CDN objects were pairing the wrong name. */
export function collapseDuplicateProductImages<T extends { image?: string | null; imageUrl?: string | null }>(items: T[]): T[] {
    const seen = new Set<string>();
    const out: T[] = [];
    for (const item of items) {
        const key = productImageKey(item);
        if (key) {
            if (seen.has(key)) continue;
            seen.add(key);
        }
        out.push(item);
    }
    return out;
}

/**
 * Prefer an already-computed slot macro over recomputing one.
 * Production incident: the edge payload rebuilt macros from type/category
 * aliases, which silently discarded the identity-first mapping — mistagged
 * tees (type "bottoms") flipped back to 'bottom' on the way out. A mapped
 * clothing slot is always authoritative; the recomputed value is only a
 * fallback for unmapped/ambiguous rows.
 */
const CATALOG_COLOR_WORDS = [
  'black', 'white', 'offwhite', 'off-white', 'ecru', 'cream', 'beige', 'stone',
  'sand', 'taupe', 'camel', 'khaki', 'olive', 'green', 'sage', 'mint', 'blue',
  'navy', 'indigo', 'brown', 'chocolate', 'grey', 'gray', 'silver', 'charcoal',
  'red', 'burgundy', 'maroon', 'pink', 'purple', 'lilac', 'yellow', 'mustard',
  'orange', 'rust', 'tan', 'forest',
];

// Feed photo boilerplate that poisons color inference ("white-bg product
// shot" made every studio photo infer as white). Stripped before matching.
const CATALOG_BOILERPLATE_RE =
  /white-bg|white background|product shots?|front views?|back views?|side views?|flat lays?|ghost mannequins?|on models?|studio shots?|model shot|e-commerce/gi;

/**
 * Infer a garment color from catalog name/description (the table has no
 * color column). Mirrors the edge inferShopColor so client pools carry the
 * same colors the server sees — without it every pool item is 'neutral'
 * and color rules (black-pairing, palettes) judge blind.
 * Returns 'neutral' when nothing matches. NOT a substitute for vision
 * truth (a black item named without its color still reads neutral).
 */
export function inferCatalogColor(name: unknown, description?: unknown): string {
  const raw = `${name || ''} ${description || ''}`.toLowerCase();
  const blob = raw.replace(CATALOG_BOILERPLATE_RE, ' ');
  for (const c of CATALOG_COLOR_WORDS) {
    if (new RegExp(`\\b${c}\\b`).test(blob)) return c;
  }
  return 'neutral';
}

export function preferMappedMacro(mapped: unknown, fallback: string): string {
    const m = String(mapped ?? '')
        .toLowerCase()
        .trim();
    if (m === 'top' || m === 'bottom' || m === 'shoes' || m === 'outerwear') return m;
    return fallback;
}

/** Curated MD/classic rows live near sort_order head; past ~1100 is streetwear. */
export const CLASSIC_CATALOG_HEAD_SPAN = 720;

/**
 * Per-tap offset inside the classic catalog band. Pinning every old-money /
 * work / date tap at 0 collapsed 220 rows into the same ~38 names.
 */
export function classicCatalogOffset(
    total: number,
    sliceSize: number,
    salt: number,
    headSpan: number = CLASSIC_CATALOG_HEAD_SPAN,
): number {
    const size = Math.max(1, sliceSize);
    const span = Math.min(Math.max(size, headSpan), Math.max(size, total || size));
    const maxStart = Math.max(0, span - size);
    if (maxStart <= 0) return 0;
    return ((Math.abs(salt) * 1103515245 + 12345) >>> 0) % (maxStart + 1);
}

/**
 * True when a catalog row's name is displayable on a card.
 * Production incident: feed rows with missing names surfaced as raw numeric
 * ids ("63092782") on outfit cards. Blank, digit-only, or id-equal names
 * must never enter a pick pool. (User wardrobe items are exempt — a user's
 * own naming is never discarded.)
 */
export function hasUsableProductName(name: unknown, id?: unknown): boolean {
    const n = String(name ?? '').trim();
    if (!n) return false;
    if (/^[0-9\s\-]+$/.test(n)) return false;
    if (id != null && id !== '' && n === String(id)) return false;
    return true;
}
