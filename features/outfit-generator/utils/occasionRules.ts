/**
 * occasionRules — taste enforcement for the AI stylist.
 *
 * The scorer ranks STYLE; nothing ranked OCCASION, so "minimalist date
 * night" decks contained jacquard tees, espadrilles, boat shoes and quilted
 * jackets. This module is the missing judge, used in two places:
 *
 *   1. Steering (normalizeTo4Slots): formal-first pick predicates with
 *      fallback to the unconstrained pool when nothing qualifies.
 *   2. Veto (generateOutfits deck filter): drop cards carrying rejected
 *      pieces, with fallback to unfiltered when the pool is too thin.
 *
 * Plus humanizeProductName: feed rows arrive as "Front view of a white
 * long-sleeve ... - Image 0" / ALL CAPS and that leaks onto cards.
 * Pure + deterministic. Covered by __tests__/e2e/aiOccasion.e2e.test.ts.
 */

export type OccasionFormality = 'formal' | 'casual';

/** work / date / formal demand evening polish; everything else is relaxed. */
export function occasionFormality(occasionId: string | null | undefined): OccasionFormality {
    const key = String(occasionId || '').toLowerCase().trim();
    if (key === 'work' || key === 'date' || key === 'formal') return 'formal';
    return 'casual';
}

const FORMAL_SHOE_RE = /\b(sneakers?|trainers?|athletic|basketball|skate|sandals?|slides?|flip[\s-]?flops?|espadrilles?|boat\b|slippers?|clogs?|runners?|joggers?|mules?)\b/;
// Driving moccasins sit a full formality tier below loafers: unstructured
// soft sole, tie laces, pebbled nubs. Riviera-casual, never evening wear —
// and never with tailoring. (A loafer by contrast is structured:
// penny/bit/tassel go with suits.)
const DRIVER_SHOE_RE = /\b(moccasins?|drivers?|driving|gommino|loafer[s]?\s+with\s+(?:tie|laces?|nubs?)|nubbed|rubber[\s-]?nubs?)\b/;
const CLASSIC_UGLY_SHOE_COLOR_RE = /\b(olive|khaki|moss|sage|lime|chartreuse)\b/;
// Backless loafers ARE mules by construction; slip-ons readesso casual.
// ('mules?' lives in FORMAL_SHOE_RE too — this family rule covers the
// classic-styles overlay where formal gating doesn't reach.)
const MULE_RE = /\b(mules?|backless|slip[\s-]?ons?)\b/;
// Mandarin/stand/band collars cheapen classic lines (city bans). Plain
// 'collar' is everywhere (polo collars are correct) — only the variants.
const MANDARIN_RE = /\b(mandarins?|stand[\s-]?ups?[\s-]?collars?|band[\s-]?collars?|grandad[\s-]?collars?)\b/;
// Work means business: denim bottoms never belong (dark rinse included).
const WORKISH_RE = /\b(work|office|interviews?|meetings?|business|corporate|conferences?)\b/;
const PRINT_TOP_RE = /\b(graphic|logo|jacquard|camo|tie[\s-]?dye|sequins?|rhinestones?)\b/;
const CASUAL_LAYER_RE = /\b(puffer|quilted|bombers?|hoodies?|sweatshirts?|windbreakers?|parkas?|anoraks?|fleece vest|ski\b)\b/;
// Fashion-catalog zip / "total look" jackets are not tailored outerwear.
const ZIP_CASUAL_LAYER_RE = /\b(zip[\s-]?up|zipped|total[\s-]?look|overshirt|shacket)\b/;
const RELAXED_FIT_RE = /\brelaxed[\s-]?fit\b/;
const PLAID_TOP_RE = /\bplaid\b/;
const CASUAL_LOAFER_RE = /\b(split[\s-]?leather|jute|cupsole|cup[\s-]?sole|crepe[\s-]?sole|gum[\s-]?sole|white[\s-]?sole|caramel[\s-]?sole|moc[\s-]?toe)\b/;
const SLOPPY_BOTTOM_RE = /\b(cargo|ripped|distressed|sweatpants?|joggers?|basketball|board\s*shorts?|drawstrings?|elasticated|\belastic(?:ated)?[\s-]?waist(?:band)?|co-?ords?)\b/;
const FASHION_WORK_BOTTOM_RE = /\b(washed[\s-]?effect|bell[\s-]?bottoms?|splatter|uneven[\s-]?hem|twisted[\s-]?seams?|barrel[\s-]?fit)\b/;
const FASHION_WORK_LAYER_RE = /\b(washed[\s-]?effect)\b/;
const PIQUE_POLO_RE = /\bpiqu[eé]/i;
const MINIMALIST_NOISE_RE = /\b(graphic|logo|camo|neon|rhinestones?|sequins?|animal\s*print|monogram)\b/;
// Loud Valentino-scale motifs that classic dress codes never wear on cloth:
// souvenir prints, florals, lettering. Tailored patterns (checks, stripes,
// pinstripes, tweed…) are deliberately ABSENT from this list, so oxford
// stripes and Glen checks always survive the veto.
const LOUD_PRINT_RE = /\b(graphics?|logos?|florals?|paisleys?|prints?|printed|camos?|tie[\s-]?dyes?|neons?|sequins?|rhinestones?|souvenirs?|sukajans?|embroidered|embroidery|animal\s*prints?|lettering|slogans?|texts?)\b/;
// Fit doctrine (STYLE_KNOWLEDGE #18: a jacket must cover the seat): cropped,
// boxy, or shrunken tops/layers are fashion fits, never classic.
// (Patch pockets are Ivy staples — 'patch' itself is NOT vetoed.)
const UNCLASSIC_FIT_RE = /\b(cropped|boxy|oversized|shrunken)\b/;
// Team-sport polos read as fan wear. Rugby/tennis/golf stay legal — they are
// preppy staples; soccer is not.
const SOCCER_RE = /\bsoccers?\b/;
// Sleepwear must never surface as daywear in polished or classic dress codes.
// Real log offender: "Red and white striped pajama set" picked as the TOP of
// an old_money/work outfit (it passes every other formal rule).
const SLEEPWEAR_RE = /\b(pajamas?|pyjamas?|bathrobes?|slippers?|loungewear|sleepwear|nightgowns?|nightwear|dressing\s*gowns?)\b/;
// Styles whose pillars forbid casual/sloppy pieces in EVERY occasion
// (mirrors the minimalist overlay precedent below).
const CLASSIC_STYLES = new Set(['old_money', 'classic', 'business_casual']);

export type ClimateWeather = { temp?: number | null; condition?: string | null };

/**
 * When GPS/weather APIs fail, dress for the calendar — not a fake 20°C "mild"
 * day. October in Tashkent is ~15°C; serving 100% linen is how you lose trust.
 * Southern-hemisphere timezones invert the month.
 */
export function inferSeasonalWeather(now: Date = new Date()): { temp: number; condition: string } {
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || '';
    const south = /Australia|Auckland|Pacific\/Auckland|Chile|Argentina|Sao_Paulo|Johannesburg|NZ|Perth|Sydney|Melbourne/i.test(tz);
    const month = now.getMonth();
    const m = south ? (month + 6) % 12 : month;
    if (m >= 10 || m <= 1) return { temp: 6, condition: 'cold' };
    if (m === 2 || m === 9) return { temp: 15, condition: 'cool' };
    if (m === 3 || m === 8) return { temp: 18, condition: 'mild' };
    if (m === 4 || m === 7) return { temp: 22, condition: 'mild' };
    return { temp: 28, condition: 'warm' };
}

const SUMMER_OPEN_SHOE_RE = /\b(espadrilles?|sandals?|slides?|flip[\s-]?flops?|jandals?)\b/;
const HEAVY_WINTER_LAYER_RE = /\b(puffer|parka|ski\b|down\s+jacket|overcoats?|topcoats?)\b/;
const WARM_FIBER_RE = /\b(wool|merino|cashmere|flannel|tweed|felt|fleece|boiler)\b/;
const WARM_LAYER_RE = /\b(coat|jacket|blazer|cardigan|sweater|pullover|knit|trench|peacoat|hoodie|overshirt)\b/;

/** True for garments that only belong in warm weather (pure linen, seersucker…). Wool-linen sportcoats pass. */
export function isWarmWeatherOnlyFabric(item: JudgeableItem | null | undefined): boolean {
    if (!item) return false;
    const blob = `${item.name || ''} ${item.type || ''} ${item.description || ''}`.toLowerCase();
    if (/\b(seersucker|voile|gauze|tropical\s+wool)\b/.test(blob)) return true;
    const linen = /\blinen\b/.test(blob);
    if (!linen) return false;
    if (/\b(knit|sweater|cardigan|pullover|jumper)\b/.test(blob)) return false;
    return !WARM_FIBER_RE.test(blob);
}

/**
 * Climate veto — "how can you wear linen at 15°C?" Lives beside style/occasion
 * so a 10k-user stylist never ships a Riviera look on a cold day.
 */
export function rejectForClimate(
    item: JudgeableItem | null | undefined,
    weather?: ClimateWeather | null,
): string | null {
    if (!item || !weather || typeof weather.temp !== 'number' || Number.isNaN(weather.temp)) return null;
    const temp = weather.temp;
    const blob = `${item.name || ''} ${item.type || ''} ${item.brand || ''} ${item.description || ''}`.toLowerCase();
    const blobSansSleeves = blob.replace(STRIP_SHORT_SLEEVE_RE, ' ');
    const slot = String(item.macroCategory || '').toLowerCase();
    const cold = temp < 18;
    const hot = temp >= 26;

    if (cold && isWarmWeatherOnlyFabric(item)) {
        return `${temp}°C is too cold for linen / summer cloth`;
    }
    if (cold && SUMMER_OPEN_SHOE_RE.test(blob)) {
        return `${temp}°C is too cold for open summer shoes`;
    }
    if (cold && /\b(shorts?|bermudas?|trunks?|swim)\b/.test(blobSansSleeves) && (slot === 'bottom' || slot === '')) {
        return `${temp}°C is too cold for shorts`;
    }
    if (hot && slot === 'outerwear' && HEAVY_WINTER_LAYER_RE.test(blob)) {
        return `${temp}°C is too warm for heavy winter outerwear`;
    }
    if (temp >= 28 && /\b(tweed|flannel)\b/.test(blob) && slot !== 'shoes') {
        return `${temp}°C is too warm for tweed/flannel`;
    }
    return null;
}

/** Cool days need a real (non-linen) layer — a tee + linen pants is not dressed. */
export function rejectOutfitClimate(
    items: JudgeableItem[] | null | undefined,
    weather?: ClimateWeather | null,
): string | null {
    if (!items || items.length === 0) return null;
    if (!weather || typeof weather.temp !== 'number' || Number.isNaN(weather.temp)) return null;
    if (weather.temp >= 18) return null;
    const hasWarmLayer = items.some((it) => {
        const slot = String(it.macroCategory || '').toLowerCase();
        const blob = `${it.name || ''} ${it.type || ''} ${it.description || ''}`.toLowerCase();
        if (slot !== 'outerwear' && slot !== '') return false;
        if (!WARM_LAYER_RE.test(blob) && slot !== 'outerwear') return false;
        return !isWarmWeatherOnlyFabric(it);
    });
    if (!hasWarmLayer) return `${weather.temp}°C needs a warm layer, not summer cloth`;
    return null;
}

export interface JudgeableItem {
    name?: string | null;
    type?: string | null;
    brand?: string | null;
    description?: string | null;
    /** Outfit slot when known — print rules only judge cloth (tops/layers). */
    macroCategory?: string | null;
    /** Free-text color (drives the black-pairing and suit-match heuristics). */
    color?: string | null;
}

// Square toes cheapen everything; exotics read vulgar (universal transcript
// condemnation, not occasion-bound).
const SQUARE_TOE_RE = /\bsquare[\s-]?toes?\b/;
const EXOTIC_RE = /\b(crocodile|alligator|pythons?|stingray|ostrich|lizard)(\s*(leather|skin))?\b|\bexotic\s*leathers?\b/;
// Basic tees/tanks under tailoring. Knit polos never match (no 'polo'
// alternative here) — they are the CORRECT base layer, not the violation.
const BASIC_TEE_RE = /\bt-?shirts?\b|\btees?\b|\btanks?\b/;

/**
 * Returns a rejection reason when the item has no business in the given
 * style+occasion, else null. Formal occasions veto beach/athletic shoes,
 * printed tops, casual layers and sloppy bottoms; minimalist vetoes loud
 * pieces in every occasion.
 */
export function rejectForStyleOccasion(
    item: JudgeableItem | null | undefined,
    styleId: string | null | undefined,
    occasionId: string | null | undefined,
): string | null {
    if (!item) return 'empty item';
    const blob = `${item.name || ''} ${item.type || ''} ${item.brand || ''} ${item.description || ''} ${item.color || ''}`.toLowerCase();
    // Underwear is never visible daywear, in any style or occasion.
    if (UNDERWEAR_RE.test(blob)) return 'underwear must never surface as visible clothing';
    if (/\b(iphone|phone[\s-]?cases?|airpods?)\b/.test(blob)) return 'tech accessories are not outfit pieces';
    const style = String(styleId || '').toLowerCase().replace(/[\s-]+/g, '_');
    const formal = occasionFormality(occasionId) === 'formal';
    const slot = String(item.macroCategory || '').toLowerCase();
    // Print rules judge garments, never shoes/accessories (a floral tie or
    // pocket square is dandy, not sloppy). Bottoms are judged for formal
    // occasions (printed barrel-fit drawstring pants are not workwear) but
    // spared under classic styles, so beach linen keeps its prints.
    // Unknown slot = judged.
    const isTopOrLayer = slot === '' || slot === 'top' || slot === 'outerwear';
    const isBottom = slot === 'bottom';

    const occasion = String(occasionId || '').toLowerCase().trim();
    if (occasion === 'travel') {
        if ((slot === 'outerwear' || slot === '' || isTopOrLayer) && BOARDROOM_LAYER_RE.test(blob)) {
            return 'boardroom suiting is not a travel look';
        }
        if ((slot === 'outerwear' || slot === '') && /\b(tuxedos?|suit\s*jackets?)\b/.test(blob)) {
            return 'suit jackets are not a travel look';
        }
    }
    if (formal) {
        if (FORMAL_SHOE_RE.test(blob)) return 'beach/athletic shoes are not evening wear';
        // Chunky lug soles read streetwear, even on derby shapes (verified
        // on a live Old Money card). Knitwear chunk is untouched (shoes only).
        if (isShoeish(item, blob) && /\bchunky\b/.test(blob)) return 'chunky soles cheapen polished footwear';
        if (DRIVER_SHOE_RE.test(blob)) return 'driving moccasins are a tier below evening footwear';
        if (MULE_RE.test(blob)) return 'mules and backless shoes are not evening wear';
        if (isTopOrLayer && MANDARIN_RE.test(blob)) return 'mandarin collars cheapen polished lines';
        if (PRINT_TOP_RE.test(blob)) return 'printed tops clash with a polished evening look';
        if ((isTopOrLayer || isBottom) && hasLoudPrint(blob)) return 'loud prints clash with a polished evening look';
        if (isTopOrLayer && UNCLASSIC_FIT_RE.test(blob)) return 'cropped/boxy fits break classic proportions';
        if (isTopOrLayer && SOCCER_RE.test(blob)) return 'soccer kits are fan wear, not evening wear';
        if (isTopOrLayer && (CASUAL_LAYER_RE.test(blob) || ZIP_CASUAL_LAYER_RE.test(blob))) {
            return 'casual outerwear underdresses the occasion';
        }
        if (slot === 'top' && PLAID_TOP_RE.test(blob)) return 'fashion plaid shirts underdress a polished look';
        if (isTopOrLayer && RELAXED_FIT_RE.test(blob)) return 'relaxed-fit fashion tops break polished proportions';
        if (isBottom && RELAXED_FIT_RE.test(blob)) return 'relaxed-fit trousers underdress a polished look';
        if ((slot === 'top' || slot === '') && SHORT_SLEEVE_RE.test(blob) && !/\b(knit|polo)\b/.test(blob)) {
            return 'short sleeves underdress a polished look';
        }
        if (isShoeish(item, blob) && CASUAL_LOAFER_RE.test(blob)) {
            return 'split-leather / gum-sole loafers are not evening footwear';
        }
        if (SLOPPY_BOTTOM_RE.test(blob)) return 'sloppy bottoms underdress the occasion';
        if (isBottom && FASHION_WORK_BOTTOM_RE.test(blob)) return 'fashion denim/splatter bottoms underdress work';
        if (isTopOrLayer && FASHION_WORK_LAYER_RE.test(blob)) return 'washed fashion jackets underdress the occasion';
        if ((slot === 'top' || slot === '') && PIQUE_POLO_RE.test(blob) && /\bpolo\b/.test(blob)) {
            return 'piqué polos are sporty, not workwear';
        }
        if (SLEEPWEAR_RE.test(blob)) return 'sleepwear is not daywear for a polished occasion';
        if (isBottom && /\bjeans?\b/.test(blob)) return 'denim bottoms underdress a formal look';
    }
    const workish = WORKISH_RE.test(String(occasionId || '').toLowerCase());
    if ((CLASSIC_STYLES.has(style) || workish) && isBottom && /\bjeans?\b/.test(blob)) {
        return 'denim bottoms never belong in classic or work decks';
    }
    if (workish && isTopOrLayer && /\bdenim\b/.test(blob) && /\bshirts?\b/.test(blob)) {
        return 'denim shirts underdress a work look';
    }
    if (workish && (slot === 'outerwear' || slot === '') && isKnitOnlyLayerBlob(blob)) {
        return 'knit sweaters are not work outerwear';
    }
    if (CLASSIC_STYLES.has(style)) {
        if (isShoeish(item, blob) && /\bchunky\b/.test(blob)) return 'chunky soles break classic discipline';
        if (SLEEPWEAR_RE.test(blob)) return 'sleepwear breaks classic discipline';
        if (DRIVER_SHOE_RE.test(blob)) return 'driving moccasins never pair with classic tailoring';
        if (isShoeish(item, blob) && FORMAL_SHOE_RE.test(blob)) {
            return 'boat/athletic shoes never pair with classic tailoring';
        }
        if (isShoeish(item, blob) && CLASSIC_UGLY_SHOE_COLOR_RE.test(blob)) {
            return 'olive/khaki casual shoes clash with classic tailoring';
        }
        if (isShoeish(item, blob) && CASUAL_LOAFER_RE.test(blob)) {
            return 'split-leather / gum-sole loafers never pair with classic tailoring';
        }
        if (MULE_RE.test(blob)) return 'mules never pair with classic tailoring';
        if (isTopOrLayer && (CASUAL_LAYER_RE.test(blob) || ZIP_CASUAL_LAYER_RE.test(blob))) {
            return 'casual outerwear breaks classic discipline';
        }
        if (isTopOrLayer && RELAXED_FIT_RE.test(blob)) return 'relaxed-fit fashion tops break classic proportions';
        if (isBottom && /\b(ripped|distressed|sweatpants?|joggers?|track|velour|baggy|cargo)\b/.test(blob)) {
            return 'sloppy/track bottoms break classic discipline';
        }
        if (isBottom && FASHION_WORK_BOTTOM_RE.test(blob) && formal) {
            return 'fashion denim/splatter bottoms break classic discipline';
        }
        if ((macroOf(item) === 'top' || macroOf(item) === '') && BASIC_TEE_RE.test(blob) && !/\bknit\b/.test(blob)) {
            return 'basic tees break classic discipline';
        }
        if (isTopOrLayer && hasLoudPrint(blob)) return 'loud prints break classic discipline';
        if (isTopOrLayer && UNCLASSIC_FIT_RE.test(blob)) return 'cropped/boxy fits break classic proportions';
        if (isTopOrLayer && SOCCER_RE.test(blob)) return 'soccer kits break classic discipline';
        if (isTopOrLayer && MANDARIN_RE.test(blob)) return 'mandarin collars break classic lines';
    }
    // Slot integrity is style-independent: a tee is never pants, in any
    // aesthetic. Catches feed mistags that survive pool mapping.
    if (detectSlotMismatch(item.macroCategory, item.name, item.type) !== null) {
        return 'garment name and outfit slot disagree (mislabeled catalog row)';
    }
    if (style === 'minimalist' && MINIMALIST_NOISE_RE.test(blob)) {
        return 'loud piece breaks minimalist discipline';
    }
    // NOTE: 'pleated' needs its own alternative — \bpleats?\b never matches
    // inside 'pleated' (no boundary before the 'ed').
    if (style === 'minimalist' && /\bpleats?\b|\bpleated\b/.test(blob)) {
        return 'minimalism takes no pleats';
    }
    if (style === 'minimalist' && isBottom && /\b(ripped|distressed|baggy)\b/.test(blob)) {
        return 'ripped/baggy bottoms break minimalist discipline';
    }
    // Refined Casual DNA: knit polos are the standard; the only allowed
    // athletic shoe is the minimal white leather sneaker — chunky/athletic
    // pairs, hoodies, and basic (non-knit) cotton tees belong to the gym.
    if (style === 'casual') {
        if (/\b(chunky|athletic|basketball|running|skate)\b/.test(blob) && /\bsneakers?\b|\btrainers?\b/.test(blob)) {
            return 'refined casual takes only minimal leather sneakers';
        }
        if (/\bhoodies?|hooded\b/.test(blob)) return 'refined casual never wears hoodies';
        if (macroOf(item) !== 'shoes' && BASIC_TEE_RE.test(blob) && !/\bknit\b/.test(blob)) {
            return 'refined casual wears knit polos, not basic cotton tees';
        }
    }
    // A plain (non-knit) tee as a top is underdressed for any formal
    // occasion — knit polos and knit tees are exempt.
    if (formal && (macroOf(item) === 'top' || macroOf(item) === '')
        && BASIC_TEE_RE.test(blob) && !/\bknit\b/.test(blob)) {
        return 'basic tees are underdressed for polished occasions';
    }
    // Square toes cheapen everything; exotics read vulgar with classics.
    if (SQUARE_TOE_RE.test(blob)) return 'square toes cheapen any outfit';
    if ((CLASSIC_STYLES.has(style) || formal) && EXOTIC_RE.test(blob)) return 'exotic leathers read vulgar with classics';
    return null;
}

/**
 * True theatre prints (graphic/floral/paisley/souvenir/lettering…).
 * Tailored patterns — checks, stripes, pinstripes, tweed — are explicitly
 * NOT prints, so oxford stripes and Glen checks always survive.
 */
function hasLoudPrint(blob: string): boolean {
    if (!LOUD_PRINT_RE.test(blob)) return false;
    return true;
}

const FORMAL_LAYER_RE = /\b(blazers?|suit\s*jackets?|sport\s*coats?|overcoats?|top\s*coats?|topcoats?|trenchs?|peacoats?|pea\s*coats?|tuxedos?)\b/;
const BOARDROOM_LAYER_RE = /\b(double[\s-]?breasted|suit\s*(jacket|blazer)|cool wool suit)\b/;

function isKnitOnlyLayerBlob(blob: string): boolean {
    if (/\b(blazer|jacket|coat|trench|peacoat|bomber)\b/.test(blob)) return false;
    return /\b(sweater|pullover|jumper|cardigan)\b/.test(blob);
}

/** Work wants tailored shells; date should not clone a boardroom suit. */
export function matchesOccasionSilhouette(
    item: JudgeableItem | null | undefined,
    occasionId: string | null | undefined,
): boolean {
    if (!item) return false;
    const occasion = String(occasionId || '').toLowerCase().trim();
    const blob = blobOf(item);
    const slot = macroOf(item);
    if (occasion === 'work') {
        if (slot === 'outerwear' || slot === '') {
            if (isKnitOnlyLayerBlob(blob)) return false;
            if (slot === 'outerwear') return FORMAL_LAYER_RE.test(blob);
        }
        if ((slot === 'top' || slot === '') && /\bdenim\b/.test(blob) && /\bshirts?\b/.test(blob)) return false;
        if ((slot === 'bottom' || slot === '') && /\bjeans?\b/.test(blob)) return false;
        return true;
    }
    if (occasion === 'date') {
        if ((slot === 'outerwear' || slot === '') && BOARDROOM_LAYER_RE.test(blob)) return false;
        return true;
    }
    if (occasion === 'travel') {
        if ((slot === 'outerwear' || slot === '') && BOARDROOM_LAYER_RE.test(blob)) return false;
        if ((slot === 'outerwear' || slot === '') && /\b(tuxedos?|suit\s*jackets?)\b/.test(blob)) return false;
        return true;
    }
    return true;
}
const DRESS_SHIRT_RE = /\b(oxford|button[\s-]?downs?|dress\s*shirts?)\b/;
const SNEAKER_RE = /\bsneakers?|trainers?\b/;
const BLACK_RE = /\bblack\b/;
const WHITE_RE = /\bwhite\b/;
const STRIPED_LAYER_RE = /\bstripes?|pinstripes?\b/;

function blobOf(item: JudgeableItem): string {
    // Color field included: catalog names often omit the color
    // ("Linen suit trousers" in black) while the color column has it.
    return `${item.name || ''} ${item.type || ''} ${item.brand || ''} ${item.description || ''} ${item.color || ''}`.toLowerCase();
}

function macroOf(item: JudgeableItem): string {
    return String(item.macroCategory || '').toLowerCase();
}

/** Catalog rows often arrive as footwear / shoe, not the canonical `shoes`. */
function isShoeish(item: JudgeableItem, blob: string): boolean {
    const m = macroOf(item);
    if (m === 'shoes' || m === 'shoe' || m === 'footwear' || m === 'foot_wear') return true;
    return /\b(loafers?|shoes?|moccasins?|derb(?:y|ies)|oxfords?|sneakers?|trainers?|boots?)\b/.test(blob);
}

/**
 * Outfit-level coherence vetoes — rules that only make sense across pieces
 * (a blazer is fine, sneakers are fine, TOGETHER they are not). Returns a
 * reason or null. Scoped to formal occasions + classic styles, where the
 * transcripts are absolute; streetwear stays free.
 */
export function rejectOutfitIncoherence(
    items: JudgeableItem[] | null | undefined,
    styleId: string | null | undefined,
    occasionId: string | null | undefined,
): string | null {
    if (!items || items.length === 0) return null;
    const style = String(styleId || '').toLowerCase().replace(/[\s-]+/g, '_');
    const formal = occasionFormality(occasionId) === 'formal';
    const classic = CLASSIC_STYLES.has(style);
    if (!formal && !classic) return null;

    const hasFormalLayer = items.some(
        (it) => (macroOf(it) === 'outerwear' || macroOf(it) === '') && FORMAL_LAYER_RE.test(blobOf(it)),
    );
    const hasDressShirt = items.some(
        (it) => (macroOf(it) === 'top' || macroOf(it) === '') && DRESS_SHIRT_RE.test(blobOf(it)),
    );
    const hasSneakers = items.some(
        (it) => (macroOf(it) === 'shoes' || macroOf(it) === '') && SNEAKER_RE.test(blobOf(it)),
    );
    // Minimalist white sneakers live with KNITWEAR (Smart Low) — only the
    // tailored top pairing is banned. Knit polos never trigger dressShirt.
    if (hasSneakers && (hasFormalLayer || hasDressShirt)) {
        return 'sneakers never pair with tailoring or dress shirts';
    }

    const hasBasicTee = items.some((it) => {
        if (macroOf(it) !== 'top' && macroOf(it) !== '') return false;
        const b = blobOf(it);
        return BASIC_TEE_RE.test(b) && !/\bknit\b/.test(b);
    });
    if (hasBasicTee && hasFormalLayer) {
        return 'basic tees never go under tailoring (knit polos do)';
    }

    // Piqué polos are sporty — fine with chinos, never under a blazer.
    // NOTE: no trailing \b — it is ASCII-only, so é never forms a boundary
    // and /\bpiqué\b/ silently never matches. The stem alone is safe
    // ('unique' contains no 'piqu' sequence).
    const hasPiqueTop = items.some((it) => {
        if (macroOf(it) !== 'top' && macroOf(it) !== '') return false;
        return /\bpiqu[eé]/i.test(blobOf(it));
    });
    if (hasPiqueTop && hasFormalLayer) {
        return 'piqué polos are sporty — knit polos go under tailoring';
    }

    // Classic short-sleeve shirts read cheap — except Hawaiian/camp-collar
    // linen, the one sanctioned exception.
    const hasCheapShortSleeve = items.some((it) => {
        if (macroOf(it) !== 'top' && macroOf(it) !== '') return false;
        const b = blobOf(it);
        return SHORT_SLEEVE_RE.test(b)
            && (/\bshirts?\b/.test(b) || BASIC_TEE_RE.test(b))
            && !/\bhawaiian|camp[\s-]?collar|cuban\b/.test(b);
    });
    if (hasCheapShortSleeve && (classic || formal)) {
        return 'classic short-sleeve shirts read cheap (Hawaiian camp excepted)';
    }

    // Suede belongs to Smart Casual, never to a classic suit (leather
    // oxfords/derbies or leather loafers instead).
    const hasSuedeShoes = items.some(
        (it) => (macroOf(it) === 'shoes' || macroOf(it) === '') && /\bsuede\b/.test(blobOf(it)),
    );
    const hasSuitPiece = items.some((it) => /\bsuits?\b/.test(blobOf(it)));
    if (hasSuedeShoes && hasSuitPiece) {
        return 'suede stays in smart casual — suits take leather';
    }

    // A white shirt and jeans are things from different worlds.
    const hasWhiteShirt = items.some((it) => {
        if (macroOf(it) !== 'top' && macroOf(it) !== '') return false;
        const b = blobOf(it);
        return /\bwhite\b/.test(b) && /\bshirts?|oxford\b/.test(b);
    });
    const hasJeans = items.some((it) => {
        if (macroOf(it) !== 'bottom' && macroOf(it) !== '') return false;
        return /\bjeans?|denim\b/.test(blobOf(it));
    });
    if (hasWhiteShirt && hasJeans) {
        return 'a white shirt never goes with jeans';
    }

    // Work means business: denim bottoms never belong in workish decks
    // (classic ranking already demotes them; this is the hard veto).
    const workish = WORKISH_RE.test(String(occasionId || '').toLowerCase());
    if ((classic || workish) && hasJeans) {
        return 'denim never belongs in classic or work decks';
    }

    // Black cloth needs white somewhere (sandwich exception); classic black
    // shoes (oxfords at funerals) are never the problem. Minimalist evening
    // owns head-to-toe black, so it is exempt.
    if (classic || (formal && style !== 'minimalist')) {
        const hasBlackCloth = items.some((it) => {
            const m = macroOf(it);
            return (m === 'top' || m === 'outerwear' || m === 'bottom' || m === '') && BLACK_RE.test(blobOf(it));
        });
        const hasWhite = items.some((it) => WHITE_RE.test(blobOf(it)));
        if (hasBlackCloth && !hasWhite) return 'black pairs with nothing but white';
    }

    // Striped suit jackets only go with their matching trousers. A striped
    // layer over foreign bottoms is vetoed — unless the bottom is also
    // striped in the SAME color (presumed suit, the only legal pairing).
    const stripedLayer = items.find((it) => {
        const m = macroOf(it);
        const looksOuter = m === 'outerwear' || (m === '' && /\b(jacket|coat|blazer|sport\s*coat)\b/.test(blobOf(it)));
        return looksOuter && STRIPED_LAYER_RE.test(blobOf(it));
    });
    if (stripedLayer && (classic || formal)) {
        const bottoms = items.filter((it) => macroOf(it) === 'bottom');
        const layerColor = String((stripedLayer as JudgeableItem).color || '').toLowerCase().trim();
        const matched = bottoms.some((b) => {
            const bBlob = blobOf(b);
            if (!STRIPED_LAYER_RE.test(bBlob)) return false;
            const bColor = String(b.color || '').toLowerCase().trim();
            return layerColor !== '' && bColor !== '' && layerColor === bColor;
        });
        if (!matched) return 'striped jackets only go with their matching trousers';
    }

    return null;
}

export type SlotMacro = 'top' | 'outerwear' | 'bottom' | 'shoes';

const TOP_WORDS_RE = /\bt-?shirts?\b|\btees?\b|\bpolos?\b|\bblouses?\b|\btanks?\b|\bhenleys?\b|\bshirts?\b|\boxfords?\b|\bbutton[\s-]?downs?\b/;
const OUTER_WORDS_RE = /\bjackets?|coats?|blazers?|hoodies?|cardigans?|sweaters?|pullovers?|puffers?|bombers?|vests?|outerwears?|trenchs?|peacoats?|sweatshirts?|knitwears?|jumpers?|fleece|overshirts?|shackets?|total[\s-]?look\b/;
const BOTTOM_WORDS_RE = /\bpants?|trousers?|jeans?|bottoms?|shorts|bermudas?|skirts?|chinos?|slacks?|joggers?|trunks?|swim/;
const SHOE_WORDS_RE = /\bshoes?|sneakers?|boots?|loafers?|sandals?|heels?|trainers?|derbys?|derbies|moccasins?|drivers?|mules?\b/;
// Underwear must never surface as visible daywear in any slot.
export const UNDERWEAR_RE = /\bboxers?|briefs?|underwears?|panties?|thongs?|lingerie|boxer[\s-]?shorts?|nightwear|pyjamas?|pajamas?\b/;
export const ACCESSORY_PROP_RE = /\b(sunglasses?|eyewear|glasses|card[\s-]?holders?|wallets?|belts?|bags?|totes?|scar(?:f|ves)|hats?|caps?|beanies?|watches?|jewelry|earrings?|necklaces?|ties?|neckties?)\b/;
// Adjective form is "short-sleeved" — `sleeves?` does not match it.
export const SHORT_SLEEVE_RE = /\bshort[\s-]?sleeve(?:s|d)?\b/;
export const STRIP_SHORT_SLEEVE_RE = /\bshort[\s-]?sleeve(?:s|d)?\b/gi;

/** True for boxers/briefs/underwear — never valid as visible outfit pieces. */
export function isUnderwearItem(item: { name?: string | null; type?: string | null; brand?: string | null; description?: string | null } | null | undefined): boolean {
    if (!item) return false;
    return UNDERWEAR_RE.test(`${item.name || ''} ${item.type || ''} ${item.brand || ''} ${(item as any).description || ''}`.toLowerCase());
}

/** Belts, sunglasses, wallets — never fill a clothing slot. */
export function isAccessoryPropItem(item: { name?: string | null; type?: string | null; brand?: string | null; description?: string | null; macroCategory?: string | null } | null | undefined): boolean {
    if (!item) return false;
    const macro = String(item.macroCategory || '').toLowerCase();
    if (macro === 'accessory' || macro === 'accessories') return true;
    return ACCESSORY_PROP_RE.test(`${item.name || ''} ${item.type || ''} ${item.brand || ''} ${(item as any).description || ''}`.toLowerCase());
}

/**
 * detectSlotMismatch — feed rows are mistagged at scale (a T-shirt stored
 * as bottoms, rendered as PANTS). Returns the correct slot when the NAME
 * definitively belongs to another family, else null.
 *
 * Conservative by construction: reports ONLY when the claimed slot has NO
 * supporting word AND another family positively matches (boot-cut jeans
 * keep 'boot' + 'jeans', so no false hit; vague names like 'Blue summer
 * thing' match nothing and pass through).
 */
export function detectSlotMismatch(
    macro: string | null | undefined,
    name: string | null | undefined,
    type?: string | null | undefined,
): SlotMacro | null {
    // Canonicalize first: feed uses tops/bottoms/lower_body/footwear etc.
    // Without this "bottoms" !== "bottom" skipped repair entirely (live bug:
    // T-shirt stored as bottoms rendered as PANTS).
    const raw = String(macro || '').toLowerCase().trim();
    const alias: Record<string, SlotMacro> = {
        top: 'top', tops: 'top', upper_body: 'top', 'upper-body': 'top', upperbody: 'top',
        dress: 'top', dresses: 'top',
        bottom: 'bottom', bottoms: 'bottom', lower_body: 'bottom', 'lower-body': 'bottom', lowerbody: 'bottom',
        pant: 'bottom', pants: 'bottom',
        outerwear: 'outerwear', outer: 'outerwear',
        shoe: 'shoes', shoes: 'shoes', footwear: 'shoes',
    };
    const claimed = alias[raw];
    if (!claimed) {
        return null;
    }
    // NAME ONLY — type/category carry the same feed mistag as the macro and
    // would self-confirm it (a 'bottoms'-typed T-shirt must still read top).
    const blob = String(name || '').toLowerCase();
    const blobSansSleeves = blob.replace(STRIP_SHORT_SLEEVE_RE, ' ');
    const hasTop = TOP_WORDS_RE.test(blob);
    const hasOuter = OUTER_WORDS_RE.test(blob);
    // Strip "short-sleeved" first: `\bshorts?` / `\bshorts` otherwise
    // treat shirts as bottoms (screenshot: black short-sleeved shirt as PANTS).
    const hasBottom = BOTTOM_WORDS_RE.test(blobSansSleeves);
    const hasShoe = SHOE_WORDS_RE.test(blob);
    // Short-sleeve knits/shirts are base tops, never layers or pants.
    // "Wool-blend short-sleeve knit sweater" used to fall through to
    // outerwear because OUTER_WORDS includes "sweater".
    if (SHORT_SLEEVE_RE.test(blob)) {
        if (claimed === 'top') return null;
        return 'top';
    }
    // A jacket/blazer stored as "top" is still a layer — otherwise the grid
    // clones it into both TOP tiles (screenshot: two identical suit jackets).
    if ((claimed === 'top' || claimed === 'bottom') && hasOuter && !hasTop && /\b(jacket|coat|blazer|trench|peacoat|bomber|puffer)\b/.test(blob)) {
        return 'outerwear';
    }
    const supportsClaim =
        (claimed === 'top' && hasTop)
        || (claimed === 'outerwear' && hasOuter)
        || (claimed === 'bottom' && hasBottom)
        || (claimed === 'shoes' && hasShoe);
    if (supportsClaim) return null;
    // Most distinctive family first (a 'shoe bag' keeps shoes; everything
    // else falls through to the genuine match).
    if (hasShoe) return 'shoes';
    if (hasBottom) return 'bottom';
    if (hasOuter) return 'outerwear';
    if (hasTop) return 'top';
    return null;
}

/** True when the outfit survives the judge (every item unrejected). */
export function passesStyleJudge(
    items: JudgeableItem[],
    styleId: string | null | undefined,
    occasionId: string | null | undefined,
    weather?: ClimateWeather | null,
): boolean {
    if (!items || items.length === 0) return false;
    if (!items.every((it) => rejectForStyleOccasion(it, styleId, occasionId) === null)) return false;
    if (!items.every((it) => rejectForClimate(it, weather) === null)) return false;
    if (rejectOutfitClimate(items, weather) !== null) return false;
    return rejectOutfitIncoherence(items, styleId, occasionId) === null;
}

const VIEW_PREFIX_RE = /^(front|side|back|close[\s-]?up|detail)\s+view\s+of\s+(?:(a|an|the)\s+)?/i;
const IMAGE_TAIL_RE = /\s*[-–:]\s*(image|picture|photo)\s*\d+\s*$/i;

/**
 * Turn feed photography captions into product names:
 * "Front view of a white long-sleeve ... - Image 0" → "White Long-Sleeve ..."
 * "Zara RELAXED FIT STRIPED JACQUARD T-SHIRT - Blue - Image 0" → "Zara Relaxed Fit ..."
 * Only ALL-CAPS words are re-cased (mixed-case names pass through untouched),
 * so "100% algodón" and "T-shirt" styling is preserved.
 */
export function humanizeProductName(raw: string | null | undefined): string {
    if (!raw) return '';
    let out = String(raw).replace(/\s+/g, ' ').trim();
    out = out.replace(VIEW_PREFIX_RE, '');
    out = out.replace(IMAGE_TAIL_RE, '');
    out = out
        .split(' ')
        .map((word) => {
            const letters = word.replace(/[^A-Za-zÀ-ÿ]/g, '');
            if (letters.length > 1 && letters === letters.toUpperCase()) {
                return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
            }
            return word;
        })
        .join(' ')
        .trim();
    // Sentence case to match catalog style ("Slim fit ...").
    if (out) out = out.charAt(0).toUpperCase() + out.slice(1);
    return out;
}
