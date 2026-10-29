/**
 * occasionRules — taste enforcement for the AI stylist.
 *
 * The scorer ranks STYLE; nothing ranked OCCASION, so "minimalist date
 * night" decks contained jacquard tees, espadrilles, boat shoes and quilted
 * jackets. This module is the missing judge, used in two places:
 *
 *   1. Steering (normalizeTo4Slots): formal-first pick predicates with
 *      selecting only candidates that meet the request.
 *   2. Veto (generateOutfits deck filter): drop cards carrying rejected
 *      pieces; an empty accepted set stays empty.
 *
 * Plus humanizeProductName: feed rows arrive as "Front view of a white
 * long-sleeve ... - Image 0" / ALL CAPS and that leaks onto cards.
 * Pure + deterministic. Covered by __tests__/e2e/aiOccasion.e2e.test.ts.
 */


/** Normalize multilingual catalog evidence before applying the same rules. */
export function normalizeGarmentText(value: string): string {
    const aliases: Array<[RegExp, string]> = [
        [/свитшот/gi, 'sweatshirt'], [/худи/gi, 'hoodie'], [/логотип|monologo/gi, 'logo'],
        [/лофер[а-я]*/gi, 'loafers'], [/кроссов[а-я]*|кеды/gi, 'sneakers'],
        [/рубаш[а-я]*/gi, 'shirt'], [/брюк[а-я]*/gi, 'trousers'],
        [/пиджак[а-я]*/gi, 'blazer'], [/смокинг[а-я]*/gi, 'tuxedo'],
        [/замш[а-я]*/gi, 'suede'], [/коричнев[а-я]*/gi, 'brown'],
        [/черн[а-я]*|чёрн[а-я]*/gi, 'black'], [/бел[а-я]*/gi, 'white'],
        [/футбол[а-я]*/gi, 't-shirt'], [/спортивн[а-я]*/gi, 'athletic'],
        [/трениров[а-я]*/gi, 'training'], [/шорт[а-я]*/gi, 'shorts'],
    ];
    return aliases.reduce((text, [pattern, replacement]) => text.replace(pattern, ` ${replacement} `), value.toLowerCase());
}

export function isWorkoutOccasion(value?: string | null): boolean {
    return /\b(gym|workout|fitness|training|sport|sports)\b|трениров|спортзал|фитнес/i.test(value || '');
}

/** Individual admission rule shared by candidate selection and the final judge. */
export function rejectForActivity(item: JudgeableItem, occasion?: string | null): string | null {
    if (!isWorkoutOccasion(occasion)) return null;
    const slot = macroOf(item);
    const text = blobOf(item);
    if (['top', 'bottom', 'shoes'].includes(slot) && !/\b(athletic|training|running|workout|performance|sportswear|activewear|gym)\b/.test(text)) {
        return `workout requires verified athletic ${slot}`;
    }
    if (/\b(dress[ -]shirt|formal[ -]shirt|blazer|loafers?|suede)\b/.test(text)) return 'dress clothing is not workout clothing';
    return null;
}

function activityReason(items: JudgeableItem[], occasion: string): string | null {
    const text = occasion.toLowerCase();
    if (isWorkoutOccasion(text)) {
        for (const slot of ['top', 'bottom', 'shoes']) {
            const candidates = items.filter(it => macroOf(it) === slot);
            if (!candidates.length || candidates.some(it => !/\b(athletic|training|running|workout|performance|sportswear|activewear|gym)\b/.test(blobOf(it)))) {
                return `workout requires verified athletic ${slot}`;
            }
        }
        for (const item of items) {
            const reason = rejectForActivity(item, occasion);
            if (reason) return reason;
        }
    }
    if (/black[ -]?tie|блэк[ -]?тай|смокинг/.test(text)) {
        const has = (slot: string, pattern: RegExp) => items.some(it => macroOf(it) === slot && pattern.test(blobOf(it)));
        // Evening dress codes require actual evening garments.
        if (!has('outerwear', /tuxedo|dinner jacket/) || !has('bottom', /tuxedo|dinner suit|black trousers|black pants|formal trousers/) || !has('top', /tuxedo|dress shirt|formal shirt|white shirt/)) return 'black tie requires a tuxedo and formal shirt';
        if (!has('shoes', /black/) || !has('shoes', /patent|oxford|opera pump/) || items.some(it => macroOf(it) === 'shoes' && /brown|suede|loafer/.test(blobOf(it)))) return 'black tie requires black formal evening shoes';
    }
    return null;
}

/** Admit scarce required pieces before ranking, rather than discovering them after truncation. */
export function rejectForRequestIntent(item: JudgeableItem, request: string, weather?: ClimateWeather | null): string | null {
    const slot = macroOf(item), text = blobOf(item);
    if (slot === 'outerwear') {
        // Keep the fourth slot, choosing a different garment when a jacket
        // or coat is unwanted. A cardigan, vest or overshirt can still qualify.
        const noJacket = /\b(no|without|skip|exclude)\s+(?:a\s+|any\s+)?jackets?\b|без\s+(?:куртк|пиджак)/i.test(request);
        const noCoat = /\b(no|without|skip|exclude)\s+(?:a\s+|any\s+)?coats?\b|без\s+пальто/i.test(request);
        const softLayer = /\b(no|without|skip|exclude)\s+(?:a\s+|any\s+)?(?:layers?|outerwear)\b/i.test(request);
        if (noJacket && /\b(jackets?|blazers?|shackets?)\b/.test(text)) return 'requested a non-jacket layer';
        if (noCoat && /\b(coats?|overcoats?|topcoats?|parkas?|peacoats?)\b/.test(text)) return 'requested a non-coat layer';
        if (softLayer && !/\b(cardigans?|vests?|overshirts?)\b/.test(text)) return 'requested a light alternative layer';
    }
    if (weather?.temp != null && weather.temp < 5) {
        if (slot === 'outerwear' && !/\b(puffer|parka|down|insulated|overcoat|topcoat|winter coat|shearling|quilted)\b/.test(text)) return 'cold request requires insulated coat';
        if (slot === 'shoes' && !/\b(boots?|chelsea|winter|insulated|snow)\b/.test(text)) return 'cold request requires winter boots';
    }
    if (/black[ -]?tie|блэк[ -]?тай|смокинг/i.test(request)) {
        if (slot === 'outerwear' && !/tuxedo|dinner jacket/.test(text)) return 'black tie requires dinner jacket';
        if (slot === 'bottom' && !/tuxedo|dinner suit|black trousers|black pants|formal trousers/.test(text)) return 'black tie requires evening trousers';
        if (slot === 'top' && !/tuxedo|dress shirt|formal shirt|white shirt/.test(text)) return 'black tie requires formal shirt';
        if (slot === 'shoes' && (!/black/.test(text) || !/patent|oxford|opera pump/.test(text) || /brown|suede|loafer/.test(text))) return 'black tie requires black evening shoes';
    }
    return null;
}

export type OccasionFormality = 'formal' | 'casual';

/** work / date / formal demand evening polish; everything else is relaxed. */
export function occasionFormality(occasionId: string | null | undefined): OccasionFormality {
    const key = String(occasionId || '').toLowerCase().trim();
    if (/\b(work|office|professional|business|interview|date|formal|black[ -]?tie)\b|офис|работ|делов|свидани/.test(key)) return 'formal';
    return 'casual';
}

// Cyrillic alternatives must sit OUTSIDE \b groups: \b is ASCII-only, so a
// boundary before 'к' never matches (same trap as the é note at Piqué).
const FORMAL_SHOE_RE = /\b(sneakers?|trainers?|athletic|basketball|skate|sandals?|slides?|flip[\s-]?flops?|espadrilles?|boat\b|slippers?|clogs?|runners?|joggers?|mules?)\b|кроссовки|кеды|сандалии|шлёпанцы|шлепанцы|эспадрильи/;
// Driving moccasins sit a full formality tier below loafers: unstructured
// soft sole, tie laces, pebbled nubs. Riviera-casual, never evening wear —
// and never with tailoring. (A loafer by contrast is structured:
// penny/bit/tassel go with suits.)
const DRIVER_SHOE_RE = /\b(moccasins?|drivers?|driving|gommino|loafer[s]?\s+with\s+(?:tie|laces?|nubs?)|nubbed|rubber[\s-]?nubs?)\b/;
/** A stitched moccasin toe is a construction detail, not evidence of a driving sole. */
export function isDrivingFootwear(item: JudgeableItem): boolean {
    let text = normalizeGarmentText(`${item.name || ''} ${item.type || ''} ${item.description || ''}`);
    const structuredLoafer = /\b(penny|tassel|bit|structured)\b/.test(text) && /\bloafer/.test(text)
        && /\b(blake|leather outsoles?|leather soles?|structured)\b/.test(text);
    if (structuredLoafer) text = text.replace(/\bmoccasin[ -]+toe\b/g, 'stitched toe');
    return DRIVER_SHOE_RE.test(text);
}
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
// 'jogging' (e.g. "Basic Jogging Trousers") must match alongside 'joggers':
// the judge only rejects what the regex sees, and a jogger pants named by
// fabric/action slipped old-money decks while 'joggers?' never matched it.
const SLOPPY_BOTTOM_RE = /\b(cargo|ripped|distressed|sweatpants?|joggers?|jogging|basketball|board\s*shorts?|drawstrings?|elasticated|\belastic(?:ated)?[\s-]?waist(?:band)?|co-?ords?|tracksuits?|track[\s-]?(?:pants?|trousers?|suits?)|balloon[\s-]?fit)\b/;
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
const HEAVY_WINTER_LAYER_RE = /\b(puffer|parka|ski\b|down\s+jacket|overcoats?|topcoats?|peacoats?|duffle|greatcoats?)\b/;
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
    if (hot && slot !== 'shoes'
        && /\b(wool|merino|alpaca|cashmere|mohair|angora|lambswool|fleece|shearling|chunky|heavyweight|insulated|padded)\b/.test(blob)
        && !/\btropical\s+wool\b/.test(blob)) {
        return `${temp}°C requires ${slot === 'outerwear' ? 'a light layer' : 'light clothing'}, not warm winter fabric`;
    }
    if (hot && slot === 'outerwear'
        && (/\b(leather|suede|corduroy)\b/.test(blob)
            || !/\b(lightweight|light[ -]weight|fine[ -]gauge|open[ -]knit|summer|linen|cotton|breathable|unlined)\b/.test(blob))) {
        return `${temp}°C requires a documented lightweight layer`;
    }
    if (hot && slot === 'outerwear'
        && /\b(sweaters?|pullovers?|hoodies?|sweatshirts?)\b|funnel[ -]neck/.test(blob)
        && !/\b(lightweight|light[ -]weight|fine[ -]gauge|open[ -]knit|summer|linen)\b/.test(blob)) {
        return `${temp}°C requires a documented lightweight layer`;
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
    // Below freezing a blazer/knit alone is underdressed (observed: sub-zero
    // decks with no insulated coat or winter boots). Require both.
    if (weather.temp < 5) {
        const hasInsulated = items.some((it) => macroOf(it) === 'outerwear'
            && /\b(puffer|parka|down|insulated|overcoat|topcoat|winter coat|shearling|quilted)\b/.test(blobOf(it)));
        if (!hasInsulated) return `${weather.temp}°C requires an insulated coat`;
        const hasWinterBoots = items.some((it) => macroOf(it) === 'shoes'
            && /\b(boots?|chelsea|winter|insulated|snow)\b/.test(blobOf(it)));
        if (!hasWinterBoots) return `${weather.temp}°C requires winter boots`;
    }
    return null;
}

export interface JudgeableItem {
    id?: string | number;
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
    const refinedDate = refinedDateProductReason(item,styleId,occasionId);
    if (refinedDate) return refinedDate;
    const blob = normalizeGarmentText(`${item.name || ''} ${item.type || ''} ${item.description || ''} ${item.color || ''}`);
    // Underwear is never visible daywear, in any style or occasion.
    if (UNDERWEAR_RE.test(blob)) return 'underwear must never surface as visible clothing';
    if (/\b(iphone|phone[\s-]?cases?|airpods?)\b/.test(blob)) return 'tech accessories are not outfit pieces';
    const style = isWorkoutOccasion(occasionId) ? 'sport' : String(styleId || '').toLowerCase().replace(/[\s-]+/g, '_');
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
        if (isDrivingFootwear(item)) return 'driving moccasins are a tier below evening footwear';
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
        if (isDrivingFootwear(item)) return 'driving moccasins never pair with classic tailoring';
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
        if (isBottom && /\b(ripped|distressed|sweatpants?|joggers?|jogging|track|tracksuits?|velour|baggy|cargo|balloon[\s-]?fit)\b/.test(blob)) {
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
    if (style === 'minimalist' && isBottom && /\b(ripped|distressed|joggers?|jogging|baggy|track|tracksuits?|balloon[\s-]?fit)\b/.test(blob)) {
        return 'ripped/baggy bottoms break minimalist discipline';
    }
    // Semi-classic DNA: the scorer penalises track/baggy/cargo/oversized
    // bottoms and chunky footwear, but the judge had no branch — so those
    // pieces survived into rendered cards (verified: "Balloon Fit Trousers",
    // "Technical Tracksuit" bottoms, "Chunky Moc Toe Shoes").
    if (style === 'semi_classic') {
        if (isShoeish(item, blob) && /\bchunky\b/.test(blob)) return 'chunky soles break semi-classic polish';
        if (isBottom && /\b(ripped|distressed|sweatpants?|joggers?|jogging|tracksuits?|track[\s-]?(?:pants?|trousers?)|baggy|cargo|oversized|balloon[\s-]?fit)\b/.test(blob)) {
            return 'sloppy/track bottoms break semi-classic discipline';
        }
    }
    // Refined Casual DNA: knit polos are the standard; the only allowed
    // athletic shoe is the minimal white leather sneaker — chunky/athletic
    // pairs, hoodies, and basic (non-knit) cotton tees belong to the gym.
    if (style === 'casual') {
        if (/\b(chunky|athletic|basketball|running|skate|fabric|mesh)\b/.test(blob) && SNEAKER_RE.test(blob)) {
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
    if (occasion === 'sport') {
        // Sport wants trainers; loafers/derbies/Chelsea fail the silhouette
        // (the shoes pick tiers then prefer an athletic shoe when the pool
        // has one, and fall back to the old behaviour when it does not).
        if (slot === 'shoes') return SPORT_SHOE_RE.test(blob);
        return true;
    }
    return true;
}
const DRESS_SHIRT_RE = /\b(oxford|button[\s-]?downs?|dress\s*shirts?)\b/;
const SNEAKER_RE = /\b(?:sneakers?|trainers?)\b|кроссовки|кеды|кеда/;
const BLACK_RE = /\bblack\b/;
const WHITE_RE = /\bwhite\b/;
const STRIPED_LAYER_RE = /\bstripes?|pinstripes?\b/;

function blobOf(item: JudgeableItem): string {
    // Brand names are not garment evidence (Thursday Boot Co. also sells loafers).
    // Color field included: catalog names often omit the color
    // ("Linen suit trousers" in black) while the color column has it.
    return normalizeGarmentText(`${item.name || ''} ${item.type || ''} ${item.description || ''} ${item.color || ''}`);
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

/** Tailored trousers: marked by formality/fabric words, never by shape
 * alone — "wool straight-leg pants" and "wool jogger fit pants" stay
 * casual, "Wool Striped Suit Trousers" and "flannel pants" do not. */
const TAILORED_PANTS_RE =
    /\b(?:suit|tailored|pinstriped?|flannel|total[\s-]?look)\b[\w\s-]{0,30}\b(?:trousers?|pants?|slacks?)\b|\b(?:trousers?|pants?|slacks?)\b[\w\s-]{0,30}\b(?:suit|tailored|pinstriped?|flannel|total[\s-]?look)\b/;
/** Outer-coat class: blazer, overcoat, puffer, parka, vest… (EN+RU). */
const OUTER_LAYER_CLASS_RE =
    /\b(?:blazers?|suit\s*jackets?|sport\s*coats?|overcoats?|top\s*coats?|trench(?:es)?|peacoats?|puffers?|parkas?|bombers?|gilets?|vests?|windbreakers?|coats?|jackets?|outerwears?)\b|пальто|куртк|жилет|жакет|парк/i;
/** Pullover-knit class: sweater, cardigan, knit polo, mock neck… (EN+RU). */
const PULLOVER_KNIT_CLASS_RE =
    /\b(?:sweaters?|pullovers?|jumpers?|cardigans?|knit(?:ted)?|mock[\s-]?necks?|polos?)\b|свитт?ер|пуловер|кардиган|водолазк/i;
/** Athletic-sole footwear: real sneakers plus hybrid cupsole dress shoes. */
const ATHLETIC_SOLE_RE = /\b(?:sneakers?|trainers?|hybrids?)\b|кроссовки|кеды|кеда/;
/** Sport-occasion shoe silhouette: trainers only. */
const SPORT_SHOE_RE = /\b(?:sneakers?|trainers?|running|training|athletic|gym|workout)s?\b|кроссовки|кеды/;

/** Neutrals and muted colors support the palette; three vivid hues compete. */
export function outfitPaletteReason(items: JudgeableItem[]): string | null {
    const vivid = new Set<string>();
    for (const item of items) {
        const color = normalizeGarmentText(String(item.color || ''));
        if (/\b(navy|olive|khaki|sage|muted|dusty|pastel|pale|dark|burgundy|maroon|beige|cream|brown|grey|gray|black|white|neutral)\b/.test(color)) continue;
        for (const hue of ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'pink']) {
            if (new RegExp(`\\b${hue}\\b`).test(color)) vivid.add(hue);
        }
    }
    return vivid.size >= 3 ? 'too many competing vivid colors' : null;
}

/**
 * Outfit-level coherence vetoes — rules that only make sense across pieces
 * (a blazer is fine, sneakers are fine, TOGETHER they are not). Returns a
 * reason or null. Scoped to formal occasions + classic styles, where the
 * transcripts are absolute; streetwear stays free. The first block of
 * universal vetoes runs for EVERY style × occasion — formality-tier clashes
 * and duplicate-layer nonsense are not aesthetic preferences.
 */
export function rejectOutfitIncoherence(
    items: JudgeableItem[] | null | undefined,
    styleId: string | null | undefined,
    occasionId: string | undefined | null,
): string | null {
    if (!items || items.length === 0) return null;
    const style = isWorkoutOccasion(occasionId) ? 'sport' : String(styleId || '').toLowerCase().replace(/[\s-]+/g, '_');
    const formal = occasionFormality(occasionId) === 'formal';
    const classic = CLASSIC_STYLES.has(style);

    const palette = outfitPaletteReason(items);
    if (palette) return palette;

    // ── Universal vetoes (all styles, all occasions) ───────────────────
    // Athletic-sole footwear (sneakers, hybrid cupsole derbies) never pairs
    // with tailored trousers — "no sneakers with suits" is universal.
    const hasAthleticSole = items.some(
        (it) => (macroOf(it) === 'shoes' || macroOf(it) === '') && ATHLETIC_SOLE_RE.test(blobOf(it)),
    );
    const hasTailoredPants = items.some(
        (it) => (macroOf(it) === 'bottom' || macroOf(it) === '') && TAILORED_PANTS_RE.test(blobOf(it)),
    );
    if (hasAthleticSole && hasTailoredPants) {
        return 'sneakers never pair with tailored trousers';
    }
    // Two outer coats or two pullover knits are duplicates, not layering.
    // (Blazer + sweater passes: different classes.)
    const tops = items.filter((it) => {
        const m = macroOf(it);
        return m === 'top' || m === 'outerwear' || m === '';
    });
    for (let i = 0; i < tops.length; i++) {
        for (let j = i + 1; j < tops.length; j++) {
            const a = blobOf(tops[i]);
            const b = blobOf(tops[j]);
            if (OUTER_LAYER_CLASS_RE.test(a) && OUTER_LAYER_CLASS_RE.test(b)) {
                return 'two outer coats never layer together';
            }
            const openCardigan = (item: JudgeableItem, text: string) => macroOf(item) === 'outerwear'
                && /\bcardigans?\b/.test(text) && !/\bpullover\b/.test(text);
            if (PULLOVER_KNIT_CLASS_RE.test(a) && PULLOVER_KNIT_CLASS_RE.test(b)
                && !openCardigan(tops[i], a) && !openCardigan(tops[j], b)) {
                return 'two pullover knits never layer together';
            }
        }
    }

    // Accessory props (towels, key rings, bags…) never occupy a garment slot.
    if (
        items.some(
            (it) => (macroOf(it) === 'top' || macroOf(it) === 'bottom' || macroOf(it) === 'outerwear') && isAccessoryPropItem(it as JudgeableItem),
        )
    ) {
        return 'accessory prop never fills a garment slot';
    }

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
const BOTTOM_WORDS_RE = /\bpants?|trousers?|jeans?|bottoms?|shorts|bermudas?|skirts?|chinos?|slacks?|joggers?|jogging|trunks?|swim/;
const SHOE_WORDS_RE = /\bshoes?|sneakers?|boots?|loafers?|sandals?|heels?|trainers?|derbys?|derbies|moccasins?|drivers?|mules?|clogs?|espadrilles?\b|кроссовки|кеды|туфли|ботинки|лоферы|сапоги|мокасины|сандалии/;
// Underwear must never surface as visible daywear in any slot.
export const UNDERWEAR_RE = /\bboxers?|briefs?|underwears?|panties?|thongs?|lingerie|boxer[\s-]?shorts?|nightwear|pyjamas?|pajamas?\b/;
export const ACCESSORY_PROP_RE = /\b(sunglasses?|eyewear|glasses|card[\s-]?holders?|wallets?|belts?|bags?|totes?|scar(?:f|ves)|hats?|caps?|beanies?|watch(?:es)?|jewelry|earrings?|necklaces?|ties?|neckties?|key[\s-]?rings?|keyrings?|towels?)\b|брелок|полотен/;
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

/** First judge failure reason (item-level style/occasion, climate, or
 * outfit coherence), or null when the outfit passes. Diagnostic twin of
 * passesStyleJudge — keep the two in sync. */
export function styleJudgeReason(
    items: JudgeableItem[],
    styleId: string | null | undefined,
    occasionId: string | null | undefined,
    weather?: ClimateWeather | null,
    lockedItemId?: string | null,
): string | null {
    if (!items || items.length === 0) return 'empty outfit';
    const activity = activityReason(items, occasionId || '');
    if (activity) return activity;
    if (isWorkoutOccasion(occasionId)) styleId = 'sport';
    for (const it of items) {
        // The locked piece overrides aesthetic taste, never dress code, activity or weather.
        const r = rejectForStyleOccasion(it, lockedItemId && String(it.id) === lockedItemId ? 'casual' : styleId, occasionId);
        if (r !== null) return `item "${String((it as any)?.name || '?').slice(0, 40)}": ${r}`;
    }
    for (const it of items) {
        const intent = rejectForRequestIntent(it, occasionId || '', weather);
        if (intent) return `request: ${intent}`;
        const r = rejectForClimate(it, weather);
        if (r !== null) return `climate "${String((it as any)?.name || '?').slice(0, 40)}": ${r}`;
    }
    const oc = rejectOutfitClimate(items, weather);
    if (oc !== null) return `outfit climate: ${oc}`;
    return rejectOutfitIncoherence(items, styleId, occasionId)
        || assessOutfitQuality(items,styleId,occasionId,weather).reasons[0] || null;
}

/** True when the outfit survives the judge (every item unrejected). */
export function passesStyleJudge(
    items: JudgeableItem[],
    styleId: string | null | undefined,
    occasionId: string | null | undefined,
    weather?: ClimateWeather | null,
    lockedItemId?: string | null,
): boolean {
    return styleJudgeReason(items, styleId, occasionId, weather, lockedItemId) === null;
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

export interface OutfitQualityAssessment {
    dimensions: {styleFit:number;occasionFit:number;colorHarmony:number;materialHarmony:number;formalityFit:number;weatherFit:number};
    score:number;
    reasons:string[];
    evidence:string[];
}
function isOldMoneyDate(style?:string|null,occasion?:string|null):boolean {
    return String(style||'').toLowerCase().replace(/[\s-]+/g,'_')==='old_money'
        && /\bdate\b|свидани/i.test(occasion||'');
}
/** Context plus multiple garment cues, never a global one-word product ban. */
export function refinedDateProductReason(item:JudgeableItem,style?:string|null,occasion?:string|null):string|null {
    if(!isOldMoneyDate(style,occasion))return null;
    const text=blobOf(item),slot=macroOf(item);
    if(slot==='top' && /\bwestern[ -]shirts?\b|\bshirts?\b.{0,25}\bwestern\b/.test(text))return 'Western shirt construction conflicts with refined evening styling';
    if(slot==='top' && /\bripstop\b/.test(text) && /\bshirts?\b/.test(text))return 'utility ripstop shirt construction underdresses the date';
    if(slot==='outerwear' && /\bdenim\b/.test(text) && /\b(blazers?|jackets?)\b/.test(text))return 'denim tailoring remains too casual for this refined date';
    if(slot==='outerwear' && /\btechnical\b/.test(text) && /\b(leisure|elastic|sports?|running)\b/.test(text))return 'technical leisure construction conflicts with refined evening tailoring';
    if(slot==='bottom' && /\b(sports?|athletic|training|performance)\b/.test(text) && /\b(pants?|trousers?)\b/.test(text))return 'athletic trouser construction is not refined evening wear';
    if(slot==='top' && /\brelaxed\b/.test(text) && /\b(shirt|button[ -]down|poplin|pinpoint)\b/.test(text))return 'relaxed shirt silhouette underdresses this polished date';
    if(['top','outerwear','bottom'].includes(slot) && /\b(acid[ -]washed|stone[ -]washed|destroyed|distressed)\b/.test(text))return 'distressed surface treatment conflicts with understated evening clothing';
    if(slot==='top' && /\b(terry|towelling|toweling|towel)\b/.test(text) && /\b(polo|shirt|knit)\b/.test(text))return 'towelling resort texture underdresses a refined evening';
    if(slot==='outerwear' && /\b(flight|racer|racing|waxed|technical|track)\b/.test(text) && /\b(jacket|cotton|nylon|zip|pocket)\b/.test(text))return 'sport/utility construction underdresses the refined date';
    if(slot==='outerwear' && /\b(nylon|contrasting bands|contrast panels)\b/.test(text) && /\b(zip|zipped|jacket|elastic)\b/.test(text))return 'technical or athletic outerwear conflicts with refined evening garments';
    if(slot==='outerwear' && /\bjacket\b/.test(text)
        && !/\b(blazer|tailored|single[ -]breasted|double[ -]breasted|suit|wool|cashmere|unstructured|tweed|herringbone|linen)\b/.test(text))return 'jacket lacks evidence of refined tailoring for this date';
    if(slot==='top' && /\bdenim\b/.test(text) && /\b(shirts?|western)\b/.test(text))return 'denim/Western shirt underdresses a refined date';
    if(slot==='outerwear' && /\b(biker|motorcycle|moto|trucker)\b/.test(text)
        && /\b(leather|suede|shearling|jacket|asymmetric|hardware)\b/.test(text))return 'riding/workwear jacket dominates the refined date silhouette';
    if(slot==='outerwear' && /\b(deck|field|military|utility)\b/.test(text)
        && /\b(jacket|cargo|pockets|epaulettes)\b/.test(text))return 'utility outerwear is too casual for this refined date';
    if(slot==='shoes' && /\b(rugged|stormking|hiking|combat|workboot)\b/.test(text)
        && /\b(lug|anti-slip|reinforced|rugged|workboot)\b/.test(text))return 'rugged lug footwear conflicts with refined evening clothing';
    if(slot==='bottom' && /\b(drawstring|jogger|track|interlock|jersey)\b/.test(text)
        && /\b(trousers?|pants?|elasticated|jersey)\b/.test(text))return 'sporting trouser construction underdresses the date';
    return null;
}
function qualityColor(item:JudgeableItem):string {
    const text=normalizeGarmentText(`${item.color||''} ${item.name||''}`);
    const groups:Array<[string,RegExp]>=[['white',/\b(white|ivory|ecru|cream|offwhite)\b/],['navy',/\b(navy|indigo|midnight)\b/],['black',/\bblack\b/],['grey',/\b(grey|gray|charcoal|silver)\b/],['earth',/\b(brown|tan|camel|beige|taupe|sand|cognac|brandy|chocolate)\b/],['green',/\b(olive|sage|khaki|forest|green)\b/],['red',/\b(burgundy|maroon|red)\b/],['blue',/\bblue\b/],['yellow',/\b(yellow|mustard)\b/],['orange',/\borange\b/],['purple',/\b(purple|lilac)\b/],['pink',/\bpink\b/]];
    const explicit=normalizeGarmentText(String(item.color||''));
    return groups.find(([,pattern])=>pattern.test(explicit))?.[0] || groups.find(([,pattern])=>pattern.test(text))?.[0]||'unknown';
}
/** Six equally reported dimensions. Missing evidence stays neutral, not invented. */
export function assessOutfitQuality(items:JudgeableItem[],style?:string|null,occasion?:string|null,weather?:ClimateWeather|null):OutfitQualityAssessment {
    const dimensions={styleFit:1,occasionFit:1,colorHarmony:1,materialHarmony:1,formalityFit:1,weatherFit:1};
    const reasons:string[]=[],evidence:string[]=[];
    const relevant=isOldMoneyDate(style,occasion);
    for(const item of items){
        const reason=refinedDateProductReason(item,style,occasion);
        if(reason){reasons.push(reason);dimensions.styleFit=0;dimensions.occasionFit=0;dimensions.formalityFit=0;}
        if(rejectForClimate(item,weather))dimensions.weatherFit=0;
    }
    const bySlot=(slot:string)=>items.find(item=>macroOf(item)===slot);
    const pairs=[['outerwear','top'],['top','bottom'],['bottom','shoes'],['outerwear','shoes']];
    let conflictingPairs=0;
    for(const [a,b] of pairs){
        const first=bySlot(a),second=bySlot(b);if(!first||!second)continue;
        const x=qualityColor(first),y=qualityColor(second);
        // Bright complementary colors need deliberate styling; muted earth,
        // navy, burgundy and olive palettes remain free to mix.
        const vivid=(item:JudgeableItem)=>/\b(bright|neon|electric|vivid|fluorescent)\b/.test(blobOf(item));
        const complement=[['red','green'],['blue','orange'],['yellow','purple']].some(p=>p.includes(x)&&p.includes(y)&&x!==y);
        if(complement&&vivid(first)&&vivid(second)){conflictingPairs++;evidence.push(`${a}/${b}: competing vivid complementary colors`);}
    }
    if(conflictingPairs){dimensions.colorHarmony=0;reasons.push('competing vivid complementary colors across outfit pairs');}
    const palette=outfitPaletteReason(items);if(palette){dimensions.colorHarmony=0;reasons.push(palette);}
    if(relevant){
        const layer=bySlot('outerwear'),top=bySlot('top'),bottom=bySlot('bottom');
        const upper=[layer,top].filter(Boolean).map(item=>blobOf(item!)).join(' ');
        const lower=bottom?blobOf(bottom):'';
        const layerEvidence=layer?blobOf(layer):'';
        const winterCoat=/\b(coat|trench|overcoat|peacoat)\b/.test(layerEvidence)
            && /\b(wool|cashmere|winter|insulated|double[ -]breasted)\b/.test(layerEvidence)
            && !/\b(lightweight|tropical|summer|unlined)\b/.test(layerEvidence);
        if((winterCoat || /\b(shearling|alpaca|mohair|fleece|chunky|heavyweight|rugged|padded)\b/.test(upper))&&/\b(linen|seersucker|gauze)\b/.test(`${lower} ${top?blobOf(top):''}`)){
            dimensions.materialHarmony=0;reasons.push('heavy rugged upper textures conflict with airy summer trousers');
        }
        const utility=items.filter(item=>/\b(western|military|utility|workwear|leisure)\b/.test(blobOf(item)));
        if(utility.length>=2){dimensions.styleFit=0;reasons.push('multiple utility/leisure pieces dominate the refined date');}
        if(utility.length===1){dimensions.styleFit=0.75;evidence.push('one casual detail balanced by refined companions');}
        const layerText=layer?blobOf(layer):'';
        if(/\b(leather|suede)\b/.test(layerText)&&!reasons.length){dimensions.styleFit=Math.min(dimensions.styleFit,0.75);evidence.push('understated leather layer: smart evening rather than classic tailoring');}
    }
    const score=Object.values(dimensions).reduce((sum,n)=>sum+n,0)/6;
    return {dimensions,score,reasons:[...new Set(reasons)],evidence};
}

/** Preserve valid alternatives across garment families before bounded candidate trims. */
export function spreadQualityFamilies<T>(items:T[],familyOf:(item:T)=>string):T[] {
    const families=new Map<string,T[]>();
    for(const item of items){const key=familyOf(item);const group=families.get(key)||[];group.push(item);families.set(key,group);}
    const result:T[]=[];
    for(let i=0;result.length<items.length;i++)for(const group of families.values())if(i<group.length)result.push(group[i]);
    return result;
}
