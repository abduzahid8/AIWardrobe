/**
 * colorHarmony — deterministic whole-outfit color relationship scorer.
 *
 * Scores how four garments' colors work TOGETHER, not how restrained the
 * palette is. A restrained palette of clashing saturated colors scores
 * poorly; a tonal olive/brown story or a balanced navy/white contrast
 * scores well. No combination of specific colors is preferred — bonuses
 * reward relationships (tonal, contrast, classic pairings), never hues.
 *
 * Evidence discipline:
 *  - Only canonical color tokens (name / color-field evidence) are read.
 *    Photography-background phrases are stripped before tokenizing, so a
 *    "white background" studio note is never proof of a white garment.
 *  - No RGB values, undertones or temperatures are invented. When fewer
 *    than two garments carry usable color evidence the result is marked
 *    uncertain and clamped to neutral scoring.
 *  - Pure + dependency-free (imports only the token tables): safe in the
 *    Expo client and the Deno edge function.
 */

export interface HarmonyGarment {
    /** Canonical color tokens, e.g. ['navy', 'white']. Empty = unknown. */
    colors: string[];
    /** Coarse family, e.g. 'dark' | 'light' | 'earth'. 'unknown' allowed. */
    colorFamily: string;
    /** Pattern profile: 'solid' | 'stripe' | 'check' | 'texture' | 'print'. */
    pattern: string;
}

export interface ColorHarmony {
    /** 0..1. Soft preference — never a hard validity verdict. */
    score: number;
    reasons: string[];
    evidence: string[];
    /** True when color evidence was too thin to judge confidently. */
    uncertain: boolean;
}

/** Achromatic + off-white: combine with anything, never an accent. */
const CORE_NEUTRALS = new Set(['white', 'black', 'grey', 'charcoal', 'cream']);
/** Working neutrals: chromatic but combinable; count as support, not accent. */
const WORKHORSE_NEUTRALS = new Set(['navy', 'brown', 'tan', 'beige']);
/** Loud hues: one is an accent, two fight, three shout. */
const SATURATED = new Set(['red', 'rust', 'pink', 'yellow', 'green', 'light-blue', 'blue']);
const LIGHT_TOKENS = new Set(['white', 'cream', 'beige', 'light-blue', 'grey']);
const DARK_TOKENS = new Set(['black', 'charcoal', 'navy', 'dark-neutral', 'brown', 'burgundy']);
/**
 * Relationship bonuses, not templates. Each pair is one example of a
 * well-established neutral pairing; the bonus fires at most once and is
 * small, so no palette is systematically preferred.
 */
const CLASSIC_PAIRS: Array<[string, string]> = [
    ['navy', 'brown'],
    ['grey', 'navy'],
    ['black', 'white'],
    ['white', 'navy'],
    ['beige', 'brown'],
    ['cream', 'beige'],
];

const clamp01 = (value: number): number => Math.max(0, Math.min(1, value));

export function scoreColorHarmony(garments: HarmonyGarment[]): ColorHarmony {
    const reasons: string[] = [];
    const evidence: string[] = [];
    const list = Array.isArray(garments) ? garments : [];
    const withColor = list.filter((g) => g && g.colors.length > 0);
    const unknownCount = list.length - withColor.length;
    const uncertain = unknownCount >= 2 || withColor.length < 2;

    let score = 0.8;

    const tokenSet = new Set<string>();
    for (const garment of withColor) for (const token of garment.colors) tokenSet.add(token);
    const tokens = [...tokenSet];
    const accents = tokens.filter((token) => SATURATED.has(token));
    const chromatics = tokens.filter(
        (token) => !CORE_NEUTRALS.has(token) && !WORKHORSE_NEUTRALS.has(token),
    );

    if (accents.length === 0) {
        score += 0.05;
        evidence.push('restrained_palette_no_loud_accents');
    } else if (accents.length === 1) {
        const supported = tokens.some(
            (token) => CORE_NEUTRALS.has(token) || WORKHORSE_NEUTRALS.has(token),
        );
        score += supported ? 0.1 : 0.05;
        evidence.push(`single_accent_${accents[0]}_on_neutral_base`);
    } else {
        // Two loud hues compete; three or more shout over the outfit.
        score -= 0.15 + 0.2 * (accents.length - 2);
        reasons.push('overloaded_accent_combination');
    }

    // Tonal story: one chromatic idea (plus any neutrals) reads intentional.
    const chromaticFamilies = new Set(
        withColor.flatMap((g) =>
            g.colors
                .filter((t) => !CORE_NEUTRALS.has(t) && !WORKHORSE_NEUTRALS.has(t))
                .map(() => g.colorFamily),
        ),
    );
    if (withColor.length > 0 && chromatics.length > 0 && chromaticFamilies.size <= 1) {
        score += 0.1;
        evidence.push('tonal_palette_single_chromatic_story');
    }

    // Light/dark contrast keeps neutrals from reading flat.
    const hasLight = tokens.some((token) => LIGHT_TOKENS.has(token));
    const hasDark = tokens.some((token) => DARK_TOKENS.has(token));
    if (hasLight && hasDark) {
        score += 0.05;
        evidence.push('balanced_light_dark_contrast');
    }

    // Monochrome discipline: one token (or one token over neutrals) is
    // deliberate in any hue — black, olive and burgundy score identically.
    const nonNeutral = tokens.filter((token) => !CORE_NEUTRALS.has(token));
    if (tokens.length > 0 && nonNeutral.length <= 1) {
        score += 0.05;
        evidence.push('monochromatic_discipline');
    }

    if (CLASSIC_PAIRS.some(([a, b]) => tokenSet.has(a) && tokenSet.has(b))) {
        score += 0.05;
        evidence.push('classic_neutral_pairing');
    }

    // Print discipline: two competing prints, or a print fighting a stripe,
    // break harmony even when the colors agree.
    const prints = list.filter((g) => g && g.pattern === 'print').length;
    const stripes = list.filter((g) => g && (g.pattern === 'stripe' || g.pattern === 'check')).length;
    if (prints >= 2 || (prints >= 1 && stripes >= 1)) {
        score -= 0.1;
        reasons.push('clashing_prints_break_harmony');
    }

    if (uncertain) {
        score = Math.max(0.45, Math.min(0.75, score));
        evidence.push('uncertain_palette_insufficient_color_evidence');
    }

    return { score: clamp01(Math.round(score * 1000) / 1000), reasons, evidence, uncertain };
}
