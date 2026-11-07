/**
 * Color harmony: relationships, not restraint. Tonal stories and balanced
 * contrast score well in ANY hue; overloaded saturated accents and
 * print fights score poorly; unknown palettes stay neutral, never certain.
 */
import { scoreColorHarmony, type HarmonyGarment } from '../../src/domain/colorHarmony';

const g = (colors: string[], colorFamily = 'neutral', pattern = 'solid'): HarmonyGarment => ({
    colors,
    colorFamily,
    pattern,
});

describe('color harmony scorer', () => {
    it('rewards a harmonious tonal earth story', () => {
        const result = scoreColorHarmony([
            g(['beige'], 'earth'),
            g(['cream'], 'earth'),
            g(['olive'], 'earth'),
            g(['brown'], 'earth'),
        ]);
        expect(result.score).toBeGreaterThanOrEqual(0.85);
        expect(result.reasons).toHaveLength(0);
    });

    it('rewards a balanced high-contrast pairing', () => {
        const result = scoreColorHarmony([
            g(['navy'], 'cool-dark'),
            g(['white'], 'light'),
            g(['charcoal'], 'dark'),
            g(['brown'], 'earth'),
        ]);
        expect(result.score).toBeGreaterThanOrEqual(0.85);
    });

    it('penalizes overloaded saturated accents', () => {
        const calm = scoreColorHarmony([
            g(['red'], 'warm'),
            g(['white'], 'light'),
            g(['charcoal'], 'dark'),
            g(['brown'], 'earth'),
        ]);
        const overloaded = scoreColorHarmony([
            g(['red'], 'warm'),
            g(['rust'], 'warm'),
            g(['pink'], 'warm'),
            g(['yellow'], 'warm'),
        ]);
        expect(calm.score).toBeGreaterThan(overloaded.score + 0.2);
        expect(overloaded.reasons).toContain('overloaded_accent_combination');
    });

    it('does not systematically prefer black/white/beige: olive tonality matches monochrome', () => {
        const oliveTonal = scoreColorHarmony([
            g(['olive'], 'earth'),
            g(['brown'], 'earth'),
            g(['cream'], 'earth'),
            g(['tan'], 'earth'),
        ]);
        const bwMono = scoreColorHarmony([
            g(['black'], 'dark'),
            g(['white'], 'light'),
            g(['grey'], 'mid'),
            g(['black'], 'dark'),
        ]);
        expect(Math.abs(oliveTonal.score - bwMono.score)).toBeLessThanOrEqual(0.15);
        expect(oliveTonal.score).toBeGreaterThanOrEqual(0.8);
    });

    it('treats unknown palettes as uncertain with neutral scoring', () => {
        const result = scoreColorHarmony([g([]), g([]), g(['navy'], 'cool-dark'), g(['brown'], 'earth')]);
        expect(result.uncertain).toBe(true);
        expect(result.score).toBeGreaterThanOrEqual(0.45);
        expect(result.score).toBeLessThanOrEqual(0.75);
        expect(result.evidence).toContain('uncertain_palette_insufficient_color_evidence');
    });

    it('penalizes competing prints even when colors agree', () => {
        const result = scoreColorHarmony([
            g(['navy'], 'cool-dark', 'print'),
            g(['white'], 'light', 'print'),
            g(['brown'], 'earth'),
            g(['cream'], 'earth'),
        ]);
        expect(result.reasons).toContain('clashing_prints_break_harmony');
    });
});
