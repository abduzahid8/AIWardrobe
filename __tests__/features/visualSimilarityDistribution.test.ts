/**
 * The score distribution that justifies MAX_VISUAL_SIMILARITY and
 * FAMILY_ESCAPE_SIMILARITY.
 *
 * The thresholds are only meaningful if the regimes they separate do not
 * overlap. Measured on the fixture matrix below:
 *
 *   identical formula, different SKUs   -> 0.75 .. 0.83   REJECTED
 *   same formula, only the shoes differ -> 0.54 .. 0.60   REJECTED
 *   same concept, one slot swapped      -> 0.54 .. 0.67   REJECTED
 *   clearly different concepts           -> 0.09 .. 0.52   ACCEPTED
 *
 * 0.65 sits inside the "same concept" band, which is exactly why the
 * categorical archetype rule (sameArchetype) and the shared-family rule
 * exist alongside the numeric score: the score alone cannot separate
 * "same formula" from "same formula minus one slot", and a look whose only
 * change is the shoes can score BELOW the threshold. The archetype rule
 * rejects it categorically.
 *
 * This test asserts the DECISION (would the gate accept the pair?), not
 * just the raw numbers.
 */
import {
    buildOutfitFingerprint,
    FAMILY_ESCAPE_SIMILARITY,
    MAX_VISUAL_SIMILARITY,
    sharedVisualFamilies,
    sameArchetype,
    visualSimilarity,
} from '../../features/outfit-generator/utils/outfitDiversity';

const item = (id: string, slot: string, name: string, extra: any = {}) => ({
    id, name, macroCategory: slot, image: `https://cdn/${id}.jpg`, ...extra,
});

const looks: Record<string, any[]> = {
    A: [item('a1', 'outerwear', 'Navy wool blazer'), item('a2', 'top', 'White oxford shirt'), item('a3', 'bottom', 'Grey tailored trousers'), item('a4', 'shoes', 'Brown penny loafers')],
    B: [item('b1', 'outerwear', 'Charcoal blazer'), item('b2', 'top', 'Blue poplin shirt'), item('b3', 'bottom', 'Black trousers'), item('b4', 'shoes', 'Penny loafer')],
    C: [item('c1', 'outerwear', 'Beige blazer'), item('c2', 'top', 'Striped shirt'), item('c3', 'bottom', 'Navy trousers'), item('c4', 'shoes', 'Tassel loafer')],
    D: [item('d1', 'outerwear', 'Suede jacket'), item('d2', 'top', 'Knitted polo'), item('d3', 'bottom', 'Beige chinos'), item('d4', 'shoes', 'Minimal leather sneakers')],
    E: [item('e1', 'outerwear', 'Cardigan'), item('e2', 'top', 'Fine gauge knit'), item('e3', 'bottom', 'Dark denim jeans'), item('e4', 'shoes', 'Derby shoes')],
    F: [item('f1', 'outerwear', 'Navy blazer'), item('f2', 'top', 'White oxford shirt'), item('f3', 'bottom', 'Grey tailored trousers'), item('f4', 'shoes', 'Brown derby shoes')],
    G: [item('g1', 'outerwear', 'Coat'), item('g2', 'top', 'Chunky knit sweater'), item('g3', 'bottom', 'Dark jeans'), item('g4', 'shoes', 'Chelsea boots')],
    H: [item('h1', 'outerwear', 'Navy peacoat'), item('h2', 'top', 'Merino sweater'), item('h3', 'bottom', 'Indigo jeans'), item('h4', 'shoes', 'Desert boots')],
};

const fingerprints = Object.fromEntries(
    Object.entries(looks).map(([key, items]) => [key, buildOutfitFingerprint(items)]),
);

/** The strict rule of the deck gate, mirrored here as the contract. */
const gateAccepts = (a: string, b: string): boolean => {
    const left = fingerprints[a];
    const right = fingerprints[b];
    const similarity = visualSimilarity(left, right);
    if (similarity >= MAX_VISUAL_SIMILARITY) return false;
    if (sameArchetype(left, right)) return false;
    if (sharedVisualFamilies(left, right) > 1 && similarity >= FAMILY_ESCAPE_SIMILARITY) return false;
    return true;
};

describe('visual similarity distribution', () => {
    it('identical formulas score far above the threshold', () => {
        for (const [a, b] of [['A', 'B'], ['A', 'C'], ['B', 'C']]) {
            expect(visualSimilarity(fingerprints[a], fingerprints[b])).toBeGreaterThan(MAX_VISUAL_SIMILARITY);
        }
    });

    it('the categorical rule catches identical formulas even in the low band', () => {
        // A/B/C: one formula, three colourways. A/F, B/F, C/F: same garment
        // roles with different shoes — still one formula. Footwear is a
        // finishing detail, not a styling direction.
        for (const [a, b] of [
            ['A', 'B'], ['A', 'C'], ['B', 'C'],
            ['A', 'F'], ['B', 'F'], ['C', 'F'],
        ]) {
            expect(sameArchetype(fingerprints[a], fingerprints[b])).toBe(true);
        }
    });

    it('the gate rejects every "same formula" and "same concept" pair', () => {
        const sameConceptPairs: Array<[string, string]> = [
            ['A', 'B'], ['A', 'C'], ['B', 'C'],   // identical formula
            ['A', 'F'], ['B', 'F'], ['C', 'F'],   // one slot swapped
            ['G', 'H'],                          // same concept, different coat
        ];
        for (const [a, b] of sameConceptPairs) {
            expect(gateAccepts(a, b)).toBe(false);
        }
    });

    it('the gate accepts every clearly different concept pair', () => {
        const differentConceptPairs: Array<[string, string]> = [
            ['A', 'D'], ['A', 'E'], ['A', 'G'], ['A', 'H'],
            ['B', 'D'], ['B', 'E'], ['C', 'D'], ['C', 'E'],
            ['D', 'E'], ['D', 'G'], ['D', 'H'], ['F', 'G'], ['F', 'H'],
        ];
        for (const [a, b] of differentConceptPairs) {
            expect(gateAccepts(a, b)).toBe(true);
        }
    });

    it('the whole matrix separates the regimes', () => {
        const keys = Object.keys(looks);
        const sameFormula: number[] = [];
        const accepted: number[] = [];
        for (let i = 0; i < keys.length; i++) {
            for (let j = i + 1; j < keys.length; j++) {
                const a = keys[i];
                const b = keys[j];
                const value = visualSimilarity(fingerprints[a], fingerprints[b]);
                if (sameArchetype(fingerprints[a], fingerprints[b])) sameFormula.push(value);
                if (gateAccepts(a, b)) accepted.push(value);
            }
        }
        // Every accepted pair is strictly below the threshold …
        expect(accepted.length).toBeGreaterThan(0);
        expect(Math.max(...accepted)).toBeLessThan(MAX_VISUAL_SIMILARITY);
        // … and every same-formula pair is rejected by the gate, even the
        // ones that score below the threshold (only the shoes changed). The
        // numeric rule alone does NOT separate them; the categorical rule
        // does, which is why both exist.
        expect(Math.min(...sameFormula)).toBeLessThan(MAX_VISUAL_SIMILARITY);
        for (const [a, b] of [
            ['A', 'B'], ['A', 'C'], ['B', 'C'],
            ['A', 'F'], ['B', 'F'], ['C', 'F'],
        ]) {
            expect(sameArchetype(fingerprints[a], fingerprints[b])).toBe(true);
            expect(gateAccepts(a, b)).toBe(false);
        }
    });
});
