/**
 * Visual similarity is a SOFT ranking signal, not the deck gate.
 *
 * The score distribution below is still useful: identical formulas score
 * higher than clearly different concepts. But the deck invariant is product
 * uniqueness. Two looks with the same formula and different actual products
 * are allowed; the similarity score only breaks ties between valid decks.
 *
 *   identical formula, different SKUs   -> high band
 *   same formula, only the shoes differ -> threshold band
 *   clearly different concepts           -> low band
 */
import {
    assembleDiverseDeck,
    buildOutfitFingerprint,
    MAX_VISUAL_SIMILARITY,
    productIdentityKeys,
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

describe('visual similarity distribution', () => {
    it('identical formulas score above the strict reference threshold', () => {
        for (const [a, b] of [['A', 'B'], ['A', 'C'], ['B', 'C']]) {
            expect(visualSimilarity(fingerprints[a], fingerprints[b])).toBeGreaterThan(MAX_VISUAL_SIMILARITY);
        }
    });

    it('sameArchetype describes the formula, including when only the shoes differ', () => {
        for (const [a, b] of [
            ['A', 'B'], ['A', 'C'], ['B', 'C'],
            ['A', 'F'], ['B', 'F'], ['C', 'F'],
        ]) {
            expect(sameArchetype(fingerprints[a], fingerprints[b])).toBe(true);
        }
    });

    it('clearly different concepts score lower and share fewer families', () => {
        for (const [a, b] of [['A', 'D'], ['A', 'E'], ['D', 'E'], ['D', 'G'], ['E', 'H']]) {
            expect(visualSimilarity(fingerprints[a], fingerprints[b]))
                .toBeLessThan(visualSimilarity(fingerprints.A, fingerprints.B));
            expect(sharedVisualFamilies(fingerprints[a], fingerprints[b]))
                .toBeLessThan(sharedVisualFamilies(fingerprints.A, fingerprints.B));
        }
    });

    it('does not reject same-formula looks when their products are unique', () => {
        const candidates = ['A', 'B', 'C'].map((key) => ({ id: key, items: looks[key] }));
        const result = assembleDiverseDeck<{ id: string; items: any[] }>({
            candidates,
            itemsOf: (look) => look.items,
            keysOf: productIdentityKeys,
            usedKeys: new Set<string>(),
            poolItems: Object.values(looks).flat(),
        });
        expect(result.deck).toHaveLength(3);
        expect(result.relaxed).toBe(false);
        expect(new Set(result.archetypes).size).toBe(1);
        expect(result.productDiversity.nonAnchorProductReuseCount).toBe(0);
    });
});
