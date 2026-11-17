/**
 * MMR deck selection: within one quality band diversity decides, across
 * bands quality dominates, and recent-show resemblance demotes instead of
 * the old sub-bucket subtraction (which bucketing absorbed to zero).
 */
import {
    assembleDiverseDeck,
    productIdentityKeys,
} from '../../features/outfit-generator/utils/outfitDiversity';
import { outfitVisualSignature, recentLookSimilarity } from '../../src/domain/outfitVisualSignature';
import type { DiversitySlot } from '../../features/outfit-generator/utils/outfitDiversity';

type Product = { id: string; name: string; macroCategory: DiversitySlot; image?: string };
const product = (id: string, macroCategory: DiversitySlot, name: string): Product => ({
    id,
    name,
    macroCategory,
    image: `https://cdn.test/${id}.jpg`,
});
type Look = { id: string; items: Product[]; score: number };
const look = (id: string, items: Product[], score: number): Look => ({ id, items, score });
const itemsOf = (value: Look) => value.items;

const blazerLook = (prefix: string, topName: string, bottomName: string, shoeName: string, score: number): Look =>
    look(prefix, [
        product(`${prefix}_o`, 'outerwear', 'Navy wool blazer'),
        product(`${prefix}_t`, 'top', topName),
        product(`${prefix}_b`, 'bottom', bottomName),
        product(`${prefix}_s`, 'shoes', shoeName),
    ], score);

const cardiganLook = (prefix: string, score: number): Look =>
    look(prefix, [
        product(`${prefix}_o`, 'outerwear', 'Beige cashmere cardigan'),
        product(`${prefix}_t`, 'top', 'Cream merino crewneck'),
        product(`${prefix}_b`, 'bottom', 'Olive chino trousers'),
        product(`${prefix}_s`, 'shoes', 'Brown suede loafers'),
    ], score);

const baseOptions = {
    itemsOf,
    keysOf: productIdentityKeys,
    qualityOf: (entry: Look) => entry.score,
    usedKeys: new Set<string>(),
    target: 2,
};

describe('MMR deck selection', () => {
    it('prefers the diverse look over a near-duplicate inside one quality band', () => {
        const anchor = blazerLook('m1', 'White oxford shirt', 'Grey tailored trousers', 'Brown penny loafers', 0.95);
        const nearDupe = blazerLook('m2', 'Blue poplin shirt', 'Charcoal tailored trousers', 'Black penny loafers', 0.93);
        const diverse = cardiganLook('m3', 0.90);
        const result = assembleDiverseDeck<Look>({ ...baseOptions, candidates: [anchor, nearDupe, diverse] });
        // The diverse cardigan look must win a slot; which of the two
        // near-identical blazer formulas keeps the other slot is a tie.
        const ids = result.deck.map((entry) => entry.id);
        expect(ids).toHaveLength(2);
        expect(ids).toContain('m3');
        expect(ids.filter((id) => id === 'm1' || id === 'm2')).toHaveLength(1);
    });

    it('lets a genuine quality gap dominate visual similarity', () => {
        const best = blazerLook('q1', 'White oxford shirt', 'Grey tailored trousers', 'Brown penny loafers', 0.95);
        // Distinct product (charcoal, not navy) but the same blazer formula.
        const nearDupe = look('q2', [
            product('q2_o', 'outerwear', 'Charcoal wool blazer'),
            product('q2_t', 'top', 'Blue poplin shirt'),
            product('q2_b', 'bottom', 'Charcoal tailored trousers'),
            product('q2_s', 'shoes', 'Black penny loafers'),
        ], 0.93);
        const weak = cardiganLook('q3', 0.50);
        const result = assembleDiverseDeck<Look>({ ...baseOptions, candidates: [best, nearDupe, weak] });
        // 0.45 quality gap dwarfs any variety term: the strong pair wins.
        expect(result.deck.map((entry) => entry.id)).toEqual(['q1', 'q2']);
    });

    it('demotes a look identical to recent history via recencyOf', () => {
        const shown = blazerLook('h1', 'White oxford shirt', 'Grey tailored trousers', 'Brown penny loafers', 0.9);
        const shownSig = outfitVisualSignature(shown.items);
        const repeat = blazerLook('r1', 'White oxford shirt', 'Grey tailored trousers', 'Brown penny loafers', 0.93);
        expect(recentLookSimilarity(repeat.items, [shownSig])).toBe(1);
        // Distinct products, same formula as the shown look, one slot off.
        const nearRepeat = look('r1b', [
            product('r1b_o', 'outerwear', 'Navy wool blazer'),
            product('r1b_t', 'top', 'Blue poplin shirt'),
            product('r1b_b', 'bottom', 'Charcoal tailored trousers'),
            product('r1b_s', 'shoes', 'Brown penny loafers'),
        ], 0.92);
        expect(recentLookSimilarity(nearRepeat.items, [shownSig])).toBeLessThan(1);
        const fresh = cardiganLook('r2', 0.91);
        const result = assembleDiverseDeck<Look>({
            ...baseOptions,
            candidates: [repeat, nearRepeat, fresh],
            recencyOf: (entry) => recentLookSimilarity(entry.items, [shownSig]),
        });
        // All three pairings land in one quality band, so the near-repeat
        // loses to the fresh look instead of the exact repeat winning.
        const ids = result.deck.map((entry) => entry.id);
        expect(ids).toHaveLength(2);
        expect(ids).toContain('r2');
        expect(ids).not.toContain('r1');
    });

    it('never starves the deck when every candidate resembles history', () => {
        const shown = blazerLook('h1', 'White oxford shirt', 'Grey tailored trousers', 'Brown penny loafers', 0.9);
        const shownSig = outfitVisualSignature(shown.items);
        const a = blazerLook('s1', 'White oxford shirt', 'Grey tailored trousers', 'Brown penny loafers', 0.9);
        const b = blazerLook('s2', 'White pinpoint shirt', 'Grey tailored trousers', 'Brown penny loafers', 0.88);
        const result = assembleDiverseDeck<Look>({
            ...baseOptions,
            candidates: [a, b],
            recencyOf: (entry) => recentLookSimilarity(entry.items, [shownSig]),
        });
        // Thin or stale pools still return a full deck — recency never vetoes.
        expect(result.deck).toHaveLength(2);
    });
});
