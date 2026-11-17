/**
 * Layering (C), footwear (D) and evidence-hierarchy (E) additions.
 *
 * Hard rules fire only on construction evidence; unknown thickness, fit
 * or provenance never convicts. Marketing words alone never promote.
 */
import { layerCompatibility } from '../../src/domain/layerCompatibility';
import { describeGarment } from '../../src/domain/garmentDescriptor';
import { outfitCoherence } from '../../src/domain/outfitCoherence';
import { colorFamilyOf, colorTokensOf } from '../../src/domain/outfitDiversity';
import { stylingContext } from '../../src/domain/stylingContext';

const top = (name: string, extra: Record<string, unknown> = {}) => ({
    name,
    macroCategory: 'top',
    ...extra,
});
const layer = (name: string) => ({ name, macroCategory: 'outerwear' });
const bottom = (name: string) => ({ name, macroCategory: 'bottom' });
const shoes = (name: string) => ({ name, macroCategory: 'shoes' });
const casual = stylingContext({ style: 'casual' });
const look = (names: [string, string, string, string]) => [
    top(names[0]),
    layer(names[1]),
    bottom(names[2]),
    shoes(names[3]),
];

describe('layering compatibility additions', () => {
    it('rejects a hoodie and gym-wear under a tailored blazer', () => {
        expect(
            layerCompatibility(top('Heavy cotton hoodie'), layer('Wool blazer'), casual).reasons
        ).toContain('hood_under_tailored_blazer');
        expect(
            layerCompatibility(top('Athletic training tee'), layer('Structured wool blazer'), casual)
                .reasons
        ).toContain('tailored_layer_over_athletic_base');
    });

    it('rejects only provably bulky knits under a blazer, never unknown thickness', () => {
        expect(
            layerCompatibility(top('Heavy oversized wool sweater'), layer('Slim wool blazer'), casual)
                .valid
        ).toBe(false);
        const unknown = layerCompatibility(
            top('Cashmere crewneck sweater'),
            layer('Wool blazer'),
            casual
        );
        expect(unknown.valid).toBe(true);
        expect(unknown.evidence).toContain('unverified_knit_bulk_under_blazer');
    });

    it('explicitly accepts fine knitwear under a blazer', () => {
        const result = layerCompatibility(
            top('Fine-knit merino sweater'),
            layer('Navy wool blazer'),
            casual
        );
        expect(result.valid).toBe(true);
        expect(result.evidence).toContain('fine_knit_under_blazer');
    });

    it('rejects athletic outerwear over a formal base', () => {
        expect(
            layerCompatibility(
                top('White formal dress shirt'),
                layer('Training track jacket'),
                stylingContext({ style: 'casual' })
            ).reasons
        ).toContain('athletic_outerwear_over_formal_base');
    });
});

describe('footwear relationship additions', () => {
    it('rejects formal shoes with athletic bottoms in any context', () => {
        const items = look([
            'White poplin shirt',
            'Navy overshirt',
            'Stripe track pants',
            'Black leather oxford shoes',
        ]);
        expect(
            outfitCoherence(items, stylingContext({ style: 'casual' })).reasons
        ).toContain('formal_footwear_with_athletic_bottom');
    });

    it('soft-penalizes technical runners under a tailored layer, exempts minimal sneakers', () => {
        const runner = look([
            'White oxford shirt',
            'Navy wool blazer',
            'Grey tailored trousers',
            'Technical running trainers',
        ]);
        const runnerOut = outfitCoherence(runner, stylingContext({ style: 'casual' }));
        expect(runnerOut.dimensions.footwear).toBeLessThanOrEqual(0.4);
        expect(runnerOut.evidence).toContain('sport_sneaker_with_tailored_layer');
        expect(runnerOut.valid).toBe(true);
        const minimal = look([
            'White oxford shirt',
            'Navy wool blazer',
            'Grey tailored trousers',
            'Minimal white leather sneakers',
        ]);
        const minimalOut = outfitCoherence(minimal, stylingContext({ style: 'casual' }));
        expect(minimalOut.evidence).not.toContain('sport_sneaker_with_tailored_layer');
    });
});

describe('evidence hierarchy', () => {
    it('never treats a white studio background as proof of a white garment', () => {
        const item = {
            name: 'Charcoal wool blazer',
            description: 'Shot on white background, ghost mannequin, flat lay studio photo',
            macroCategory: 'outerwear',
        };
        expect(colorTokensOf(item)).not.toContain('white');
        expect(colorFamilyOf(item)).toBe('dark');
    });

    it('requires construction evidence for dress-loafer status, not the bare word', () => {
        // Contradicted claims stay smart even with leather-adjacent words.
        expect(
            describeGarment({ name: 'Brown suede dress loafers', macroCategory: 'shoes' })
                .formality
        ).toBe('smart');
        expect(
            describeGarment({ name: 'Dress loafers', macroCategory: 'shoes' }).formality
        ).toBe('smart');
        expect(
            describeGarment({
                name: 'Blake-stitched leather-sole loafers',
                macroCategory: 'shoes',
            }).formality
        ).toBe('formal');
        // Named loafer constructions (penny/tassel/horsebit) are tailoring
        // evidence, not marketing: they go with suits even without leather.
        expect(
            describeGarment({ name: 'DRESS PENNY LOAFERS', macroCategory: 'shoes' })
                .formality
        ).toBe('formal');
        expect(
            describeGarment({ name: 'Brown penny loafers', macroCategory: 'shoes' })
                .formality
        ).toBe('formal');
        expect(
            describeGarment({ name: 'Black tassel loafers', macroCategory: 'shoes' })
                .formality
        ).toBe('formal');
    });

    it('never mistakes lounge or terry bottoms for tailoring', () => {
        expect(
            describeGarment({ name: 'Lounge terry drawstring pants', macroCategory: 'bottom' })
                .subtype
        ).not.toBe('tailored-trouser');
        expect(
            describeGarment({ name: 'Tailored wool trousers', macroCategory: 'bottom' }).formality
        ).toBe('formal');
    });
});
