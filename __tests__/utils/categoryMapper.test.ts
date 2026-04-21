import { normalizeCategory, mapDbCategory, getMacroCategory } from '../../src/utils/categoryMapper';

describe('categoryMapper', () => {
    describe('normalizeCategory', () => {
        it.each([
            ['top', 'top'], ['Tops', 'top'], ['tops', 'top'],
            ['bottom', 'bottom'], ['Bottoms', 'bottom'],
            ['shoes', 'shoes'], ['shoe', 'shoes'], ['Shoes', 'shoes'],
            ['outerwear', 'outerwear'],
            ['dress', 'dress'], ['dresses', 'dress'],
            ['accessory', 'accessory'], ['accessories', 'accessory'],
            ['unknown', 'other'], ['', 'other'],
        ])('"%s" → "%s"', (input, expected) => {
            expect(normalizeCategory(input)).toBe(expected);
        });
    });

    describe('mapDbCategory', () => {
        it('handles legacy PascalCase', () => {
            expect(mapDbCategory('Tops')).toBe('top');
            expect(mapDbCategory('Bottoms')).toBe('bottom');
        });

        it('passes through lowercase', () => {
            expect(mapDbCategory('shoes')).toBe('shoes');
        });

        it('defaults to other for unknowns', () => {
            expect(mapDbCategory('xyz')).toBe('other');
        });
    });

    describe('getMacroCategory', () => {
        it('detects outerwear from type string', () => {
            expect(getMacroCategory('top', 'blazer')).toBe('outerwear');
            expect(getMacroCategory('outerwear', 'jacket')).toBe('outerwear');
            expect(getMacroCategory('clothing', 'puffer')).toBe('outerwear');
        });

        it('detects tops', () => {
            expect(getMacroCategory('top', 't-shirt')).toBe('top');
            expect(getMacroCategory('clothing', 'blouse')).toBe('top');
        });

        it('detects bottoms', () => {
            expect(getMacroCategory('bottom', 'jeans')).toBe('bottom');
            expect(getMacroCategory('clothing', 'trouser')).toBe('bottom');
        });

        it('detects shoes', () => {
            expect(getMacroCategory('shoes', 'sneaker')).toBe('shoes');
        });

        it('detects accessories', () => {
            expect(getMacroCategory('accessory', 'hat')).toBe('accessory');
            expect(getMacroCategory('other', 'belt')).toBe('accessory');
        });

        it('falls back to other', () => {
            expect(getMacroCategory('misc', 'thing')).toBe('other');
        });
    });
});
