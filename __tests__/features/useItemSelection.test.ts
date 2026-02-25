/**
 * Tests for the item selection hook used in outfit generation
 */

import { getMacroCategory } from '../../features/outfit-generator/hooks/useItemSelection';

describe('getMacroCategory', () => {
    it('categorizes sweaters correctly', () => {
        expect(getMacroCategory('sweater')).toBe('sweater');
        expect(getMacroCategory('Hoodie')).toBe('sweater');
        expect(getMacroCategory('cardigan')).toBe('sweater');
    });

    it('categorizes tops correctly', () => {
        expect(getMacroCategory('t-shirt')).toBe('top');
        expect(getMacroCategory('polo shirt')).toBe('top');
        expect(getMacroCategory('blouse')).toBe('top');
        expect(getMacroCategory('Tank Top')).toBe('top');
    });

    it('categorizes bottoms correctly', () => {
        expect(getMacroCategory('pants')).toBe('bottom');
        expect(getMacroCategory('jeans')).toBe('bottom');
        expect(getMacroCategory('shorts')).toBe('bottom');
        expect(getMacroCategory('skirt')).toBe('bottom');
        expect(getMacroCategory('trousers')).toBe('bottom');
    });

    it('categorizes shoes correctly', () => {
        expect(getMacroCategory('shoes')).toBe('shoes');
        expect(getMacroCategory('sneakers')).toBe('shoes');
        expect(getMacroCategory('boots')).toBe('shoes');
        expect(getMacroCategory('sandals')).toBe('shoes');
        expect(getMacroCategory('loafers')).toBe('shoes');
    });

    it('categorizes outerwear correctly', () => {
        expect(getMacroCategory('jacket')).toBe('outerwear');
        expect(getMacroCategory('coat')).toBe('outerwear');
        expect(getMacroCategory('blazer')).toBe('outerwear');
        expect(getMacroCategory('vest')).toBe('outerwear');
    });

    it('returns "other" for unrecognized types', () => {
        expect(getMacroCategory('hat')).toBe('other');
        expect(getMacroCategory('accessory')).toBe('other');
        expect(getMacroCategory('bag')).toBe('other');
        expect(getMacroCategory('')).toBe('other');
    });
});
