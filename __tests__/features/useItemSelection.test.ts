/**
 * Tests for the item selection hook used in outfit generation
 */

import { getMacroCategory } from '../../features/outfit-generator/hooks/useItemSelection';

describe('getMacroCategory', () => {
    it('categorizes sweaters/hoodies as outerwear', () => {
        expect(getMacroCategory('sweater')).toBe('outerwear');
        expect(getMacroCategory('Hoodie')).toBe('outerwear');
        expect(getMacroCategory('cardigan')).toBe('outerwear');
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

    it('categorizes accessories correctly', () => {
        expect(getMacroCategory('hat')).toBe('accessory');
        expect(getMacroCategory('bag')).toBe('accessory');
        expect(getMacroCategory('belt')).toBe('accessory');
    });

    it('returns "other" for unrecognized types', () => {
        expect(getMacroCategory('')).toBe('other');
        expect(getMacroCategory('something random')).toBe('other');
    });
});
