import { mapCategoryToType, mapColorToId } from '../../src/utils/mappingUtils';

describe('mappingUtils', () => {
  describe('mapColorToId', () => {
    it('returns exact match for direct colors', () => {
      expect(mapColorToId('black')).toBe('black');
      expect(mapColorToId('navy')).toBe('navy');
    });

    it('maps known synonyms correctly', () => {
      expect(mapColorToId('charcoal')).toBe('black');
      expect(mapColorToId('ivory')).toBe('white');
      expect(mapColorToId('sand')).toBe('beige');
      expect(mapColorToId('burgundy')).toBe('red');
      expect(mapColorToId('forest')).toBe('green');
    });

    it('handles case-insensitivity and whitespace', () => {
      expect(mapColorToId(' BURGUNDY ')).toBe('red');
      expect(mapColorToId('Light Blue')).toBe('blue');
    });

    it('falls back to grey for unknown colors', () => {
      expect(mapColorToId('transparent')).toBe('grey');
      expect(mapColorToId('')).toBe('grey');
      expect(mapColorToId(null as any)).toBe('grey');
    });
  });

  describe('mapCategoryToType', () => {
    it('evaluates conditions sequentially based on original logic', () => {
      expect(mapCategoryToType('random shirt', 'bottoms')).toBe('tops');
    });

    it('maps by category keyword when section is empty', () => {
      expect(mapCategoryToType('t-shirt')).toBe('tops');
      expect(mapCategoryToType('denim jean')).toBe('bottoms');
      expect(mapCategoryToType('winter coat')).toBe('outerwear');
      expect(mapCategoryToType('running sneaker')).toBe('shoes');
      expect(mapCategoryToType('leather belt')).toBe('accessories');
    });

    it('falls back to tops for unknown categories', () => {
      expect(mapCategoryToType('unknown')).toBe('tops');
      expect(mapCategoryToType('')).toBe('tops');
    });
  });
});
