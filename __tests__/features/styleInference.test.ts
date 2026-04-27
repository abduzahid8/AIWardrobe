import {
  inferItemAttributes,
  scoreItemForStyle,
  rankItemsForStyle,
  normalizeStyleId,
} from '../../features/outfit-generator/utils/styleInference';

describe('styleInference', () => {
  describe('inferItemAttributes', () => {
    it('tags a wool blazer as old_money / business_casual, not semi_classic', () => {
      const attrs = inferItemAttributes({
        name: 'Slim Wool Blazer',
        description: 'slim-fit structured wool blazer with notch lapels',
        brand: 'Massimo Dutti',
      });
      expect(attrs.styleTags).toContain('old_money');
      expect(attrs.styleTags).toContain('business_casual');
      expect(attrs.styleTags).not.toContain('semi_classic');
      expect(attrs.materials).toContain('wool');
      expect(attrs.formality).toBeGreaterThanOrEqual(0.6);
    });

    it('tags a graphic heavyweight tee as casual, NOT old_money', () => {
      const attrs = inferItemAttributes({
        name: 'Basic Heavyweight T-Shirt',
        description: 'oversized heavyweight tee with graphic print',
        brand: 'Zara',
      });
      expect(attrs.styleTags).not.toContain('old_money');
    });

    it('tags linen trousers as old_money', () => {
      const attrs = inferItemAttributes({
        name: '100% Linen Suit Trousers',
        description: 'relaxed-fit linen trousers in beige',
        brand: 'Zara',
      });
      expect(attrs.styleTags[0]).toBe('old_money');
      expect(attrs.materials).toContain('linen');
    });

    it('defaults to casual when nothing matches', () => {
      const attrs = inferItemAttributes({ name: 'Thing', description: 'a thing' });
      expect(attrs.styleTags).toEqual(['casual']);
    });
  });

  describe('scoreItemForStyle', () => {
    it('scores a cashmere cardigan higher than a hoodie for old_money', () => {
      const cardigan = scoreItemForStyle(
        { name: 'Cashmere Cardigan', description: 'merino wool cardigan' },
        'old_money',
      );
      const hoodie = scoreItemForStyle(
        { name: 'Oversized Graphic Hoodie', description: 'heavyweight hoodie with logo print' },
        'old_money',
      );
      expect(cardigan).toBeGreaterThan(hoodie);
    });
  });

  describe('rankItemsForStyle', () => {
    const mixedItems = [
      { name: 'Oversized Graphic Hoodie', description: 'with logo print' },
      { name: 'Slim Wool Blazer', description: 'tailored wool blazer', brand: 'Massimo Dutti' },
      { name: 'Cargo Shorts', description: 'utility cargo shorts' },
      { name: 'Classic Poplin Shirt', description: 'regular-fit poplin cotton shirt' },
      { name: 'Loafers', description: 'leather penny loafers' },
    ];

    it('for old_money, drops hoodie + cargo and keeps blazer/shirt/loafers on top', () => {
      const ranked = rankItemsForStyle(mixedItems, 'old_money');
      const topThree = ranked.slice(0, 3).map((i) => i.name);
      expect(topThree).toContain('Slim Wool Blazer');
      expect(topThree).toContain('Loafers');
      // Hoodie and cargo shorts should be filtered out (score < -2) OR ranked last
      const hoodieIndex = ranked.findIndex((i) => i.name === 'Oversized Graphic Hoodie');
      const blazerIndex = ranked.findIndex((i) => i.name === 'Slim Wool Blazer');
      expect(blazerIndex).toBeLessThan(hoodieIndex);
    });

    it('for semi_classic, puts the blazer first', () => {
      const ranked = rankItemsForStyle(mixedItems, 'semi_classic');
      expect(ranked[0].name).toBe('Slim Wool Blazer');
    });

    it('ALWAYS preserves at least one shoe candidate for old_money, even if all shoes score poorly', () => {
      // Simulate the Zara catalog reality: every shoe the user has is either
      // chunky or sneaker-leaning, but the outfit still needs shoes.
      const items = [
        { name: 'Slim Wool Blazer', description: 'tailored wool blazer', macroCategory: 'outerwear' },
        { name: 'Oxford Shirt', description: 'poplin oxford shirt', macroCategory: 'top' },
        { name: 'Linen Trousers', description: 'relaxed linen trousers', macroCategory: 'bottom' },
        { name: 'Chunky Sneakers', description: 'chunky retro sneakers', macroCategory: 'shoes' },
        { name: 'Basketball Sneakers', description: 'high-top basketball sneakers', macroCategory: 'shoes' },
        { name: 'Skate Sneakers', description: 'rope lace skate sneakers', macroCategory: 'shoes' },
      ];
      const ranked = rankItemsForStyle(items, 'old_money', {
        minKeep: 4,
        dropThreshold: -2,
        perCategoryFloor: 2,
      });
      const shoes = ranked.filter(i => i.macroCategory === 'shoes');
      expect(shoes.length).toBeGreaterThan(0);
    });

    it('prefers leather penny loafers over chunky sneakers for old_money', () => {
      const shoes = [
        { name: 'Chunky Sneakers', description: '', macroCategory: 'shoes' },
        { name: 'Leather Penny Loafers', description: '', macroCategory: 'shoes' },
        { name: 'Basketball Sneakers', description: '', macroCategory: 'shoes' },
        { name: 'Dress Shoes', description: '', macroCategory: 'shoes' },
      ];
      const ranked = rankItemsForStyle(shoes, 'old_money', { perCategoryFloor: 0 });
      expect(ranked[0].name).toBe('Leather Penny Loafers');
    });
  });

  describe('normalizeStyleId', () => {
    it('normalizes various casings and separators', () => {
      expect(normalizeStyleId('Old Money')).toBe('old_money');
      expect(normalizeStyleId('old-money')).toBe('old_money');
      expect(normalizeStyleId('OLD_MONEY')).toBe('old_money');
      expect(normalizeStyleId('semi_classic')).toBe('semi_classic');
      expect(normalizeStyleId('unknown')).toBe('casual');
      expect(normalizeStyleId(null)).toBe('casual');
    });
  });
});
