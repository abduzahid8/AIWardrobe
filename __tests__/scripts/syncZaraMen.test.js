const {
    areRowsTooSimilar,
    buildProductKey,
    buildSimilarityKey,
    categoryToGarmentType,
    isSupportedMensClothing,
    normalizeText,
} = require('../../scripts/sync-zara-men.js');

describe('sync-zara-men helpers', () => {
    it('normalizes duplicate product names consistently', () => {
        expect(normalizeText('  DRESS PENNY LOAFERS  ')).toBe('dress penny loafers');
        expect(normalizeText('Dress-Penny Loafers')).toBe('dress penny loafers');
    });

    it('builds the same product key for duplicate Zara URL slugs', () => {
        const left = buildProductKey({
            name: 'DRESS PENNY LOAFERS',
            category: 'shoes',
            source_url: 'https://www.zara.com/us/en/dress-penny-loafers-p12685720.html',
        });
        const right = buildProductKey({
            name: 'DRESS PENNY LOAFERS',
            category: 'shoes',
            source_url: 'https://www.zara.com/us/en/dress-penny-loafers-p12628720.html',
        });

        expect(left).toBe('shoes:dress-penny-loafers');
        expect(left).toBe(right);
    });

    it('falls back to normalized category and name when no product URL exists', () => {
        const left = buildProductKey({
            name: 'Regular Fit Shirt',
            category: 'tops',
        });
        const right = buildProductKey({
            name: 'REGULAR-FIT SHIRT',
            category: 'tops',
        });

        expect(left).toBe('tops:regular fit shirt');
        expect(left).toBe(right);
    });

    it('filters out sporty and technical menswear', () => {
        expect(
            isSupportedMensClothing(
                { name: 'TRAINING SNEAKERS' },
                'Shoes',
                'https://www.zara.com/us/en/training-sneakers-p12381720.html',
            ),
        ).toBe(false);

        expect(
            isSupportedMensClothing(
                { name: 'LIGHTWEIGHT WATER REPELLENT TECHNICAL JACKET' },
                'Jackets',
                'https://www.zara.com/us/en/lightweight-water-repellent-technical-jacket-p03918206.html',
            ),
        ).toBe(false);

        expect(
            isSupportedMensClothing(
                { name: 'COMFORT JOGGER WAIST PANTS' },
                'Trousers',
                'https://www.zara.com/us/en/comfort-jogger-waist-pants-p06861693.html',
            ),
        ).toBe(false);

        expect(
            isSupportedMensClothing(
                { name: '3-PACK OF BASIC MEDIUM WEIGHT T-SHIRTS /02' },
                'Basic T-Shirts',
                'https://www.zara.com/us/en/3-pack-of-basic-medium-weight-t-shirts-p00000002.html',
            ),
        ).toBe(false);
    });

    it('keeps classic menswear items', () => {
        expect(
            isSupportedMensClothing(
                { name: '100% LINEN SUIT PANTS' },
                'Trousers',
                'https://www.zara.com/us/en/100-linen-suit-pants-p04410411.html',
            ),
        ).toBe(true);

        expect(
            isSupportedMensClothing(
                { name: 'DRESS PENNY LOAFERS' },
                'Shoes',
                'https://www.zara.com/us/en/dress-penny-loafers-p12685720.html',
            ),
        ).toBe(true);

        expect(
            isSupportedMensClothing(
                { name: 'WOOL SPORT COAT' },
                'Blazers',
                'https://www.zara.com/us/en/wool-sport-coat-p04391200.html',
            ),
        ).toBe(true);
    });

    it('builds the same similarity key for plain Zara basics', () => {
        const left = buildSimilarityKey({
            name: 'REGULAR FIT COTTON T-SHIRT',
            category: 'tops',
        });
        const right = buildSimilarityKey({
            name: 'SOFT TOUCH TEE',
            category: 'tops',
        });

        expect(left).toBe('tops:tee');
        expect(left).toBe(right);
    });

    it('filters near-duplicate Zara basics but keeps distinct silhouettes', () => {
        expect(
            areRowsTooSimilar(
                {
                    name: 'REGULAR FIT COTTON T-SHIRT',
                    category: 'tops',
                    source_url: 'https://www.zara.com/us/en/regular-fit-cotton-t-shirt-p123.html',
                },
                {
                    name: 'SOFT TOUCH TEE',
                    category: 'tops',
                    source_url: 'https://www.zara.com/us/en/soft-touch-tee-p456.html',
                },
            ),
        ).toBe(true);

        expect(
            areRowsTooSimilar(
                {
                    name: 'BASIC HEAVYWEIGHT T-SHIRT /03',
                    category: 'tops',
                    source_url: 'https://www.zara.com/us/en/basic-heavyweight-t-shirt-p123.html',
                },
                {
                    name: 'RELAXED FIT INTERLOCK T-SHIRT /04',
                    category: 'tops',
                    source_url: 'https://www.zara.com/us/en/relaxed-fit-interlock-t-shirt-p456.html',
                },
            ),
        ).toBe(true);

        expect(
            areRowsTooSimilar(
                {
                    name: 'LINEN SHIRT',
                    category: 'tops',
                    source_url: 'https://www.zara.com/us/en/linen-shirt-p123.html',
                },
                {
                    name: 'OXFORD SHIRT',
                    category: 'tops',
                    source_url: 'https://www.zara.com/us/en/oxford-shirt-p456.html',
                },
            ),
        ).toBe(false);
    });

    it('maps core clothing types into the expected catalog categories', () => {
        expect(categoryToGarmentType('dress penny loafers')).toEqual({
            garment_type: 'shoes',
            category: 'shoes',
        });

        expect(categoryToGarmentType('regular fit shirt')).toEqual({
            garment_type: 'upper_body',
            category: 'tops',
        });

        expect(categoryToGarmentType('100% linen suit pants')).toEqual({
            garment_type: 'lower_body',
            category: 'bottoms',
        });
    });

    it('classifies "STRIPED OXFORD SHIRT" as a top, not shoes', () => {
        expect(categoryToGarmentType('striped oxford shirt')).toEqual({
            garment_type: 'upper_body',
            category: 'tops',
        });
        expect(categoryToGarmentType('oxford shirt')).toEqual({
            garment_type: 'upper_body',
            category: 'tops',
        });
        // Bare "oxford" (as used for oxford dress shoes in Zara feeds) still
        // correctly routes to shoes.
        expect(categoryToGarmentType('leather oxford')).toEqual({
            garment_type: 'shoes',
            category: 'shoes',
        });
    });
});
