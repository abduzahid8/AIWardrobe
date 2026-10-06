import {
    buildCatalogSimilarityKey,
    spreadSimilarCatalogItems,
    hasUsableProductName,
    inferCatalogColor,
    isTrustedProductImage,
    collapseDuplicateProductImages,
    getCatalogShoeFamily,
    garmentVarietyFamily,
    inferShopRowMacro,
} from '../../src/utils/shopCatalogOrder';

type TestCatalogItem = {
    id: string;
    name: string;
    garmentType: 'upper_body' | 'lower_body' | 'shoes' | 'outfit';
};

function makeItem(
    id: string,
    name: string,
    garmentType: TestCatalogItem['garmentType'] = 'upper_body',
): TestCatalogItem {
    return { id, name, garmentType };
}

describe('shopCatalogOrder', () => {
    it('collapses Zara basic tee variants into the same similarity key', () => {
        const heavyweight = makeItem('1', 'BASIC HEAVYWEIGHT T-SHIRT /03');
        const slimFit = makeItem('2', 'BASIC SLIM FIT T-SHIRT /01');
        const relaxed = makeItem('3', 'RELAXED FIT INTERLOCK T-SHIRT /04');

        expect(buildCatalogSimilarityKey(heavyweight)).toBe('upper_body:tee');
        expect(buildCatalogSimilarityKey(heavyweight)).toBe(buildCatalogSimilarityKey(slimFit));
        expect(buildCatalogSimilarityKey(heavyweight)).toBe(buildCatalogSimilarityKey(relaxed));
    });

    it('spreads similar items apart when other options exist', () => {
        const ordered = spreadSimilarCatalogItems([
            makeItem('1', 'BASIC HEAVYWEIGHT T-SHIRT /03'),
            makeItem('2', 'BASIC SLIM FIT T-SHIRT /01'),
            makeItem('3', 'RELAXED FIT INTERLOCK T-SHIRT /04'),
            makeItem('4', 'OXFORD SHIRT'),
            makeItem('5', 'KNIT POLO'),
            makeItem('6', 'STRAIGHT FIT JEANS', 'lower_body'),
        ]);

        const keys = ordered.map(buildCatalogSimilarityKey);
        for (let index = 1; index < keys.length; index++) {
            expect(keys[index]).not.toBe(keys[index - 1]);
        }
    });

    it('avoids repeating the last visible similarity key on appended pages', () => {
        const previousItems = [
            makeItem('1', 'LEATHER LOAFERS', 'shoes'),
            makeItem('2', 'OXFORD SHIRT'),
        ];

        const nextPage = spreadSimilarCatalogItems(
            [
                makeItem('3', 'BASIC SLIM FIT T-SHIRT /01'),
                makeItem('4', 'RELAXED FIT INTERLOCK T-SHIRT /04'),
                makeItem('5', 'STRAIGHT FIT JEANS', 'lower_body'),
            ],
            previousItems,
        );

        const lastVisibleKey = buildCatalogSimilarityKey(previousItems[previousItems.length - 1]);
        expect(buildCatalogSimilarityKey(nextPage[0])).not.toBe(lastVisibleKey);
        expect(buildCatalogSimilarityKey(nextPage[1])).not.toBe(buildCatalogSimilarityKey(nextPage[0]));
    });
});

describe('hasUsableProductName', () => {
    it('rejects blank, digit-only, and id-equal names', () => {
        expect(hasUsableProductName('', 'x')).toBe(false);
        expect(hasUsableProductName('   ', 'x')).toBe(false);
        expect(hasUsableProductName('63092782', 'md-63092782')).toBe(false);
        expect(hasUsableProductName('md-63092782', 'md-63092782')).toBe(false);
    });

    it('keeps real product names (even odd ones)', () => {
        expect(hasUsableProductName('Cool wool suit blazer', 'md-1')).toBe(true);
        expect(hasUsableProductName('100% linen shirt', 'md-2')).toBe(true);
        expect(hasUsableProductName('Levi\u2019s 501', 'lz-1')).toBe(true);
    });
});

describe('trusted product images', () => {
    it('accepts brand CDNs and files with image extensions', () => {
        expect(isTrustedProductImage('https://static.massimodutti.net/assets/public/x/blazer.jpg')).toBe(true);
        expect(isTrustedProductImage('https://example.com/loafers.jpg')).toBe(true);
        expect(isTrustedProductImage('https://image.uniqlo.com/UQ/ST3/WesternC/first.jpg')).toBe(true);
        expect(isTrustedProductImage('https://image.uniqlo.com/UQ/ST3/WesternC/no-ext')).toBe(true);
        expect(isTrustedProductImage('https://fyqpifmrsftsfqibhwhy.supabase.co/storage/v1/object/public/shop/x.jpg')).toBe(true);
    });

    it('rejects empty, non-http, and extension-less supabase roots', () => {
        expect(isTrustedProductImage('')).toBe(false);
        expect(isTrustedProductImage('placeholder_shoes')).toBe(false);
        expect(isTrustedProductImage('https://fyqpifmrsftsfqibhwhy.supabase.co')).toBe(false);
    });

    it('collapses rows that share one photo', () => {
        const url = 'https://static.massimodutti.net/assets/a.jpg';
        const out = collapseDuplicateProductImages([
            { id: '1', name: 'Wool blazer', imageUrl: url },
            { id: '2', name: 'Zip hoodie', imageUrl: url },
            { id: '3', name: 'Oxford', imageUrl: 'https://static.massimodutti.net/assets/b.jpg' },
        ]);
        expect(out.map((x) => x.id)).toEqual(['1', '3']);
    });
});

describe('inferCatalogColor', () => {
  it('reads colors from names, ignoring photo boilerplate', () => {
    expect(inferCatalogColor('Navy blazer', 'wool blazer')).toBe('navy');
    expect(inferCatalogColor('Long-sleeve striped t-shirt', 'ZARA · t-shirt · white-bg product shot')).toBe('neutral');
    expect(inferCatalogColor('Front view of a light beige tank top', '')).toBe('beige');
    expect(inferCatalogColor('Brown penny loafers', '')).toBe('brown');
    expect(inferCatalogColor('Mystery garment', 'no color words here')).toBe('neutral');
  });
});

describe('inferShopRowMacro', () => {
  it('does not put trousers in LAYER because the description mentions a blazer', () => {
    expect(inferShopRowMacro('lower_body', 'Cotton blend tailored pants', 'Pair with a cool wool suit blazer')).toBe('bottom');
  });

  it('drops a blazer title on a lower_body photo', () => {
    expect(inferShopRowMacro('lower_body', '100% cool wool suit blazer', '')).toBe('other');
  });

  it('keeps a real blazer stored as upper_body', () => {
    expect(inferShopRowMacro('upper_body', '100% cool wool suit blazer', '')).toBe('outerwear');
  });
});

describe('garmentVarietyFamily', () => {
  it('groups loafer clones and separates oxfords', () => {
    expect(getCatalogShoeFamily('Soft oiled leather penny loafers')).toBe('loafer');
    expect(getCatalogShoeFamily('Soft oiled leather loafers')).toBe('loafer');
    expect(getCatalogShoeFamily('Black oxford shoes')).toBe('dress-shoe');
    expect(garmentVarietyFamily({ macroCategory: 'shoes', name: 'Soft oiled leather penny loafers' }))
      .toBe(garmentVarietyFamily({ macroCategory: 'shoes', name: 'Leather loafers with penny strap' }));
    expect(garmentVarietyFamily({ macroCategory: 'shoes', name: 'Black oxford shoes' }))
      .not.toBe(garmentVarietyFamily({ macroCategory: 'shoes', name: 'Soft oiled leather loafers' }));
  });
});
