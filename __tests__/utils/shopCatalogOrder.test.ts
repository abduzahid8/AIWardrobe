import {
    buildCatalogSimilarityKey,
    spreadSimilarCatalogItems,
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
