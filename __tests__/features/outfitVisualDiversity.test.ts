/**
 * Product-level deck diversity — the invariant the Personal Stylist needs.
 *
 * The bug was never "three looks share an archetype". Same archetype is a
 * valid styling choice. The bug is showing the same actual PRODUCT twice
 * when the catalog has valid alternatives.
 *
 * Covered (A–J):
 *   A. exact product reuse is rejected
 *   B. same archetype + unique products is accepted
 *   C. different products of the same family are allowed
 *   D. duplicate catalog rows (same normalized name) count as one product
 *   E. duplicate images (thumbnail query variants) count as one product
 *   F. the locked anchor may repeat; every other product must be unique
 *   G. genuinely thin inventory relaxes with `insufficient_unique_products`
 *   H. enough products means zero non-anchor reuse
 *   I. archetype variety is never forced
 *   J. product uniqueness outranks a small quality difference
 */
import {
    analyzeDeckProductDiversity,
    assembleDiverseDeck,
    assessInventoryDiversity,
    buildOutfitFingerprint,
    garmentVisualFamily,
    productIdentityKeys,
    productIdentityOf,
    type DiversitySlot,
} from '../../features/outfit-generator/utils/outfitDiversity';

type Product = {
    id: string;
    name: string;
    macroCategory: DiversitySlot;
    image?: string;
    color?: string;
    material?: string;
};

const product = (
    id: string,
    macroCategory: DiversitySlot,
    name: string,
    extra: Partial<Product> = {},
): Product => ({
    id,
    name,
    macroCategory,
    image: `https://cdn.test/${id}.jpg`,
    ...extra,
});

type Look = { id: string; items: Product[]; score?: number };
const look = (id: string, items: Product[], score = 0.9): Look => ({ id, items, score });
const itemsOf = (value: Look) => value.items;

const formula = (prefix: string, label = 'look'): Look =>
    look(prefix, [
        product(`${prefix}_o`, 'outerwear', `${label} navy wool blazer`),
        product(`${prefix}_t`, 'top', `${label} white oxford shirt`),
        product(`${prefix}_b`, 'bottom', `${label} grey tailored trousers`),
        product(`${prefix}_s`, 'shoes', `${label} brown penny loafers`),
    ]);

const richPool = (): Product[] => {
    const pool: Product[] = [];
    const push = (prefix: string, slot: DiversitySlot, names: string[]) => {
        names.forEach((name, index) => pool.push(product(`${prefix}_${index}`, slot, name)));
    };
    push('blz', 'outerwear', [
        'Navy wool blazer', 'Charcoal wool blazer', 'Beige linen blazer',
        'Olive tweed blazer', 'Brown houndstooth blazer', 'Black dinner jacket',
    ]);
    push('top', 'top', [
        'White oxford shirt', 'Blue poplin shirt', 'Striped cotton shirt',
        'Navy knit polo', 'Grey merino sweater', 'Cream cashmere knit',
    ]);
    push('bot', 'bottom', [
        'Grey tailored trousers', 'Navy pleated trousers', 'Beige chinos',
        'Dark rinse jeans', 'Olive chinos', 'Black suit trousers',
    ]);
    push('sho', 'shoes', [
        'Brown penny loafers', 'Black penny loafers', 'Tan tassel loafers',
        'Brown derby shoes', 'Minimal white sneakers', 'Chelsea boots',
    ]);
    return pool;
};

const thinPool = (): Product[] => [
    product('thin_o', 'outerwear', 'Navy wool blazer'),
    product('thin_t', 'top', 'White oxford shirt'),
    product('thin_b', 'bottom', 'Grey tailored trousers'),
    product('thin_s', 'shoes', 'Brown penny loafers'),
];

/** Mirrors the production picker: product-fresh first, family only a soft preference. */
const fillBuilder = (pool: Product[]) => {
    const slots: DiversitySlot[] = ['outerwear', 'top', 'bottom', 'shoes'];
    return (usedKeys: Set<string>, attempt = 0): Look | null => {
        const items: Product[] = [];
        for (const [slotIndex, slot] of slots.entries()) {
            const inSlot = pool.filter((item) => item.macroCategory === slot);
            if (!inSlot.length) return null;
            const fresh = inSlot.filter((item) =>
                !productIdentityKeys(item).some((key) => usedKeys.has(key)),
            );
            const choices = fresh.length ? fresh : inSlot;
            const familyFresh = choices.filter((item) => {
                const family = garmentVisualFamily(item);
                return !family || !usedKeys.has(`__vf:${family}`);
            });
            const ordered = [...familyFresh, ...choices.filter((item) => !familyFresh.includes(item))];
            const pick = ordered[(attempt + slotIndex) % ordered.length];
            items.push(pick);
            productIdentityKeys(pick).forEach((key) => usedKeys.add(key));
            const family = garmentVisualFamily(pick);
            if (family) usedKeys.add(`__vf:${family}`);
        }
        return look(`fill_${attempt}`, items, 0.85);
    };
};

const deckProducts = (deck: Look[]) =>
    analyzeDeckProductDiversity(deck.map((entry) => entry.items));

describe('A. exact product reuse fails', () => {
    it('replaces a duplicated look with a unique one', () => {
        const base = formula('a', 'soft');
        const clone = look('clone', base.items.map((item) => ({ ...item })), 0.98);
        const third = formula('c', 'minimal');
        const fourth = formula('d', 'weekend');
        const result = assembleDiverseDeck<Look>({
            candidates: [look('base', base.items, 0.99), clone, third, fourth],
            itemsOf,
            keysOf: productIdentityKeys,
            usedKeys: new Set<string>(),
            poolItems: richPool(),
            buildFill: fillBuilder(richPool()),
        });
        expect(result.deck).toHaveLength(3);
        expect(result.deck.map((entry) => entry.id)).not.toContain('clone');
        expect(deckProducts(result.deck).nonAnchorProductReuseCount).toBe(0);
        expect(result.relaxed).toBe(false);
    });

    it('counts different rows of the same product as one product', () => {
        const a = product('sku-a', 'top', 'White oxford shirt', { image: 'https://cdn.test/a.jpg' });
        const b = product('sku-b', 'top', 'White Oxford Shirt', { image: 'https://cdn.test/b.jpg' });
        const diversity = analyzeDeckProductDiversity([[a], [b]]);
        expect(diversity.nonAnchorProductReuseCount).toBe(1);
        expect(diversity.repeatedProductKeys).toContain(productIdentityOf(a).canonicalKey);
    });
});

describe('B. same archetype + unique products is accepted', () => {
    it('keeps three blazer + shirt + trousers + loafer looks when the products differ', () => {
        const candidates = [formula('b1', 'navy'), formula('b2', 'charcoal'), formula('b3', 'beige')];
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf: productIdentityKeys,
            usedKeys: new Set<string>(),
            poolItems: richPool(),
        });
        expect(result.deck).toHaveLength(3);
        expect(result.relaxed).toBe(false);
        expect(new Set(result.archetypes).size).toBe(1);
        expect(deckProducts(result.deck).nonAnchorProductReuseCount).toBe(0);
    });
});

describe('C. different products of the same family are allowed', () => {
    it('does not treat two loafers as the same product', () => {
        const candidates = [
            look('c1', [
                product('c1_o', 'outerwear', 'Navy wool blazer'),
                product('c1_t', 'top', 'White oxford shirt'),
                product('c1_b', 'bottom', 'Grey tailored trousers'),
                product('c1_s', 'shoes', 'Brown penny loafers'),
            ]),
            look('c2', [
                product('c2_o', 'outerwear', 'Charcoal wool blazer'),
                product('c2_t', 'top', 'Blue poplin shirt'),
                product('c2_b', 'bottom', 'Navy pleated trousers'),
                product('c2_s', 'shoes', 'Black penny loafers'),
            ]),
            look('c3', [
                product('c3_o', 'outerwear', 'Beige linen blazer'),
                product('c3_t', 'top', 'Striped cotton shirt'),
                product('c3_b', 'bottom', 'Beige chinos'),
                product('c3_s', 'shoes', 'Tan tassel loafers'),
            ]),
        ];
        const families = candidates.map((entry) =>
            garmentVisualFamily(entry.items.find((item) => item.macroCategory === 'shoes')!),
        );
        expect(new Set(families)).toEqual(new Set(['shoe:loafer']));
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf: productIdentityKeys,
            usedKeys: new Set<string>(),
            poolItems: richPool(),
        });
        expect(result.deck).toHaveLength(3);
        expect(result.relaxed).toBe(false);
        expect(deckProducts(result.deck).nonAnchorProductReuseCount).toBe(0);
    });
});

describe('D. duplicate catalog rows are blocked', () => {
    it('normalizes the product name for identity', () => {
        const first = product('dup-a', 'top', 'White oxford shirt');
        const second = product('dup-b', 'top', '  WHITE   Oxford, Shirt! ');
        expect(productIdentityOf(second).canonicalKey).toBe(productIdentityOf(first).canonicalKey);
        expect(productIdentityKeys(second).some((key) => productIdentityKeys(first).includes(key))).toBe(true);
    });

    it('never puts both duplicate rows in the same deck', () => {
        const first = product('dup-a', 'top', 'White oxford shirt');
        const second = product('dup-b', 'top', 'White oxford shirt');
        const candidates = [
            look('d1', [
                product('d1_o', 'outerwear', 'Navy wool blazer'),
                first,
                product('d1_b', 'bottom', 'Grey tailored trousers'),
                product('d1_s', 'shoes', 'Brown penny loafers'),
            ]),
            look('d2', [
                product('d2_o', 'outerwear', 'Charcoal wool blazer'),
                second,
                product('d2_b', 'bottom', 'Navy pleated trousers'),
                product('d2_s', 'shoes', 'Black penny loafers'),
            ]),
            formula('d3', 'minimal'),
            formula('d4', 'weekend'),
        ];
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf: productIdentityKeys,
            usedKeys: new Set<string>(),
            poolItems: richPool(),
        });
        expect(result.deck).toHaveLength(3);
        expect(deckProducts(result.deck).nonAnchorProductReuseCount).toBe(0);
        const names = result.deck.flatMap((entry) => entry.items.map((item) => item.id));
        expect(names.filter((id) => id === 'dup-a').length + names.filter((id) => id === 'dup-b').length).toBeLessThan(2);
    });
});

describe('E. duplicate images are blocked', () => {
    it('strips thumbnail query params and protocol before comparing', () => {
        const first = product('img-a', 'shoes', 'Brown penny loafers', {
            image: 'https://cdn.test/product.jpg?width=300',
        });
        const second = product('img-b', 'shoes', 'Brown penny loafers large', {
            image: 'http://cdn.test/product.jpg?width=1200#hero',
        });
        expect(productIdentityOf(first).imageKey).toBe('i:cdn.test/product.jpg');
        expect(productIdentityOf(second).imageKey).toBe('i:cdn.test/product.jpg');
        expect(productIdentityKeys(first).some((key) => productIdentityKeys(second).includes(key))).toBe(true);
        expect(analyzeDeckProductDiversity([[first], [second]]).nonAnchorProductReuseCount).toBe(1);
    });

    it('rejects a pair whose only overlap is the product photo', () => {
        const first = product('photo-a', 'top', 'White oxford shirt', {
            image: 'https://cdn.test/shared.jpg?width=300',
        });
        const second = product('photo-b', 'top', 'White shirt regular fit', {
            image: 'https://cdn.test/shared.jpg?width=900',
        });
        const candidates = [
            look('e1', [
                product('e1_o', 'outerwear', 'Navy wool blazer'),
                first,
                product('e1_b', 'bottom', 'Grey tailored trousers'),
                product('e1_s', 'shoes', 'Brown penny loafers'),
            ]),
            look('e2', [
                product('e2_o', 'outerwear', 'Charcoal wool blazer'),
                second,
                product('e2_b', 'bottom', 'Navy pleated trousers'),
                product('e2_s', 'shoes', 'Black penny loafers'),
            ]),
            formula('e3', 'minimal'),
            formula('e4', 'weekend'),
        ];
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf: productIdentityKeys,
            usedKeys: new Set<string>(),
            poolItems: richPool(),
        });
        expect(result.deck).toHaveLength(3);
        expect(deckProducts(result.deck).nonAnchorProductReuseCount).toBe(0);
    });
});

describe('F. the anchor repeats by design', () => {
    it('excludes the anchor from product reuse while every other product rotates', () => {
        const anchor = product('anchor_blazer', 'outerwear', 'Navy wool blazer');
        const candidates = [
            look('f1', [anchor, product('f1_t', 'top', 'White oxford shirt'), product('f1_b', 'bottom', 'Grey tailored trousers'), product('f1_s', 'shoes', 'Brown penny loafers')]),
            look('f2', [anchor, product('f2_t', 'top', 'Blue poplin shirt'), product('f2_b', 'bottom', 'Navy pleated trousers'), product('f2_s', 'shoes', 'Black penny loafers')]),
            look('f3', [anchor, product('f3_t', 'top', 'Striped cotton shirt'), product('f3_b', 'bottom', 'Beige chinos'), product('f3_s', 'shoes', 'Tan tassel loafers')]),
        ];
        const anchorKeys = new Set(productIdentityKeys(anchor));
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf: productIdentityKeys,
            anchorKeys,
            anchorSlot: 'outerwear',
            usedKeys: new Set<string>(),
            poolItems: richPool(),
        });
        expect(result.deck).toHaveLength(3);
        for (const entry of result.deck) {
            expect(entry.items.some((item) => item.id === anchor.id)).toBe(true);
        }
        const diversity = analyzeDeckProductDiversity(
            result.deck.map((entry) => entry.items),
            { anchorKeys },
        );
        expect(diversity.nonAnchorProductReuseCount).toBe(0);
        expect(diversity.anchorLookCount).toBe(3);
        expect(diversity.anchorRepetitionCount).toBe(2);
    });
});

describe('G. genuinely thin inventory relaxes honestly', () => {
    it('reports insufficient_unique_products instead of pretending to be diverse', () => {
        const assessment = assessInventoryDiversity(thinPool(), { target: 3 });
        expect(assessment.supportsUniqueProducts).toBe(false);
        expect(assessment.maxDistinctProducts).toBe(1);
        expect(assessment.uniqueProductsPerSlot.top).toBe(1);

        const result = assembleDiverseDeck<Look>({
            candidates: [],
            itemsOf,
            keysOf: productIdentityKeys,
            usedKeys: new Set<string>(),
            poolItems: thinPool(),
            buildFill: fillBuilder(thinPool()),
        });
        // A wardrobe with ONE garment per slot can produce exactly ONE
        // distinct look. Repeating that look would show the user the same
        // card two or three times, so the deck is one card and says so.
        expect(result.deck).toHaveLength(1);
        expect(result.relaxed).toBe(true);
        expect(result.relaxReason).toBe('insufficient_unique_products');
    });

    it('repeats only the thin slot and keeps a different look beside it', () => {
        // Two tops, one bottom, one pair of shoes: the bottom and shoes are
        // forced, but the two looks still DIFFER (different tops), so both
        // cards are shown.
        const pool = [
            product('t1', 'top', 'White oxford shirt'),
            product('t2', 'top', 'Blue poplin shirt'),
            product('b1', 'bottom', 'Grey tailored trousers'),
            product('s1', 'shoes', 'Brown penny loafers'),
        ];
        const first = look('t1-first', [pool[0], pool[2], pool[3]]);
        const second = look('t2-second', [pool[1], pool[2], pool[3]]);
        const copy = look('t1-copy', [pool[0], pool[2], pool[3]]);
        const result = assembleDiverseDeck<Look>({
            candidates: [first, second, copy],
            itemsOf,
            keysOf: productIdentityKeys,
            usedKeys: new Set<string>(),
            poolItems: pool,
            activeSlots: ['top', 'bottom', 'shoes'],
            target: 2,
        });
        expect(result.deck).toHaveLength(2);
        expect(result.deck.map((entry) => entry.id)).toEqual(['t1-first', 't2-second']);
        expect(result.inventory.forcedRepeatSlots).toEqual(expect.arrayContaining(['bottom', 'shoes']));
        expect(result.productDiversity.illegalProductReuseCount).toBe(0);
    });
});

describe('H. enough products means zero non-anchor reuse', () => {
    it('fills a strict deck from a rich pool without repeating a product', () => {
        const result = assembleDiverseDeck<Look>({
            candidates: [],
            itemsOf,
            keysOf: productIdentityKeys,
            usedKeys: new Set<string>(),
            poolItems: richPool(),
            buildFill: fillBuilder(richPool()),
        });
        expect(result.deck).toHaveLength(3);
        expect(result.relaxed).toBe(false);
        expect(result.relaxReason).toBeNull();
        expect(result.inventory.supportsUniqueProducts).toBe(true);
        expect(result.productDiversity.nonAnchorProductReuseCount).toBe(0);
        expect(deckProducts(result.deck).nonAnchorProductReuseCount).toBe(0);
    });
});

describe('I. archetype variety is never forced', () => {
    it('returns three same-formula looks when their products are unique', () => {
        const candidates = [formula('i1', 'navy'), formula('i2', 'charcoal'), formula('i3', 'beige')];
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf: productIdentityKeys,
            usedKeys: new Set<string>(),
            poolItems: richPool(),
        });
        expect(result.deck).toHaveLength(3);
        expect(new Set(result.archetypes).size).toBe(1);
        expect(result.relaxReason).toBeNull();
        expect(result.productDiversity.nonAnchorProductReuseCount).toBe(0);
    });

    it('still reports formula similarity as a soft signal', () => {
        const fingerprints = [formula('j1', 'navy'), formula('j2', 'charcoal')].map((entry) =>
            buildOutfitFingerprint(entry.items),
        );
        expect(fingerprints[0].archetype).toBe(fingerprints[1].archetype);
    });
});

describe('J. uniqueness outranks a small quality difference', () => {
    it('drops a slightly better look when it repeats a product', () => {
        const shared = product('j_shared_o', 'outerwear', 'Navy wool blazer');
        const best = look('best', [
            shared,
            product('j_best_t', 'top', 'White oxford shirt'),
            product('j_best_b', 'bottom', 'Grey tailored trousers'),
            product('j_best_s', 'shoes', 'Brown penny loafers'),
        ], 0.99);
        const nearBest = look('near', [
            shared,
            product('j_near_t', 'top', 'Blue poplin shirt'),
            product('j_near_b', 'bottom', 'Navy pleated trousers'),
            product('j_near_s', 'shoes', 'Black penny loafers'),
        ], 0.98);
        const unique = look('unique', [
            product('j_unique_o', 'outerwear', 'Charcoal wool blazer'),
            product('j_unique_t', 'top', 'Striped cotton shirt'),
            product('j_unique_b', 'bottom', 'Beige chinos'),
            product('j_unique_s', 'shoes', 'Tan tassel loafers'),
        ], 0.70);
        const result = assembleDiverseDeck<Look>({
            candidates: [best, nearBest, unique],
            itemsOf,
            keysOf: productIdentityKeys,
            qualityOf: (entry) => entry.score ?? 0.9,
            usedKeys: new Set<string>(),
            poolItems: richPool(),
            target: 2,
        });
        expect(result.deck).toHaveLength(2);
        expect(result.deck.map((entry) => entry.id)).toEqual(['best', 'unique']);
        expect(result.productDiversity.nonAnchorProductReuseCount).toBe(0);
    });
});
