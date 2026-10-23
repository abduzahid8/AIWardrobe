/**
 * Deck selection contract for the Personal Stylist.
 *
 * The invariant: a non-anchor actual PRODUCT may appear in more than one
 * final look ONLY when its slot is genuinely too small to fill the deck.
 * Everything else — the outfit formula, the garment family, the colour — is
 * free.
 *
 * Covered (A–J):
 *   A. the greedy trap — a strong look that conflicts with three others must
 *      not starve the deck
 *   B. the same formula with different products is a valid deck
 *   C. two catalog rows with the same name are ONE product
 *   D. non-Latin names keep their identity (Cyrillic)
 *   E. an irrelevant outerwear slot must not make a non-layered request thin
 *   F. the anchor slot is excluded from capacity; the anchor may repeat
 *   G. a thin slot may repeat; a rich slot in the SAME deck may not
 *   H. a rich inventory never relaxes — it returns a smaller unique deck
 *   I. a larger deck target is judged against the larger target
 *   J. the authoritative capacity verdict (the edge) wins over the client's
 */
import {
    analyzeDeckProductDiversity,
    assembleDiverseDeck,
    assessInventoryDiversity,
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

const makeLook = (prefix: string, label: string, slotProducts: Partial<Record<DiversitySlot, Product>>): Look =>
    look(prefix, [
        slotProducts.outerwear ?? product(`${prefix}_o`, 'outerwear', `${label} navy wool blazer`),
        slotProducts.top ?? product(`${prefix}_t`, 'top', `${label} white oxford shirt`),
        slotProducts.bottom ?? product(`${prefix}_b`, 'bottom', `${label} grey tailored trousers`),
        slotProducts.shoes ?? product(`${prefix}_s`, 'shoes', `${label} brown penny loafers`),
    ]);

/** Pool with several distinct products in every slot. */
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

/** Builds replacement looks that rotate through the pool, product-fresh first. */
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
            const pick = choices[(attempt + slotIndex) % choices.length];
            items.push(pick);
            productIdentityKeys(pick).forEach((key) => usedKeys.add(key));
        }
        return look(`fill_${attempt}`, items, 0.85);
    };
};

describe('A. the greedy trap', () => {
    it('drops the strongest look when it would starve the deck', () => {
        // A conflicts with every other candidate. B, C and D are mutually
        // unique, so the correct deck is B + C + D even though A scores
        // higher than all of them.
        // A conflicts with every other candidate: it shares the top with B,
        // the bottom with C and the shoes with D. Accepting A first would
        // leave a deck of two.
        const sharedTop = product('trap_top', 'top', 'White oxford shirt');
        const sharedBottom = product('trap_bottom', 'bottom', 'Grey tailored trousers');
        const sharedShoes = product('trap_shoes', 'shoes', 'Brown penny loafers');
        const a = look('A', [
            product('a_o', 'outerwear', 'Navy wool blazer'),
            sharedTop,
            sharedBottom,
            sharedShoes,
        ], 0.99);
        const b = makeLook('b', 'charcoal', { top: sharedTop });
        const c = makeLook('c', 'beige', { bottom: sharedBottom });
        const d = makeLook('d', 'olive', { shoes: sharedShoes });
        const result = assembleDiverseDeck<Look>({
            candidates: [a, b, c, d],
            itemsOf,
            keysOf: productIdentityKeys,
            qualityOf: (entry) => entry.score ?? 0.9,
            usedKeys: new Set<string>(),
            poolItems: richPool(),
            target: 3,
        });
        expect(result.deck).toHaveLength(3);
        expect(result.deck.map((entry) => entry.id)).not.toContain('A');
        expect(result.deck.map((entry) => entry.id).sort()).toEqual(['b', 'c', 'd']);
        expect(result.productDiversity.illegalProductReuseCount).toBe(0);
    });
});

describe('B. the same formula is legal', () => {
    it('accepts three blazer + shirt + trousers + loafer looks with unique products', () => {
        const candidates = [
            makeLook('b1', 'navy', {}),
            makeLook('b2', 'charcoal', {}),
            makeLook('b3', 'beige', {}),
        ];
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf: productIdentityKeys,
            usedKeys: new Set<string>(),
            poolItems: richPool(),
            target: 3,
        });
        expect(result.deck).toHaveLength(3);
        expect(result.relaxed).toBe(false);
        expect(result.relaxReason).toBeNull();
        expect(new Set(result.archetypes).size).toBe(1);
        expect(result.productDiversity.illegalProductReuseCount).toBe(0);
    });
});

describe('C. two rows of one product are one product', () => {
    it('keeps only one of two rows that share a normalized name', () => {
        const first = product('row-a', 'top', 'White oxford shirt');
        const second = product('row-b', 'top', '  WHITE   Oxford, Shirt! ');
        expect(productIdentityOf(second).canonicalKey).toBe(productIdentityOf(first).canonicalKey);

        const candidates = [
            makeLook('c1', 'navy', { top: first }),
            makeLook('c2', 'charcoal', { top: second }),
            makeLook('c3', 'beige', {}),
        ];
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf: productIdentityKeys,
            usedKeys: new Set<string>(),
            poolItems: richPool(),
            target: 3,
        });
        // Two of the three candidates share one product, so the largest
        // UNIQUE deck is two looks — never a repeat.
        expect(result.deck).toHaveLength(2);
        expect(result.relaxReason).toBe('insufficient_unique_candidates');
        expect(result.productDiversity.illegalProductReuseCount).toBe(0);
        const topIds = result.deck.flatMap((entry) =>
            entry.items.filter((item) => item.macroCategory === 'top').map((item) => item.id),
        );
        expect(new Set(topIds).size).toBe(topIds.length);
    });
});

describe('D. non-Latin names keep their identity', () => {
    it('does not collapse two different Cyrillic products into one', () => {
        const first = product('cyr-a', 'top', 'Белая рубашка');
        const second = product('cyr-b', 'top', 'Синяя рубашка');
        expect(productIdentityOf(first).canonicalKey).not.toBe(productIdentityOf(second).canonicalKey);
        expect(productIdentityKeys(first).some((key) => productIdentityKeys(second).includes(key))).toBe(false);
    });

    it('still treats the same Cyrillic name as the same product', () => {
        const first = product('cyr-c', 'top', 'Белая рубашка');
        const second = product('cyr-d', 'top', 'Белая  рубашка!');
        expect(productIdentityOf(second).canonicalKey).toBe(productIdentityOf(first).canonicalKey);
        expect(analyzeDeckProductDiversity([[first], [second]]).nonAnchorProductReuseCount).toBe(1);
    });

    it('does not let a non-Latin name collapse to an empty identity', () => {
        const item = product('cyr-e', 'top', 'Рубашка');
        expect(productIdentityOf(item).nameKey.length).toBeGreaterThan(0);
    });
});

describe('E. an irrelevant outerwear slot is not capacity', () => {
    it('does not mark a non-layered request thin because of one outerwear item', () => {
        const pool = richPool().filter((item) => item.macroCategory !== 'outerwear');
        pool.push(product('single_outer', 'outerwear', 'Navy wool blazer'));

        const assessment = assessInventoryDiversity(pool, {
            target: 3,
            activeSlots: ['top', 'bottom', 'shoes'],
        });
        expect(assessment.activeSlots).toEqual(['top', 'bottom', 'shoes']);
        expect(assessment.supportsUniqueProducts).toBe(true);
        expect(assessment.forcedRepeatSlots).toEqual([]);
        expect(assessment.strictSlots).toEqual(['top', 'bottom', 'shoes']);
    });

    it('keeps a three-slot deck unique when the pool has one outerwear item', () => {
        const pool = richPool().filter((item) => item.macroCategory !== 'outerwear');
        pool.push(product('single_outer', 'outerwear', 'Navy wool blazer'));
        const candidates = [
            makeLook('e1', 'navy', {}),
            makeLook('e2', 'charcoal', {}),
            makeLook('e3', 'beige', {}),
        ];
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf: productIdentityKeys,
            usedKeys: new Set<string>(),
            poolItems: pool,
            activeSlots: ['top', 'bottom', 'shoes'],
            target: 3,
        });
        expect(result.deck).toHaveLength(3);
        expect(result.relaxed).toBe(false);
        expect(result.productDiversity.illegalProductReuseCount).toBe(0);
    });
});

describe('F. the anchor slot is excluded from capacity', () => {
    it('lets the anchor repeat while every other product rotates', () => {
        const anchor = product('anchor_blazer', 'outerwear', 'Navy wool blazer');
        const candidates = [
            makeLook('f1', 'navy', { outerwear: anchor }),
            makeLook('f2', 'charcoal', { outerwear: anchor }),
            makeLook('f3', 'beige', { outerwear: anchor }),
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
            target: 3,
        });
        expect(result.deck).toHaveLength(3);
        expect(result.deck.every((entry) => entry.items.some((item) => item.id === anchor.id))).toBe(true);
        expect(result.inventory.activeSlots).not.toContain('outerwear');
        const diversity = analyzeDeckProductDiversity(
            result.deck.map((entry) => entry.items),
            { anchorKeys },
        );
        expect(diversity.nonAnchorProductReuseCount).toBe(0);
        expect(diversity.anchorRepetitionCount).toBe(2);
    });
});

describe('G. only a genuinely thin slot may repeat', () => {
    it('repeats the thin shoe slot but never the rich top slot', () => {
        const onlyShoes = product('one_shoes', 'shoes', 'Brown penny loafers');
        const pool = richPool().filter((item) => item.macroCategory !== 'shoes');
        pool.push(onlyShoes);

        const candidates = [
            makeLook('g1', 'navy', { shoes: onlyShoes }),
            makeLook('g2', 'charcoal', { shoes: onlyShoes }),
            makeLook('g3', 'beige', { shoes: onlyShoes }),
        ];
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf: productIdentityKeys,
            usedKeys: new Set<string>(),
            poolItems: pool,
            activeSlots: ['top', 'bottom', 'shoes'],
            target: 3,
        });
        expect(result.deck).toHaveLength(3);
        expect(result.inventory.forcedRepeatSlots).toEqual(['shoes']);
        expect(result.inventory.strictSlots).toEqual(expect.arrayContaining(['top', 'bottom']));
        expect(result.relaxed).toBe(true);
        expect(result.relaxReason).toBe('insufficient_unique_products');
        // The shoe repeat is legitimate; no OTHER product repeats.
        expect(result.productDiversity.forcedRepeatProducts).toHaveLength(1);
        expect(result.productDiversity.illegalProductReuseCount).toBe(0);
    });
});

describe('H. a rich inventory never relaxes', () => {
    it('returns the largest unique subset instead of repeating a product', () => {
        // Two candidates that share a product cannot both be in the deck, and
        // there is no third candidate: the answer is ONE card, not a repeat.
        const shared = product('h_shared', 'top', 'White oxford shirt');
        const candidates = [
            makeLook('h1', 'navy', { top: shared }),
            makeLook('h2', 'charcoal', { top: shared }),
        ];
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf: productIdentityKeys,
            usedKeys: new Set<string>(),
            poolItems: richPool(),
            target: 3,
        });
        expect(result.deck).toHaveLength(1);
        expect(result.relaxed).toBe(true);
        expect(result.relaxReason).toBe('insufficient_unique_candidates');
        expect(result.productDiversity.illegalProductReuseCount).toBe(0);
        expect(result.inventory.supportsUniqueProducts).toBe(true);
    });
});

describe('H2. a verbatim duplicate look is never shown twice', () => {
    it('caps the deck instead of repeating the same products in every slot', () => {
        // Every slot is thin (one bottom, one pair of shoes), so repetition
        // IS allowed there — but a look that repeats the products of another
        // look in EVERY slot is the same card. The deck must shrink instead.
        const oneBottom = product('dup_b', 'bottom', 'Grey tailored trousers');
        const oneShoes = product('dup_s', 'shoes', 'Brown penny loafers');
        const layer = product('dup_o', 'outerwear', 'Navy wool blazer');
        const pool = [
            product('dup_t1', 'top', 'White oxford shirt'),
            product('dup_t2', 'top', 'Blue poplin shirt'),
            oneBottom,
            oneShoes,
            layer,
        ];
        // All three looks carry the same layer as well, so x1 and the copy
        // are the SAME card (identical products in every slot).
        const first = makeLook('x1', 'navy', { outerwear: layer, top: pool[0], bottom: oneBottom, shoes: oneShoes });
        const second = makeLook('x2', 'charcoal', { outerwear: layer, top: pool[1], bottom: oneBottom, shoes: oneShoes });
        const copy = makeLook('x3', 'beige', { outerwear: layer, top: pool[0], bottom: oneBottom, shoes: oneShoes });
        const result = assembleDiverseDeck<Look>({
            candidates: [first, second, copy],
            itemsOf,
            keysOf: productIdentityKeys,
            usedKeys: new Set<string>(),
            poolItems: pool,
            activeSlots: ['top', 'outerwear', 'bottom', 'shoes'],
            target: 3,
        });
        expect(result.deck).toHaveLength(2);
        const topIds = (result.deck as Look[]).map((entry) =>
            entry.items.filter((item) => item.macroCategory === 'top').map((item) => item.id).join('|'),
        );
        expect(new Set(topIds).size).toBe(2);
        expect(result.relaxReason).toBe('insufficient_unique_products');
        expect(result.productDiversity.illegalProductReuseCount).toBe(0);
    });
});

describe('I. the deck target drives the capacity', () => {
    it('judges the slot against the requested deck size', () => {
        const threeShoes = [
            product('i_s1', 'shoes', 'Brown penny loafers'),
            product('i_s2', 'shoes', 'Black penny loafers'),
            product('i_s3', 'shoes', 'Tan tassel loafers'),
        ];
        const pool = richPool().filter((item) => item.macroCategory !== 'shoes');
        pool.push(...threeShoes);

        const small = assessInventoryDiversity(pool, { target: 3, activeSlots: ['top', 'bottom', 'shoes'] });
        expect(small.supportsUniqueProducts).toBe(true);
        expect(small.forcedRepeatSlots).toEqual([]);

        const large = assessInventoryDiversity(pool, { target: 5, activeSlots: ['top', 'bottom', 'shoes'] });
        expect(large.supportsUniqueProducts).toBe(false);
        expect(large.forcedRepeatSlots).toEqual(expect.arrayContaining(['shoes']));
    });
});

describe('J. the authoritative capacity verdict wins', () => {
    it('accepts the edge\'s forced repeat even when the client pool looks rich', () => {
        const onlyShoes = product('j_shoes', 'shoes', 'Brown penny loafers');
        const candidates = [
            makeLook('j1', 'navy', { shoes: onlyShoes }),
            makeLook('j2', 'charcoal', { shoes: onlyShoes }),
            makeLook('j3', 'beige', { shoes: onlyShoes }),
        ];
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf: productIdentityKeys,
            usedKeys: new Set<string>(),
            // The client pool is rich (it has six pairs of shoes) ...
            poolItems: richPool(),
            activeSlots: ['top', 'bottom', 'shoes'],
            // ... but the edge, which owns the valid pool, says the shoe slot
            // is thin. The edge verdict wins.
            capacity: {
                activeSlots: ['top', 'bottom', 'shoes'],
                strictSlots: ['top', 'bottom'],
                forcedRepeatSlots: ['shoes'],
            },
            target: 3,
        });
        expect(result.deck).toHaveLength(3);
        expect(result.inventory.forcedRepeatSlots).toContain('shoes');
        expect(result.productDiversity.illegalProductReuseCount).toBe(0);
    });

    it('forbids a repeat the edge says is strict, even when the client pool looks thin', () => {
        const shared = product('k_shared', 'top', 'White oxford shirt');
        const candidates = [
            makeLook('k1', 'navy', { top: shared }),
            makeLook('k2', 'charcoal', { top: shared }),
            makeLook('k3', 'beige', {}),
        ];
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf: productIdentityKeys,
            usedKeys: new Set<string>(),
            poolItems: [shared],
            activeSlots: ['top', 'bottom', 'shoes'],
            capacity: {
                activeSlots: ['top', 'bottom', 'shoes'],
                strictSlots: ['top', 'bottom', 'shoes'],
                forcedRepeatSlots: [],
            },
            target: 3,
        });
        // k1 and k2 share one product, so the largest UNIQUE deck is two.
        expect(result.deck).toHaveLength(2);
        expect(result.productDiversity.illegalProductReuseCount).toBe(0);
    });
});
