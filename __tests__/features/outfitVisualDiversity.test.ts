/**
 * outfitVisualDiversity — the VISUAL / STYLISTIC diversity layer.
 *
 * Product dedupe guarantees that two looks never repeat the same SKU. That
 * is not enough: a user seeing
 *
 *   navy blazer  + white oxford  + grey tailored trousers + brown loafers
 *   charcoal     + blue poplin   + black trousers       + penny loafers
 *   beige        + striped shirt + navy trousers       + tassel loafers
 *
 * has seen ONE outfit three times. These tests pin the invariant that the
 * three cards must be three DIFFERENT STYLING DIRECTIONS whenever the
 * inventory allows it.
 *
 * Covered: A (same formula, different ids) · B (same family, different ids)
 * · C (truly diverse) · D (same style, different concepts) · E (black tie)
 * · F (anchor) · G (thin inventory) · H (large inventory never relaxes)
 * · I (full-look archetype regression) · J (pairwise similarity ceiling).
 */
import {
    assembleDiverseDeck,
    assessInventoryDiversity,
    buildOutfitFingerprint,
    describeFingerprint,
    garmentVisualFamily,
    MAX_VISUAL_SIMILARITY,
    sameArchetype,
    sharedVisualFamilies,
    visualSimilarity,
    type DiversitySlot,
} from '../../features/outfit-generator/utils/outfitDiversity';

// ── Fixtures ──────────────────────────────────────────────────────────────

type Product = {
    id: string;
    name: string;
    macroCategory: DiversitySlot;
    image?: string;
    color?: string;
    material?: string;
    description?: string;
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

const itemsOf = (l: Look) => l.items;
const keysOf = (item: Product) => [String(item.id)];

/**
 * Fill builder mirroring the production slot picker: prefer a garment whose
 * VISUAL FAMILY is not used yet (a different garment TYPE, not another SKU
 * of the same one), then fall back to any unused product.
 */
const fillBuilder = (pool: Product[]) => {
    const slots: DiversitySlot[] = ['outerwear', 'top', 'bottom', 'shoes'];
    return (usedKeys: Set<string>, attempt = 0): Look | null => {
        const items: Product[] = [];
        for (const [slotIndex, slot] of slots.entries()) {
            const inSlot = pool.filter((p) => p.macroCategory === slot);
            if (!inSlot.length) return null;
            const fresh = inSlot.filter((p) => !usedKeys.has(String(p.id)));
            const unusedFamily = fresh.filter((p) => !usedKeys.has(`__vf:${garmentVisualFamily(p)}`));
            const choices = unusedFamily.length ? unusedFamily : fresh.length ? fresh : inSlot;
            const rotation = attempt + slotIndex * 2;
            const pick = choices[rotation % choices.length];
            items.push(pick);
            usedKeys.add(String(pick.id));
            usedKeys.add(`__vf:${garmentVisualFamily(pick)}`);
        }
        return look(`fill_${attempt}`, items, 0.85);
    };
};

/** The §10 catalogue: rich enough for three genuinely different directions. */
const richPool = (): Product[] => {
    const pool: Product[] = [];
    const push = (
        prefix: string,
        slot: DiversitySlot,
        names: string[],
        extra: Partial<Product> = {},
    ) => {
        names.forEach((name, index) => {
            pool.push(product(`${prefix}_${index}`, slot, name, extra));
        });
    };
    push('blz', 'outerwear', [
        'Navy wool blazer', 'Charcoal wool blazer', 'Beige linen blazer',
        'Navy double breasted blazer', 'Grey unstructured blazer',
        'Black structured blazer', 'Olive wool blazer', 'Brown tweed blazer',
    ]);
    push('coat', 'outerwear', ['Camel wool topcoat', 'Navy peacoat']);
    push('ovr', 'outerwear', ['Chambray overshirt', 'Waxed cotton overshirt', 'Suede jacket']);
    push('shrt', 'top', [
        'White oxford shirt', 'Blue poplin shirt', 'Striped shirt', 'Linen shirt', 'Flannel shirt',
    ]);
    push('polo', 'top', ['Navy knit polo', 'White piqué polo', 'Green polo', 'Grey polo']);
    push('knit', 'top', [
        'Grey merino sweater', 'Navy fine gauge knit', 'Cream cashmere knit',
        'Charcoal roll neck', 'Brown crewneck sweater',
    ]);
    push('trsr', 'bottom', [
        'Grey tailored trousers', 'Black suit trousers', 'Navy pleated trousers',
        'Charcoal wool trousers', 'Stone tailored trousers', 'Dark tailored trousers',
    ]);
    push('chno', 'bottom', ['Beige chinos', 'Navy chinos', 'Olive chinos', 'Stone chinos']);
    push('jean', 'bottom', ['Dark rinse jeans', 'Indigo jeans', 'Black jeans']);
    push('loaf', 'shoes', [
        'Brown penny loafers', 'Black penny loafers', 'Tan tassel loafers',
        'Suede loafers', 'Dark brown loafers',
    ]);
    push('drby', 'shoes', ['Brown derby shoes', 'Black derby shoes', 'Tan derby shoes']);
    push('snkr', 'shoes', [
        'White leather sneakers', 'Minimal white sneakers', 'Grey suede sneakers', 'Black sneakers',
    ]);
    return pool;
};

/** The §10 thin catalogue: one blazer, one shirt, one trouser, one shoe. */
const thinPool = (): Product[] => [
    product('thin_blazer', 'outerwear', 'Navy wool blazer'),
    product('thin_shirt', 'top', 'White oxford shirt'),
    product('thin_trousers', 'bottom', 'Grey tailored trousers'),
    product('thin_loafers', 'shoes', 'Brown penny loafers'),
];

const archetypesOf = (deck: Look[]) =>
    deck.map((l) => buildOutfitFingerprint(l.items).archetype);

const pairwise = (deck: Look[]) => {
    const fps = deck.map((l) => buildOutfitFingerprint(l.items));
    const out: number[] = [];
    for (let i = 0; i < fps.length; i++) {
        for (let j = i + 1; j < fps.length; j++) out.push(visualSimilarity(fps[i], fps[j]));
    }
    return out;
};

// ── Tests ────────────────────────────────────────────────────────────────

describe('outfit visual diversity — fingerprints', () => {
    it('describes a complete look as an outfit formula, not a product list', () => {
        const fingerprint = buildOutfitFingerprint([
            product('a1', 'outerwear', 'Navy wool blazer'),
            product('a2', 'top', 'White oxford shirt'),
            product('a3', 'bottom', 'Grey tailored trousers'),
            product('a4', 'shoes', 'Brown penny loafers'),
        ]);
        // The archetype token is the coarse ROLE in the formula, so an oxford
        // and a poplin shirt are the same role.
        expect(fingerprint.archetype).toBe('blazer-shirt-tailored-loafer');
        expect(fingerprint.silhouette).toBe('tailored');
        expect(fingerprint.footwearFamily).toBe('shoe:loafer');
        expect(fingerprint.families).toEqual([
            'outer:blazer', 'top:dress-shirt', 'bottom:tailored', 'shoe:loafer',
        ]);
        expect(describeFingerprint(fingerprint)).toContain('blazer-shirt-tailored-loafer');
    });

    it('keeps colour the weakest signal: same formula, different palette, still similar', () => {
        const navy = buildOutfitFingerprint([
            product('a1', 'outerwear', 'Navy wool blazer'),
            product('a2', 'top', 'White oxford shirt'),
            product('a3', 'bottom', 'Grey tailored trousers'),
            product('a4', 'shoes', 'Brown penny loafers'),
        ]);
        const beige = buildOutfitFingerprint([
            product('b1', 'outerwear', 'Beige linen blazer'),
            product('b2', 'top', 'Striped shirt'),
            product('b3', 'bottom', 'Navy trousers'),
            product('b4', 'shoes', 'Tassel loafer'),
        ]);
        expect(visualSimilarity(navy, beige)).toBeGreaterThan(MAX_VISUAL_SIMILARITY);
    });
});

describe('A. different product ids, identical outfit formula', () => {
    const candidates: Look[] = [
        look('A', [
            product('a1', 'outerwear', 'Navy wool blazer'),
            product('a2', 'top', 'White oxford shirt'),
            product('a3', 'bottom', 'Grey tailored trousers'),
            product('a4', 'shoes', 'Brown penny loafers'),
        ]),
        look('B', [
            product('b1', 'outerwear', 'Charcoal blazer'),
            product('b2', 'top', 'Blue poplin shirt'),
            product('b3', 'bottom', 'Black trousers'),
            product('b4', 'shoes', 'Penny loafer'),
        ]),
        look('C', [
            product('c1', 'outerwear', 'Beige blazer'),
            product('c2', 'top', 'Striped shirt'),
            product('c3', 'bottom', 'Navy trousers'),
            product('c4', 'shoes', 'Tassel loafer'),
        ]),
    ];

    it('must not accept all three as strictly diverse', () => {
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf,
            usedKeys: new Set<string>(),
            poolItems: richPool(),
        });
        // The pool is rich, so the deck must NOT be presented as STRICT — the
        // honest reason has to be reported instead of silently shipping three
        // colourways of one formula as "diverse".
        expect(result.relaxed).toBe(true);
        expect(result.relaxReason).toBe('diversity_unavailable_despite_varied_inventory');
    });

    it('the three cards share the same archetype token tuple', () => {
        const fingerprints = candidates.map((c) => buildOutfitFingerprint(c.items));
        expect(sameArchetype(fingerprints[0], fingerprints[1])).toBe(true);
        expect(sameArchetype(fingerprints[0], fingerprints[2])).toBe(true);
        expect(sameArchetype(fingerprints[1], fingerprints[2])).toBe(true);
        expect(new Set(fingerprints.map((f) => f.archetype)).size).toBe(1);
    });

    it('rejects the pair outright on the archetype rule', () => {
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf,
            usedKeys: new Set<string>(),
            poolItems: richPool(),
            buildFill: fillBuilder(richPool()),
        });
        expect(result.relaxed).toBe(false);
        expect(result.deck).toHaveLength(3);
        // Three DIFFERENT formulas — the reported bug produced one.
        expect(new Set(result.archetypes).size).toBe(3);
    });
});

describe('B. similar products with different ids', () => {
    it('shares the garment family and counts towards visual similarity', () => {
        const left = buildOutfitFingerprint([
            product('x1', 'outerwear', 'Navy wool blazer'),
            product('x2', 'top', 'White oxford shirt'),
            product('x3', 'bottom', 'Grey tailored trousers'),
            product('x4', 'shoes', 'Brown penny loafers'),
        ]);
        const right = buildOutfitFingerprint([
            product('y1', 'outerwear', 'Charcoal wool blazer'),
            product('y2', 'top', 'Blue oxford shirt'),
            product('y3', 'bottom', 'Black tailored trousers'),
            product('y4', 'shoes', 'Dark brown loafers'),
        ]);
        // Same family per slot, different ids / names / photos.
        expect(sharedVisualFamilies(left, right)).toBe(4);
        expect(visualSimilarity(left, right)).toBeGreaterThan(MAX_VISUAL_SIMILARITY);
    });

    it('does not punish two items of the same family that look clearly different', () => {
        const plain = buildOutfitFingerprint([
            product('p1', 'outerwear', 'Navy wool blazer'),
            product('p2', 'top', 'White oxford shirt'),
            product('p3', 'bottom', 'Grey tailored trousers'),
            product('p4', 'shoes', 'Brown penny loafers'),
        ]);
        const plainCopy = buildOutfitFingerprint([
            product('q1', 'outerwear', 'Navy wool blazer'),
            product('q2', 'top', 'White oxford shirt'),
            product('q3', 'bottom', 'Grey tailored trousers'),
            product('q4', 'shoes', 'Brown penny loafers'),
        ]);
        const distinct = buildOutfitFingerprint([
            product('t1', 'outerwear', 'Beige linen blazer'),
            product('t2', 'top', 'White oxford shirt'),
            product('t3', 'bottom', 'Grey tailored trousers'),
            product('t4', 'shoes', 'Brown penny loafers'),
        ]);
        // The outerwear is a different VARIANT (colour + material), so it does
        // not count as a repeated garment family …
        expect(sharedVisualFamilies(plain, distinct)).toBe(3);
        expect(sharedVisualFamilies(plain, plainCopy)).toBe(4);
        // … and the variant is recognised as a real difference: a beige linen
        // blazer is measurably further from a navy wool blazer than another
        // navy wool blazer is.
        expect(visualSimilarity(plain, distinct))
            .toBeLessThan(visualSimilarity(plain, plainCopy));
    });

    it('decides at outfit level: a same-family blazer does not buy a pass', () => {
        const left = look('L', [
            product('l1', 'outerwear', 'Structured navy double breasted wool blazer'),
            product('l2', 'top', 'White oxford shirt'),
            product('l3', 'bottom', 'Grey tailored trousers'),
            product('l4', 'shoes', 'Brown penny loafers'),
        ]);
        const right = look('R', [
            product('r1', 'outerwear', 'Soft beige unstructured linen blazer'),
            product('r2', 'top', 'Blue poplin shirt'),
            product('r3', 'bottom', 'Navy tailored trousers'),
            product('r4', 'shoes', 'Black derby shoes'),
        ]);
        // Both looks are still "blazer + shirt + tailored trousers + formal
        // shoes": the family rule no longer counts the blazer, but the
        // outfit-level comparison still refuses the pair.
        expect(sharedVisualFamilies(
            buildOutfitFingerprint(left.items),
            buildOutfitFingerprint(right.items),
        )).toBe(2);
        const result = assembleDiverseDeck<Look>({
            candidates: [left, right],
            itemsOf,
            keysOf,
            usedKeys: new Set<string>(),
            poolItems: richPool(),
        });
        expect(result.relaxed).toBe(true);
    });

    it('does not let different shoes — or a different colourway — create a new formula', () => {
        // The hole the simulator found: blazer + shirt + trousers twice with
        // only the shoes (and the blazer's colour) changed scored 0.55 and
        // slipped under the numeric threshold. Footwear is a finishing
        // detail, not a styling direction.
        const derby = buildOutfitFingerprint([
            product('s1', 'outerwear', 'Beige unstructured linen blazer'),
            product('s2', 'top', 'Chambray button down shirt'),
            product('s3', 'bottom', 'Charcoal wool trousers'),
            product('s4', 'shoes', 'Black derby shoes'),
        ]);
        const boot = buildOutfitFingerprint([
            product('t1', 'outerwear', 'Navy wool tailored blazer'),
            product('t2', 'top', 'Blue poplin shirt'),
            product('t3', 'bottom', 'Grey tailored trousers'),
            product('t4', 'shoes', 'Chelsea boots'),
        ]);
        expect(visualSimilarity(derby, boot)).toBeLessThan(MAX_VISUAL_SIMILARITY);
        expect(sameArchetype(derby, boot)).toBe(true);
    });
});

describe('C. a genuinely diverse deck is accepted', () => {
    it('keeps three different styling directions', () => {
        const candidates: Look[] = [
            look('A', [
                product('a1', 'outerwear', 'Navy wool blazer'),
                product('a2', 'top', 'White oxford shirt'),
                product('a3', 'bottom', 'Grey tailored trousers'),
                product('a4', 'shoes', 'Brown penny loafers'),
            ]),
            look('B', [
                product('b1', 'outerwear', 'Suede jacket'),
                product('b2', 'top', 'Knitted polo'),
                product('b3', 'bottom', 'Beige chinos'),
                product('b4', 'shoes', 'Minimal leather sneakers'),
            ]),
            look('C', [
                product('c1', 'outerwear', 'Cardigan'),
                product('c2', 'top', 'Fine gauge knit'),
                product('c3', 'bottom', 'Dark denim jeans'),
                product('c4', 'shoes', 'Derby shoes'),
            ]),
        ];
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf,
            usedKeys: new Set<string>(),
            poolItems: richPool(),
        });
        expect(result.relaxed).toBe(false);
        expect(result.deck).toHaveLength(3);
        expect(new Set(result.archetypes).size).toBe(3);
        for (const score of pairwise(result.deck)) {
            expect(score).toBeLessThan(MAX_VISUAL_SIMILARITY);
        }
    });
});

describe('D. same style, different concepts', () => {
    it('three old-money looks still have to be three directions', () => {
        const oldMoney = (id: string, names: [string, string, string, string]): Look =>
            look(id, [
                product(`${id}_o`, 'outerwear', names[0]),
                product(`${id}_t`, 'top', names[1]),
                product(`${id}_b`, 'bottom', names[2]),
                product(`${id}_s`, 'shoes', names[3]),
            ]);
        const candidates: Look[] = [
            oldMoney('A', ['Navy wool blazer', 'White oxford shirt', 'Grey tailored trousers', 'Brown penny loafers']),
            oldMoney('B', ['Suede jacket', 'Knitted polo', 'Beige chinos', 'Minimal leather sneakers']),
            oldMoney('C', ['Cardigan', 'Fine gauge knit', 'Dark denim jeans', 'Derby shoes']),
        ];
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf,
            usedKeys: new Set<string>(),
            poolItems: richPool(),
        });
        expect(result.relaxed).toBe(false);
        expect(new Set(result.archetypes).size).toBe(3);
        expect(result.deck).toHaveLength(3);
    });
});

describe('E. formal occasion — diversity never introduces casual clothing', () => {
    it('three black-tie looks stay formal and still differ', () => {
        const candidates: Look[] = [
            look('A', [
                product('a1', 'outerwear', 'Classic black tuxedo dinner jacket'),
                product('a2', 'top', 'White tuxedo formal shirt'),
                product('a3', 'bottom', 'Black tuxedo trousers'),
                product('a4', 'shoes', 'Black patent Oxford shoes'),
            ]),
            look('B', [
                product('b1', 'outerwear', 'Midnight navy dinner jacket'),
                product('b2', 'top', 'White formal dress shirt'),
                product('b3', 'bottom', 'Midnight tuxedo trousers'),
                product('b4', 'shoes', 'Black velvet loafers'),
            ]),
            look('C', [
                product('c1', 'outerwear', 'Shawl collar dinner jacket'),
                product('c2', 'top', 'White pleated formal shirt'),
                product('c3', 'bottom', 'Black tuxedo trousers'),
                product('c4', 'shoes', 'Polished black formal shoes'),
            ]),
        ];
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf,
            usedKeys: new Set<string>(),
            poolItems: candidates.flatMap((c) => c.items),
        });
        expect(result.deck).toHaveLength(3);
        // Every garment is still evening/formal — no sneaker, denim or
        // overshirt slipped in to buy diversity.
        for (const item of result.deck.flatMap((l) => l.items)) {
            expect(/sneaker|denim|jeans|overshirt|chinos|polo/i.test(item.name)).toBe(false);
        }
        for (const item of result.deck.flatMap((l) => l.items)) {
            const family = garmentVisualFamily(item);
            if (item.macroCategory === 'top') expect(family).toBe('top:dress-shirt');
            if (item.macroCategory === 'shoes') expect(family).not.toBe('shoe:sneaker');
        }
    });

    it('never relaxes the dress code to reach diversity', () => {
        const tuxedo = (id: string, outer: string, shoe: string): Look =>
            look(id, [
                product(`${id}_o`, 'outerwear', outer),
                product(`${id}_t`, 'top', 'White formal dress shirt'),
                product(`${id}_b`, 'bottom', 'Black tuxedo trousers'),
                product(`${id}_s`, 'shoes', shoe),
            ]);
        const candidates: Look[] = [
            tuxedo('A', 'Black tuxedo dinner jacket', 'Black patent Oxford shoes'),
            tuxedo('B', 'Midnight navy dinner jacket', 'Black velvet loafers'),
            tuxedo('C', 'Shawl collar dinner jacket', 'Polished black formal shoes'),
        ];
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf,
            usedKeys: new Set<string>(),
            poolItems: candidates.flatMap((c) => c.items),
        });
        expect(result.deck).toHaveLength(3);
        for (const item of result.deck.flatMap((l) => l.items)) {
            expect(/sneaker|denim|overshirt/i.test(item.name)).toBe(false);
        }
    });
});

describe('F. an explicit anchor repeats — everything else must vary', () => {
    it('keeps the anchor in all three looks while the structure changes', () => {
        const anchor = product('anchor_blazer', 'outerwear', 'Navy wool blazer');
        const candidates: Look[] = [
            look('A', [
                anchor,
                product('a2', 'top', 'White oxford shirt'),
                product('a3', 'bottom', 'Grey tailored trousers'),
                product('a4', 'shoes', 'Brown penny loafers'),
            ]),
            look('B', [
                anchor,
                product('b2', 'top', 'Knitted polo'),
                product('b3', 'bottom', 'Beige chinos'),
                product('b4', 'shoes', 'Minimal leather sneakers'),
            ]),
            look('C', [
                anchor,
                product('c2', 'top', 'Fine gauge knit'),
                product('c3', 'bottom', 'Dark denim jeans'),
                product('c4', 'shoes', 'Derby shoes'),
            ]),
        ];
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf,
            anchorKeys: new Set([String(anchor.id)]),
            anchorSlot: 'outerwear',
            usedKeys: new Set<string>(),
            poolItems: richPool(),
        });
        expect(result.deck).toHaveLength(3);
        for (const l of result.deck) {
            expect(l.items.some((i) => i.id === anchor.id)).toBe(true);
        }
        // The anchor is excluded from the visual comparison, so the three
        // looks still have to differ in the other three slots.
        expect(new Set(result.archetypes).size).toBe(3);
        for (const score of pairwise(result.deck)) {
            expect(score).toBeLessThan(MAX_VISUAL_SIMILARITY);
        }
    });
});

describe('G. genuinely thin inventory allows controlled repetition', () => {
    it('completes the deck instead of failing', () => {
        const pool = thinPool();
        const result = assembleDiverseDeck<Look>({
            candidates: [],
            itemsOf,
            keysOf,
            usedKeys: new Set<string>(),
            poolItems: pool,
            buildFill: fillBuilder(pool),
        });
        expect(result.deck.length).toBeGreaterThanOrEqual(1);
        expect(result.relaxed).toBe(true);
        expect(result.relaxReason).toBe('thin_inventory');
    });

    it('detects the thin inventory before relaxing', () => {
        const assessment = assessInventoryDiversity(thinPool(), { target: 3 });
        expect(assessment.homogeneous).toBe(true);
        expect(assessment.supportsTargetConcepts).toBe(false);
        expect(assessment.conceptCapacity).toBe(1);
    });

    it('a rich inventory is never treated as thin', () => {
        const assessment = assessInventoryDiversity(richPool(), { target: 3 });
        expect(assessment.homogeneous).toBe(false);
        expect(assessment.supportsTargetConcepts).toBe(true);
        expect(assessment.conceptCapacity).toBeGreaterThanOrEqual(3);
    });
});

describe('H. a large inventory must never relax', () => {
    it('finds three different directions and stays STRICT', () => {
        const pool = richPool();
        const result = assembleDiverseDeck<Look>({
            candidates: [],
            itemsOf,
            keysOf,
            usedKeys: new Set<string>(),
            poolItems: pool,
            buildFill: fillBuilder(pool),
        });
        expect(result.deck).toHaveLength(3);
        expect(result.relaxed).toBe(false);
        expect(result.relaxReason).toBeNull();
        expect(new Set(result.archetypes).size).toBe(3);
        for (const score of pairwise(result.deck)) {
            expect(score).toBeLessThan(MAX_VISUAL_SIMILARITY);
        }
    });

    it('refuses to relax even when every edge candidate is one formula', () => {
        const pool = richPool();
        const sameFormula = (id: string, outer: string, top: string, bottom: string, shoe: string): Look =>
            look(id, [
                product(`${id}_o`, 'outerwear', outer),
                product(`${id}_t`, 'top', top),
                product(`${id}_b`, 'bottom', bottom),
                product(`${id}_s`, 'shoes', shoe),
            ]);
        const candidates: Look[] = [
            sameFormula('A', 'Navy wool blazer', 'White oxford shirt', 'Grey tailored trousers', 'Brown penny loafers'),
            sameFormula('B', 'Charcoal blazer', 'Blue poplin shirt', 'Black trousers', 'Penny loafer'),
            sameFormula('C', 'Beige blazer', 'Striped shirt', 'Navy trousers', 'Tassel loafer'),
        ];
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf,
            usedKeys: new Set<string>(),
            poolItems: pool,
            buildFill: fillBuilder(pool),
        });
        expect(result.relaxed).toBe(false);
        expect(new Set(result.archetypes).size).toBe(3);
    });

    it('reports the honest reason when a varied inventory still cannot be diversified', () => {
        const pool = richPool();
        const sameFormula = (id: string, outer: string, top: string, bottom: string, shoe: string): Look =>
            look(id, [
                product(`${id}_o`, 'outerwear', outer),
                product(`${id}_t`, 'top', top),
                product(`${id}_b`, 'bottom', bottom),
                product(`${id}_s`, 'shoes', shoe),
            ]);
        const candidates: Look[] = [
            sameFormula('A', 'Navy wool blazer', 'White oxford shirt', 'Grey tailored trousers', 'Brown penny loafers'),
            sameFormula('B', 'Charcoal blazer', 'Blue poplin shirt', 'Black trousers', 'Penny loafer'),
            sameFormula('C', 'Beige blazer', 'Striped shirt', 'Navy trousers', 'Tassel loafer'),
        ];
        // No fill builder: the deck can only be built from these three.
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf,
            usedKeys: new Set<string>(),
            poolItems: pool,
        });
        expect(result.relaxed).toBe(true);
        expect(result.relaxReason).toBe('diversity_unavailable_despite_varied_inventory');
    });
});

describe('I. full-look archetype regression', () => {
    it('never ships three looks with one composition', () => {
        const pool = richPool();
        const candidates: Look[] = [
            look('A', [
                product('a1', 'outerwear', 'Navy wool blazer'),
                product('a2', 'top', 'White oxford shirt'),
                product('a3', 'bottom', 'Grey tailored trousers'),
                product('a4', 'shoes', 'Brown penny loafers'),
            ]),
            look('B', [
                product('b1', 'outerwear', 'Charcoal wool blazer'),
                product('b2', 'top', 'Blue poplin shirt'),
                product('b3', 'bottom', 'Black tailored trousers'),
                product('b4', 'shoes', 'Dark brown loafers'),
            ]),
            look('C', [
                product('c1', 'outerwear', 'Beige linen blazer'),
                product('c2', 'top', 'White striped shirt'),
                product('c3', 'bottom', 'Navy tailored trousers'),
                product('c4', 'shoes', 'Tassel loafer'),
            ]),
        ];
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            keysOf,
            usedKeys: new Set<string>(),
            poolItems: pool,
            buildFill: fillBuilder(pool),
        });
        expect(result.deck).toHaveLength(3);
        const archetypes = result.archetypes;
        expect(new Set(archetypes).size).toBe(3);
        expect(archetypes).not.toContain('blazer-dress-shirt-tailored-loafer|blazer-dress-shirt-tailored-loafer');
    });
});

describe('J. pairwise similarity ceiling of the final deck', () => {
    it('every pair stays below the strict threshold', () => {
        const pool = richPool();
        const result = assembleDiverseDeck<Look>({
            candidates: [],
            itemsOf,
            keysOf,
            usedKeys: new Set<string>(),
            poolItems: pool,
            buildFill: fillBuilder(pool),
        });
        expect(result.similarities).toHaveLength(3);
        for (const entry of result.similarities) {
            expect(entry.score).toBeLessThan(MAX_VISUAL_SIMILARITY);
        }
    });

    it('the deck is chosen as a COMBINATION, not by accepting looks one by one', () => {
        const pool = richPool();
        const lookWith = (id: string, outer: string, top: string, bottom: string, shoe: string, score: number): Look =>
            look(id, [
                product(`${id}_o`, 'outerwear', outer),
                product(`${id}_t`, 'top', top),
                product(`${id}_b`, 'bottom', bottom),
                product(`${id}_s`, 'shoes', shoe),
            ], score);
        const candidates: Look[] = [
            lookWith('A', 'Navy wool blazer', 'White oxford shirt', 'Grey tailored trousers', 'Brown penny loafers', 0.94),
            lookWith('B', 'Charcoal blazer', 'Blue poplin shirt', 'Black trousers', 'Penny loafer', 0.93),
            lookWith('C', 'Beige blazer', 'Striped shirt', 'Navy trousers', 'Tassel loafer', 0.92),
        ];
        const result = assembleDiverseDeck<Look>({
            candidates,
            itemsOf,
            qualityOf: (l) => l.score ?? 0.9,
            keysOf,
            usedKeys: new Set<string>(),
            poolItems: pool,
            buildFill: fillBuilder(pool),
        });
        // The three 0.94/0.93/0.92 looks must NOT win every slot: the deck
        // trades a little quality for three real directions.
        expect(result.deck).toHaveLength(3);
        expect(new Set(result.archetypes).size).toBe(3);
        const keptScore = Math.max(...result.deck.map((l) => l.score ?? 0.9));
        expect(keptScore).toBeGreaterThanOrEqual(0.9);
    });
});
