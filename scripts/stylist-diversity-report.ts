/**
 * stylist-diversity-report — the Personal Stylist simulator.
 *
 * Runs the REAL production chain for 20 style × occasion scenarios:
 *
 *   valid pool (style / occasion / climate filtered)
 *     → candidate looks (the "AI" output)
 *     → assembleDiverseDeck()            (the diversity gate)
 *        → createSlotNormalizer()        (the real slot picker builds the
 *                                          targeted replacement looks)
 *     → final deck + pairwise metrics
 *
 * For every deck it records the outfit archetype of each look, the three
 * pairwise similarity scores, the diversity mode (STRICT / RELAXED) and the
 * reason when the mode is relaxed. The report makes the quality of the
 * Personal Stylist MEASURABLE instead of subjective.
 *
 * Usage:  npx tsx scripts/stylist-diversity-report.ts
 * Output: stdout (and /tmp/stylist-diversity-report.md when --out is passed)
 */
import { writeFileSync } from 'fs';
import {
    analyzeDeckProductDiversity,
    assembleDiverseDeck,
    buildOutfitFingerprint,
    canonicalProductKey,
    MAX_VISUAL_SIMILARITY,
    productIdentityKeys,
    sameArchetype,
    visualSimilarity,
    type DiversitySlot,
} from '../features/outfit-generator/utils/outfitDiversity';
import {
    createSlotNormalizer,
    isCompleteNormalizedOutfit,
    isFourSlotLook,
} from '../features/outfit-generator/utils/normalizeSlots';
import {
    passesStyleJudge,
    rejectForClimate,
    rejectForStyleOccasion,
} from '../features/outfit-generator/utils/occasionRules';

// React Native defines this global; the simulator runs in plain Node.
(globalThis as any).__DEV__ = process.env.NODE_ENV !== 'production';

// ── Catalogue ─────────────────────────────────────────────────────────────

type CatalogItem = {
    id: string;
    name: string;
    type: string;
    category: string;
    macroCategory: DiversitySlot;
    color: string;
    brand: string;
    description: string;
    material: string;
    image: string;
    price: number;
    currency: string;
};

const CATALOG: Array<[string, string, string]> = [
    ['outerwear', 'Navy wool tailored blazer', 'navy'],
    ['outerwear', 'Charcoal double breasted wool blazer', 'charcoal'],
    ['outerwear', 'Beige unstructured linen blazer', 'beige'],
    ['outerwear', 'Olive tweed sport coat', 'olive'],
    ['outerwear', 'Camel wool topcoat', 'camel'],
    ['outerwear', 'Navy peacoat', 'navy'],
    ['outerwear', 'Chambray overshirt', 'blue'],
    ['outerwear', 'Waxed cotton field jacket', 'olive'],
    ['outerwear', 'Brown suede jacket', 'brown'],
    ['outerwear', 'Grey wool cardigan', 'grey'],
    ['outerwear', 'Navy merino cardigan', 'navy'],
    ['outerwear', 'Black tuxedo dinner jacket', 'black'],
    ['outerwear', 'Midnight navy dinner jacket', 'navy'],
    ['outerwear', 'Shawl collar dinner jacket', 'black'],
    ['outerwear', 'White cotton hoodie', 'white'],
    ['top', 'White oxford shirt', 'white'],
    ['top', 'Blue poplin shirt', 'blue'],
    ['top', 'Chambray button down shirt', 'blue'],
    ['top', 'Striped cotton shirt', 'white'],
    ['top', 'Linen shirt', 'beige'],
    ['top', 'Navy knit polo', 'navy'],
    ['top', 'White piqué polo', 'white'],
    ['top', 'Grey merino sweater', 'grey'],
    ['top', 'Navy fine gauge knit', 'navy'],
    ['top', 'Cream cashmere sweater', 'cream'],
    ['top', 'White cotton t-shirt', 'white'],
    ['top', 'White tuxedo formal shirt', 'white'],
    ['top', 'White pleated formal shirt', 'white'],
    ['bottom', 'Grey tailored trousers', 'grey'],
    ['bottom', 'Navy pleated trousers', 'navy'],
    ['bottom', 'Charcoal wool trousers', 'charcoal'],
    ['bottom', 'Black suit trousers', 'black'],
    ['bottom', 'Beige chinos', 'beige'],
    ['top', 'Navy chinos', 'navy'],
    ['bottom', 'Olive chinos', 'olive'],
    ['bottom', 'Dark rinse jeans', 'indigo'],
    ['bottom', 'Black jeans', 'black'],
    ['bottom', 'Grey sweatpants', 'grey'],
    ['bottom', 'Black tuxedo trousers', 'black'],
    ['shoes', 'Brown penny loafers', 'brown'],
    ['shoes', 'Black penny loafers', 'black'],
    ['shoes', 'Tan suede loafers', 'tan'],
    ['shoes', 'Brown derby shoes', 'brown'],
    ['shoes', 'Black derby shoes', 'black'],
    ['shoes', 'Minimal white leather sneakers', 'white'],
    ['shoes', 'Grey suede sneakers', 'grey'],
    ['shoes', 'Chelsea boots', 'black'],
    ['shoes', 'Black patent Oxford shoes', 'black'],
    ['shoes', 'Black velvet loafers', 'black'],
];

const MATERIALS: Record<string, string> = {
    blazer: 'wool', coat: 'wool', jacket: 'suede', cardigan: 'wool',
    shirt: 'cotton', polo: 'cotton', sweater: 'merino', knit: 'cashmere',
    't-shirt': 'cotton', trousers: 'wool', chinos: 'cotton', jeans: 'denim',
    sweatpants: 'cotton', loafers: 'leather', 'derby shoes': 'leather',
    sneakers: 'leather', boots: 'leather', 'oxford shoes': 'patent',
};

const buildCatalog = (): CatalogItem[] =>
    CATALOG.map(([slot, name, color], index) => {
        const material =
            MATERIALS[Object.keys(MATERIALS).find((key) => name.includes(key)) ?? ''] ?? 'cotton';
        return {
            id: `cat_${index}`,
            name,
            type: slot === 'outerwear' ? 'outerwear' : slot === 'top' ? 'top' : slot === 'bottom' ? 'bottom' : 'shoes',
            category: slot === 'shoes' ? 'shoes' : `${slot}s`,
            macroCategory: slot as DiversitySlot,
            color,
            brand: 'Atelier',
            description: `${material} ${name}`,
            material,
            image: `https://cdn.test/${index}.jpg`,
            price: 60 + index * 3,
            currency: 'USD',
        };
    });

// ── Scenarios ─────────────────────────────────────────────────────────────

export interface Scenario {
    id: number;
    style: string;
    occasion: string;
    weather: { temp: number; condition: string };
}

const SCENARIO_MATRIX: Array<[string, string]> = [
    ['work', 'old_money'],
    ['work', 'classic'],
    ['work', 'business_casual'],
    ['work', 'minimalist'],
    ['work', 'semi_classic'],
    ['date', 'old_money'],
    ['date', 'classic'],
    ['date', 'casual'],
    ['date', 'business_casual'],
    ['travel', 'casual'],
    ['travel', 'minimalist'],
    ['travel', 'semi_classic'],
    ['casual', 'casual'],
    ['casual', 'minimalist'],
    ['casual', 'semi_classic'],
    ['formal', 'classic'],
    ['formal', 'old_money'],
    ['formal', 'business_casual'],
    ['weekend', 'casual'],
    ['dinner', 'old_money'],
];

export const SCENARIOS: Scenario[] = SCENARIO_MATRIX.map(([occasion, style], index) => ({
    id: index + 1,
    style,
    occasion,
    weather: index % 3 === 0
        ? { temp: 9, condition: 'cool' }
        : index % 3 === 1
            ? { temp: 17, condition: 'mild' }
            : { temp: 24, condition: 'warm' },
}));

// ── The simulator ─────────────────────────────────────────────────────────

export interface ScenarioReport {
    scenario: Scenario;
    poolSize: number;
    archetypes: string[];
    /** Archetype tokens per look, in slot order. Tokens may contain hyphens
     * ('dinner-jacket', 'dress-shoe'), so the display must not re-split them. */
    looks: string[][];
    similarities: number[];
    mode: 'STRICT' | 'RELAXED';
    relaxReason: string | null;
    /** Canonical product key per item, per look. */
    productKeys: string[][];
    /** Number of distinct non-anchor products appearing in >1 look. */
    nonAnchorProductReuseCount: number;
    repeatedProductKeys: string[];
    repeatedCanonicalKeys: string[];
    anchorRepetitionCount: number;
    supportsUniqueProducts: boolean;
    pass: boolean;
    notes: string[];
}

/**
 * Run one scenario through the production chain.
 *
 * `candidateLooks` stands in for the AI output: the first valid looks of
 * the pool, in catalogue order. The gate then has to find three genuinely
 * different styling directions among them and the replacements it builds.
 */
export function runScenario(scenario: Scenario, catalog: CatalogItem[] = buildCatalog()): ScenarioReport {
    const valid = catalog.filter((item) =>
        rejectForStyleOccasion(item, scenario.style, scenario.occasion) === null
        && rejectForClimate(item, scenario.weather) === null,
    );

    // "AI output": the naive baseline — three complete looks built by
    // rotating through the catalogue in slot order. This is exactly the
    // reported failure mode (three colourways of one formula); the gate has
    // to find three real styling directions among them and the replacements
    // it builds.
    const bySlot = (slot: DiversitySlot) => valid.filter((item) => item.macroCategory === slot);
    const slotPools = {
        outerwear: bySlot('outerwear'),
        top: bySlot('top'),
        bottom: bySlot('bottom'),
        shoes: bySlot('shoes'),
    };
    const candidates = [0, 1, 2].map((step) => ({
        id: `ai_${step}`,
        items: [
            slotPools.outerwear[step],
            slotPools.top[step],
            slotPools.bottom[step],
            slotPools.shoes[step],
        ].filter(Boolean) as CatalogItem[],
    }));

    const generationNonceRef = { current: 1 };
    const slotDeckUsed = new Set<string>();
    const normalize = createSlotNormalizer({
        weather: scenario.weather,
        currentSource: 'shop',
        selectedOccasion: scenario.occasion,
        generationNonceRef,
        poolFallback: valid,
    });

    const judge = (items: any[]) =>
        passesStyleJudge(items, scenario.style, `${scenario.occasion}`, scenario.weather);

    // The real slot picker builds the targeted replacement looks — the same
    // builder the AIOutfitmaker screen uses.
    const buildFill = (_used: Set<string>, attempt: number) => {
        const items = normalize([], scenario.style, attempt, scenario.id, valid, slotDeckUsed);
        if (!(isFourSlotLook(items) || isCompleteNormalizedOutfit(items))) return null;
        if (!judge(items)) return null;
        return { id: `fill_${attempt}`, items };
    };

    const result = assembleDiverseDeck<{ id: string; items: any[] }>({
        candidates,
        itemsOf: (look) => look.items,
        qualityOf: () => 0.9,
        keysOf: (item: any) => productIdentityKeys(item),
        usedKeys: slotDeckUsed,
        buildFill,
        poolItems: valid,
        target: 3,
    });

    const fingerprints = result.deck.map((look) => buildOutfitFingerprint(look.items));
    const archetypes = fingerprints.map((fingerprint) => fingerprint.archetype);
    const looks = fingerprints.map((fingerprint) =>
        Object.values(fingerprint.archetypeTokens).filter(Boolean) as string[],
    );
    const productDiversity = analyzeDeckProductDiversity(
        result.deck.map((look) => look.items),
    );
    const productKeys = result.deck.map((look) => look.items.map((item) => canonicalProductKey(item)));
    const similarities: number[] = [];
    for (let i = 0; i < fingerprints.length; i++) {
        for (let j = i + 1; j < fingerprints.length; j++) {
            similarities.push(visualSimilarity(fingerprints[i], fingerprints[j]));
        }
    }

    const notes: string[] = [];
    let pass = true;
    if (result.deck.length < 3) {
        pass = false;
        notes.push(`only ${result.deck.length} look(s) produced`);
    }
    for (const score of similarities) {
        if (score >= MAX_VISUAL_SIMILARITY) {
            pass = false;
            notes.push(`pairwise similarity ${score.toFixed(2)} >= ${MAX_VISUAL_SIMILARITY}`);
        }
    }
    if (result.inventory.supportsUniqueProducts && productDiversity.nonAnchorProductReuseCount > 0) {
        pass = false;
        notes.push(`${productDiversity.nonAnchorProductReuseCount} non-anchor product(s) reused: ${productDiversity.repeatedProductKeys.join(', ')}`);
    }
    if (!result.inventory.supportsUniqueProducts && productDiversity.nonAnchorProductReuseCount > 0) {
        // Genuinely thin inventory: repetition is allowed, but it must be
        // reported with the honest reason.
        notes.push(`thin inventory allowed ${productDiversity.nonAnchorProductReuseCount} reused product(s)`);
    }
    if (new Set(archetypes).size < Math.min(3, result.deck.length)) {
        // Soft signal only: the same formula with unique products is valid.
        notes.push(`shared archetype formula (allowed): ${archetypes.join(' / ')}`);
    }
    // The diversity layer must never break the dress code.
    for (const look of result.deck) {
        if (!judge(look.items)) {
            pass = false;
            notes.push(`a look fails the style judge for ${scenario.style}/${scenario.occasion}`);
        }
    }

    return {
        scenario,
        poolSize: valid.length,
        archetypes,
        looks,
        similarities,
        mode: result.relaxed ? 'RELAXED' : 'STRICT',
        relaxReason: result.relaxReason,
        productKeys,
        nonAnchorProductReuseCount: productDiversity.nonAnchorProductReuseCount,
        repeatedProductKeys: productDiversity.repeatedProductKeys,
        repeatedCanonicalKeys: productDiversity.repeatedCanonicalKeys,
        anchorRepetitionCount: productDiversity.anchorRepetitionCount,
        supportsUniqueProducts: result.inventory.supportsUniqueProducts,
        pass,
        notes,
    };
}

export function runAllScenarios(): ScenarioReport[] {
    const catalog = buildCatalog();
    return SCENARIOS.map((scenario) => runScenario(scenario, catalog));
}

// ── Report ────────────────────────────────────────────────────────────────

const describeLook = (tokens: string[] | undefined, archetype: string) =>
    (tokens && tokens.length ? tokens : archetype.split('-')).join(' + ');

export function formatReport(reports: ScenarioReport[]): string {
    const lines: string[] = [];
    lines.push('AIWardrobe — Personal Stylist diversity report');
    lines.push('='.repeat(64));
    for (const report of reports) {
        const { scenario } = report;
        lines.push('');
        lines.push(
            `Scenario ${String(scenario.id).padStart(2, '0')} — ${scenario.occasion} / ${scenario.style}`
            + `  (pool ${report.poolSize}, ${scenario.weather.temp}°C ${scenario.weather.condition})`,
        );
        lines.push('');
        report.archetypes.forEach((archetype, index) => {
            lines.push(`  Look ${index + 1}: ${describeLook(report.looks?.[index], archetype)}`);
        });
        lines.push('');
        if (report.similarities.length) {
            lines.push('  Visual similarity:');
            const labels = ['1↔2', '1↔3', '2↔3'];
            report.similarities.forEach((score, index) => {
                lines.push(`    ${labels[index]} = ${score.toFixed(2)}`);
            });
        }
        lines.push('');
        lines.push(`  Products:`);
        report.productKeys.forEach((keys, index) => {
            lines.push(`    Look ${index + 1}: ${keys.join(' | ')}`);
        });
        lines.push('');
        lines.push(`  Product uniqueness: nonAnchorProductReuseCount=${report.nonAnchorProductReuseCount} · supportsUniqueProducts=${report.supportsUniqueProducts}${report.repeatedProductKeys.length ? ` · repeated=${report.repeatedProductKeys.join(', ')}` : ''}`);
        lines.push(`  Diversity mode: ${report.mode}${report.relaxReason ? ` (${report.relaxReason})` : ''}`);
        lines.push(`  ${report.pass ? 'PASS' : 'FAIL'}${report.notes.length ? ` — ${report.notes.join('; ')}` : ''}`);
    }
    lines.push('');
    lines.push('='.repeat(64));
    const strict = reports.filter((r) => r.mode === 'STRICT').length;
    const failed = reports.filter((r) => !r.pass).length;
    const reuse = reports.reduce((sum, r) => sum + r.nonAnchorProductReuseCount, 0);
    const worst = reports.reduce((max, r) => Math.max(max, ...r.similarities), 0);
    lines.push(`scenarios: ${reports.length} · strict: ${strict} · relaxed: ${reports.length - strict} · failed: ${failed}`);
    lines.push(`non-anchor product reuse across all decks: ${reuse} (target 0)`);
    lines.push(`worst pairwise similarity across all decks: ${worst.toFixed(2)} (soft signal, threshold ${MAX_VISUAL_SIMILARITY})`);
    return lines.join('\n');
}

const isMain = (() => {
    try {
        return require.main === module;
    } catch {
        return false;
    }
})();

if (isMain) {
    const report = formatReport(runAllScenarios());
    // eslint-disable-next-line no-console
    console.log(report);
    if (process.argv.includes('--out')) {
        writeFileSync('/tmp/stylist-diversity-report.md', report);
        // eslint-disable-next-line no-console
        console.log('\nwrote /tmp/stylist-diversity-report.md');
    }
}
