/** @jest-environment node */
const fs = require('fs'), path = require('path'), ts = require('typescript');
const root = process.env.NVIDIA_REQUEST_SOURCE_ROOT || path.resolve(__dirname, '../../backend/supabase/functions');

// The shared helpers the edge function imports.
const shared = {};
for (const file of ['_shared/occasionRules.ts', '_shared/outfitStructure.ts', '_shared/outfitDiversity.ts']) {
    new Function('exports', ts.transpileModule(fs.readFileSync(path.join(root, file), 'utf8'), {
        compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 },
    }).outputText)(shared);
}
const strip = source => source.replace(/^import .*\n/gm, '').replace(/\bexport\s+/g, '');
const source = ['_shared/modelOutfitResponse.ts', 'generate-outfits/index.ts']
    .map(file => strip(fs.readFileSync(path.join(root, file), 'utf8')))
    .join('\n');
const compiled = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.None, target: ts.ScriptTarget.ES2022 },
}).outputText;
const api = new Function(
    'serve', 'createClient', 'Deno',
    ...Object.keys(shared),
    compiled + '\nreturn { filterValidOutfits, localFallback, isShopOnlyDeck };',
)(() => {}, () => {}, { env: { get: () => undefined } }, ...Object.values(shared));

const item = (id, macroCategory, name, extra = {}) => ({
    id,
    macroCategory,
    name,
    type: macroCategory,
    color: 'navy',
    imageUrl: `https://cdn.test/${id}.jpg`,
    isWardrobe: true,
    ...extra,
});

const look = (id, items) => ({ description: `${id} description`, items: items.map(entry => ({ id: entry.id })) });

/** A pool with one garment per slot: the thin inventory. */
const thinPool = () => [
    item('thin_o', 'outerwear', 'Navy wool blazer'),
    item('thin_t', 'top', 'White oxford shirt'),
    item('thin_b', 'bottom', 'Grey tailored trousers'),
    item('thin_s', 'shoes', 'Brown penny loafers'),
];

/** A pool with several garments per slot. */
const richPool = () => {
    const pool = [];
    const push = (prefix, macroCategory, names) => names.forEach((name, index) => pool.push(item(`${prefix}_${index}`, macroCategory, name)));
    push('blz', 'outerwear', ['Navy wool blazer', 'Charcoal wool blazer', 'Beige linen blazer', 'Olive tweed blazer']);
    push('top', 'top', ['White oxford shirt', 'Blue poplin shirt', 'Striped cotton shirt', 'Navy knit polo']);
    push('bot', 'bottom', ['Grey tailored trousers', 'Navy pleated trousers', 'Beige chinos', 'Dark rinse jeans']);
    push('sho', 'shoes', ['Brown penny loafers', 'Black penny loafers', 'Tan tassel loafers', 'Brown derby shoes']);
    return pool;
};

const itemMapOf = items => new Map(items.map(entry => [String(entry.id), entry]));

describe('the edge validates every look independently', () => {
    test('a look that shares a product with an earlier look is still kept', () => {
        const pool = richPool();
        const sharedTop = pool.find(entry => entry.id === 'top_0');
        const first = look('l1', [pool.find(e => e.id === 'blz_0'), sharedTop, pool.find(e => e.id === 'bot_0'), pool.find(e => e.id === 'sho_0')]);
        const second = look('l2', [pool.find(e => e.id === 'blz_1'), sharedTop, pool.find(e => e.id === 'bot_1'), pool.find(e => e.id === 'sho_1')]);
        const kept = api.filterValidOutfits([first, second], 'semi_classic', itemMapOf(pool), true, undefined, 'Work');
        expect(kept).toHaveLength(2);
        expect(kept[0].items.map(entry => entry.id)).toContain('top_0');
        expect(kept[1].items.map(entry => entry.id)).toContain('top_0');
    });

    test('an exact duplicate look is still rejected', () => {
        const pool = richPool();
        const items = [pool.find(e => e.id === 'blz_0'), pool.find(e => e.id === 'top_0'), pool.find(e => e.id === 'bot_0'), pool.find(e => e.id === 'sho_0')];
        const kept = api.filterValidOutfits([look('l1', items), look('l2', items)], 'semi_classic', itemMapOf(pool), true, undefined, 'Work');
        expect(kept).toHaveLength(1);
    });

    test('an unknown item id is rejected', () => {
        const pool = richPool();
        const kept = api.filterValidOutfits(
            [look('l1', [...pool.filter(e => ['blz_0', 'top_0', 'bot_0'].includes(e.id)), item('ghost', 'shoes', 'Ghost shoes')])],
            'semi_classic', itemMapOf(pool), true, undefined, 'Work',
        );
        expect(kept).toHaveLength(0);
    });
});

describe('the deterministic fallback builds candidates, it does not judge the deck', () => {
    test('it never repeats an exact look', async () => {
        const pool = thinPool();
        const candidates = await api.localFallback(pool, 'semi_classic', 'Work', 3, true);
        const signatures = candidates.map(entry => entry.items.map(item => String(item.id)).sort().join('|'));
        expect(new Set(signatures).size).toBe(signatures.length);
    });

    test('it does not enforce cross-look product uniqueness (the global selector owns that rule)', async () => {
        // Two layers and two tops, but only ONE bottom and ONE pair of shoes:
        // the rotation can only produce looks that share those two products.
        // The fallback offers them anyway — whether the deck may repeat them
        // is the global selector's decision, not the fallback's.
        const pool = [
            item('o_0', 'outerwear', 'Navy wool blazer'),
            item('o_1', 'outerwear', 'Charcoal wool blazer'),
            item('t_0', 'top', 'White oxford shirt'),
            item('t_1', 'top', 'Blue poplin shirt'),
            item('b_0', 'bottom', 'Grey tailored trousers'),
            item('s_0', 'shoes', 'Brown penny loafers'),
        ];
        const candidates = await api.localFallback(pool, 'semi_classic', 'Work', 3, true);
        expect(candidates.length).toBeGreaterThan(1);
        const bottomIds = candidates.map(entry => (entry.items || []).filter(item => item.macroCategory === 'bottom').map(item => String(item.id)).join('|'));
        expect(new Set(bottomIds).size).toBeLessThan(bottomIds.length);
    });

    test('it returns no candidates for a non-positive limit', async () => {
        expect(await api.localFallback(thinPool(), 'semi_classic', 'Work', 0, true)).toEqual([]);
    });
});

describe('provider routing', () => {
    const edgeSource = fs.readFileSync(path.join(root, 'generate-outfits/index.ts'), 'utf8');

    test('generation is NVIDIA-only: the unreachable second provider is gone', () => {
        expect(edgeSource).not.toMatch(/callGemini/);
        expect(edgeSource).not.toMatch(/geminiKey/);
        expect(edgeSource).not.toMatch(/geminiModelPrimary/);
        expect(edgeSource).not.toMatch(/gemini_token/);
    });

    test('the routing policy is documented in the code', () => {
        expect(edgeSource).toMatch(/PROVIDER_ROUTING/);
    });

    test('the retry budget counts logical provider calls, not HTTP attempts', () => {
        expect(edgeSource).toMatch(/providerCalls/);
        expect(edgeSource).not.toMatch(/nvidiaAttempts/);
    });

    test('a provider failure falls through to the deterministic fallback', () => {
        expect(edgeSource).toMatch(/PROVIDER_GENERATION_FAILED/);
        expect(edgeSource).toMatch(/DETERMINISTIC_FALLBACK_USED/);
    });
});
