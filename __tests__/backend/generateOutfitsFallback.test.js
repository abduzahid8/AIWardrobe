/** @jest-environment node */
const fs = require('fs'), path = require('path'), ts = require('typescript');
const root = process.env.NVIDIA_REQUEST_SOURCE_ROOT || path.resolve(__dirname, '../../backend/supabase/functions');

// The shared helpers the edge function imports.
const shared = {};
for (const file of ['../../../src/domain/occasionRules.ts', '../../../src/domain/outfitStructure.ts', '../../../src/domain/outfitDiversity.ts']) {
    Object.assign(shared, require(path.join(root,file)));
}
shared.styleInference = require('../../src/domain/styleInference');
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
    compiled + '\nreturn { filterValidOutfits, isShopOnlyDeck };',
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

// Real handler with the shared offline transport fixtures.
const wireHarness = fs
    .readFileSync(path.join(__dirname, 'outfitBudgetWire.test.js'), 'utf8')
    .split('const owned=')[0];
const loadWire = new Function('require', '__dirname', 'jest', wireHarness + '\nreturn load;')(
    require,
    __dirname,
    jest
);
const owned = [
    ['wardrobe-shirt', 'top', 'White cotton shirt'],
    ['wardrobe-pants', 'bottom', 'Navy tailored cotton trousers'],
    ['wardrobe-shoes', 'shoes', 'Brown leather dress loafers'],
    ['wardrobe-layer', 'outerwear', 'Beige linen blazer'],
].map(([id, macroCategory, name]) => ({ id, macroCategory, name, type: macroCategory, color: 'neutral', isWardrobe: true, imageUrl: `https://example.invalid/${id}.jpg`, currency: 'USD' }));

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

// The pre-refactor deterministic `localFallback` helper was replaced by the
// bounded graph planner (`createOutfitPlan`), which is covered directly by
// __tests__/services/validatedQuartets.test.ts.

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

    test('the edge makes one bounded NVIDIA ranking call and owns no second provider', () => {
        expect(edgeSource.match(/rankNvidiaLooks\(/g)).toHaveLength(1);
        expect(edgeSource).toMatch(/providerTrace\.push\(ranking\.attempt\)/);
        expect(edgeSource).not.toMatch(/nvidiaAttempts|callGemini/);
    });

    test('a provider failure falls through to the deterministic fallback', async () => {
        const { handler, fetchMock } = loadWire();
        fetchMock.mockRejectedValue(new Error('provider outage'));
        const response = await handler(
            new Request('https://example.invalid', {
                method: 'POST',
                headers: { Authorization: 'Bearer fixture-user' },
                body: JSON.stringify({
                    prompt: 'Create one casual outfit.',
                    stylePreferences: 'casual',
                    occasion: 'Everyday',
                    weather: { temp: 20 },
                    limit: 1,
                    wardrobeItems: owned,
                    useProvidedWardrobeOnly: true,
                }),
            })
        );
        expect(response.status).toBe(200);
        const data = await response.json();
        // The feasible planner seed survives a failed rank call.
        expect(data.outfits.length).toBeGreaterThan(0);
        expect(data.outfits[0].selectionSource).toBe('deterministic_fallback');
    });
});
