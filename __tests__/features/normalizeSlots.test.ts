/**
 * normalizeSlots.test.ts — REAL integration tests for the outfit pick code.
 *
 * Unlike the mirror helpers elsewhere, these drive createSlotNormalizer
 * (the exact function the screen calls) with incident-realistic pools.
 * Added after user screenshots showed: a tee rendered as LAYER, a tee
 * rendered as PANTS, and driving moccasins in Old Money decks.
 */
import { createSlotNormalizer, ensureSlotFloors, isFourSlotLook } from '../../features/outfit-generator/utils/normalizeSlots';

function ctx(overrides: Record<string, unknown> = {}) {
  return {
    weather: undefined,
    baseItem: null,
    baseItemId: null,
    currentSource: 'shop',
    selectedOccasion: 'work',
    generationNonceRef: { current: 1 },
    poolFallback: [],
    ...overrides,
  };
}

const item = (id: string, macro: string, name: string, extra: Record<string, unknown> = {}) => ({
  id,
  name,
  type: macro,
  macroCategory: macro,
  image: `https://example.com/${id}.jpg`,
  imageUrl: `https://example.com/${id}.jpg`,
  color: 'neutral',
  brand: '',
  ...extra,
});

describe('normalizeTo4Slots — screenshot incidents', () => {
  it('never renders a tee as LAYER or PANTS (mistagged feed rows get repaired)', () => {
    const normalize = createSlotNormalizer(ctx());
    const incoming = [
      item('edge-tee-layer', 'outerwear', '100% cotton short sleeve T-shirt'),
      item('edge-shirt', 'top', 'Light indigo shirt with white stripes'),
      item('edge-tee-pants', 'bottom', 'Short sleeve linen and cotton T-shirt'),
      item('edge-loafer', 'shoes', 'Leather loafers with penny strap'),
    ];
    const pool = [
      item('pool-blazer', 'outerwear', 'Cool wool suit blazer'),
      item('pool-oxford', 'top', 'White oxford shirt'),
      item('pool-trousers', 'bottom', 'Charcoal pleated trousers'),
      item('pool-derby', 'shoes', 'Black derby shoes'),
    ];
    const out = normalize(incoming, 'old_money', 0, 7, pool);
    const byMacro = Object.fromEntries(out.map((o: any) => [o.macroCategory, o]));
    // Layer slot holds a real layer, pants slot holds real bottoms.
    expect(byMacro.outerwear).toBeDefined();
    expect(/t-shirt|tee/i.test(byMacro.outerwear.name || '')).toBe(false);
    expect(byMacro.bottom).toBeDefined();
    expect(/t-shirt|tee/i.test(byMacro.bottom.name || '')).toBe(false);
    expect(out.length).toBe(4);
  });

  it('keeps a clean blazer deck intact (does not swap good AI picks)', () => {
    const normalize = createSlotNormalizer(ctx());
    const incoming = [
      item('in-blazer', 'outerwear', 'Cool wool suit blazer'),
      item('in-shirt', 'top', '100% linen slim fit shirt'),
      item('in-trousers', 'bottom', 'Wool blend cool wool darted pants'),
      item('in-loafer', 'shoes', 'Leather loafers with penny strap'),
    ];
    const pool = [
      item('pool-pajama', 'top', 'Red and white striped pajama set'),
      item('pool-hoodie', 'outerwear', 'Grey hoodie'),
      item('pool-shorts', 'bottom', 'Cargo shorts'),
      item('pool-runner', 'shoes', 'Chunky running sneakers'),
    ];
    const out = normalize(incoming, 'old_money', 0, 7, pool);
    const ids = out.map((o: any) => o.id);
    // Every incoming piece is valid — the steady deck survives untouched.
    expect(ids).toEqual(['in-blazer', 'in-shirt', 'in-trousers', 'in-loafer']);
  });

  it('replaces a rejected incoming layer with a pool layer (anorak case)', () => {
    const normalize = createSlotNormalizer(ctx());
    const incoming = [
      item('in-anorak', 'outerwear', 'Navy blue anorak with high neck'),
      item('in-shirt', 'top', 'White oxford shirt'),
      item('in-trousers', 'bottom', 'Charcoal pleated trousers'),
      item('in-loafer', 'shoes', 'Brown penny loafers'),
    ];
    const pool = [item('pool-blazer', 'outerwear', 'Navy blazer with metal buttons')];
    const out = normalize(incoming, 'old_money', 0, 7, pool);
    const layer = (out as any[]).find((o: any) => o.macroCategory === 'outerwear');
    expect(layer.id).toBe('pool-blazer');
  });

  it('locks the anchor into its slot in wardrobe mode', () => {
    const anchor = item('anchor-loafer', 'shoes', 'Brown penny loafers');
    const normalize = createSlotNormalizer(
      ctx({ currentSource: 'wardrobe', baseItem: anchor, baseItemId: 'anchor-loafer' }),
    );
    const incoming = [
      item('w-top', 'top', 'White shirt'),
      item('w-bot', 'bottom', 'Navy trousers'),
      item('w-shoe', 'shoes', 'Black derbies'),
    ];
    const out = normalize(incoming, 'classic', 0, 7, []);
    expect((out as any[]).some((o: any) => String(o.id) === 'anchor-loafer')).toBe(true);
  });

  it('prefers deck-unused ids across sequential cards', () => {
    const deckUsed = new Set<string>();
    const pool = [
      item('pants-a', 'bottom', 'Charcoal pleated trousers'),
      item('pants-b', 'bottom', 'Beige chinos'),
      item('top-a', 'top', 'White oxford shirt'),
      item('shoe-a', 'shoes', 'Brown loafers'),
    ];
    const first: any[] = createSlotNormalizer(ctx())([], 'casual', 0, 7, pool, deckUsed);
    const second: any[] = createSlotNormalizer(ctx())([], 'casual', 1, 7, pool, deckUsed);
    const bottom1 = first.find((o: any) => o.macroCategory === 'bottom')?.id;
    const bottom2 = second.find((o: any) => o.macroCategory === 'bottom')?.id;
    // With two bottoms available, consecutive cards must not repeat.
    expect(bottom1).not.toBe(bottom2);
  });

  it('never clones a jacket into the top slot or parks a shirt as pants', () => {
    const normalize = createSlotNormalizer(ctx());
    const incoming = [
      item('j1', 'top', 'Cool wool suit jacket'),
      item('j1-copy', 'top', 'Cool wool suit jacket'),
      item('shirt-as-pants', 'bottom', 'Black short-sleeved cotton shirt'),
      item('loafers', 'shoes', 'Soft oiled leather penny loafers'),
    ];
    const pool = [
      item('pool-blazer', 'outerwear', 'Navy unstructured blazer'),
      item('pool-oxford', 'top', 'White oxford shirt'),
      item('pool-trousers', 'bottom', 'Tapered fit linen pants'),
      item('pool-derby', 'shoes', 'Black derby shoes'),
    ];
    const out = normalize(incoming, 'old_money', 0, 7, pool);
    const names = (out as any[]).map((o) => String(o.name || ''));
    const jacketTiles = names.filter((n) => /suit jacket|blazer/i.test(n));
    expect(jacketTiles.length).toBeLessThanOrEqual(1);
    const pants = (out as any[]).find((o) => o.macroCategory === 'bottom');
    expect(/short-sleeved|shirt/i.test(pants?.name || '')).toBe(false);
    expect(/pant|trouser|chino/i.test(pants?.name || '')).toBe(true);
    const macros = (out as any[]).map((o) => o.macroCategory);
    expect(macros).toContain('bottom');
    expect(macros).toContain('shoes');
    expect(macros.filter((m: string) => m === 'top').length).toBeGreaterThanOrEqual(1);
  });

  it('swaps driving moccasins and olive casual shoes out of classic decks', () => {
    const normalize = createSlotNormalizer(ctx({ selectedOccasion: 'casual' }));
    const incoming = [
      item('in-jacket', 'outerwear', 'Navy blue jacket with pockets'),
      item('in-shirt', 'top', '100% linen regular fit shirt'),
      item('in-pants', 'bottom', 'Tapered fit linen pants'),
      item('in-mocs', 'shoes', 'Soft leather loafers with tie', { color: 'olive' }),
    ];
    const pool = [
      item('pool-penny', 'shoes', 'Leather loafers with penny strap', { color: 'brown' }),
    ];
    const out = normalize(incoming, 'old_money', 0, 7, pool);
    const shoes = (out as any[]).find((o) => o.macroCategory === 'shoes');
    expect(shoes?.id).toBe('pool-penny');
  });

  it('honors comment work/old-money over leftover casual chips and swaps zip/plaid/loafers', () => {
    const normalize = createSlotNormalizer(ctx({
      selectedOccasion: 'casual',
      budgetText: 'work tomorrow, old money, $400',
    }));
    const incoming = [
      item('in-zip', 'outerwear', '100% linen total look jacket', { price: 90, isShopItem: true }),
      item('in-plaid', 'top', 'Zara Relaxed Fit Plaid Shirt', { price: 40, isShopItem: true }),
      item('in-pants', 'bottom', 'Charcoal pleated trousers', { price: 80, isShopItem: true }),
      item('in-split', 'shoes', 'Split leather loafers with white sole', { price: 70, isShopItem: true }),
    ];
    const pool = [
      item('pool-blazer', 'outerwear', 'Navy wool blazer', { price: 120, isShopItem: true }),
      item('pool-oxford', 'top', 'White oxford shirt', { price: 45, isShopItem: true }),
      item('pool-trousers', 'bottom', 'Wool blend trousers', { price: 90, isShopItem: true }),
      item('pool-penny', 'shoes', 'Brown penny loafers', { price: 80, isShopItem: true }),
    ];
    const out = normalize(incoming, 'old_money', 0, 7, pool);
    const layer = (out as any[]).find((o) => o.macroCategory === 'outerwear');
    const top = (out as any[]).find((o) => o.macroCategory === 'top');
    const shoes = (out as any[]).find((o) => o.macroCategory === 'shoes');
    expect(layer?.id).toBe('pool-blazer');
    expect(top?.id).toBe('pool-oxford');
    expect(shoes?.id).toBe('pool-penny');
  });

  it('does not repeat the same layer product name across a deck', () => {
    const normalize = createSlotNormalizer(ctx({ selectedOccasion: 'casual' }));
    const deck = new Set<string>();
    const shared = [
      item('shirt-a', 'top', 'White oxford shirt'),
      item('pants-a', 'bottom', 'Beige chinos'),
      item('shoe-a', 'shoes', 'Brown penny loafers'),
    ];
    const pool = [
      item('zip-b', 'outerwear', '100% linen total look jacket'),
      item('blazer', 'outerwear', 'Navy wool blazer'),
      ...shared,
    ];
    const out0 = normalize([item('zip-a', 'outerwear', '100% linen total look jacket'), ...shared], 'casual', 0, 1, pool, deck);
    const out1 = normalize([item('zip-b2', 'outerwear', '100% linen total look jacket'), ...shared], 'casual', 1, 1, pool, deck);
    const layer0 = (out0 as any[]).find((o) => o.macroCategory === 'outerwear')?.name;
    const layer1 = (out1 as any[]).find((o) => o.macroCategory === 'outerwear')?.name;
    expect(layer0).toBe('100% linen total look jacket');
    expect(layer1).toBe('Navy wool blazer');
    expect((out1 as any[]).find((o) => o.macroCategory === 'bottom')?.id).toBe('pants-a');
    expect((out1 as any[]).find((o) => o.macroCategory === 'shoes')?.id).toBe('shoe-a');
  });

  it('rotates same-named catalog clones so three cards are not the same look', () => {
    const normalize = createSlotNormalizer(ctx({ selectedOccasion: 'work' }));
    const deck = new Set<string>();
    const pool = [
      item('oxford', 'top', 'White oxford shirt'),
      item('blue-linen', 'top', 'Light blue linen shirt'),
      item('blazer-grey', 'outerwear', 'Grey wool blazer'),
      item('blazer-db', 'outerwear', 'Navy double-breasted blazer'),
      item('pants-navy', 'bottom', 'Navy wool trousers'),
      item('pants-beige', 'bottom', 'Beige chinos'),
      item('shoe-oxford', 'shoes', 'Black oxford shoes'),
      item('shoe-burgundy', 'shoes', 'Burgundy leather loafers'),
    ];
    const outs = [0, 1, 2].map((i) =>
      normalize(
        [
          item(`in-b-${i}`, 'outerwear', 'Navy wool blazer'),
          item(`in-t-${i}`, 'top', '100% linen slim fit shirt'),
          item(`in-p-${i}`, 'bottom', 'Charcoal wool trousers'),
          item(`in-s-${i}`, 'shoes', 'Brown penny loafers'),
        ],
        'old_money',
        i,
        1,
        pool,
        deck,
      ),
    );
    const names = (macro: string) => outs.map((o) => (o as any[]).find((x) => x.macroCategory === macro)?.name);
    expect(new Set(names('top')).size).toBe(3);
    expect(new Set(names('bottom')).size).toBe(3);
    expect(new Set(names('outerwear')).size).toBe(3);
    expect(new Set(names('shoes')).size).toBe(3);
  });

  it('three cards from a rich pool share at most one piece', () => {
    const normalize = createSlotNormalizer(ctx({ selectedOccasion: 'work', currentSource: 'shop' }));
    const deck = new Set<string>();
    const pool = [
      item('blazer-1', 'outerwear', 'Navy wool blazer'),
      item('blazer-2', 'outerwear', 'Grey wool blazer'),
      item('blazer-3', 'outerwear', 'Camel sport coat'),
      item('shirt-1', 'top', 'White oxford shirt'),
      item('shirt-2', 'top', 'Blue oxford shirt'),
      item('shirt-3', 'top', 'Poplin dress shirt'),
      item('pants-1', 'bottom', 'Charcoal wool trousers'),
      item('pants-2', 'bottom', 'Navy wool trousers'),
      item('pants-3', 'bottom', 'Beige chinos'),
      item('shoe-1', 'shoes', 'Black oxford shoes'),
      item('shoe-2', 'shoes', 'Brown penny loafers'),
      item('shoe-3', 'shoes', 'Burgundy leather loafers'),
    ];
    const outs = [0, 1, 2].map((i) => normalize([], 'old_money', i, 9, pool, deck));
    for (let a = 0; a < 3; a++) {
      for (let b = a + 1; b < 3; b++) {
        const idsA = new Set((outs[a] as any[]).map((x) => x.id));
        const shared = (outs[b] as any[]).filter((x) => idsA.has(x.id)).length;
        expect(shared).toBeLessThan(2);
      }
    }
  });

  it('does not put two loafer clones in one deck when oxfords exist', () => {
    const normalize = createSlotNormalizer(ctx({ selectedOccasion: 'work', currentSource: 'shop' }));
    const deck = new Set<string>();
    const pool = [
      item('blazer-1', 'outerwear', 'Navy wool blazer'),
      item('blazer-2', 'outerwear', 'Grey wool blazer'),
      item('shirt-1', 'top', 'White oxford shirt'),
      item('shirt-2', 'top', 'Blue oxford shirt'),
      item('pants-1', 'bottom', 'Charcoal wool trousers'),
      item('pants-2', 'bottom', 'Beige chinos'),
      item('shoe-1', 'shoes', 'Soft oiled leather penny loafers'),
      item('shoe-2', 'shoes', 'Soft oiled leather loafers'),
      item('shoe-3', 'shoes', 'Black oxford shoes'),
    ];
    const a = normalize([], 'old_money', 0, 9, pool, deck);
    const b = normalize([], 'old_money', 1, 9, pool, deck);
    const shoes = [...a, ...b]
      .filter((x: any) => x.macroCategory === 'shoes')
      .map((x: any) => String(x.name).toLowerCase());
    expect(shoes.filter((n) => n.includes('loafer')).length).toBeLessThan(2);
  });

  it('never fills a clothing slot with sunglasses or a card holder', () => {
    const normalize = createSlotNormalizer(ctx({ selectedOccasion: 'casual', currentSource: 'shop' }));
    const pool = [
      item('sun', 'top', 'Rectangle Sunglasses'),
      item('card', 'top', 'Nappa leather card holder'),
      item('tee', 'top', 'White cotton t-shirt'),
      item('hoodie', 'outerwear', 'Navy zip hoodie'),
      item('pants', 'bottom', 'Grey sweatpants'),
      item('shoe', 'shoes', 'White sneakers'),
    ];
    const out = normalize([], 'minimalist', 0, 1, pool, new Set());
    const names = (out as any[]).map((x) => String(x.name).toLowerCase()).join(' ');
    expect(names).not.toMatch(/sunglass|card holder/);
  });

  it('never fills TOP with a tie or PANTS with pyjamas', () => {
    const normalize = createSlotNormalizer(ctx({ selectedOccasion: 'formal', currentSource: 'shop' }));
    const pool = [
      item('blazer', 'outerwear', 'Navy wool blazer'),
      item('tie', 'top', '100% silk polka dot tie'),
      item('shirt', 'top', 'White oxford shirt'),
      item('pj', 'bottom', 'Short sleeve pyjama trousers'),
      item('pants', 'bottom', 'Charcoal wool trousers'),
      item('shoe', 'shoes', 'Black oxford shoes'),
    ];
    const out = normalize([], 'old_money', 0, 1, pool, new Set());
    const names = (out as any[]).map((x) => String(x.name).toLowerCase()).join(' ');
    expect(names).not.toMatch(/tie|pyjama|pajama/);
    expect(names).toMatch(/oxford shirt/);
    expect(names).toMatch(/wool trousers/);
  });

  it('always returns complete outfits, falling back to placeholders', () => {
    const normalize = createSlotNormalizer(ctx());
    const out = normalize([], 'casual', 0, 7, []);
    expect(out.length).toBeGreaterThan(0);
    const macros = (out as any[]).map((o: any) => o.macroCategory);
    expect(macros).toContain('bottom');
    expect(macros).toContain('shoes');
  });

  it('never lets a 250$ budget pick a look over the cap', () => {
    const normalize = createSlotNormalizer(ctx({ budgetText: '250$' }));
    const incoming = [
      item('shop_md-exp-jacket', 'outerwear', 'Cool wool suit jacket', { price: 220, isShopItem: true }),
      item('shop_md-exp-shirt', 'top', '100% linen slim fit shirt', { price: 90, isShopItem: true }),
      item('shop_md-exp-pants', 'bottom', 'Wool blend cool wool darted pants', { price: 140, isShopItem: true }),
      item('shop_md-exp-shoes', 'shoes', 'Smart nappa loafers', { price: 180, isShopItem: true }),
    ];
    const pool = [
      item('md-cheap-jacket', 'outerwear', 'Linen jacket with pockets', { price: 70, isShopItem: true }),
      item('md-cheap-shirt', 'top', 'White oxford shirt', { price: 45, isShopItem: true }),
      item('md-cheap-pants', 'bottom', 'Tapered fit linen pants', { price: 59, isShopItem: true }),
      item('md-cheap-shoes', 'shoes', 'Leather loafers with penny strap', { price: 69, isShopItem: true }),
      item('zara-overshirt-pants', 'bottom', 'Short-sleeved pink overshirt', { price: 40, isShopItem: true }),
    ];
    const out = normalize(incoming, 'old_money', 0, 7, pool);
    const spend = (out as any[]).reduce((s, o) => s + Number(o.price || 0), 0);
    expect(spend).toBeLessThanOrEqual(250);
    const pants = (out as any[]).find((o) => o.macroCategory === 'bottom');
    expect(/overshirt/i.test(pants?.name || '')).toBe(false);
  });
});

describe('ensureSlotFloors + incomplete edge looks', () => {
  it('fills shirts/shoes when taste-filter starved the payload', () => {
    const thin = [
      item('j1', 'outerwear', 'Cool wool suit jacket'),
      item('p1', 'bottom', 'Wool blend cool wool darted pants'),
      item('s-bad', 'footwear', 'Split leather loafers with moc toe'),
    ];
    const extras = [
      item('ox1', 'top', 'White oxford shirt'),
      item('ox2', 'top', 'Light blue linen shirt'),
      item('bl2', 'outerwear', 'Navy wool blazer'),
      item('tr2', 'bottom', 'Charcoal wool trousers'),
      item('penny', 'shoes', 'Brown penny loafers'),
      item('derby', 'shoes', 'Black derby shoes'),
    ];
    const floored = ensureSlotFloors(thin.filter((x) => !/split leather/i.test(x.name)), extras, 'old_money', 'work');
    const macros = floored.map((x) => x.macroCategory);
    expect(macros.filter((m) => m === 'top').length).toBeGreaterThanOrEqual(2);
    expect(macros).toContain('shoes');
    expect(floored.some((x) => /split leather|moc toe/i.test(x.name))).toBe(false);
  });

  it('fills a missing shirt so jacket+pants+shoes become a 4-slot look', () => {
    const normalize = createSlotNormalizer(ctx());
    const incoming = [
      item('md-63001443', 'outerwear', 'Cool wool suit jacket'),
      item('md-63001892', 'bottom', 'Wool blend cool wool darted pants'),
      item('shop_md-63047958', 'shoes', 'Split leather loafers with moc toe'),
    ];
    const pool = [
      item('pool-oxford', 'top', 'White oxford shirt'),
      item('pool-penny', 'shoes', 'Leather loafers with penny strap'),
      item('pool-blazer', 'outerwear', 'Navy wool blazer'),
      item('pool-trousers', 'bottom', 'Charcoal pleated trousers'),
    ];
    const out = normalize(incoming, 'old_money', 0, 7, pool);
    expect(isFourSlotLook(out)).toBe(true);
    const shoes = (out as any[]).find((o) => o.macroCategory === 'shoes');
    expect(/split leather|moc toe/i.test(shoes?.name || '')).toBe(false);
    const top = (out as any[]).find((o) => o.macroCategory === 'top');
    expect(top?.id).toBe('pool-oxford');
  });

  it('work and date from the same pool do not clone the same boardroom look', () => {
    const incoming = [
      item('db-suit', 'outerwear', '100% cool wool double-breasted suit blazer'),
      item('oxford', 'top', '100% cotton regular fit Oxford shirt'),
      item('wool-pants', 'bottom', 'Darted wool trousers'),
      item('loafers', 'shoes', 'Smart leather loafers'),
    ];
    const pool = [
      item('linen-blazer', 'outerwear', '100% linen blazer'),
      item('knit', 'outerwear', 'Cotton and linen mouliné knit sweater'),
      item('denim-shirt', 'top', '100% cotton regular fit denim shirt'),
      item('jeans', 'bottom', 'Selvedge fit jeans'),
    ];
    const work = createSlotNormalizer(ctx({ selectedOccasion: 'work', weather: { temp: 20, condition: 'mild' } }))(
      incoming, 'old_money', 0, 3, pool, new Set(),
    );
    const date = createSlotNormalizer(ctx({ selectedOccasion: 'date', weather: { temp: 20, condition: 'mild' } }))(
      incoming, 'old_money', 0, 3, pool, new Set(),
    );
    const workLayer = work.find((o: any) => o.macroCategory === 'outerwear')?.name || '';
    const dateLayer = date.find((o: any) => o.macroCategory === 'outerwear')?.name || '';
    const workTop = work.find((o: any) => o.macroCategory === 'top')?.name || '';
    const workBottom = work.find((o: any) => o.macroCategory === 'bottom')?.name || '';
    expect(/denim shirt/i.test(workTop)).toBe(false);
    expect(/jeans/i.test(workBottom)).toBe(false);
    expect(/double-breasted/i.test(workLayer)).toBe(true);
    expect(/double-breasted|suit blazer/i.test(dateLayer)).toBe(false);
  });

  it('travel never keeps a boardroom double-breasted suit as the layer', () => {
    const incoming = [
      item('db-suit', 'outerwear', '100% cool wool double-breasted suit blazer'),
      item('oxford', 'top', '100% cotton regular fit Oxford shirt'),
      item('wool-pants', 'bottom', 'Darted wool trousers'),
      item('loafers', 'shoes', 'Smart leather loafers'),
    ];
    const pool = [
      item('overshirt', 'outerwear', 'Linen overshirt'),
      item('knit', 'outerwear', 'Cotton and linen mouliné knit sweater'),
      item('chino', 'bottom', 'Slim stretch chinos'),
      item('sneaker', 'shoes', 'White leather sneakers'),
    ];
    const travel = createSlotNormalizer(ctx({ selectedOccasion: 'travel', weather: { temp: 22, condition: 'mild' } }))(
      incoming, 'semi_classic', 0, 3, pool, new Set(),
    );
    const layer = travel.find((o: any) => o.macroCategory === 'outerwear')?.name || '';
    expect(/double-breasted|suit blazer|cool wool suit/i.test(layer)).toBe(false);
    expect(/overshirt|knit|sweater/i.test(layer)).toBe(true);
  });

  it('at 15°C swaps a linen suit for wool, not summer cloth', () => {
    const incoming = [
      item('linen-blazer', 'outerwear', '100% linen blazer'),
      item('linen-shirt', 'top', '100% linen slim fit shirt'),
      item('linen-pants', 'bottom', '100% linen trousers'),
      item('espadrille', 'shoes', 'Split leather espadrille'),
    ];
    const pool = [
      item('wool-blazer', 'outerwear', 'Navy wool blazer'),
      item('oxford', 'top', 'White oxford shirt'),
      item('wool-pants', 'bottom', 'Darted wool trousers'),
      item('loafers', 'shoes', 'Leather penny loafers'),
    ];
    const out = createSlotNormalizer(ctx({ selectedOccasion: 'casual', weather: { temp: 15, condition: 'cool' } }))(
      incoming, 'semi_classic', 0, 3, pool, new Set(),
    );
    const names = out.map((o: any) => o.name).join(' | ');
    expect(/linen/i.test(names)).toBe(false);
    expect(/espadrille/i.test(names)).toBe(false);
    expect(/wool blazer/i.test(names)).toBe(true);
    expect(/oxford/i.test(names)).toBe(true);
    expect(/wool trousers/i.test(names)).toBe(true);
    expect(/loafer/i.test(names)).toBe(true);
  });
});
