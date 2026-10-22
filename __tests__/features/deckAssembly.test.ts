/**
 * deckAssembly — the production deck gate used by AIOutfitmaker.
 *
 * Regression: 3 looks that share the same blazer + trousers in every card.
 * Strict dedupe rejected them, but the screen then relaxed to exact-only
 * WITHOUT first generating fresh fills, so the clones came back.
 * assembleDeck must: strict candidates → strict fills (seeded with the
 * accepted looks' product keys) → relaxed fills → relaxed candidates.
 */
import { assembleDeck } from '../../features/outfit-generator/utils/deckDedupe';

type Item = { id: string; name: string; macroCategory: string; image?: string };
type Look = { items: Item[] };

const it_ = (id: string, macroCategory: string, name = id): Item => ({ id, macroCategory, name, image: `img://${id}` });
const keysOf = (item: Item): string[] => [String(item.id), `__n:${item.name.toLowerCase()}`, `__img:${item.image}`];
const itemsOf = (look: Look): Item[] => look.items;

function look(prefix: string): Look {
  return {
    items: [
      it_(`${prefix}-layer`, 'outerwear', `${prefix} blazer`),
      it_(`${prefix}-top`, 'top', `${prefix} shirt`),
      it_(`${prefix}-pants`, 'bottom', `${prefix} trousers`),
      it_(`${prefix}-shoes`, 'shoes', `${prefix} loafers`),
    ],
  };
}

/** Fill builder mirroring normalizeTo4Slots: fresh-per-slot first (rotated by attempt), reuse only when a slot is exhausted. */
function poolFillBuilder(pool: Item[]) {
  return (usedKeys: Set<string>, attempt = 0): Look | null => {
    const slots = ['outerwear', 'top', 'bottom', 'shoes'];
    const items: Item[] = [];
    for (const [slotIndex, slot] of slots.entries()) {
      const inSlot = pool.filter((p) => p.macroCategory === slot && !items.some((x) => x.id === p.id));
      const fresh = inSlot.filter((p) => !keysOf(p).some((k) => usedKeys.has(k)));
      const choices = fresh.length > 0 ? fresh : inSlot;
      const rotation = slot === 'shoes' ? attempt + Math.floor(attempt / 2) : attempt + slotIndex;
      const pick = choices[rotation % Math.max(choices.length, 1)];
      if (!pick) return null;
      items.push(pick);
      keysOf(pick).forEach((k) => usedKeys.add(k));
    }
    return { items };
  };
}

function bigPool(): Item[] {
  const pool: Item[] = [];
  for (const p of ['a', 'b', 'c', 'd']) pool.push(...look(p).items);
  return pool;
}

function sharedCount(a: Look, b: Look, skipIds = new Set<string>()): number {
  const bKeys = new Set(b.items.flatMap(keysOf));
  return a.items.filter((i) => !skipIds.has(i.id) && keysOf(i).some((k) => bKeys.has(k))).length;
}

function expectPairwiseDiverse(deck: Look[], skipIds = new Set<string>()) {
  for (let i = 0; i < deck.length; i++) {
    for (let j = i + 1; j < deck.length; j++) {
      expect(sharedCount(deck[i], deck[j], skipIds)).toBeLessThanOrEqual(1);
    }
  }
}

describe('assembleDeck diversity invariant', () => {
  it('A. large pool: 3 looks, every pair shares at most 1 item, preferring 0 repeats', () => {
    const { deck } = assembleDeck({
      candidates: [look('a')],
      itemsOf, keysOf,
      buildFill: poolFillBuilder(bigPool()),
      usedKeys: new Set<string>(),
    });
    expect(deck).toHaveLength(3);
    expectPairwiseDiverse(deck);
    const allIds = deck.flatMap((l) => l.items.map((i) => i.id));
    expect(new Set(allIds).size).toBe(allIds.length);
  });

  it('regression: 3 looks must not all share the same two non-anchor items when alternatives exist', () => {
    const sharedLayer = it_('navy-blazer', 'outerwear', 'navy blazer');
    const sharedPants = it_('grey-trousers', 'bottom', 'grey trousers');
    const candidates: Look[] = ['a', 'b', 'c'].map((p) => ({
      items: [sharedLayer, it_(`${p}-top`, 'top', `${p} shirt`), sharedPants, it_(`${p}-shoes`, 'shoes', `${p} shoes`)],
    }));
    const { deck, relaxed } = assembleDeck({
      candidates, itemsOf, keysOf,
      buildFill: poolFillBuilder(bigPool()),
      usedKeys: new Set<string>(),
    });
    expect(deck).toHaveLength(3);
    expect(relaxed).toBe(false);
    expectPairwiseDiverse(deck);
    const inAll = deck[0].items.filter((i) => deck.every((l) => l.items.some((x) => x.id === i.id)));
    expect(inAll).toHaveLength(0);
  });

  it('D. edge near-duplicates (A+B+C+D / A+B+E+F / A+B+G+H) are replaced by fresh fills, not relaxed in', () => {
    const A = it_('A', 'outerwear', 'blazer a');
    const B = it_('B', 'bottom', 'trousers b');
    const candidates: Look[] = [
      { items: [A, B, it_('C', 'top', 'shirt c'), it_('D', 'shoes', 'shoes d')] },
      { items: [A, B, it_('E', 'top', 'shirt e'), it_('F', 'shoes', 'shoes f')] },
      { items: [A, B, it_('G', 'top', 'shirt g'), it_('H', 'shoes', 'shoes h')] },
    ];
    const { deck } = assembleDeck({
      candidates, itemsOf, keysOf,
      buildFill: poolFillBuilder(bigPool()),
      usedKeys: new Set<string>(),
    });
    expect(deck).toHaveLength(3);
    expect(deck[0]).toBe(candidates[0]);
    expect(deck).not.toContain(candidates[1]);
    expect(deck).not.toContain(candidates[2]);
    expectPairwiseDiverse(deck);
  });

  it('B. anchor repeats in all 3 looks; every other piece rotates', () => {
    const anchor = it_('anchor-blazer', 'outerwear', 'selected blazer');
    const candidates: Look[] = ['a', 'b', 'c'].map((p) => ({
      items: [anchor, it_(`${p}-top`, 'top', `${p} shirt`), it_(`${p}-pants`, 'bottom', `${p} trousers`), it_(`${p}-shoes`, 'shoes', `${p} shoes`)],
    }));
    // Fill pool also carries the anchor as its only outerwear.
    const pool = [anchor, ...bigPool().filter((i) => i.macroCategory !== 'outerwear')];
    const { deck } = assembleDeck({
      candidates, itemsOf, keysOf,
      anchorKeys: new Set(keysOf(anchor)),
      buildFill: poolFillBuilder(pool),
      usedKeys: new Set<string>(),
    });
    expect(deck).toHaveLength(3);
    for (const l of deck) expect(l.items.some((i) => i.id === anchor.id)).toBe(true);
    expectPairwiseDiverse(deck, new Set([anchor.id]));
  });

  it('C. thin pool: one valid shoe is reused, other slots rotate, generation completes', () => {
    const onlyShoe = it_('only-shoe', 'shoes', 'only loafers');
    const pool: Item[] = [onlyShoe];
    for (const p of ['a', 'b', 'c']) {
      pool.push(it_(`${p}-layer`, 'outerwear', `${p} blazer`), it_(`${p}-top`, 'top', `${p} shirt`), it_(`${p}-pants`, 'bottom', `${p} trousers`));
    }
    const { deck } = assembleDeck({
      candidates: [], itemsOf, keysOf,
      buildFill: poolFillBuilder(pool),
      usedKeys: new Set<string>(),
    });
    expect(deck).toHaveLength(3);
    for (const l of deck) expect(l.items).toHaveLength(4);
    expectPairwiseDiverse(deck, new Set([onlyShoe.id]));
  });

  it('C2. exhausted pool still completes via relaxed fills instead of returning 1 look', () => {
    const tiny: Item[] = [
      it_('l1', 'outerwear', 'blazer'), it_('t1', 'top', 'shirt'),
      it_('p1', 'bottom', 'trousers'), it_('s1', 'shoes', 'loafers'),
      it_('t2', 'top', 'polo'), it_('s2', 'shoes', 'derbies'),
    ];
    const { deck, relaxed } = assembleDeck({
      candidates: [], itemsOf, keysOf,
      buildFill: poolFillBuilder(tiny),
      usedKeys: new Set<string>(),
    });
    expect(deck).toHaveLength(3);
    expect(relaxed).toBe(true);
    const sigs = deck.map((l) => l.items.map((i) => i.id).sort().join('|'));
    expect(new Set(sigs).size).toBe(3);
  });

  it('E. same product under different IDs (same name/image) counts as repeated', () => {
    const base = look('a');
    const clone: Look = {
      items: base.items.map((i) => ({ ...i, id: `${i.id}-sku2` })),
    };
    const { deck } = assembleDeck({
      candidates: [base, clone], itemsOf, keysOf,
      buildFill: poolFillBuilder(bigPool()),
      usedKeys: new Set<string>(),
    });
    expect(deck).toHaveLength(3);
    expect(deck).not.toContain(clone);
    expectPairwiseDiverse(deck);
  });

  it('seeds usedKeys from accepted candidates before the first fill', () => {
    const seenAtFill: string[][] = [];
    const builder = (usedKeys: Set<string>): Look | null => {
      seenAtFill.push([...usedKeys]);
      return poolFillBuilder(bigPool())(usedKeys);
    };
    assembleDeck({ candidates: [look('a')], itemsOf, keysOf, buildFill: builder, usedKeys: new Set<string>() });
    expect(seenAtFill.length).toBeGreaterThan(0);
    expect(seenAtFill[0]).toContain('a-layer');
    expect(seenAtFill[0]).toContain('__n:a blazer');
  });

  it('no fill builder: near-duplicate candidates are only relaxed in as a last resort', () => {
    const A = it_('A', 'outerwear', 'blazer a');
    const B = it_('B', 'bottom', 'trousers b');
    const candidates: Look[] = [
      { items: [A, B, it_('C', 'top', 'shirt c'), it_('D', 'shoes', 'shoes d')] },
      { items: [A, B, it_('E', 'top', 'shirt e'), it_('F', 'shoes', 'shoes f')] },
    ];
    const { deck, relaxed } = assembleDeck({ candidates, itemsOf, keysOf, usedKeys: new Set<string>() });
    expect(deck).toHaveLength(2);
    expect(relaxed).toBe(true);
  });
});
