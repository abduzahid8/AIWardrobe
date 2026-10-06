/**
 * slotPickOrder.test.ts — orderSlotCandidates contract.
 *
 * Production incident: AIOutfitmaker shuffled edge picks + pool picks
 * TOGETHER, so the edge's careful choices (blazer + linen shirt for
 * old_money/work) were replaced by random pool items (pajama top). And with
 * no memory across the 3 normalize calls, thin formal pools repeated the
 * same pants + shoes in 2 of 3 cards.
 *
 * Order produced: incoming (edge/AI rank) first, then seeded-shuffled pool;
 * each group stable-partitioned deck-unused-first (falls back to used ids
 * when the pool is exhausted, so decks never shrink).
 */
import { orderSlotCandidates, selectSlotPick, shuffleWithSeed } from '../../features/outfit-generator/utils/deckDedupe';
import { rejectForStyleOccasion } from '../../features/outfit-generator/utils/occasionRules';

const item = (id: string) => ({ id, name: id });

describe('orderSlotCandidates', () => {
  it('puts incoming matches before pool matches (AI intent wins)', () => {
    const ordered = orderSlotCandidates([item('edge-blazer')], [item('pool-jacket'), item('pool-coat')], { seed: 42 });
    expect(ordered.map((x) => x.id)).toEqual(['edge-blazer', expect.any(String), expect.any(String)]);
    expect(ordered[0].id).toBe('edge-blazer');
    expect(ordered.slice(1).map((x) => x.id).sort()).toEqual(['pool-coat', 'pool-jacket']);
  });

  it('shuffles the pool deterministically by seed', () => {
    const pool = [item('a'), item('b'), item('c'), item('d'), item('e')];
    const once = orderSlotCandidates([], pool, { seed: 7 });
    const twice = orderSlotCandidates([], pool, { seed: 7 });
    const other = orderSlotCandidates([], pool, { seed: 8 });
    expect(once.map((x) => x.id)).toEqual(twice.map((x) => x.id));
    // Different seeds (almost surely) order differently on 5 items.
    expect(other.map((x) => x.id)).not.toEqual(once.map((x) => x.id));
    expect([...once.map((x) => x.id)].sort()).toEqual(['a', 'b', 'c', 'd', 'e']);
  });

  it('prefers deck-unused ids within each group, falls back when exhausted', () => {
    const used = new Set(['edge-blazer', 'pool-jacket']);
    const ordered = orderSlotCandidates(
      [item('edge-blazer'), item('edge-shirt')],
      [item('pool-jacket'), item('pool-coat')],
      { seed: 42, deckUsedIds: used },
    );
    // Fresh incoming first, then used incoming, then fresh pool, then used pool.
    expect(ordered.map((x) => x.id)[0]).toBe('edge-shirt');
    expect(ordered.map((x) => x.id)).toEqual(['edge-shirt', 'edge-blazer', 'pool-coat', 'pool-jacket']);
  });

  it('treats same-name clones as already used via keysOf', () => {
    const used = new Set(['__n:penny loafer']);
    const ordered = orderSlotCandidates(
      [{ id: 'shoe-17', name: 'penny loafer' }],
      [{ id: 'shoe-21', name: 'burgundy loafer' }],
      {
        seed: 1,
        deckUsedIds: used,
        keysOf: (c: any) => (c.name === 'penny loafer' ? ['__n:penny loafer'] : []),
      },
    );
    // Incoming still ranks first, but the used clone is partitioned after
    // unused pool when selectSlotPick skips keysOf matches.
    expect(ordered.map((x) => x.id)).toEqual(['shoe-17', 'shoe-21']);
    expect(selectSlotPick(ordered, {
      deckUsedIds: used,
      keysOf: (c: any) => (c.name === 'penny loafer' ? ['__n:penny loafer'] : []),
    })?.id).toBe('shoe-21');
  });

  it('returns everything (used included) when the pool is exhausted', () => {
    const used = new Set(['a', 'b']);
    const ordered = orderSlotCandidates([], [item('a'), item('b')], { seed: 1, deckUsedIds: used });
    expect(ordered.map((x) => x.id).sort()).toEqual(['a', 'b']);
  });

  it('ignores an empty deck set (identical to no memory)', () => {
    const pool = [item('a'), item('b'), item('c')];
    const noMem = orderSlotCandidates([item('z')], pool, { seed: 3 });
    const emptyMem = orderSlotCandidates([item('z')], pool, { seed: 3, deckUsedIds: new Set() });
    expect(emptyMem.map((x) => x.id)).toEqual(noMem.map((x) => x.id));
  });

  it('handles empty inputs', () => {
    expect(orderSlotCandidates([], [], { seed: 1 })).toEqual([]);
  });
});

/**
 * Incident replay: old_money + work deck from the logs.
 * Pool tops contained a "Red and white striped pajama set" that won the
 * formal top pick, and 2 formal-clean bottoms repeated across 2 of 3 cards.
 * Simulates pickCandidate's formal-first tier + shared deck memory:
 *  - pajama never survives the formal tier (sleepwear veto),
 *  - 3 sequential picks over 2 bottoms use both before repeating.
 */
describe('incident replay: old_money/work thin formal pool', () => {
  const tops = [
    { id: 'pajama', name: 'Red and white striped pajama set. Long-sleeved shirt with buttons.' },
    { id: 'polo', name: 'White long-sleeve polo shirt. Loose fit, V-neck collar.' },
    { id: 'linen', name: '100% linen slim fit shirt' },
  ];
  const bottoms = [
    { id: 'darted', name: 'Wool blend cool wool darted pants' },
    { id: 'tapered', name: 'Tapered fit linen pants' },
  ];
  const formalTopTier = (list: Array<{ id: string; name: string }>) =>
    list.filter((c) => rejectForStyleOccasion(c, 'old_money', 'work') === null);

  it('pajama never clears the formal top tier', () => {
    const clean = formalTopTier(tops);
    expect(clean.map((c) => c.id).sort()).toEqual(['linen', 'polo']);
  });

  it('three cards over two bottoms use both before repeating', () => {
    // Mirrors pickCandidate: order, then rotation-index inside the fresh
    // partition, then mark deck-used.
    const deckUsed = new Set<string>();
    const picks: string[] = [];
    for (let card = 0; card < 3; card++) {
      const ordered = orderSlotCandidates([], bottoms, { seed: 1234, deckUsedIds: deckUsed });
      const found = selectSlotPick(ordered, { rotationOffset: card, slotOffset: 6, deckUsedIds: deckUsed });
      picks.push(found!.id);
      deckUsed.add(found!.id);
    }
    // Both bottoms appear; no card clones another card's single pick set here
    // (with 2 items / 3 cards one repeat is pigeonhole-unavoidable).
    expect(new Set(picks).size).toBe(2);
    expect(picks[0]).not.toBe(picks[1]);
  });
});

describe('selectSlotPick', () => {
  const a = { id: 'a' };
  const b = { id: 'b' };
  const c = { id: 'c' };

  it('returns undefined for an empty list', () => {
    expect(selectSlotPick([], {})).toBeUndefined();
  });

  it('rotates by rotationOffset + slotOffset without deck memory', () => {
    const ordered = [a, b, c];
    expect(selectSlotPick(ordered, { rotationOffset: 0, slotOffset: 0 })?.id).toBe('a');
    expect(selectSlotPick(ordered, { rotationOffset: 1, slotOffset: 0 })?.id).toBe('b');
    expect(selectSlotPick(ordered, { rotationOffset: 0, slotOffset: 9 })?.id).toBe('a');
    expect(selectSlotPick(ordered, { rotationOffset: 1, slotOffset: 9 })?.id).toBe('b');
  });

  it('stays inside the fresh partition while one exists', () => {
    const ordered = [a, b, c];
    // 'a' used: rotation would land on 'a' (index 0) but fresh wins.
    expect(selectSlotPick(ordered, { rotationOffset: 0, slotOffset: 0, deckUsedIds: new Set(['a']) })?.id).toBe('b');
  });

  it('falls back to the full list when everything is used', () => {
    const ordered = [a, b];
    const used = new Set(['a', 'b']);
    expect(selectSlotPick(ordered, { rotationOffset: 0, slotOffset: 0, deckUsedIds: used })?.id).toBe('a');
    expect(selectSlotPick(ordered, { rotationOffset: 1, slotOffset: 0, deckUsedIds: used })?.id).toBe('b');
  });
});

describe('shuffleWithSeed (moved, behavior preserved)', () => {
  it('is deterministic and complete', () => {
    const arr = [1, 2, 3, 4, 5, 6];
    expect(shuffleWithSeed(arr, 99)).toEqual(shuffleWithSeed(arr, 99));
    expect([...shuffleWithSeed(arr, 99)].sort()).toEqual([1, 2, 3, 4, 5, 6]);
    // Does not mutate the input.
    expect(arr).toEqual([1, 2, 3, 4, 5, 6]);
  });
});
