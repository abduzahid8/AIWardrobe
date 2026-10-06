/**
 * aiDecks.e2e.test.ts — executable spec of the deck-uniqueness rule.
 *
 * Regression this guards: "3 outfits, 2 of them the same". The app used to
 * clone an existing card under a fresh id (`dupfill_*`) and to re-add
 * near-dupes when topping up to 3. Those paths are gone; this suite pins the
 * invariant against the REAL shared implementation:
 *
 *   features/outfit-generator/utils/deckDedupe.ts  (also used by AIOutfitmaker)
 *
 * plus a deck-assembly simulation over REAL catalog ids that mirrors the
 * app's assemble gate (only non-dupes are ever added).
 */

import {
  createDeckDedupe,
  deckSignature,
  countSharedIds,
  prioritizeUnseen,
  NEAR_DUPE_THRESHOLD,
} from '../../features/outfit-generator/utils/deckDedupe';
import { SHOP_CATALOG_ITEMS } from '../../data/shopCatalogItems';

const ids = (...xs: string[]) => xs;

describe('deckDedupe — real implementation', () => {
  it('exact duplicates are rejected regardless of order', () => {
    const d = createDeckDedupe();
    d.track(ids('a', 'b', 'c', 'd'));
    expect(d.hasExact(ids('d', 'c', 'b', 'a'))).toBe(true);
    expect(d.isDupe(ids('a', 'b', 'c', 'd'))).toBe(true);
    expect(d.size).toBe(1);
  });

  it(`near-duplicates sharing >= ${NEAR_DUPE_THRESHOLD} ids are rejected`, () => {
    const d = createDeckDedupe();
    d.track(ids('top1', 'bottom1', 'shoes1', 'layer1'));
    // Same shoes + bottom, different top+layer → still the same trousers+shoes.
    expect(d.isDupe(ids('top2', 'bottom1', 'shoes1', 'layer2'))).toBe(true);
  });

  it('outfits that share a single item are diverse enough to keep', () => {
    const d = createDeckDedupe();
    d.track(ids('top1', 'bottom1', 'shoes1', 'layer1'));
    expect(d.isDupe(ids('top2', 'bottom2', 'shoes1', 'layer2'))).toBe(false);
  });

  it('locked anchor is excluded from the shared count (wardrobe mode)', () => {
    const d = createDeckDedupe();
    const skip = new Set(['anchor-uuid']);
    d.track(['anchor-uuid', 'b1', 's1', 'l1']);
    // Shares two non-anchor pieces (b1, s1) → clone, even with skip.
    expect(d.isDupe(['anchor-uuid', 'b1', 's1', 'l2'], skip)).toBe(true);
    // Shares only the anchor + 1 other → allowed.
    expect(d.isDupe(['anchor-uuid', 'b1', 's2', 'l2'], skip)).toBe(false);
    // Without the skip set, the anchor counts like any other item.
    expect(countSharedIds(['anchor-uuid', 'b1', 's1', 'l2'], new Set(['anchor-uuid', 'b1', 's1', 'l1']))).toBe(3);
  });

  it('signature is order-invariant and string-coerced', () => {
    expect(deckSignature(['b', 'a', 3])).toBe(deckSignature([3, 'a', 'b']));
  });

  it('replaying the old dupfill attack yields 1 card, never clones', () => {
    // The deleted code path pushed {...sameItems, id: dupfill_*} until the
    // deck hit 3. The ONLY correct behavior: the gate refuses all clones.
    const d = createDeckDedupe();
    const victim = ids('t', 'b', 's', 'l');
    d.track(victim);
    const deck = [victim];
    for (let k = 0; k < 5; k++) {
      if (!d.isDupe(victim)) {
        d.track(victim);
        deck.push(victim);
      }
    }
    expect(deck).toHaveLength(1);
  });
});

describe('deck assembly e2e — real catalog ids, app assembly gate', () => {
  // Mirrors AIOutfitmaker: iterate candidates, keep iff !isDupe, track kept.
  function assemble(candidates: string[][]): string[][] {
    const d = createDeckDedupe();
    const kept: string[][] = [];
    for (const c of candidates) {
      if (kept.length >= 3) break;
      if (d.isDupe(c)) continue;
      d.track(c);
      kept.push(c);
    }
    return kept;
  }

  const realIds = SHOP_CATALOG_ITEMS.slice(0, 24).map((i) => String(i.id));

  it('assembled deck has pairwise <2 shared items', () => {
    const candidates: string[][] = [];
    for (let k = 0; k < 12; k++) {
      candidates.push([realIds[k], realIds[k + 1], realIds[k + 2], realIds[k + 3]]);
    }
    const deck = assemble(candidates);
    expect(deck.length).toBeGreaterThanOrEqual(2);
    for (let a = 0; a < deck.length; a++) {
      for (let b = a + 1; b < deck.length; b++) {
        const shared = countSharedIds(deck[a], new Set(deck[b]));
        expect(shared).toBeLessThan(NEAR_DUPE_THRESHOLD);
      }
    }
  });

  it('edge returning 3x the same outfit assembles to exactly 1 card', () => {
    const same = [realIds[0], realIds[1], realIds[2], realIds[3]];
    expect(assemble([same, same, same])).toHaveLength(1);
  });

  it('tiny pool (1 bottom + 1 shoes) cannot fake a 3-card deck', () => {
    // Documents WHY the UI may show fewer than 3 cards instead of clones:
    // with one shared bottom+shoes pair every candidate shares ≥3 with the
    // first kept card once tops repeat.
    const tops = realIds.slice(0, 2);
    const candidates = [
      [tops[0], realIds[10], realIds[11], realIds[12]],
      [tops[1], realIds[10], realIds[11], realIds[12]],
      [tops[0], realIds[10], realIds[11], realIds[12]],
    ];
    const deck = assemble(candidates);
    expect(deck.length).toBeLessThanOrEqual(2);
    const sigs = deck.map(deckSignature);
    expect(new Set(sigs).size).toBe(sigs.length);
  });
});

describe('prioritizeUnseen — cross-tap recency (same pants every tap)', () => {
  const pool = ['pants-A', 'pants-B', 'shirt-A', 'shirt-B', 'shoes-A'].map((id) => ({ id }));

  it('unseen items sort first, order stable inside groups', () => {
    const out = prioritizeUnseen(pool, new Set(['pants-A', 'shoes-A']), (i) => i.id).map((i) => i.id);
    expect(out).toEqual(['pants-B', 'shirt-A', 'shirt-B', 'pants-A', 'shoes-A']);
  });

  it('empty memory returns the pool untouched', () => {
    expect(prioritizeUnseen(pool, new Set(), (i) => i.id)).toBe(pool);
  });

  it('all-seen pool keeps working (deprioritize, never starve)', () => {
    const out = prioritizeUnseen(pool, new Set(pool.map((i) => i.id)), (i) => i.id);
    expect(out).toHaveLength(pool.length);
  });

  it('a top-N trim after prioritizing drops the recently shown', () => {
    // Mirrors the edge slice(0, 40): the trim budget goes to fresh items.
    const out = prioritizeUnseen(pool, new Set(['pants-A']), (i) => i.id)
      .slice(0, 3)
      .map((i) => i.id);
    expect(out).not.toContain('pants-A');
  });

  it('treats an item as seen when any of its keys match', () => {
    const pool = [
      { id: 'shoe-17', name: 'penny loafer' },
      { id: 'shoe-21', name: 'oxford' },
    ];
    const out = prioritizeUnseen(pool, new Set(['__n:penny loafer']), (i) => [
      i.id,
      `__n:${i.name}`,
    ]);
    expect(out.map((i) => i.id)).toEqual(['shoe-21', 'shoe-17']);
  });
});
