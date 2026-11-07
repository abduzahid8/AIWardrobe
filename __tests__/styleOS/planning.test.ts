/**
 * planning.test.ts — §43: capsule reuse/budget/climate; travel luggage/
 * laundry; events/locking (no fabricated laundry); seasonal; goals.
 */
import {
  buildCapsule,
  createWardrobeGoal,
  createWardrobeEvent,
  goalProgress,
  lockOutfit,
  planPacking,
  recommendOutfitsForEvent,
  seasonalRotation,
} from '../../src/services/style/planning';
import { item, NOW } from './helpers';

const pool = [
  item({ id: 't1', category: 'top', subCategory: 'tee' }),
  item({ id: 't2', category: 'top', subCategory: 'shirt' }),
  item({ id: 'b1', category: 'bottom', subCategory: 'jeans' }),
  item({ id: 'b2', category: 'bottom', subCategory: 'chinos' }),
  item({ id: 's1', category: 'shoes', subCategory: 'sneakers' }),
  item({ id: 's2', category: 'shoes', subCategory: 'loafers' }),
  item({ id: 'o1', category: 'outerwear', subCategory: 'jacket' }),
];

test('university capsule reuses owned first and respects size', () => {
  const cap = buildCapsule({ kind: 'university_15', userId: 'u1', maxItems: 7 }, pool);
  expect(cap.itemIds.length).toBeLessThanOrEqual(7);
  expect(cap.reusedOwnedIds.length).toBe(cap.itemIds.length);
  expect(cap.outfitCount).toBeGreaterThan(0);
});

test('summer capsule filters climate; budget recorded without shopping', () => {
  const winter = item({ id: 'w1', category: 'outerwear', subCategory: 'parka', seasons: ['winter'] });
  const cap = buildCapsule(
    { kind: 'summer_budget', userId: 'u1', climate: { tempC: 32 }, budget: { amount: 200, currency: 'USD' } },
    [...pool, winter],
  );
  expect(cap.itemIds).not.toContain('w1');
  expect(cap.evidence.join(' ')).toMatch(/200 USD/);
});

test('travel packing honors luggage and laundry; names missing footwear', () => {
  const noShoes = pool.filter((i) => i.category !== 'shoes');
  const res = planPacking('u1', { days: 7, activities: ['office'], laundryAvailable: false, luggageLimit: 5 }, noShoes);
  expect(res.packingList.length).toBeLessThanOrEqual(5);
  expect(res.missingItems.join(' ')).toMatch(/shoes/);
  expect(res.outfitSchedule).toHaveLength(7);
  expect(res.evidence.join(' ')).toMatch(/No laundry/);
  const withLaundry = planPacking('u1', { days: 7, activities: ['office'], laundryAvailable: true }, pool);
  expect(withLaundry.repeatStrategy).toMatch(/Laundry/);
});

test('event flow: recommend, lock with unknown laundry, never fabricate', () => {
  const ev = createWardrobeEvent({ userId: 'u1', title: 'Dinner', startsAt: '2026-12-01T19:00:00Z', contextId: 'formal' });
  const recs = recommendOutfitsForEvent(ev, [
    { id: 'o1', itemIds: ['t1'], occasion: 'formal', wornCount: 5 },
    { id: 'o2', itemIds: ['t2'], occasion: 'gym', wornCount: 9 },
  ]);
  expect(recs[0].outfitId).toBe('o1');
  const { event, locked } = lockOutfit(ev, 'o1', ['t1'], {});
  expect(event.locked).toBe(true);
  expect(locked.readiness).toBe('needs_attention');
  expect(locked.availability.t1).toBe('unknown');
  expect(locked.evidence.join(' ')).toMatch(/never|reported|unknown/i);
  const blocked = lockOutfit(ev, 'o1', ['t1'], { t1: 'missing' });
  expect(blocked.locked.readiness).toBe('blocked');
});

test('seasonal rotation suggests storage; never auto-archives', () => {
  const items = [
    item({ id: 'tee', seasons: ['summer'] }),
    item({ id: 'coat', seasons: ['winter'] }),
  ];
  const rot = seasonalRotation(items, 6); // July
  expect(rot.outOfSeasonIds).toContain('coat');
  expect(rot.activeIds).toContain('tee');
  // resurface = items for next season
  expect(rot.evidence.join(' ')).toMatch(/manual|Current season/);
});

test('goals carry progress metrics per type', () => {
  const g = createWardrobeGoal({ userId: 'u1', type: 'reduce_size', priority: 'high' });
  expect(g.status).toBe('active');
  expect(goalProgress(g, { itemCount: 42, wornShare: 0.5, formalCount: 2, duplicateRoles: 3 }).itemCount).toBe(42);
  void NOW;
});
