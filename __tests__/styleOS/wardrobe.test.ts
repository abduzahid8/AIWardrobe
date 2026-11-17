/**
 * wardrobe.test.ts — §43: health redundancy/gaps/fit uncertainty;
 * lifecycle/wear count/CPW; underused; gap graph; graph bounds; query
 * determinism and bounds.
 */
import {
  computeWardrobeHealth,
  costPerWear,
  defaultLifecycle,
  detectGaps,
  detectUnderused,
  redundancyMap,
  updateLifecycle,
  validatedCompanions,
} from '../../src/services/style/wardrobeHealth';
import {
  buildWardrobeGraph,
  neighborhood,
  sanitizedGraphStats,
} from '../../src/services/style/wardrobeGraph';
import {
  getBestItemsForContext,
  getHighConfidenceOutfits,
  getNextPurchaseCandidates,
  getNoBuyEvidence,
  getRedundantRoles,
  getSubstitutes,
  getTopWardrobeGaps,
  getUnderusedItems,
  getValidatedCompanions,
} from '../../src/services/style/wardrobeQueries';
import { recordWear } from '../../src/services/style/outfitMemory';
import { item, NOW, daysAgo } from './helpers';

test('health returns per-dimension grades, never one vanity score', () => {
  const items = [
    item({ id: 't1', category: 'top', subCategory: 'tee' }),
    item({ id: 'b1', category: 'bottom', subCategory: 'jeans' }),
    item({ id: 's1', category: 'shoes', subCategory: 'sneakers' }),
  ];
  const h = computeWardrobeHealth(items, [], { t1: 'high' }, NOW);
  expect(h.dimensions.length).toBeGreaterThan(5);
  expect(h).not.toHaveProperty('score');
  expect(h.dimensions.find((d) => d.key === 'coverage')?.grade).toBe('strong');
});

test('empty wardrobe is explicit, not zero-scored', () => {
  const h = computeWardrobeHealth([], [], {}, NOW);
  expect(h.itemCount).toBe(0);
  expect(h.summary.join(' ')).toMatch(/empty/i);
});

test('lifecycle wearCount only from confirmed wears; CPW never fabricates', () => {
  const lc = defaultLifecycle('t1', 'u1');
  const updated = updateLifecycle(lc, { acquisitionPrice: 100, currency: 'USD', status: 'active' }, 4);
  expect(updated.wearCount).toBe(4);
  const cpw = costPerWear('t1', updated, 4);
  expect(cpw.costPerWear).toBe(25);
  expect(cpw.estimated).toBe(false);
  const unknown = costPerWear('t2', { acquisitionPrice: null, currency: null }, 0);
  expect(unknown.costPerWear).toBeNull();
  expect(unknown.explanation).toMatch(/No acquisition price/);
  const est = costPerWear('t3', { acquisitionPrice: null, currency: null }, 2, { amount: 60, currency: 'USD' });
  expect(est.estimated).toBe(true);
  expect(est.explanation).toMatch(/ESTIMATED/);
});

test('underused skips archived/new/frequent; flags reasons and actions', () => {
  const items = [
    item({ id: 'old', createdAt: daysAgo(120), wearCount: 1 }),
    item({ id: 'new', createdAt: daysAgo(5), wearCount: 0 }),
    item({ id: 'loved', createdAt: daysAgo(120), wearCount: 40 }),
  ];
  const out = detectUnderused(items, {}, [], NOW);
  expect(out.map((o) => o.itemId)).toContain('old');
  expect(out.map((o) => o.itemId)).not.toContain('new');
  expect(out.map((o) => o.itemId)).not.toContain('loved');
  const archived = detectUnderused(items, { old: { ...defaultLifecycle('old', 'u1'), status: 'archived' } }, [], NOW);
  expect(archived.map((o) => o.itemId)).not.toContain('old');
});

test('redundancy groups by role with wear distribution; never prescribes selling', () => {
  const items = [
    item({ id: 'a', subCategory: 'tee', primaryColor: 'black', wearCount: 10 }),
    item({ id: 'b', subCategory: 'tee', primaryColor: 'black', wearCount: 0 }),
    item({ id: 'c', subCategory: 'jeans', primaryColor: 'navy', wearCount: 3 }),
  ];
  const map = redundancyMap(items);
  expect(map).toHaveLength(1);
  expect(map[0].itemIds).toHaveLength(2);
  expect(map[0].favoriteItemId).toBe('a');
  expect(map[0].note).toMatch(/Never|never|report/i);
});

test('gap graph names blocked contexts with urgency and confidence', () => {
  const gaps = detectGaps('u1', [item({ id: 't1' })], ['formal'], NOW);
  expect(gaps.length).toBeGreaterThan(0);
  expect(gaps[0].blockedContexts).toContain('formal');
  expect(['low', 'medium', 'high']).toContain(gaps[0].confidence);
});

test('graph build is bounded; neighborhood capped at depth 2', () => {
  const items = Array.from({ length: 30 }, (_, i) =>
    item({ id: `g${i}`, category: i % 3 === 0 ? 'top' : i % 3 === 1 ? 'bottom' : 'shoes' }));
  let mem = recordWear([], { userId: 'u1', outfitId: 'o1', itemIds: ['g0', 'g1'], wornDate: '2026-10-01', rating: 'loved' });
  const g = buildWardrobeGraph(items, mem);
  expect(g.edges.length).toBeLessThanOrEqual(2000);
  const nb = neighborhood(g, 'garment:g0');
  expect(nb.nodes.length).toBeLessThanOrEqual(60);
  const stats = sanitizedGraphStats(g);
  expect(JSON.stringify(stats)).not.toMatch(/g0|u1/);
});

test('validated companions rank loved co-wears first', () => {
  let mem = recordWear([], { userId: 'u1', outfitId: 'o1', itemIds: ['a', 'b'], wornDate: '2026-10-01', rating: 'loved' });
  mem = recordWear(mem, { userId: 'u1', outfitId: 'o2', itemIds: ['a', 'c'], wornDate: '2026-10-02' });
  const comps = validatedCompanions('a', mem);
  expect(comps[0].itemId).toBe('b');
});

test('query engine is deterministic and bounded', () => {
  const items = [
    item({ id: 't1', category: 'top', occasions: ['work'] }),
    item({ id: 't2', category: 'top', occasions: ['sport'] }),
    item({ id: 't3', category: 'top', occasions: ['work'] }),
  ];
  const ctx = { userId: 'u1', items, now: NOW };
  const a = getBestItemsForContext(ctx, 'work');
  const b = getBestItemsForContext(ctx, 'work');
  expect(a.map((i) => i.id)).toEqual(b.map((i) => i.id));
  expect(getBestItemsForContext(ctx, 'work', 1)).toHaveLength(1);
  expect(getUnderusedItems(ctx)).toBeDefined();
  expect(getRedundantRoles(ctx)).toBeDefined();
  expect(getTopWardrobeGaps(ctx)).toBeDefined();
  expect(getSubstitutes({ ...ctx, items: [item({ id: 'x', subCategory: 'tee' }), item({ id: 'y', subCategory: 'tee' })] }, 'x')).toHaveLength(1);
  expect(getValidatedCompanions({ ...ctx, memory: [] }, 't1')).toHaveLength(0);
  let mem = recordWear([], { userId: 'u1', outfitId: 'o1', itemIds: ['t1'], wornDate: '2026-10-01', rating: 'loved' });
  mem = recordWear(mem, { userId: 'u1', outfitId: 'o1', itemIds: ['t1'], wornDate: '2026-10-05', rating: 'good' });
  expect(getHighConfidenceOutfits({ ...ctx, memory: mem })).toHaveLength(1);
  const gaps = getTopWardrobeGaps(ctx);
  expect(getNextPurchaseCandidates(ctx, gaps)).toBeDefined();
  const nb = getNoBuyEvidence({ ...ctx, items: Array.from({ length: 5 }, (_, i) => item({ id: `s${i}`, category: 'shoes' })) }, 'shoes');
  expect(nb.recommendation).toBe('SKIP_DUPLICATE');
});
