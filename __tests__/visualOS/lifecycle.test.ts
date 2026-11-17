/**
 * lifecycle.test.ts — condition intelligence, repair/replace/keep,
 * outfit board, try-on orchestration.
 */
import { conditionFromHints, repairReplaceKeep } from '../../src/services/vision/condition';
import { createBoard, boardAdd, boardRemove, boardSwap, boardLock, boardSnapshot, createTryOnRequest, tryOnUnavailable, planMultiItemTryOn, syntheticFitDisclaimer } from '../../src/services/vision/outfitBoard';

test('condition: no invented damage; user confirmation wins', () => {
  const none = conditionFromHints(['good lighting'], false);
  expect(none.condition).toBe('unknown');
  const hints = conditionFromHints(['visible pilling on elbows'], false);
  expect(hints.condition).toBe('worn');
  expect(hints.evidence.join(' ')).toMatch(/user review required/i);
  const tear = conditionFromHints(['small tear at seam'], false);
  expect(tear.condition).toBe('needs_attention');
  const user = conditionFromHints(['tear'], true, 'repair');
  expect(user.condition).toBe('repair');
});

test('repair decision: age alone never disposes; insufficient evidence explicit', () => {
  expect(repairReplaceKeep({ itemId: 'i', condition: 'unknown', wearCount: 0 }).decision).toBe('INSUFFICIENT_EVIDENCE');
  expect(repairReplaceKeep({ itemId: 'i', condition: 'good', wearCount: 120, satisfaction: 5 }).decision).toBe('KEEP');
  expect(repairReplaceKeep({ itemId: 'i', condition: 'repair', wearCount: 40 }).decision).toBe('REPAIR');
  expect(repairReplaceKeep({ itemId: 'i', condition: 'replace_candidate', wearCount: 200, replacementAvailable: true }).decision).toBe('REPLACE_WHEN_NEEDED');
  // old but good + loved => KEEP, never auto-dispose
  expect(repairReplaceKeep({ itemId: 'i', condition: 'good', wearCount: 500, satisfaction: 5 }).decision).toBe('KEEP');
  expect(repairReplaceKeep({ itemId: 'i', condition: 'good', wearCount: 10, satisfaction: 1, duplicateCount: 3 }).decision).toBe('SELL');
});

test('board: add/remove/swap/lock/version; locks respected', () => {
  let b = createBoard('u', ['a', 'b']);
  b = boardAdd(b, 'c');
  expect(b.itemIds).toEqual(['a', 'b', 'c']);
  b = boardLock(b, 'a');
  b = boardRemove(b, 'a');
  expect(b.itemIds).toContain('a'); // locked stays
  b = boardSwap(b, 'b', 'd');
  expect(b.itemIds).toEqual(['a', 'd', 'c']);
  expect(b.alternativesRequestedFor).toBe('d');
  b = boardSwap(b, 'a', 'z'); // locked source ignored
  expect(b.itemIds).toContain('a');
  b = boardSnapshot(b, 'v2');
  expect(b.versions).toHaveLength(2);
});

test('try-on: synthetic by default; multi-item degrades gracefully', () => {
  const req = createTryOnRequest({ personAssetId: 'p', garmentVariantId: 'v1', garmentImageAssetId: 'g', category: 'top' });
  expect(req.preservationConstraints).toContain('preserve face');
  const un = tryOnUnavailable(req, 'Provider offline.');
  expect(un.status).toBe('unsupported');
  expect(un.synthetic).toBe(true);
  expect(syntheticFitDisclaimer(un)).toMatch(/never feeds FitEvidence/i);
  const r2 = createTryOnRequest({ personAssetId: 'p', garmentVariantId: 'v2', garmentImageAssetId: 'g2', category: 'shoes' });
  const plan = planMultiItemTryOn([req, r2]);
  // default provider: max 1 item, no shoes
  expect(plan.planned).toHaveLength(1);
  expect(plan.skipped.length).toBeGreaterThanOrEqual(1);
  expect(plan.skipped.join(' ')).not.toMatch(/fake/i);
});
