/**
 * memory.test.ts — §43: feedback one-tap/dismiss/cooldown; planned vs
 * worn; repetition verdicts; proven combinations.
 */
import {
  createOutfitFeedback,
  provenCombinations,
  recentlyWornItemIds,
  recordWear,
  repetitionVerdict,
  shouldPromptForFeedback,
  FEEDBACK_COOLDOWN_DAYS,
} from '../../src/services/style/outfitMemory';
import { NOW, daysAgo } from './helpers';

test('feedback prompt respects cooldown; dismissal is cooldown-shaped', () => {
  const fb = [createOutfitFeedback({ userId: 'u1', outfitId: 'o1', rating: 'good', createdAt: daysAgo(1) })];
  expect(shouldPromptForFeedback('o1', fb, NOW)).toBe(false);
  const old = [createOutfitFeedback({ userId: 'u1', outfitId: 'o1', rating: 'good', createdAt: daysAgo(FEEDBACK_COOLDOWN_DAYS + 1) })];
  expect(shouldPromptForFeedback('o1', old, NOW)).toBe(true);
  expect(shouldPromptForFeedback('o2', fb, NOW)).toBe(true);
});

test('recordWear increments repeats on same combination; planned flag kept', () => {
  let mem = recordWear([], {
    userId: 'u1', outfitId: 'o1', itemIds: ['a', 'b'],
    wornDate: '2026-10-01', plannedDate: '2026-09-30',
  });
  expect(mem[0].planned).toBe(true);
  expect(mem[0].confirmed).toBe(true);
  expect(mem[0].repeatCount).toBe(1);
  mem = recordWear(mem, { userId: 'u1', outfitId: 'o1', itemIds: ['b', 'a'], wornDate: '2026-10-10' });
  expect(mem[1].repeatCount).toBe(2);
  expect(mem[1].combinationKey).toBe(mem[0].combinationKey);
});

test('repetition is not automatically bad: loved repeats endorsed', () => {
  let mem = recordWear([], { userId: 'u1', outfitId: 'o1', itemIds: ['a'], wornDate: '2026-10-01', rating: 'loved' });
  mem = recordWear(mem, { userId: 'u1', outfitId: 'o1', itemIds: ['a'], wornDate: daysAgo(40).slice(0, 10), rating: 'loved' });
  const v = repetitionVerdict('o1', mem, { varietySeeking: 'low' }, NOW);
  expect(v?.verdict).toBe('proven_repeat');
  expect(v?.timesWorn).toBe(2);
});

test('frequent recent repeats nudge variety only for variety-seekers', () => {
  let mem: ReturnType<typeof recordWear> = [];
  for (const d of [2, 5, 9]) {
    mem = recordWear(mem, { userId: 'u1', outfitId: 'o1', itemIds: ['a'], wornDate: daysAgo(d).slice(0, 10) });
  }
  expect(repetitionVerdict('o1', mem, { varietySeeking: 'high' }, NOW)?.verdict).toBe('consider_variety');
  expect(repetitionVerdict('o1', mem, { varietySeeking: 'low' }, NOW)?.verdict).not.toBe('consider_variety');
});

test('unrated combinations never surface as proven; unconfirmed never count', () => {
  let mem = recordWear([], { userId: 'u1', outfitId: 'o1', itemIds: ['a'], wornDate: '2026-10-01' });
  mem = recordWear(mem, { userId: 'u1', outfitId: 'o1', itemIds: ['a'], wornDate: '2026-10-05' });
  expect(provenCombinations(mem)).toHaveLength(0);
  let loved = recordWear([], { userId: 'u1', outfitId: 'o2', itemIds: ['b'], wornDate: '2026-10-01', rating: 'loved' });
  loved = recordWear(loved, { userId: 'u1', outfitId: 'o2', itemIds: ['b'], wornDate: '2026-10-06', rating: 'good' });
  expect(provenCombinations(loved)).toHaveLength(1);
  expect(recentlyWornItemIds(loved, 300, NOW)).toContain('b');
});
