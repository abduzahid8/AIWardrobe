import {reconcileStylistBriefWithChips} from '../../features/outfit-generator/utils/stylistBriefSelection';

describe('Stylist brief follows explicit chips', () => {
  test('changing Classic / Work to Old Money / Date removes only stale tags', () => {
    expect(reconcileStylistBriefWithChips('work, classic', 'old_money','date')).toBe('');
    expect(reconcileStylistBriefWithChips(
      'work, classic, max 300 USD, avoid neon', 'old_money','date',
    )).toBe('max 300 USD, avoid neon');
  });

  test('current matching tags and freeform constraints survive intact', () => {
    expect(reconcileStylistBriefWithChips('work, classic','classic','work')).toBe('work, classic');
    expect(reconcileStylistBriefWithChips('work dinner with clients, no bright colors','classic','date'))
      .toBe('work dinner with clients, no bright colors');
    expect(reconcileStylistBriefWithChips('beige, wool, tailored','old_money','date'))
      .toBe('beige, wool, tailored');
  });

  test('partial chip changes do not erase unrelated preference or selected occasion', () => {
    expect(reconcileStylistBriefWithChips('work, classic, no red','old_money','work'))
      .toBe('work, no red');
    expect(reconcileStylistBriefWithChips('work, classic, no red','classic','date'))
      .toBe('classic, no red');
    expect(reconcileStylistBriefWithChips('date night; minimal; no graphics','minimalist','date'))
      .toBe('date night; minimal; no graphics');
  });

  test('missing selections, empty fields and repeated chip changes are stable', () => {
    expect(reconcileStylistBriefWithChips('', 'classic','work')).toBe('');
    expect(reconcileStylistBriefWithChips('work, classic',null,null)).toBe('work, classic');
    const once=reconcileStylistBriefWithChips('work, classic, no red','old_money','date');
    expect(reconcileStylistBriefWithChips(once,'old_money','date')).toBe(once);
  });
});
