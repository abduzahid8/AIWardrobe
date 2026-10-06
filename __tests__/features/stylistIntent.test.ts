import { parseStylistIntent } from '../../features/outfit-generator/utils/stylistIntent';

describe('parseStylistIntent', () => {
  it('reads occasion, style, and budget from a comment', () => {
    const intent = parseStylistIntent('Work tomorrow, old money, budget $200');
    expect(intent.style).toBe('old_money');
    expect(intent.occasion).toBe('work');
    expect(intent.budget).toEqual({ amount: 200, currency: 'USD' });
  });

  it('reads classic date night without a budget', () => {
    const intent = parseStylistIntent('date night, classic look');
    expect(intent.style).toBe('classic');
    expect(intent.occasion).toBe('date');
    expect(intent.budget).toBeNull();
  });

  it('reads Russian work + classic', () => {
    const intent = parseStylistIntent('На работу, классика, 150$');
    expect(intent.style).toBe('classic');
    expect(intent.occasion).toBe('work');
    expect(intent.budget?.amount).toBe(150);
  });

  it('returns empty intent for blank text', () => {
    expect(parseStylistIntent('')).toEqual({ style: null, occasion: null, budget: null });
  });

  it('reads weekend casual brunch without leftover chip tokens', () => {
    expect(parseStylistIntent('weekend casual brunch under $200')).toEqual({
      style: 'casual',
      occasion: 'casual',
      budget: { amount: 200, currency: 'USD' },
    });
  });

  it('would prefer work if chip ids were concatenated — generate must never do that', () => {
    const poisoned = parseStylistIntent('weekend casual brunch under $200 work old_money');
    expect(poisoned.occasion).toBe('work');
    expect(poisoned.style).toBe('old_money');
  });
});
