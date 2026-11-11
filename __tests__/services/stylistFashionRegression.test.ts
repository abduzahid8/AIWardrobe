import { describeGarment, competingUpperPatterns } from '../../src/domain/garmentDescriptor';
import { rejectForStyleOccasion, passesStyleJudge, assessOutfitQuality } from '../../src/domain/occasionRules';
import { lookConcept, outfitConcepts } from '../../src/domain/outfitConcepts';
import { wholeOutfitRegisterClash } from '../../src/domain/outfitCoherence';
import { groundedLookExplanation } from '../../src/domain/lookExplanation';
import { stylingContext } from '../../src/domain/stylingContext';
import en from '../../i18n/locales/en.json';

const item = (name: string, macroCategory: string, extra: Record<string, unknown> = {}) => ({
  id: `piece-${name}`,
  name,
  macroCategory,
  type: macroCategory,
  ...extra,
});

describe('stylist fashion regression: user screenshot scenarios', () => {
  test('Scenario A: bright buffalo check is a prominent motif, not a quiet knit', () => {
    const d = describeGarment(
      item('Bright red-and-black buffalo-check flannel shirt', 'top')
    );
    expect(d.pattern).toBe('check');
    expect(d.patternProminence).toBe('prominent');
  });

  test('Scenario A: minimalist work vetoes loud checked shirts', () => {
    const reason = rejectForStyleOccasion(
      item('Bright red-and-black buffalo-check flannel shirt', 'top'),
      'minimalist',
      'Work'
    );
    expect(reason).not.toBeNull();
  });

  test('Scenario A: minimalist still admits restrained Glen checks', () => {
    const reason = rejectForStyleOccasion(
      item('Grey Glen check oxford shirt', 'top'),
      'minimalist',
      'Work'
    );
    expect(reason).toBeNull();
  });

  test('Scenario B: same-family checks compete unless one side is subtle', () => {
    const shirt = describeGarment(item('Blue checked oxford shirt', 'top'));
    const trousers = describeGarment(
      item('Navy checked tailored trousers', 'bottom')
    );
    expect(competingUpperPatterns(shirt, trousers)).toBe(true);
    const subtleShirt = describeGarment(
      item('Subtle micro-check dress shirt', 'top')
    );
    expect(subtleShirt.patternProminence).toBe('subtle');
    expect(competingUpperPatterns(subtleShirt, trousers)).toBe(false);
  });

  test('Scenario A/C: knit-titled concepts require knit evidence', () => {
    const deck = [
      item('Bright red-and-black buffalo-check flannel shirt', 'top'),
      item('Camel cotton chinos', 'bottom'),
      item('Dark brown leather loafers', 'shoes'),
    ];
    const context = stylingContext({ style: 'minimalist', occasion: 'Work' });
    const { concept } = lookConcept(deck, context);
    expect(concept.title).not.toMatch(/knit/i);
  });

  test('Scenario C: hooded casual layer over a dress shirt clashes for old money date', () => {
    const deck = [
      item('White striped dress shirt', 'top'),
      item('Grey hooded zip knit sweater', 'outerwear'),
      item('Navy tailored wool trousers', 'bottom'),
      item('Black leather formal loafers', 'shoes'),
    ];
    const context = stylingContext({
      style: 'old_money',
      occasion: 'Date night',
    });
    expect(wholeOutfitRegisterClash(deck, context)).not.toBeNull();
  });

  test('Scenario C guard: fine knit cardigan over a dress shirt still passes', () => {
    const deck = [
      item('White dress shirt', 'top'),
      item('Navy fine-gauge knit cardigan', 'outerwear'),
      item('Navy tailored wool trousers', 'bottom'),
      item('Black leather derby shoes', 'shoes'),
    ];
    const context = stylingContext({
      style: 'old_money',
      occasion: 'Date night',
    });
    expect(wholeOutfitRegisterClash(deck, context)).toBeNull();
  });

  test('Scenario E: every emitted palette family has an editorial label', () => {
    for (const family of [
      'dark',
      'light',
      'mid',
      'earth',
      'warm',
      'cool-dark',
      'cool-mid',
    ]) {
      expect(
        (en.outfitMaker.editorialColors as Record<string, string>)[family]
      ).toBeTruthy();
    }
  });

  test('Concept fallback: zero fulfillable concepts yields grounded neutral, never a knit title', () => {
    const deck = [
      item('Black technical overshirt', 'top'),
      item('Charcoal technical stretch trousers', 'bottom'),
      item('Grey technical low-profile sneakers', 'shoes'),
    ];
    const context = stylingContext({ style: 'minimalist', occasion: 'Work' });
    const before = outfitConcepts(context);
    expect(before.length).toBeGreaterThan(0);
    const { concept } = lookConcept(deck, context);
    expect(concept.id).toBe('grounded-separates');
    expect(concept.title).toBe('Clean separates');
    expect(concept.title).not.toMatch(/knit/i);
    expect(concept.textures).toEqual([]);
    // Composition (slots, layer role) is preserved from this context.
    expect(concept.slots).toEqual(before[0].slots);
    expect(concept.layerRole).toBe(before[0].layerRole);
    // Shared concept objects are untouched: fallback is a fresh object.
    expect(before.includes(concept)).toBe(false);
    expect(before.some((c) => c.id === 'grounded-separates')).toBe(false);
    expect(before.every((c) => c.textures.length > 0)).toBe(true);
  });

  test('Scenario F: explanations stay concise and never recite templates', () => {
    const deck = [
      item('White cotton shirt', 'top'),
      item('Beige linen blazer', 'outerwear'),
      item('Navy tailored cotton trousers', 'bottom'),
      item('Brown leather dress loafers', 'shoes'),
    ];
    const context = stylingContext({ style: 'casual', occasion: 'Everyday' });
    const text = groundedLookExplanation(deck, context);
    expect(text).not.toMatch(/palette links the known colors/i);
    expect(text).not.toMatch(/defines the silhouette/i);
    expect(text).not.toMatch(/continued by the dress profile/i);
    expect(text.length).toBeLessThan(300);
  });
});

describe('workout occasions judge the quality floor as sport (edge selectionStyle parity)', () => {
  const athleticLook = [
    { name: 'Black quick-dry training t-shirt', macroCategory: 'top' },
    { name: 'Black athletic training shorts', macroCategory: 'bottom' },
    { name: 'White running sneakers', macroCategory: 'shoes' },
  ];
  it('a refined aesthetic plus a workout occasion passes the floor as sport', () => {
    expect(passesStyleJudge(athleticLook, 'old_money', 'gym workout')).toBe(true);
    // The gate performs the remap; the raw-style floor call is unchanged.
    expect(assessOutfitQuality(athleticLook, 'sport', 'gym workout').acceptance.passed).toBe(true);
  });
  it('the same aesthetic outside a workout occasion still keeps its floor discipline', () => {
    expect(passesStyleJudge([
      { name: 'Black quick-dry training t-shirt', macroCategory: 'top' },
      { name: 'Black athletic training shorts', macroCategory: 'bottom' },
      { name: 'White running sneakers', macroCategory: 'shoes' },
    ], 'old_money', 'casual')).toBe(false);
  });
});
