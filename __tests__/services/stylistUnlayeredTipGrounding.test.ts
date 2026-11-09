import { stylingExplanationKeys } from '../../src/domain/stylingExplanation';
import { stylingContext } from '../../src/domain/stylingContext';
import en from '../../i18n/locales/en.json';
import ru from '../../i18n/locales/ru.json';
import uz from '../../i18n/locales/uz.json';

test('three-piece unlayered looks never advise fitting an absent outer layer', () => {
  const items = [
    { id: 'top', macroCategory: 'top', name: 'Plain cotton tee', type: 't-shirt', color: 'white' },
    { id: 'bottom', macroCategory: 'bottom', name: 'Straight cotton trousers', type: 'trousers', color: 'black' },
    { id: 'shoe', macroCategory: 'shoes', name: 'Casual canvas sneakers', type: 'sneakers', color: 'white' },
  ] as any[];
  const context = stylingContext({ style: 'casual', occasion: 'casual' });
  const keys = stylingExplanationKeys(items, context);
  expect(keys).not.toContain('outfitMaker.qualityExplanation.fitCheck');
  expect(keys).toContain('outfitMaker.qualityExplanation.unlayeredFitCheck');
});

test('the grounded no-layer fallback is localized in every supported language', () => {
  for (const locale of [en, ru, uz]) {
    const tip = locale.outfitMaker.qualityExplanation.unlayeredFitCheck;
    expect(typeof tip).toBe('string');
    expect(tip.length).toBeGreaterThan(15);
    expect(tip).not.toBe(locale.outfitMaker.qualityExplanation.fitCheck);
  }
});
