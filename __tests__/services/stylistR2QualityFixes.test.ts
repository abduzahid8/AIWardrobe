import i18n from 'i18next';
import * as fs from 'fs';
import * as path from 'path';
import { wholeOutfitRegisterClash, outfitCoherence } from '../../src/domain/outfitCoherence';
import { describeGarment, competingUpperPatterns } from '../../src/domain/garmentDescriptor';
import { lookConcept, outfitConcepts, groundedConceptFallback } from '../../src/domain/outfitConcepts';
import { groundedLookExplanation } from '../../src/domain/lookExplanation';
import { stylingExplanationKeys } from '../../src/domain/stylingExplanation';
import { stylingContext } from '../../src/domain/stylingContext';
import {
  outfitConceptTitle,
  distinctStylingTips,
  resolveSelectionSource,
  selectionBadgeKey,
} from '../../src/services/stylingCopy';
import { parseBudgetControl, strictestOutfitBudget } from '../../src/domain/outfitBudgetSelection';

const item = (name: string, macroCategory: string, extra: Record<string, unknown> = {}) => ({
  id: `piece-${name}`,
  name,
  macroCategory,
  type: macroCategory,
  ...extra,
});

// ── Workstream 2: whole-outfit pattern and formality coherence ──────────────

test('Semi-Classic + Date Night rejects competing bold flannel/jacquard uppers', () => {
  const deck = [
    item('Burgundy black checked flannel shirt', 'top'),
    item('Brown orange geometric jacquard cardigan', 'outerwear'),
    item('Olive casual chinos', 'bottom'),
    item('Brown leather loafers', 'shoes'),
  ];
  const context = stylingContext({ style: 'semi_classic', occasion: 'Date night' });
  expect(context.dressCode).toBe('smart');
  // The jacquard knit is a visible pattern, not a quiet texture.
  expect(describeGarment(deck[1]).pattern).toBe('print');
  expect(competingUpperPatterns(describeGarment(deck[0]), describeGarment(deck[1]))).toBe(true);
  expect(wholeOutfitRegisterClash(deck, context))
    .toBe('restrained_style_rejects_competing_upper_patterns');
});

test('Minimalist + Work rejects loud statement checks', () => {
  const deck = [
    item('Bright oversized buffalo check shirt', 'top'),
    item('Grey large-scale geometric jacquard overshirt', 'outerwear'),
    item('Charcoal tailored trousers', 'bottom'),
    item('Black leather derby shoes', 'shoes'),
  ];
  const context = stylingContext({ style: 'minimalist', occasion: 'Work' });
  expect(describeGarment(deck[0]).patternProminence).toBe('prominent');
  expect(describeGarment(deck[1]).pattern).toBe('print');
  expect(wholeOutfitRegisterClash(deck, context))
    .toBe('restrained_style_rejects_competing_upper_patterns');
});

test('Old Money + Work keeps refined, restrained tailoring accepted', () => {
  const deck = [
    item('White oxford dress shirt', 'top'),
    item('Navy cashmere blazer', 'outerwear'),
    item('Charcoal pleated wool trousers', 'bottom'),
    item('Brown penny loafers', 'shoes'),
  ];
  const context = stylingContext({ style: 'old_money', occasion: 'Work' });
  expect(wholeOutfitRegisterClash(deck, context)).toBeNull();
});

test('Classic + Work keeps coherent shirt/blazer/trouser/leather-shoe ensembles', () => {
  const deck = [
    item('Light blue chambray shirt', 'top'),
    item('Navy suit jacket', 'outerwear'),
    item('Grey tailored trousers', 'bottom'),
    item('Dark brown leather derby shoes', 'shoes'),
  ];
  const context = stylingContext({ style: 'classic', occasion: 'Work' });
  expect(wholeOutfitRegisterClash(deck, context)).toBeNull();
  expect(outfitCoherence(deck, context).valid).toBe(true);
});

test('Semi-Classic + Date Night keeps polished smart-casual bridges accepted', () => {
  const deck = [
    item('Cream fine-knit polo', 'top'),
    item('Beige merino cardigan', 'outerwear'),
    item('Navy cotton chinos', 'bottom'),
    item('Brown suede loafers', 'shoes'),
  ];
  const context = stylingContext({ style: 'semi_classic', occasion: 'Date night' });
  expect(wholeOutfitRegisterClash(deck, context)).toBeNull();
});

test('Formal tailoring over performance bottoms is rejected', () => {
  const deck = [
    item('Light blue dress shirt', 'top'),
    item('Navy performance suit jacket', 'outerwear'),
    item('Black performance stretch chinos', 'bottom'),
    item('Dark brown leather derby shoes', 'shoes'),
  ];
  const context = stylingContext({ style: 'classic', occasion: 'Work' });
  expect(wholeOutfitRegisterClash(deck, context))
    .toBe('formal_tailoring_with_performance_bottom');
});

test('Polished date nights reject very casual layers', () => {
  const deck = [
    item('White cotton t-shirt', 'top'),
    item('Black tech zip hoodie', 'outerwear'),
    item('Dark indigo jeans', 'bottom'),
    item('White minimal sneakers', 'shoes'),
  ];
  const context = stylingContext({ style: 'semi_classic', occasion: 'Date night' });
  expect(wholeOutfitRegisterClash(deck, context))
    .toBe('date_night_rejects_very_casual_layer');
});

// ── Workstream 3: grounded, color-honest, distinct concept naming ────────────

const en = require('../../i18n/locales/en.json');
const ru = require('../../i18n/locales/ru.json');
const uz = require('../../i18n/locales/uz.json');

test('grounded fallback variants describe verified garments, never colors', () => {
  const context = stylingContext({ style: 'semi_classic', occasion: 'Date night' });
  const base = outfitConcepts(context)[0];
  const knit = groundedConceptFallback(base, [
    item('Cream cotton polo shirt', 'top'),
    item('Beige merino cardigan', 'outerwear'),
    item('Navy cotton chinos', 'bottom'),
    item('Brown suede loafers', 'shoes'),
  ]);
  expect(knit.id).toBe('grounded-knit-layers');
  const tailored = groundedConceptFallback(base, [
    item('White cotton dress shirt', 'top'),
    item('Navy wool blazer', 'outerwear'),
    item('Charcoal tailored trousers', 'bottom'),
    item('Black leather oxford shoes', 'shoes'),
  ]);
  expect(tailored.id).toBe('grounded-tailoring');
  const denim = groundedConceptFallback(base, [
    item('White cotton t-shirt', 'top'),
    item('Navy denim overshirt', 'outerwear'),
    item('Blue denim jeans', 'bottom'),
    item('White minimal sneakers', 'shoes'),
  ]);
  expect(denim.id).toBe('grounded-denim-ease');
  for (const concept of [knit, tailored, denim]) {
    expect(concept.title).not.toMatch(/white|black|brown|navy|beige|blue|grey|gray|cream/i);
  }
});

const COLOR_WORDS: Record<string, string[]> = {
  en: ['white', 'black', 'red', 'blue', 'navy', 'green', 'brown', 'beige', 'cream', 'grey', 'gray',
    'yellow', 'pink', 'purple', 'orange', 'burgundy', 'olive', 'khaki', 'ivory', 'charcoal',
    'tan', 'camel', 'maroon', 'mustard', 'rust', 'stone', 'sand', 'ecru', 'lilac', 'sage', 'mint'],
  ru: ['белый', 'белая', 'белое', 'белые', 'черный', 'чёрный', 'красный', 'синий', 'зеленый', 'зелёный',
    'коричневый', 'бежевый', 'серый', 'желтый', 'жёлтый', 'розовый', 'фиолетовый', 'оранжевый',
    'бордовый', 'голубой', 'хаки', 'кремовый', 'графит', 'оливковый', 'горчичный', 'молочный',
    'песочный', 'изумрудный'],
  uz: ['oq', 'qora', 'qizil', 'yashil', 'jigarrang', 'bej', 'kulrang', 'sariq', 'pushti',
    'binafsha', 'qaymoqrang', "ko'k"],
};
const tokens = (text: string) =>
  text.toLowerCase().replace(/[’‘ʼ']/g, "'").split(/[^a-z'а-яёўқғҳ]+/u).filter(Boolean);

test.each(['en', 'ru', 'uz'] as const)('no %s concept title invents a garment color', (lng) => {
  const locales = { en, ru, uz };
  const concepts = (locales[lng].outfitMaker.concepts ?? {}) as Record<string, { title?: string }>;
  expect(Object.keys(concepts).length).toBeGreaterThan(10);
  for (const [id, concept] of Object.entries(concepts)) {
    expect(typeof concept.title).toBe('string');
    const hits = tokens(concept.title || '').filter((t) => COLOR_WORDS[lng].includes(t));
    expect({ id, hits }).toEqual({ id, hits: [] });
  }
});

test('three genuinely different looks receive distinct localized titles', async () => {
  await i18n.init({ lng: 'uz', resources: { uz: { translation: uz } } });
  const context = stylingContext({ style: 'semi_classic', occasion: 'Date night' });
  const decks = [
    [
      item('Cream fine-knit polo shirt', 'top'),
      item('Beige merino cardigan', 'outerwear'),
      item('Navy cotton chinos', 'bottom'),
      item('Brown suede loafers', 'shoes'),
    ],
    [
      item('White cotton dress shirt', 'top'),
      item('Navy wool blazer', 'outerwear'),
      item('Charcoal tailored trousers', 'bottom'),
      item('Black leather derby shoes', 'shoes'),
    ],
    [
      item('White cotton t-shirt', 'top'),
      item('Navy cotton overshirt', 'outerwear'),
      item('Blue denim jeans', 'bottom'),
      item('Brown chelsea boots', 'shoes'),
    ],
  ];
  const titles = decks.map((deck) => {
    const { concept } = lookConcept(deck, context);
    const title = outfitConceptTitle(concept.id, concept.title);
    expect(title).not.toContain('outfitMaker.');
    return title;
  });
  expect(new Set(titles).size).toBe(3);
});

// ── Workstream 4: useful, non-repetitive explanations ────────────────────────

test('explanations name the actual garments instead of reciting templates', () => {
  const deck = [
    item('White cotton shirt', 'top'),
    item('Beige linen blazer', 'outerwear'),
    item('Navy tailored cotton trousers', 'bottom'),
    item('Brown leather dress loafers', 'shoes'),
  ];
  const context = stylingContext({ style: 'classic', occasion: 'Work' });
  const text = groundedLookExplanation(deck, context);
  expect(text).toContain('Brown leather dress loafers');
  expect(text).toContain('Navy tailored cotton trousers');
  expect(text).not.toContain('balance the tailored trousers');
  expect(text.length).toBeLessThan(300);
});

test('explanation keys are grounded in verified evidence and localized everywhere', async () => {
  const light = [
    item('White lightweight cotton shirt', 'top'),
    item('Beige open-knit cardigan', 'outerwear'),
    item('Navy cotton chinos', 'bottom'),
    item('Brown suede loafers', 'shoes'),
  ];
  const tailored = [
    item('White cotton dress shirt', 'top'),
    item('Navy wool blazer', 'outerwear'),
    item('Charcoal tailored trousers', 'bottom'),
    item('Black leather oxford shoes', 'shoes'),
  ];
  const context = stylingContext({ style: 'classic', occasion: 'Work' });
  expect(stylingExplanationKeys(light, context))
    .toContain('outfitMaker.qualityExplanation.lightBase');
  expect(stylingExplanationKeys(tailored, context))
    .toContain('outfitMaker.qualityExplanation.tailoredFootwear');
  for (const lng of ['en', 'ru', 'uz'] as const) {
    await i18n.init({ lng, resources: { [lng]: { translation: { en, ru, uz }[lng] } } });
    for (const deck of [light, tailored]) {
      for (const key of stylingExplanationKeys(deck, context)) {
        const rendered = i18n.t(key);
        expect(rendered).not.toBe(key);
        expect(rendered).not.toContain('{{');
      }
    }
  }
});

test('a tip that restates the description is never rendered twice', () => {
  const description = 'Brown loafers echo the line of navy chinos.';
  expect(distinctStylingTips(description, [description, ` ${description.toUpperCase()} `, 'Fit tip'])).toEqual(['Fit tip']);
});

// ── Workstream 5: complete localization of the Stylist flow ─────────────────

test('every outfitMaker key used by the Stylist screen exists in en, ru and uz', () => {
  const source = fs.readFileSync(
    path.resolve(__dirname, '../../screens/AIOutfitmaker.tsx'), 'utf8');
  const used = new Set<string>();
  for (const match of source.matchAll(/t\('outfitMaker\.([A-Za-z0-9_]+)'/g)) used.add(match[1]);
  expect(used.size).toBeGreaterThan(50);
  const missing: string[] = [];
  for (const key of used) {
    for (const [lng, locale] of Object.entries({ en, ru, uz })) {
      if ((locale.outfitMaker as Record<string, unknown>)[key] === undefined)
        missing.push(`${key} missing in ${lng}`);
    }
  }
  expect(missing).toEqual([]);
});

test.each([
  ['newOutfit', 'Yangi obraz'],
  ['generateOutfit', 'Obraz yaratish'],
  ['weatherUnavailable', 'Ob-havo mavjud emas'],
  ['fromWardrobe', 'Shkafdan'],
  ['newClothing', 'Yangi kiyimlar'],
] as const)('uz Stylist label %s renders Uzbek, not English', async (key, expected) => {
  await i18n.init({ lng: 'uz', resources: { uz: { translation: uz } } });
  expect(i18n.t(`outfitMaker.${key}`)).toBe(expected);
});

test.each([
  ['style_semi_classic', 'Yarim klassika'],
  ['dateNight', 'Romantik uchrashuv'],
  ['concepts.grounded-knit-layers.title', 'Yumshoq trikotaj qatlamlar'],
] as const)('uz style/occasion/concept label %s is translated', async (key, expected) => {
  await i18n.init({ lng: 'uz', resources: { uz: { translation: uz } } });
  expect(i18n.t(`outfitMaker.${key}`)).toBe(expected);
});

test('prompt placeholders never carry the old example budget', () => {
  for (const [lng, locale] of Object.entries({ en, ru, uz })) {
    const placeholder = String(locale.outfitMaker.promptPlaceholder || '');
    expect(placeholder).not.toContain('1 500 000');
    expect(placeholder).not.toContain('UZS');
  }
});

// ── Workstream 6: truthful AI-ranked labels ──────────────────────────────────

test.each([
  ['ai', 'ai', 'outfitMaker.aiRanked'],
  ['hybrid', 'hybrid', 'outfitMaker.styledLook'],
  ['deterministic_fallback', 'deterministic_fallback', 'outfitMaker.styledLook'],
  [undefined, 'deterministic_fallback', 'outfitMaker.styledLook'],
  ['legacy-value', 'deterministic_fallback', 'outfitMaker.styledLook'],
] as const)('source %s resolves to %s with badge %s', (source, expected, badge) => {
  expect(resolveSelectionSource(source)).toBe(expected);
  expect(selectionBadgeKey(source)).toBe(badge);
});

// ── Regression scenarios: budget contract ───────────────────────────────────

test('an absent budget imposes no hidden limit; an explicit budget is honored', () => {
  expect(parseBudgetControl('', 'UZS')).toBeNull();
  expect(parseBudgetControl('   ', 'UZS')).toBeNull();
  expect(parseBudgetControl('1 500 000', 'UZS')).toEqual({ amount: 1500000, currency: 'UZS' });
  const convert = (budget: { amount: number; currency: string }) => budget;
  expect(strictestOutfitBudget(null, null, convert)).toBeNull();
  expect(strictestOutfitBudget({ amount: 1500000, currency: 'UZS' }, null, convert))
    .toEqual({ amount: 1500000, currency: 'UZS' });
});
