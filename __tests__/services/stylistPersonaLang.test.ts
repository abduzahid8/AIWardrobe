/**
 * stylistPersonaLang — personaReply speaks the UI language (en/ru/uz).
 * The offline fallback IS the stylist voice while provider keys are down,
 * so it must answer in the user's language, not always English.
 */
import { personaReply, resolvePersonaLang } from '../../src/services/ai/stylistPersona';

describe('resolvePersonaLang', () => {
  it.each([
    ['ru', 'ru'],
    ['ru-RU', 'ru'],
    ['uz', 'uz'],
    ['uz-UZ', 'uz'],
    ['en', 'en'],
    ['en-US', 'en'],
    ['de', 'en'],
    ['', 'en'],
    [undefined, 'en'],
  ])('maps %p to %p', (input, expected) => {
    expect(resolvePersonaLang(input as string)).toBe(expected);
  });
});

describe('personaReply localization', () => {
  it('answers Russian work questions in Russian', () => {
    const r = personaReply('Дайте мне рабочий образ', 'casual', 'ru');
    expect(r.lang).toBe('ru');
    expect(r.text).toMatch(/работ/i);
    expect(r.text).not.toMatch(/For work/);
    expect(r.suggestions.length).toBe(3);
    expect(r.suggestions[0]).toMatch(/Рабочий образ/);
  });

  it('answers Uzbek greetings in Uzbek', () => {
    const r = personaReply('Menga ish uchun obraz kerak', undefined, 'uz');
    expect(r.lang).toBe('uz');
    expect(r.text).toMatch(/Ish/);
  });

  it('keeps English default and English intents', () => {
    const r = personaReply('I need a work outfit');
    expect(r.lang).toBe('en');
    expect(r.text).toMatch(/For work/);
  });

  it('detects intents in Russian (not just English keywords)', () => {
    expect(personaReply('Что надеть на свадьбу?', undefined, 'ru').text).toMatch(/дресс-код/i);
    expect(personaReply('Холодно, что надеть?', undefined, 'ru').text).toMatch(/погоде/i);
    expect(personaReply('Какая обувь подойдёт?', undefined, 'ru').text).toMatch(/формальность/i);
  });

  it('still applies the style pillar tip in every language', () => {
    for (const lang of ['en', 'ru', 'uz'] as const) {
      const r = personaReply('hello', 'old_money', lang);
      expect(r.text).toMatch(/navy/);
    }
  });
});
