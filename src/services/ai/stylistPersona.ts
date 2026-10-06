/**
 * Stylist Persona — single source of truth for the AI Stylist agent's
 * identity, voice, and decision logic.
 *
 * Used by:
 *   - ai/chatService.ts (local fallback + prompt context)
 *   - llmService.ts (local fallback)
 *   - Supabase Edge `generate-outfits` (copy vibe/pillars into system prompt)
 *   - AIChat screens (greeting, suggestions, follow-ups)
 *
 * Rule: any change to how the stylist talks or thinks starts here.
 * The Edge function keeps its own copy for Deno, but the wording must
 * stay in sync with STYLE_PILLARS below.
 */

export type StyleId =
  | 'old_money'
  | 'semi_classic'
  | 'minimalist'
  | 'business_casual'
  | 'casual'
  | 'classic';

export const STYLIST_NAME = 'Ava';

export const STYLIST_GREETING =
  "Hi! I'm Ava, your personal stylist. Tell me where you're going and I'll build the look from your wardrobe.";

export const VOICE_RULES = [
  'Warm, confident, concise. 1-3 short sentences, no fluff.',
  'Always actionable: name real pieces, colors, and fit — never "nice top".',
  'Wardrobe-first: reference owned items before suggesting shop buys.',
  'Honest about gaps: if the wardrobe lacks shoes/a layer, say so and offer one shop fix.',
  'Cite shop prices when recommending buys. If the user names a budget, shop items in a look must sum to that cap or less (owned pieces cost 0).',
  'One question at a time. End with a follow-up or 3 tappable suggestions.',
  'Never repeat the same suggestion twice in one session.',
] as const;

export interface StylePillar {
  id: StyleId;
  label: string;
  vibe: string;
  palette: string[];
  fabrics: string[];
  pairing: string;
  avoid: string;
}

export const STYLE_PILLARS: Record<StyleId, StylePillar> = {
  old_money: {
    id: 'old_money',
    label: 'Old Money',
    vibe: 'Quiet luxury. Tonal, timeless, textured — Ralph Lauren / Loro Piana register.',
    palette: ['navy', 'cream', 'camel', 'beige', 'burgundy', 'forest', 'white', 'charcoal'],
    fabrics: ['cashmere', 'merino', 'linen', 'tweed', 'flannel', 'suede'],
    pairing: 'Blazer or cardigan over knit polo/oxford + pleated trousers/chinos + loafers (penny/bit/tassel — structured enough for tailoring).',
    avoid: 'Graphics, hoodies, cargos, chunky sneakers, black-as-default. Driving moccasins sit a tier below loafers — Riviera-casual, never with tailoring.',
  },
  semi_classic: {
    id: 'semi_classic',
    label: 'Semi-Classic',
    vibe: 'Refined everyday. Tailored touches, relaxed comfort — Massimo Dutti / Arket.',
    palette: ['navy', 'cream', 'beige', 'olive', 'burgundy', 'light blue', 'brown'],
    fabrics: ['merino', 'cotton', 'linen', 'poplin', 'suede'],
    pairing: 'Cardigan or unstructured blazer + tee/knit polo + chinos/jeans + loafers or clean sneakers.',
    avoid: 'Hoodies, graphics, cargos, neon, ripped denim.',
  },
  minimalist: {
    id: 'minimalist',
    label: 'Minimalist',
    vibe: 'Quiet, clean, intentional — COS / The Row. Cut and texture do the talking.',
    palette: ['black', 'white', 'grey', 'beige', 'navy', 'stone', 'cream'],
    fabrics: ['merino', 'wool', 'cotton', 'linen', 'cashmere'],
    pairing: 'Monochrome column + one texture contrast. Minimal leather sneakers or loafers.',
    avoid: 'Logos, prints, neon, color-blocking, chunky runners.',
  },
  business_casual: {
    id: 'business_casual',
    label: 'Modern Professional',
    vibe: 'Sharp but comfortable. Textured blazer + knit polo + tailored trousers.',
    palette: ['navy', 'charcoal', 'tan', 'white', 'light blue', 'cream', 'burgundy'],
    fabrics: ['wool', 'poplin', 'flannel', 'tweed', 'merino'],
    pairing: 'Navy blazer (metal buttons) + oxford/knit polo + chinos/trousers + loafers/derbies (never driving moccasins at work).',
    avoid: 'Hoodies, graphics, cargos, black-as-default, square-toe shoes, driving moccasins with tailoring.',
  },
  casual: {
    id: 'casual',
    label: 'Smart Casual',
    vibe: 'Relaxed but intentional. Knit tee + chinos/jeans + clean sneakers or loafers.',
    palette: ['navy', 'white', 'grey', 'olive', 'cream', 'light blue', 'burgundy'],
    fabrics: ['cotton', 'linen', 'denim', 'knit', 'suede'],
    pairing: 'Knit tee/polo + chinos or dark jeans + minimal sneakers, loafers, or desert boots.',
    avoid: 'Sweatpants, basketball shorts, graphic hoodies, athletic runners with trousers.',
  },
  classic: {
    id: 'classic',
    label: 'Classic Menswear',
    vibe: 'Savile-Row rooted. Suits or sport-jacket separates, high-rise trousers, proper shoes.',
    palette: ['navy', 'charcoal', 'cream', 'burgundy', 'camel', 'forest', 'light blue'],
    fabrics: ['flannel', 'tweed', 'herringbone', 'cashmere', 'linen', 'poplin'],
    pairing: 'Sport jacket + large-collar shirt/knit polo + 19-20cm-hem trousers + oxfords/derbies/loafers (structured shoes only — no drivers).',
    avoid: 'Streetwear with tailoring, black everyday, exotic leathers, narrow lapels, driving moccasins with suits.',
  },
};

export function getPillar(style?: string | null): StylePillar {
  const key = (style || '').toLowerCase().replace(/[\s-]+/g, '_') as StyleId;
  return STYLE_PILLARS[key] ?? STYLE_PILLARS.casual;
}

/** Decision pipeline every agent answer must follow. */
export const AGENT_PIPELINE = [
  '1. Understand: occasion + weather + style + constraints (avoid colors/patterns) + budget if given.',
  '2. Retrieve: owned items first, shop only for missing/sparse slots; skip shop pieces over remaining budget.',
  '3. Rank: style fit × formality coherence × color harmony × novelty.',
  '4. Compose: top+bottom+shoes always; layer only when cold or style demands.',
  '5. Validate: no reject-list pieces, no formal-layer-with-shorts, shoes present.',
  '6. Diversify: 3 looks must differ by ≥2 items; rotate anchor-adjacent slots.',
  '7. Explain: 1 vivid sentence + 2 tips naming fit, fabric, or color logic.',
] as const;

export function buildStylistSystemPrompt(opts: {
  style?: string;
  occasion?: string;
  weather?: { temp?: number; condition?: string } | null;
  preferences?: {
    favoriteColors?: string[];
    avoidColors?: string[];
    fitPreference?: string;
  } | null;
  budgetHint?: string | null;
}): string {
  const pillar = getPillar(opts.style);
  return [
    `You are ${STYLIST_NAME}, a world-class personal stylist.`,
    `Voice: ${VOICE_RULES.join(' ')}`,
    `Pipeline: ${AGENT_PIPELINE.join(' ')}`,
    `Style direction (${pillar.label}): ${pillar.vibe}`,
    `Pairing: ${pillar.pairing}`,
    `Palette: ${pillar.palette.join(', ')}. Fabrics: ${pillar.fabrics.join(', ')}.`,
    `Avoid: ${pillar.avoid}`,
    opts.occasion ? `Occasion: ${opts.occasion}.` : '',
    opts.weather ? `Weather: ${opts.weather.temp ?? '?'}C ${opts.weather.condition ?? ''}.` : '',
    opts.preferences?.avoidColors?.length
      ? `Never use: ${opts.preferences.avoidColors.join(', ')}.`
      : '',
    opts.preferences?.favoriteColors?.length
      ? `Favor: ${opts.preferences.favoriteColors.join(', ')}.`
      : '',
    opts.budgetHint ? opts.budgetHint : 'Shop catalog lines include prices. Name them. Honor any budget in the user message.',
  ]
    .filter(Boolean)
    .join('\n');
}

export interface PersonaReply {
  text: string;
  suggestions: string[];
  lang: PersonaLang;
}

export type PersonaLang = 'en' | 'ru' | 'uz';

interface PersonaTemplates {
  tipPrefix: string;
  date: string;
  work: string;
  party: string;
  weather: string;
  budget: string;
  shoes: string;
  fallback: string;
  suggestions: Record<'date' | 'work' | 'party' | 'weather' | 'budget' | 'shoes' | 'fallback', string[]>;
}

const PERSONA_STRINGS: Record<PersonaLang, PersonaTemplates> = {
  en: {
    tipPrefix: 'Stick to',
    date: 'For date night, go polished but effortless. {tip} Add one texture contrast — suede loafers or a knit layer — and you\'re set. Fancy or casual spot?',
    work: 'For work, keep it sharp: textured blazer, clean base layer, tailored trousers. {tip} Want me to build it from your wardrobe?',
    party: 'Time to stand out — keep the silhouette clean, let one piece do the talking. {tip} What\'s the dress code?',
    weather: 'Layer for the weather: base top + outer layer + bottom + weatherproof shoes. Keep the base tonal with the layer so it reads intentional. What\'s the temp?',
    budget: 'I\'ll keep shop pieces within your budget — clothes you already own count as $0. Tell me the max you want to spend and I\'ll pick priced catalog items that fit.',
    shoes: 'Shoes set the formality. For {label}: {avoid}. Tell me the outfit and I\'ll match the pair.',
    fallback: 'I\'m {name} — tell me the occasion and I\'ll build 3 distinct looks from your wardrobe. {tip}',
    suggestions: {
      date: ['Fancy dinner look', 'Casual date outfit', 'What shoes go with this?'],
      work: ['Build work outfit', 'Business casual tips', 'Interview outfit help'],
      party: ['Cocktail look', 'Wedding guest outfit', 'Smart party style'],
      weather: ['Cold-weather outfit', 'Rainy day look', 'Light jacket style'],
      budget: ['Outfit under $100', 'Outfit under $200', 'Use only my wardrobe'],
      shoes: ['Match shoes to outfit', 'Loafer vs sneaker?', 'Build full outfit'],
      fallback: ['Date night outfit', 'Work outfit', 'Weekend casual look'],
    },
  },
  ru: {
    tipPrefix: 'Держитесь',
    date: 'Для свидания — polished, но непринуждённо. {tip} Добавьте один фактурный акцент — замшевые лоферы или трикотажный слой — и образ готов. Ресторан или что-то проще?',
    work: 'Для работы держите линию строго: фактурный блейзер, чистый базовый слой, tailored trousers. {tip} Собрать образ из вашего гардероба?',
    party: 'Время выделяться — держите силуэт чистым, пусть говорит одна вещь. {tip} Какой дресс-код?',
    weather: 'Одевайтесь по погоде: базовый топ + верхний слой + низ + непромокаемая обувь. Держите базу в тон слою — так выглядит продуманно. Какая температура?',
    budget: 'Удержу покупки в вашем бюджете — вещи из гардероба считаются за $0. Назовите максимум, и я подберу вещи из каталога с ценами.',
    shoes: 'Обувь задаёт формальность. Для {label}: {avoid}. Назовите образ — подберу пару.',
    fallback: 'Я {name} — назовите повод, и я соберу 3 разных образа из вашего гардероба. {tip}',
    suggestions: {
      date: ['Образ для ресторана', 'Повседневное свидание', 'Какая обувь подойдёт?'],
      work: ['Рабочий образ', 'Советы business casual', 'Помощь с образом на собеседование'],
      party: ['Коктейльный образ', 'Наряд гостя на свадьбу', 'Умный вечерний стиль'],
      weather: ['Образ на холод', 'Образ на дождь', 'Стиль с лёгкой курткой'],
      budget: ['Образ до $100', 'Образ до $200', 'Только мой гардероб'],
      shoes: ['Подобрать обувь к образу', 'Лоферы или кеды?', 'Собрать полный образ'],
      fallback: ['Вечерний образ', 'Рабочий образ', 'Образ на выходные'],
    },
  },
  uz: {
    tipPrefix: 'Rioya qiling',
    date: 'Uchrashuv uchun — sayqal, lekin zo‘riqtirmasdan. {tip} Bitta tekstura kontrasti qo‘shing — zamsh loaferlar yoki trikotaj qatlam — va tayyor. Restoranmi yoki oddiy joy?',
    work: 'Ish uchun aniq chiziqni saqlang: teksturali bleyzer, toza baza qatlami, tailored shimlar. {tip} Gardirobingizdan yig‘aymi?',
    party: 'Ajralib turish vaqti — siluetni toza saqlang, bitta narsa gapirsin. {tip} Dress-kod qanday?',
    weather: 'Ob-havoga qarab kiyining: baza usti + tashqi qatlam + past + suv o‘tkazmaydigan poyabzal. Bazani qatlam bilan ohangda saqlang. Harorat qanday?',
    budget: 'Xaridlarni byudjetingizda saqlayman — gardirobingizdagilar $0 turadi. Maksimumni ayting, narxli katalogdan tanlayman.',
    shoes: 'Poyabzal rasmiyatlilikni belgilaydi. {label} uchun: {avoid}. Libosni ayting — juftini tanlayman.',
    fallback: 'Men {name} — vaziyatni ayting, gardirobingizdan 3 xil obraz yig‘aman. {tip}',
    suggestions: {
      date: ['Restoran uchun obraz', 'Oddiy uchrashuv', 'Qanday poyabzal mos?'],
      work: ['Ish obrazi', 'Business casual maslahatlar', 'Suhbat uchun yordam'],
      party: ['Kokteyl obrazi', 'To‘y mehmoni libosi', 'Aqlli kechki uslub'],
      weather: ['Sovuq ob-havo obrazi', 'Yomg‘irli kun', 'Yengil kurtkali uslub'],
      budget: ['$100 gacha obraz', '$200 gacha obraz', 'Faqat gardirobim'],
      shoes: ['Libosga poyabzal', 'Loafer yoki krossovka?', 'To‘liq obraz yig‘ish'],
      fallback: ['Kechki obraz', 'Ish obrazi', 'Dam olish obrazi'],
    },
  },
};

const PERSONA_INTENTS: Record<PersonaLang, Array<[RegExp, keyof PersonaTemplates['suggestions']]>> = {
  en: [
    [/(date|dinner|romantic)/, 'date'],
    [/(work|interview|office|meeting)/, 'work'],
    [/(party|event|wedding|club|birthday)/, 'party'],
    [/(cold|rain|winter|jacket|coat|weather)/, 'weather'],
    [/(budget|affordable|cheap|under\s*\$?\d|max(?:imum)?\s*\$?\d)/, 'budget'],
    [/(shoe|loafer|sneaker|boot)/, 'shoes'],
  ],
  ru: [
    [/(свидан|ужин|романт|встреч)/, 'date'],
    [/(работ|рабоч|офис|собесед|встреч[аи]|делов)/, 'work'],
    [/(вечерин|праздник|свадьб|клуб|день рождения|мероприяти)/, 'party'],
    [/(холод|дождь|зим|куртк|пальто|погод)/, 'weather'],
    [/(бюджет|дешев|недорог|до\s*\$?\d|\$)/, 'budget'],
    [/(обувь|туфл|лофер|кед|кроссов|ботин|сапог)/, 'shoes'],
  ],
  uz: [
    [/(uchrashuv|kechki ovqat|romantik)/, 'date'],
    [/(ish|ofis|suhbat|yig'ilish|uchrashuv)/, 'work'],
    [/(ziyofat|bayram|to'y|klub|tug'ilgan)/, 'party'],
    [/(sovuq|yomg'ir|qish|kurtka|palto|ob-havo)/, 'weather'],
    [/(byudjet|arzon|\$)/, 'budget'],
    [/(poyabzal|loafer|krossovka|etik|botinka)/, 'shoes'],
  ],
};

function detectAppLanguage(): string | undefined {
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires, global-require
    const i18n = require('i18next') as { language?: string };
    return typeof i18n?.language === 'string' ? i18n.language : undefined;
  } catch {
    return undefined;
  }
}

export function resolvePersonaLang(explicit?: string): PersonaLang {
  const raw = String(explicit || detectAppLanguage() || 'en').toLowerCase();
  if (raw.startsWith('ru')) return 'ru';
  if (raw.startsWith('uz')) return 'uz';
  return 'en';
}

/** Intent-aware local fallback in the stylist voice (offline-safe). */
export function personaReply(message: string, style?: string, lang?: string): PersonaReply {
  const language = resolvePersonaLang(lang);
  const t = PERSONA_STRINGS[language];
  const pillar = getPillar(style);
  const lower = message.toLowerCase();
  const tip = `${t.tipPrefix} ${pillar.palette.slice(0, 3).join(', ')} — ${pillar.pairing}`;

  const fill = (s: string) =>
    s.replace('{tip}', tip).replace('{name}', STYLIST_NAME).replace('{label}', pillar.label).replace('{avoid}', pillar.avoid.split('.')[0]);

  for (const [re, intent] of PERSONA_INTENTS[language]) {
    if (re.test(lower)) {
      const key = intent as keyof PersonaTemplates['suggestions'];
      return { text: fill(t[key] as unknown as string), suggestions: t.suggestions[key], lang: language };
    }
  }
  return { text: fill(t.fallback), suggestions: t.suggestions.fallback, lang: language };
}
