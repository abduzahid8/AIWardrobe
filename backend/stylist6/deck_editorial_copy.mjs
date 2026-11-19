/**
 * Honest, compact styling copy for Stylist 6 New Clothing.
 * One deck can legitimately contain three looks mapped to the same native
 * concept ("Refined Tailoring"). A repeated concept is not a repeated outfit,
 * but showing identical headlines obscures what the stylist changed.
 *
 * Only request context, already-approved selected items and measured
 * direction differences are used. No images, location, body shape, skin
 * color, stock, comfort or personal taste are inferred.
 *
 * Localization (v4): the request's display language (context.language, one of
 * 'en' | 'ru' | 'uz') selects an equivalent template table with the SAME
 * uniqueness and grounding rules per language — garment, color, texture and
 * pattern vocabulary is localized, not transliterated. When any interpolated
 * fact lacks a localized term the WHOLE output string falls back to English:
 * a card string never mixes languages.
 */
import { describeGarment } from '../../src/domain/garmentDescriptor.ts';
import { lookConcept } from '../../src/domain/outfitConcepts.ts';
import { normalizeStylistLanguage } from '../../src/domain/stylingContext.ts';
import { currentVisual, confidentVisual } from '../../src/domain/visualFashion.ts';

export const DECK_COPY_VERSION = 's6-grounded-deck-copy-v5';
export const STYLIST_NOTES_VERSION = 's6-editorial-stylist-notes-v1';
const slotOrder = ['top','outerwear','bottom','shoes'];
const known = x => typeof x === 'string' && !!x &&
  !['unknown','neutral','missing'].includes(x.toLowerCase());
const titleCase = x => String(x || '').replace(/[-_]+/g,' ')
  .replace(/\b\w/g,letter => letter.toUpperCase());
const cap = s => s ? s.charAt(0).toUpperCase() + s.slice(1) : s;

// ── English role vocabulary (unchanged en behavior) ─────────────────────
const enRole = d => {
  if (!d) return '';
  const names = {
    'top': { 'fine-knit':'knitwear','t-shirt':'tee','tank-top':'tank top','shirt':'shirt','dress-shirt':'dress shirt',
      'polo':'polo','knit-polo':'knit polo','sweater':'sweater' },
    'outerwear': { 'blazer':'blazer','cardigan':'cardigan','coat':'coat','jacket':'jacket',
      'overshirt':'overshirt','bomber':'bomber' },
    'bottom': { 'chino':'chinos','jeans':'jeans','tailored-trouser':'tailored trousers',
      'trouser':'trousers','shorts':'shorts','athletic-trouser':'joggers',
      'knit-trouser':'knit trousers' },
    'shoes': { 'loafer':'loafers','derby':'derbies','oxford':'oxfords',
      'chelsea':'Chelsea boots','minimal-sneaker':'minimal sneakers',
      'sneaker':'sneakers','sport-sneaker':'training sneakers','dress-boot':'dress boots' },
  };
  return names[d.slot]?.[d.subtype] || '';
};

function details(items) {
  // Copy only: a stale cached catalog taxonomy must not call a visibly
  // named technical T-SHIRT "knitwear". No admission, SKU or vision edits.
  const descriptors=(items||[]).map(item=>{
    const d=describeGarment(item);
    if(d.slot!=='top'||typeof item?.name!=='string')return d;
    if(/\b(?:tank(?:[ -]?top)?|sleeveless[ -]?(?:top|shirt|vest))\b/i.test(item.name))
      return {...d,subtype:'tank-top'};
    if(/\bt[ -]?shirts?\b/i.test(item.name) && d.subtype!=='t-shirt')
      return {...d,subtype:'t-shirt'};
    return d;
  });
  return Object.fromEntries(slotOrder.map(slot => [slot,
    descriptors.find(d => d.slot === slot) || null]));
}

// ── Localized template tables ───────────────────────────────────────────
// Role entries carry the inflected forms Russian templates need:
//   nom — nominative ("Пиджак вместо рубашки"), gen — genitive ("цвет пиджака"),
//   acc — accusative ("добавляет пиджак"/"добавляет футболку"),
//   g   — gender used for color adjective agreement (m/f/n/p plural).

const RU = {
  roles: {
    'top': {
      'fine-knit': {nom:'трикотаж',gen:'трикотажа',acc:'трикотаж',g:'m'},
      't-shirt': {nom:'футболка',gen:'футболки',acc:'футболку',g:'f'},
      'tank-top': {nom:'майка',gen:'майки',acc:'майку',g:'f'},
      'shirt': {nom:'рубашка',gen:'рубашки',acc:'рубашку',g:'f'},
      'dress-shirt': {nom:'рубашка',gen:'рубашки',acc:'рубашку',g:'f'},
      'polo': {nom:'поло',gen:'поло',acc:'поло',g:'n'},
      'knit-polo': {nom:'трикотажное поло',gen:'трикотажного поло',acc:'трикотажное поло',g:'n'},
      'sweater': {nom:'свитер',gen:'свитера',acc:'свитер',g:'m'},
    },
    'outerwear': {
      'blazer': {nom:'пиджак',gen:'пиджака',acc:'пиджак',g:'m'},
      'cardigan': {nom:'кардиган',gen:'кардигана',acc:'кардиган',g:'m'},
      'coat': {nom:'пальто',gen:'пальто',acc:'пальто',g:'n'},
      'jacket': {nom:'куртка',gen:'куртки',acc:'куртку',g:'f'},
      'overshirt': {nom:'рубашка-куртка',gen:'рубашки-куртки',acc:'рубашку-куртку',g:'f'},
      'bomber': {nom:'бомбер',gen:'бомбера',acc:'бомбер',g:'m'},
    },
    'bottom': {
      'chino': {nom:'брюки чинос',gen:'брюк чинос',acc:'брюки чинос',g:'p'},
      'jeans': {nom:'джинсы',gen:'джинсов',acc:'джинсы',g:'p'},
      'tailored-trouser': {nom:'классические брюки',gen:'классических брюк',acc:'классические брюки',g:'p'},
      'trouser': {nom:'брюки',gen:'брюк',acc:'брюки',g:'p'},
      'athletic-trouser': {nom:'джоггеры',gen:'джоггеров',acc:'джоггеры',g:'p'},
      'knit-trouser': {nom:'трикотажные брюки',gen:'трикотажных брюк',acc:'трикотажные брюки',g:'p'},
      'shorts': {nom:'шорты',gen:'шорт',acc:'шорты',g:'p'},
    },
    'shoes': {
      'loafer': {nom:'лоферы',gen:'лоферов',acc:'лоферы',g:'p'},
      'derby': {nom:'дерби',gen:'дерби',acc:'дерби',g:'p'},
      'oxford': {nom:'оксфорды',gen:'оксфордов',acc:'оксфорды',g:'p'},
      'chelsea': {nom:'ботинки-челси',gen:'ботинок-челси',acc:'ботинки-челси',g:'p'},
      'minimal-sneaker': {nom:'минималистичные кроссовки',gen:'минималистичных кроссовок',acc:'минималистичные кроссовки',g:'p'},
      'sneaker': {nom:'кроссовки',gen:'кроссовок',acc:'кроссовки',g:'p'},
      'sport-sneaker': {nom:'спортивные кроссовки',gen:'спортивных кроссовок',acc:'спортивные кроссовки',g:'p'},
      'dress-boot': {nom:'классические ботинки',gen:'классических ботинок',acc:'классические ботинки',g:'p'},
    },
  },
  // Color adjective forms agreeing with the governed noun's gender.
  colors: {
    'white': {m:'белый',f:'белая',n:'белое',p:'белые'},
    'black': {m:'чёрный',f:'чёрная',n:'чёрное',p:'чёрные'},
    'navy': {m:'тёмно-синий',f:'тёмно-синяя',n:'тёмно-синее',p:'тёмно-синие'},
    'cream': {m:'кремовый',f:'кремовая',n:'кремовое',p:'кремовые'},
    'grey': {m:'серый',f:'серая',n:'серое',p:'серые'},
    'blue': {m:'синий',f:'синяя',n:'синее',p:'синие'},
    'brown': {m:'коричневый',f:'коричневая',n:'коричневое',p:'коричневые'},
    'beige': {m:'бежевый',f:'бежевая',n:'бежевое',p:'бежевые'},
    'charcoal': {m:'угольный',f:'угольная',n:'угольное',p:'угольные'},
    'green': {m:'зелёный',f:'зелёная',n:'зелёное',p:'зелёные'},
    'olive': {m:'оливковый',f:'оливковая',n:'оливковое',p:'оливковые'},
    'burgundy': {m:'бордовый',f:'бордовая',n:'бордовое',p:'бордовые'},
    'red': {m:'красный',f:'красная',n:'красное',p:'красные'},
    'pink': {m:'розовый',f:'розовая',n:'розовое',p:'розовые'},
    'purple': {m:'фиолетовый',f:'фиолетовая',n:'фиолетовое',p:'фиолетовые'},
    'yellow': {m:'жёлтый',f:'жёлтая',n:'жёлтое',p:'жёлтые'},
    'orange': {m:'оранжевый',f:'оранжевая',n:'оранжевое',p:'оранжевые'},
    'dark': {m:'тёмный',f:'тёмная',n:'тёмное',p:'тёмные'},
    'dark-neutral': {m:'тёмный нейтральный',f:'тёмная нейтральная',n:'тёмное нейтральное',p:'тёмные нейтральные'},
    'light': {m:'светлый',f:'светлая',n:'светлое',p:'светлые'},
    'earth': {m:'землистый',f:'землистая',n:'землистое',p:'землистые'},
    'warm': {m:'тёплый',f:'тёплая',n:'тёплое',p:'тёплые'},
    'mid': {m:'приглушённый',f:'приглушённая',n:'приглушённое',p:'приглушённые'},
    'cool-dark': {m:'глубокий тёмный',f:'глубокая тёмная',n:'глубокое тёмное',p:'глубокие тёмные'},
    'cool-mid': {m:'холодный',f:'холодная',n:'холодное',p:'холодные'},
  },
  // Predicate forms agreeing with «фактура» (feminine).
  textures: {
    'smooth':'гладкая','knit':'вязаная','textured':'фактурная',
    'suede':'замшевая','denim':'джинсовая','technical':'техническая',
  },
  // Predicate forms agreeing with «узор» (masculine).
  patterns: {
    'solid':'однотонный','stripe':'в полоску','check':'в клетку',
    'texture':'рельефный','print':'с принтом',
  },
  // Predicate forms agreeing with «силуэт» (masculine).
  volumes: {'slim':'зауженный','regular':'прямой','wide':'свободный'},
  styles: {
    'old_money':'олд мани','semi_classic':'полуклассика','minimalist':'минимализм',
    'casual':'кэжуал','classic':'классика','business_casual':'деловой кэжуал',
  },
  // Genitive forms: templates say «для …».
  occasions: {
    'work':'работы','casual':'каждого дня','date':'свидания',
    'formal':'формального выхода','sport':'спорта','travel':'поездки',
  },
  concepts: {
    'Refined Ivy':'Отточенный айви',
    'Textured tailoring':'Фактурный костюм',
    'Smart knit layering':'Трикотажная многослойность',
    'Precise office tailoring':'Точный офисный костюм',
    'Classic separates':'Классический комплект',
    'Tonal evening':'Тональный вечер',
    'Soft evening texture':'Мягкая вечерняя фактура',
    'Modern evening separates':'Современный вечерний комплект',
    'Easy transit':'Лёгкая дорога',
    'Soft utility':'Мягкая практичность',
    'Relaxed knit layers':'Расслабленный трикотаж',
    'Active layers':'Активные слои',
    'Formal tailoring':'Строгий костюм',
    'Clean lines':'Чистые линии',
    'Quiet knit texture':'Спокойная трикотажная фактура',
    'Tonal modern separates':'Тональный современный комплект',
    'Ivy Knit':'Айви-трикотаж',
    'Soft Tailoring':'Мягкий костюм',
    'Relaxed Layers':'Расслабленные слои',
    'Tailored':'Костюмный силуэт',
    'Soft Knit Layers':'Мягкие трикотажные слои',
    'Refined Tailoring':'Отточенный костюм',
    'Relaxed Denim Ease':'Расслабленный деним',
    'Clean separates':'Чистые комплекты',
    'Clean transit':'Простая дорога',
    'Soft separates':'Мягкие комплекты',
    'Textured separates':'Фактурные комплекты',
  },
  separatesFallback: 'Чистые комплекты',
  optionSuffix: n => ` · Вариант ${n}`,
  prefixes: {
    'signature': (style, occasion) => `Ближе всего к стилю «${style}» для ${occasion}.`,
    'refresh': (style, occasion) => `Иное прочтение стиля «${style}» для ${occasion}.`,
    'discovery': (_style, occasion) => `Новое стилевое направление, которое по-прежнему подходит для ${occasion}.`,
    'alternative': (_style, occasion) => `Ещё один совместимый вариант для ${occasion}, а не отдельное направление-открытие.`,
  },
  relationship: {
    dressFootwear: () => 'Классическая обувь поддерживает линию классических брюк.',
    cardigan: () => 'Кардиган образует открытый слой поверх базы.',
    blazerStructured: () => 'Пиджак и классические брюки выдерживают единую линию костюма.',
    // Same structural fact the English path states for a look with an outer
    // layer plus a base top — still real selected-garment structure.
    generic: () => 'Верхний слой и база образуют верхнюю часть образа.',
    palette: (c1, r1, c2, r2) => `Основу палитры задают ${c1} ${r1} и ${c2} ${r2}.`,
  },
  focus: {
    adds: role => `Этот вариант добавляет ${role.acc} как слой.`,
    removes: role => `Этот вариант убирает ${role.acc} из композиции.`,
    replaces: (to, from) => `${cap(to.nom)} вместо ${from.gen} меняет структуру комплекта.`,
    volume: (gen, v1, v2) => `Силуэт ${gen} меняется: был ${v1}, стал ${v2}.`,
    texture: (gen, t1, t2) => `Фактура ${gen} меняется: была ${t1}, стала ${t2}.`,
    pattern: (gen, p1, p2) => `Узор ${gen} меняется: был ${p1}, стал ${p2}.`,
    paletteShift: (gen, c1, c2) => `Цвет ${gen} меняется: был ${c1}, стал ${c2}.`,
  },
  pairTitle: (a, b) => `${cap(a)} и ${b}`,
  colorRoleTitle: (colorForm, roleNom) => `${cap(colorForm)} ${roleNom}`,
};

const UZ = {
  roles: {
    'top': {
      'fine-knit': 'trikotaj','t-shirt': 'futbolka','shirt': "ko'ylak",
      'dress-shirt': "klassik ko'ylak",'polo': 'polo','knit-polo': 'trikotaj polo',
      'tank-top': 'mayka',
      'sweater': 'sviter',
    },
    'outerwear': {
      'blazer': 'pidjak','cardigan': 'kardigan','coat': 'palto',
      'jacket': 'kurtka','overshirt': "ustki ko'ylak",'bomber': 'bomber',
    },
    'bottom': {
      'chino': 'chino shim','jeans': 'jinsi shim','tailored-trouser': "klassik shim",
      'trouser': 'shim','shorts': 'shorti','athletic-trouser': 'sport shim',
      'knit-trouser': 'trikotaj shim',
    },
    'shoes': {
      'loafer': 'lofer','derby': 'derbi','oxford': 'oksford',
      'chelsea': 'Chelsi botink','minimal-sneaker': 'minimal krossovka',
      'sneaker': 'krossovka','sport-sneaker': 'sport krossovka','dress-boot': "klassik botink",
    },
  },
  colors: {
    'white':'oq','black':'qora','cream':'krem','beige':'bej','brown':'jigarrang',
    'navy':"to'q ko'k",'blue':"ko'k",'grey':'kulrang','green':'yashil','olive':'zaytun',
    'burgundy':'bordo','red':'qizil','pink':'pushti','purple':'binafsha',
    'yellow':'sariq','orange':"to'q sariq",'charcoal':"ko'mirrang",'dark':"to'q",
    'dark-neutral':"to'q neytral",'light':'och','earth':'tabiiy','warm':'iliq',
    'mid':"mo'tadil",'cool-dark':"chuqur sovuq",'cool-mid':'sovuq',
  },
  textures: {
    'smooth':'silliq','knit':'trikotaj','textured':'teksturali',
    'suede':'zamsh','denim':'jinsi','technical':'texnik',
  },
  patterns: {
    'solid':'bir rangli','stripe':'chiziqli','check':'kvadrat naqshli',
    'texture':'teksturali','print':'bosma naqshli',
  },
  volumes: {'slim':'tor','regular':'standart','wide':'keng'},
  styles: {
    'old_money':'old money','semi_classic':'yarim klassik','minimalist':'minimalizm',
    'casual':'kundalik','classic':'klassik','business_casual':'biznes klassik',
  },
  occasions: {
    'work':'ish','casual':'kundalik hayot','date':'uchrashuv',
    'formal':'rasmiy tadbir','sport':'sport','travel':'sayohat',
  },
  concepts: {
    'Refined Ivy':'Sayqallangan ayvi',
    'Textured tailoring':'Fakturali kostyum',
    'Smart knit layering':"Aqlli trikotaj qatlamlari",
    'Precise office tailoring':"Aniq ofis kostyumi",
    'Classic separates':"Klassik to'plam",
    'Tonal evening':'Tonli kech',
    'Soft evening texture':"Yumshoq kech fakturasi",
    'Modern evening separates':"Zamonaviy kech to'plami",
    'Easy transit':"Qulay yo'l",
    'Soft utility':'Yumshoq amaliy uslub',
    'Relaxed knit layers':'Erkin trikotaj qatlamlari',
    'Active layers':'Faol qatlamlar',
    'Formal tailoring':'Rasmiy kostyum',
    'Clean lines':'Toza chiziqlar',
    'Quiet knit texture':'Sokin trikotaj fakturasi',
    'Tonal modern separates':"Tonli zamonaviy to'plam",
    'Ivy Knit':'Ayvi trikotaj',
    'Soft Tailoring':'Yumshoq kostyum',
    'Relaxed Layers':'Erkin qatlamlar',
    'Tailored':'Kostyum silueti',
    'Soft Knit Layers':'Yumshoq trikotaj qatlamlari',
    'Refined Tailoring':'Sayqallangan kostyum',
    'Relaxed Denim Ease':"Erkin jinsi qulayligi",
    'Clean separates':"Toza to'plam",
    'Clean transit':"Toza yo'l",
    'Soft separates':"Yumshoq to'plam",
    'Textured separates':"Fakturali to'plam",
  },
  separatesFallback: "Toza to'plam",
  optionSuffix: n => ` · Variant ${n}`,
  prefixes: {
    'signature': (style, occasion) => `${cap(style)} uslubingizga eng yaqin variant — ${occasion} uchun.`,
    'refresh': (style, occasion) => `${cap(style)} uslubining ${occasion} uchun boshqa talqini.`,
    'discovery': (_style, occasion) => `${cap(occasion)} uchun yaroqli yangi uslub yo'nalishi.`,
    'alternative': (_style, occasion) => `${cap(occasion)} uchun yana bir mos variant; bu alohida kashfiyot yo'nalishi emas.`,
  },
  relationship: {
    dressFootwear: () => "Klassik poyabzal klassik shim chizig'ini davom ettiradi.",
    cardigan: () => 'Kardigan asosiy kiyim ustida ochiq qatlam hosil qiladi.',
    blazerStructured: () => "Pidjak va klassik shimlar uslubning uyg'unligini saqlaydi.",
    generic: () => "Tashqi qatlam va asosiy kiyim yuqori qismini hosil qiladi.",
    palette: (c1, r1, c2, r2) => `Palitra ${c1} ${r1} va ${c2} ${r2} orqali shakllanadi.`,
  },
  focus: {
    adds: role => `Bu variant qatlam sifatida ${role.nom} qo'shadi.`,
    removes: role => `Bu variant ${role.nom} qatlamini olib tashlaydi.`,
    replaces: (to, from) => `${cap(to.nom)} ${from.nom} o'rnini egallaydi — kiyim tarkibi o'zgaradi.`,
    volume: (role, v1, v2) => `${cap(role)} silueti ${v1}dan ${v2}ga o'zgaradi.`,
    texture: (role, t1, t2) => `${cap(role)} teksturasi ${t1}dan ${t2}ga o'zgaradi.`,
    pattern: (role, p1, p2) => `${cap(role)} naqshi ${p1}dan ${p2}ga o'zgaradi.`,
    paletteShift: (role, c1, c2) => `${cap(role)} rangi ${c1}dan ${c2}ga o'zgaradi.`,
  },
  pairTitle: (a, b) => `${cap(a)} va ${b}`,
  colorRoleTitle: (color, roleNom) => `${cap(color)} ${roleNom}`,
};

const COPY_LOCALES = { ru: RU, uz: UZ };

// Localized role lookup: a role entry for the language, or null when this
// language has no template for the subtype (the candidate is then skipped,
// exactly like the English path). UZ has no case system, so its entries are
// normalized to the same {nom, gen, acc, g} shape RU uses.
function roleFor(lang, d) {
  if (!d) return null;
  if (lang === 'en') {
    const name = enRole(d);
    return name ? { name, nom: name, gen: name, acc: name } : null;
  }
  const L = COPY_LOCALES[lang];
  const entry = L.roles[d.slot]?.[d.subtype];
  if (!entry) return null;
  return typeof entry === 'string' ? { name: entry, nom: entry, gen: entry, acc: entry } : entry;
}

/** Localized color term for agreement with a role's gender, or null. */
function colorFor(lang, color, gender) {
  if (!known(color)) return null;
  if (lang === 'en') return titleCase(color);
  const L = COPY_LOCALES[lang];
  const entry = L.colors[String(color).toLowerCase()];
  if (!entry) return null;
  if (typeof entry === 'string') return entry;
  return entry[gender] || entry.m || null;
}

/** ── Localized relationship fact (null = fall back to English string) ── */
function localizedRelationship(lang, part) {
  const L = COPY_LOCALES[lang];
  if (part.bottom?.structured && part.shoes &&
      ['loafer','oxford','derby','monk','dress-boot','chelsea'].includes(part.shoes.subtype) &&
      !part.shoes.casualSole) {
    return L.relationship.dressFootwear();
  }
  if (part.outerwear?.subtype === 'cardigan' && part.top) {
    return L.relationship.cardigan();
  }
  if (part.outerwear?.subtype === 'blazer' && part.bottom?.structured) {
    return L.relationship.blazerStructured();
  }
  const top = roleFor(lang, part.top);
  const bottom = roleFor(lang, part.bottom);
  if (part.top && part.bottom && known(part.top?.color) && known(part.bottom?.color)) {
    // Unknown subtypes degrade to the same generic slot words the English
    // path uses ("top"/"bottom"), never to invented construction facts.
    const generic = lang === 'ru' ? { top: {nom:'верх',g:'m'}, bottom: {nom:'низ',g:'m'} }
      : lang === 'uz' ? { top: {nom:'yuqori kiyim'}, bottom: {nom:'pastki kiyim'} }
        : { top: {name:'top'}, bottom: {name:'bottom'} };
    const r1 = top || generic.top, r2 = bottom || generic.bottom;
    const c1 = colorFor(lang, part.top.color, r1.g || r1.gender);
    const c2 = colorFor(lang, part.bottom.color, r2.g || r2.gender);
    const name1 = lang === 'en' ? titleCase(r1.name) : r1.nom;
    const name2 = lang === 'en' ? titleCase(r2.name) : r2.nom;
    if (c1 && c2) return L.relationship.palette(c1, name1, c2, name2);
  }
  if (part.outerwear && part.top) {
    // The generic layer + base structural line, stated in the display language.
    return lang === 'en' ? 'The outer layer and base form the upper-body combination.' : L.relationship.generic();
  }
  // No relationship evidence: an empty tail is legitimate in every language
  // (missing evidence is not a missing template, so no English fallback).
  return '';
}

/**
 * ── Localized direction focus (null = fall back to English string) ──
 * Specific comparisons require knowledge of both the lead and alternative.
 * Never claim a change in shape or texture just because two SKU names differ.
 */
function localizedFocus(lang, basePart, part, direction) {
  const L = COPY_LOCALES[lang];
  const evidence = direction?.evidence || {};
  const structural = Array.isArray(evidence.structuralSlots) ? evidence.structuralSlots : [];
  const palette = Array.isArray(evidence.paletteSlots) ? evidence.paletteSlots : [];
  const clean = value => known(value) ? titleCase(value) : '';
  for (const slot of structural) {
    const before = basePart?.[slot], after = part?.[slot];
    const beforeRole = roleFor(lang, before), afterRole = roleFor(lang, after);
    if (!before && after && afterRole) return lang === 'en' ? `This version adds ${afterRole.name} as a layer.` : L.focus.adds(afterRole);
    if (before && !after && beforeRole) return lang === 'en' ? `This version removes the ${beforeRole.name} layer.` : L.focus.removes(beforeRole);
    if (!before || !after) continue;
    if (beforeRole && afterRole && beforeRole.name !== afterRole.name) {
      return lang === 'en'
        ? `The ${afterRole.name} replaces the ${beforeRole.name}, changing the outfit structure.`
        : L.focus.replaces(afterRole, beforeRole);
    }
    if (beforeRole && known(before.volume) && known(after.volume) && before.volume !== after.volume) {
      const v1 = lang === 'en' ? clean(before.volume)
        : COPY_LOCALES[lang].volumes[String(before.volume).toLowerCase()];
      const v2 = lang === 'en' ? clean(after.volume)
        : COPY_LOCALES[lang].volumes[String(after.volume).toLowerCase()];
      if (!v1 || !v2) return null;
      return lang === 'en'
        ? `The ${beforeRole.name} changes from a ${clean(before.volume)} to a ${clean(after.volume)} silhouette.`
        : L.focus.volume(beforeRole.gen || beforeRole.name, v1, v2);
    }
    if (beforeRole && known(before.texture) && known(after.texture) && before.texture !== after.texture) {
      const t1 = lang === 'en' ? clean(before.texture)
        : COPY_LOCALES[lang].textures[String(before.texture).toLowerCase()];
      const t2 = lang === 'en' ? clean(after.texture)
        : COPY_LOCALES[lang].textures[String(after.texture).toLowerCase()];
      if (!t1 || !t2) return null;
      return lang === 'en'
        ? `The ${beforeRole.name} changes from ${clean(before.texture)} to ${clean(after.texture)} texture.`
        : L.focus.texture(beforeRole.gen || beforeRole.name, t1, t2);
    }
    if (beforeRole && known(before.pattern) && known(after.pattern) && before.pattern !== after.pattern) {
      const p1 = lang === 'en' ? clean(before.pattern)
        : COPY_LOCALES[lang].patterns[String(before.pattern).toLowerCase()];
      const p2 = lang === 'en' ? clean(after.pattern)
        : COPY_LOCALES[lang].patterns[String(after.pattern).toLowerCase()];
      if (!p1 || !p2) return null;
      return lang === 'en'
        ? `The ${beforeRole.name} changes from ${clean(before.pattern)} to ${clean(after.pattern)} patterning.`
        : L.focus.pattern(beforeRole.gen || beforeRole.name, p1, p2);
    }
  }
  for (const slot of palette) {
    const before = basePart?.[slot], after = part?.[slot];
    const afterRole = roleFor(lang, after);
    if (before && after && known(before.color) && known(after.color) &&
        before.color !== after.color && afterRole) {
      if (lang === 'en') {
        return `The ${afterRole.name} ${afterRole.name.endsWith('s') ? 'shift' : 'shifts'} from ${clean(before.color)} to ${clean(after.color)}.`;
      }
      const c1 = COPY_LOCALES[lang].colors[String(before.color).toLowerCase()];
      const c2 = COPY_LOCALES[lang].colors[String(after.color).toLowerCase()];
      // RU palette shift describes «цвет» (masculine) — the .m form; UZ uses
      // the bare term.
      const term = entry => typeof entry === 'string' ? entry : entry?.m || null;
      const b = term(c1), a = term(c2);
      if (!b || !a) return null;
      return L.focus.paletteShift(afterRole.gen || afterRole.name, b, a);
    }
  }
  return '';
}

/** Localized direction prefix (null = fall back to the English sentence). */
function localizedPrefix(lang, context, direction) {
  const L = COPY_LOCALES[lang];
  const template = L.prefixes[direction?.id];
  if (!template) return null;
  const styleKey = String(context?.style || '').trim().toLowerCase();
  const rawOccasion = String(context?.occasion || '').trim().toLowerCase();
  const occasionKey = /^date(?:[_ -]?night)?$/.test(rawOccasion) ? 'date' : rawOccasion;
  const style = L.styles[styleKey];
  const occasion = L.occasions[occasionKey];
  // An unknown style/occasion id would inject foreign words into the
  // sentence: per-string English fallback instead.
  if (!style || !occasion) return null;
  return template(style, occasion);
}

/** Localized evidence-led native headline (null = keep the English title). */
function localizedNativeHeadline(lang, nativeTitle, part) {
  const L = COPY_LOCALES[lang];
  const title = String(nativeTitle || '');
  if(/^(?:shirt|polo|knitwear|blazer|sport sneakers|shoes|tee)$/i.test(title)){
    const top=roleFor(lang,part?.top),bottom=roleFor(lang,part?.bottom);
    const layer=roleFor(lang,part?.outerwear);
    if(layer&&bottom&&/blazer/i.test(title))return L.pairTitle(layer.nom,bottom.nom);
    if(top&&bottom)return L.pairTitle(top.nom,bottom.nom);
  }
  if (!(part?.outerwear || !/\blayer(?:s|ing)?\b/i.test(title))) {
    // The layering headline is ungrounded without an actual layer: same
    // replacement rule as English, in the display language.
    const top = roleFor(lang, part?.top), bottom = roleFor(lang, part?.bottom);
    const topName = top ? (lang === 'en' ? titleCase(top.name) : top.nom) : '';
    const bottomName = bottom ? (lang === 'en' ? titleCase(bottom.name) : bottom.nom) : '';
    return topName && bottomName ? L.pairTitle(topName, bottomName) : L.separatesFallback;
  }
  return L.concepts[title] ?? null;
}

function localizedHeadlineCandidates(lang, part, direction) {
  const L=COPY_LOCALES[lang],e=direction?.evidence||{};
  const focusSlots=[...new Set([
    ...(Array.isArray(e.structuralSlots)?e.structuralSlots:[]),
    ...(Array.isArray(e.paletteSlots)?e.paletteSlots:[]),...slotOrder,
  ])];
  const result=[],roleOnly=[];
  const top=roleFor(lang,part.top),bottom=roleFor(lang,part.bottom);
  const layer=roleFor(lang,part.outerwear),shoes=roleFor(lang,part.shoes);
  const pair=(a,b)=>lang==='en'
    ? titleCase(a.name)+' & '+titleCase(b.name)
    : L.pairTitle(a.nom,b.nom);
  // A pairing describes a real outfit; "Polo" alone is only one garment.
  if(layer&&bottom)result.push(pair(layer,bottom));
  if(top&&bottom)result.push(pair(top,bottom));
  if(!top&&bottom&&shoes)result.push(pair(bottom,shoes));
  for(const slot of focusSlots){
    const d=part[slot],role=roleFor(lang,d);
    if(!role)continue;
    if(known(d.color)){
      const color=colorFor(lang,d.color,role.g||role.gender);
      if(color)result.push(lang==='en'
        ? titleCase(color)+' '+titleCase(role.name)
        : L.colorRoleTitle(color,role.nom));
      else if(lang!=='en'&&enRole(d))
        result.push(titleCase(d.color)+' '+titleCase(enRole(d)));
    }
    roleOnly.push(lang==='en'?titleCase(role.name):cap(role.nom));
  }
  result.push(...roleOnly);
  return [...new Set(result)];
}

/** Full localized card (title + description) with atomic per-string English
 * fallback: a description missing any localized term is written in English
 * in full, so a card string never mixes languages. `basePart` mirrors the
 * English path: the deck's first look is the comparison base. */
function localizedCard(lang, seen, index, look, basePart, context, direction, native) {
  const L = COPY_LOCALES[lang];
  const part = details(look.items);
  const truthful = localizedNativeHeadline(lang, native.title, part);
  const options = index === 0
    ? [truthful]
    : [...localizedHeadlineCandidates(lang, part, direction), truthful];
  let title = options.find(s => s && !seen.has(s.toLowerCase()));
  if (!title) {
    // Uniqueness suffix: localized only over a localized base title, so the
    // string never mixes scripts.
    title = `${truthful ?? native.title}${truthful ? L.optionSuffix(index + 1) : ` · Option ${index + 1}`}`;
  }
  const prefix = localizedPrefix(lang, context, direction);
  const change = direction?.qualified && direction?.id !== 'signature'
    ? localizedFocus(lang, basePart, part, direction)
    : '';
  const relationship = localizedRelationship(lang, part);
  let description;
  if (prefix === null || change === null || relationship === null) {
    // Missing localized template or vocabulary: write the whole string in
    // English (never mixed).
    description = shortCopy(basePart, part, context, direction);
  } else {
    description = [prefix, change, relationship].filter(Boolean).join(' ');
  }
  seen.add(title.toLowerCase());
  return {
    concept: { id: native.id, title, referenceId: look.referenceId ?? null },
    description,
  };
}

function actualRelationship(part) {
  if (part.bottom?.structured && part.shoes &&
      ['loafer','oxford','derby','monk','dress-boot','chelsea'].includes(part.shoes.subtype) &&
      !part.shoes.casualSole)
    return 'Dress footwear follows the tailored trouser line.';
  if (part.outerwear?.subtype === 'cardigan' && part.top)
    return 'The cardigan creates an open layer over the base.';
  if (part.outerwear?.subtype === 'blazer' && part.bottom?.structured)
    return 'A blazer and structured trousers keep the tailoring consistent.';
  if (part.top && part.bottom && known(part.top.color) && known(part.bottom.color)) {
    const top = enRole(part.top) || 'top';
    const bottom = enRole(part.bottom) || 'bottom';
    return `A ${part.top.color} ${top} with ${part.bottom.color} ${bottom} defines the palette.`;
  }
  if (part.outerwear && part.top)
    return 'The outer layer and base form the upper-body combination.';
  return ''; // Unverified occasion-fit praise would add no garment evidence.
}

/**
 * Specific comparisons require knowledge of both the lead and alternative.
 * Never claim a change in shape or texture just because two SKU names differ.
 */
function focusDescription(basePart, part, direction) {
  const evidence = direction?.evidence || {};
  const structural = Array.isArray(evidence.structuralSlots) ? evidence.structuralSlots : [];
  const palette = Array.isArray(evidence.paletteSlots) ? evidence.paletteSlots : [];
  const clean = value => known(value) ? titleCase(value) : '';
  for (const slot of structural) {
    const before = basePart?.[slot], after = part?.[slot];
    if (!before && after && enRole(after)) return `This version adds ${enRole(after)} as a layer.`;
    if (before && !after && enRole(before)) return `This version removes the ${enRole(before)} layer.`;
    if (!before || !after) continue;
    const from = enRole(before), to = enRole(after);
    if (from && to && from !== to) return `The ${to} replaces the ${from}, changing the outfit structure.`;
    if (from && known(before.volume) && known(after.volume) && before.volume !== after.volume)
      return `The ${from} changes from a ${clean(before.volume)} to a ${clean(after.volume)} silhouette.`;
    if (from && known(before.texture) && known(after.texture) && before.texture !== after.texture)
      return `The ${from} changes from ${clean(before.texture)} to ${clean(after.texture)} texture.`;
    if (from && known(before.pattern) && known(after.pattern) && before.pattern !== after.pattern)
      return `The ${from} changes from ${clean(before.pattern)} to ${clean(after.pattern)} patterning.`;
  }
  for (const slot of palette) {
    const before = basePart?.[slot], after = part?.[slot];
    if (before && after && known(before.color) && known(after.color) &&
        before.color !== after.color && enRole(after))
      return `The ${enRole(after)} ${enRole(after).endsWith('s') ? 'shift' : 'shifts'} from ${clean(before.color)} to ${clean(after.color)}.`;
  }
  return '';
}

/**
 * The native concept may describe an optional knit layer even when the
 * final three-slot outfit contains only a polo or sweater. Do not display
 * "layering" unless there is actually an outer layer in the selected SKUs.
 */
export function groundedNativeHeadline(nativeTitle, part, language) {
  const title = String(nativeTitle || '');
  if (language && language !== 'en') {
    return localizedNativeHeadline(language, title, part) ?? title;
  }
  if(/^(?:shirt|polo|knitwear|blazer|sport sneakers|shoes|tee)$/i.test(title)){
    const top=enRole(part?.top),bottom=enRole(part?.bottom),layer=enRole(part?.outerwear);
    if(layer&&bottom&&/blazer/i.test(title))
      return titleCase(layer)+' & '+titleCase(bottom);
    if(top&&bottom)return titleCase(top)+' & '+titleCase(bottom);
  }
  if (part?.outerwear || !/\blayer(?:s|ing)?\b/i.test(title)) return title;
  const top = enRole(part?.top), bottom = enRole(part?.bottom);
  return top && bottom ? `${titleCase(top)} & ${titleCase(bottom)}` : 'Clean separates';
}

function headlineCandidates(part,direction) {
  const e=direction?.evidence||{};
  const focusSlots=[...new Set([
    ...(Array.isArray(e.structuralSlots)?e.structuralSlots:[]),
    ...(Array.isArray(e.paletteSlots)?e.paletteSlots:[]),...slotOrder,
  ])];
  const result=[];
  const top=enRole(part.top),bottom=enRole(part.bottom),layer=enRole(part.outerwear);
  const shoes=enRole(part.shoes);
  if(layer&&bottom)result.push(titleCase(layer)+' & '+titleCase(bottom));
  if(top&&bottom)result.push(titleCase(top)+' & '+titleCase(bottom));
  if(!top&&bottom&&shoes)result.push(titleCase(bottom)+' & '+titleCase(shoes));
  for(const slot of focusSlots){
    const d=part[slot],role=enRole(d);
    if(!role)continue;
    if(known(d.color))result.push(titleCase(d.color)+' '+titleCase(role));
    result.push(titleCase(role));
  }
  return [...new Set(result)];
}

function shortCopy(basePart, part, context, direction) {
  const style = known(context?.style) ? titleCase(context.style) : 'your selected style';
  const occasion = known(context?.occasion) ? (/^date(?:[_ -]night)?$/i.test(context.occasion) ? 'date night' : titleCase(context.occasion)) : 'your occasion';
  const prefix = direction?.id === 'signature'
    ? `Closest to your ${style} direction for ${occasion}.`
    : direction?.id === 'refresh'
      ? `A different interpretation of ${style} for ${occasion}.`
      : direction?.id === 'discovery'
        ? `A new styling direction that still fits ${occasion}.`
        : `Another compatible option for ${occasion}; not a separate discovery direction.`;
  const change = direction?.qualified && direction?.id !== 'signature'
    ? focusDescription(basePart,part,direction) : '';
  return [prefix,change,actualRelationship(part)].filter(Boolean).join(' ');
}

// ── Personalized Editorial Stylist's Notes (grounded, zero-fabrication) ──

const SPECIFIC_MATERIALS = [
  'cashmere', 'silk', 'linen', 'suede', 'corduroy', 'tweed', 'velvet',
  'leather', 'denim', 'merino', 'wool', 'cotton', 'flannel', 'poplin', 'satin',
];
const UNSAFE_CLAIMS_RE = /\b(?:in\s+stock|available\s+now|free\s+shipping|ships?\s+today|guaranteed|100%\s+[a-z]+|waterproof|viewed\s+your\s+photos?)\b/i;
const COLOR_NAME_ALIASES = new Map([
  ['#0a1931', 'navy'], ['0a1931', 'navy'], ['gray', 'grey'],
  ['ivory', 'cream'], ['off-white', 'cream'], ['ecru', 'cream'],
  ['taupe', 'beige'], ['sand', 'beige'], ['slate', 'grey'],
  ['chocolate', 'brown'], ['espresso', 'brown'], ['cognac', 'brown'],
  ['oxblood', 'burgundy'], ['bordeaux', 'burgundy'], ['wine', 'burgundy'],
]);
const WARM_TONES = new Set(['cream', 'beige', 'camel', 'tan', 'brown', 'olive', 'khaki', 'rust', 'burgundy', 'stone', 'yellow', 'orange']);
const COOL_TONES = new Set(['navy', 'blue', 'grey', 'charcoal', 'black', 'white', 'green', 'purple', 'pink']);
const LIGHT_TONES = new Set(['white', 'cream', 'beige', 'stone', 'light']);
const DARK_TONES = new Set(['black', 'navy', 'charcoal', 'brown', 'burgundy', 'dark', 'dark-neutral', 'cool-dark']);

function cleanEditorialText(raw, max = 260) {
  if (typeof raw !== 'string') return '';
  return raw
    .replace(/https?:\/\/[^\s]+|\b[\w.%+-]+@[\w.-]+\.[a-z]{2,}\b/gi, '')
    .replace(/[<>{}\u0000-\u001f]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .slice(0, max);
}

function canonicalNoteColor(raw) {
  if (typeof raw !== 'string') return null;
  const cleaned = raw.trim().toLowerCase().replace(/[\s_]+/g, '-');
  if (!cleaned || ['unknown', 'neutral', 'missing', 'other'].includes(cleaned)) return null;
  if (COLOR_NAME_ALIASES.has(cleaned)) return COLOR_NAME_ALIASES.get(cleaned);
  return cleaned;
}

function verifiedItemColor(item, descriptor) {
  const vf = item ? currentVisual(item) : null;
  const vCol = vf?.attributes?.color;
  if (confidentVisual(vCol) && typeof vCol.value === 'string') {
    const canon = canonicalNoteColor(vCol.value);
    if (canon) return canon;
  }
  const explicit = canonicalNoteColor(item?.color);
  if (explicit) return explicit;
  const desc = canonicalNoteColor(descriptor?.color);
  // Avoid generic descriptor buckets ('dark', 'cool-dark', 'earth') if specific name color exists
  if (desc && !['dark', 'dark-neutral', 'cool-dark', 'cool-mid', 'mid', 'earth', 'light', 'warm'].includes(desc)) {
    return desc;
  }
  const nameText = String(item?.name || '').toLowerCase();
  for (const c of ['navy', 'charcoal', 'cream', 'beige', 'camel', 'olive', 'burgundy', 'stone', 'brown', 'black', 'white', 'grey', 'blue', 'green', 'tan']) {
    if (new RegExp(`\\b${c}\\b`, 'i').test(nameText)) return c;
  }
  return desc || null;
}

function verifiedLookMaterials(items, part) {
  const found = new Set();
  for (const item of items || []) {
    const text = `${item?.name || ''} ${item?.texture || ''} ${Array.isArray(item?.materials) ? item.materials.join(' ') : ''}`.toLowerCase();
    for (const mat of SPECIFIC_MATERIALS) {
      if (new RegExp(`\\b${mat}\\b`, 'i').test(text)) found.add(mat);
    }
    const vf = currentVisual(item);
    const vTex = vf?.attributes?.texture;
    if (confidentVisual(vTex) && typeof vTex.value === 'string') {
      found.add(vTex.value.toLowerCase());
    }
  }
  for (const slot of slotOrder) {
    const d = part?.[slot];
    if (!d) continue;
    if (known(d.texture)) found.add(d.texture.toLowerCase());
    for (const m of Array.isArray(d.materials) ? d.materials : []) {
      found.add(String(m).toLowerCase());
    }
  }
  return found;
}

/**
 * Validate that a Creative Director or Visual Critic sentence does not fabricate
 * missing garment roles (e.g. claiming a cardigan or blazer when absent) or
 * unverified fiber claims (e.g. claiming cashmere/silk when none of the selected
 * SKUs have that verified material).
 */
function isGroundedAgainstLook(sentence, part, verifiedMaterials) {
  if (!sentence || sentence.length < 18) return false;
  if (UNSAFE_CLAIMS_RE.test(sentence)) return false;
  const lower = sentence.toLowerCase();
  // Never claim an outer layer exists when the actual outfit has no outerwear.
  if (!part?.outerwear && /\b(blazer|cardigan|overcoat|coat|jacket|overshirt|outer\s+layer|layered\s+over)\b/.test(lower)) {
    return false;
  }
  // Never misname the actual outerwear subtype (e.g., calling a blazer a cardigan).
  if (part?.outerwear?.subtype) {
    const actualOuter = part.outerwear.subtype;
    if (actualOuter !== 'cardigan' && /\bcardigans?\b/.test(lower)) return false;
    if (actualOuter !== 'blazer' && /\bblazers?\b/.test(lower)) return false;
    if (actualOuter !== 'coat' && /\b(?:overcoats?|trench\s+coats?)\b/.test(lower)) return false;
  }
  // Never fabricate rare/specific textiles not verified on any selected SKU.
  for (const mat of ['cashmere', 'silk', 'linen', 'suede', 'corduroy', 'tweed', 'velvet']) {
    if (new RegExp(`\\b${mat}\\b`, 'i').test(lower) && !verifiedMaterials.has(mat)) {
      return false;
    }
  }
  return true;
}

function extractVisualCriticInsights(look, options, part, verifiedMaterials) {
  const verdict = options?.visualVerdict || look?.visualCritique || look?.visualVerdict || look?.visualReview || null;
  if (!verdict || typeof verdict !== 'object') return [];
  const insights = [];
  if (Array.isArray(verdict.relationships)) {
    for (const rel of verdict.relationships.slice(0, 3)) {
      const obs = cleanEditorialText(rel?.observation || rel?.note || '', 180);
      if (obs && !/\b(clash|mismatch|reject|veto|poor|bad|wrong|fail|incoherent)\b/i.test(obs) &&
          isGroundedAgainstLook(obs, part, verifiedMaterials)) {
        insights.push(obs);
      }
    }
  }
  if (Array.isArray(verdict.strengths)) {
    for (const st of verdict.strengths.slice(0, 2)) {
      const text = cleanEditorialText(typeof st === 'string' ? st : st?.observation || '', 160);
      if (text && isGroundedAgainstLook(text, part, verifiedMaterials)) {
        insights.push(text);
      }
    }
  }
  if (!insights.length && verdict.observations && typeof verdict.observations === 'object') {
    for (const slot of ['outerwear', 'top', 'bottom', 'shoes']) {
      const text = cleanEditorialText(verdict.observations[slot], 150);
      if (text && !/\b(clash|mismatch|reject|veto|poor|bad|wrong|fail)\b/i.test(text) &&
          isGroundedAgainstLook(text, part, verifiedMaterials)) {
        insights.push(text);
        if (insights.length >= 1) break;
      }
    }
  }
  return [...new Set(insights)];
}

function buildSilhouetteFormula(lang, part, items) {
  const hasDress = (items || []).some(it => String(it?.macroCategory || '').toLowerCase() === 'dress');
  const formatSlot = (slot) => {
    const d = part?.[slot];
    if (!d) return null;
    const role = roleFor(lang, d);
    if (!role) return null;
    const name = lang === 'en' ? role.name : role.nom;
    if (lang === 'en') {
      if (slot === 'outerwear' && d.structured) return `structured ${name}`;
      if (slot === 'outerwear' && d.volume === 'wide') return `relaxed ${name}`;
      if (slot === 'bottom' && d.volume === 'slim') return `slim ${name}`;
      if (slot === 'bottom' && d.volume === 'wide') return `wide-leg ${name}`;
      if (slot === 'shoes' && ['oxford', 'derby', 'monk', 'loafer', 'chelsea', 'dress-boot'].includes(d.subtype) && !d.casualSole) {
        return `sleek ${name}`;
      }
    }
    return name;
  };
  const topLabel = hasDress
    ? (lang === 'ru' ? 'платье' : lang === 'uz' ? "ko'ylak-libos" : 'dress')
    : formatSlot('top');
  const outerLabel = formatSlot('outerwear');
  const bottomLabel = hasDress ? null : formatSlot('bottom');
  const shoesLabel = formatSlot('shoes');

  const upper = outerLabel && topLabel
    ? (lang === 'ru' ? `${outerLabel} поверх ${roleFor('ru', part.top)?.gen || topLabel}`
      : lang === 'uz' ? `${topLabel} ustidan ${outerLabel}`
        : `${outerLabel} over ${topLabel}`)
    : (outerLabel || topLabel);

  const pieces = [upper, bottomLabel, shoesLabel].filter(Boolean);
  if (!pieces.length) {
    return lang === 'ru' ? 'Собранный комплект из выбранных вещей'
      : lang === 'uz' ? 'Tanlangan buyumlardan uygʻun toʻplam'
        : 'Balanced complete silhouette';
  }
  return cap(pieces.join(' + '));
}

function buildColorStory(lang, part, items) {
  const verifiedColors = [];
  for (const slot of slotOrder) {
    const item = (items || []).find(it => String(it?.macroCategory || '').toLowerCase() === slot);
    const d = part?.[slot];
    if (!item && !d) continue;
    const col = verifiedItemColor(item, d);
    if (col && !verifiedColors.includes(col)) verifiedColors.push(col);
  }
  if (!verifiedColors.length) {
    return lang === 'ru' ? 'Спокойная базовая палитра без визуального шума.'
      : lang === 'uz' ? "Ortiqcha shovqinsiz sokin asosiy palitra."
        : 'Restrained neutral palette grounded in the selected pieces.';
  }
  const hasLight = verifiedColors.some(c => LIGHT_TONES.has(c));
  const hasDark = verifiedColors.some(c => DARK_TONES.has(c));
  const warmCount = verifiedColors.filter(c => WARM_TONES.has(c)).length;
  const coolCount = verifiedColors.filter(c => COOL_TONES.has(c)).length;
  const isTonal = verifiedColors.length <= 2 && !(hasLight && hasDark);

  const toneMode = isTonal ? 'tonal'
    : (hasLight && hasDark) ? 'contrast'
      : (warmCount >= coolCount && warmCount >= 2) ? 'warm'
        : (coolCount >= 2) ? 'cool' : 'balanced';

  if (lang === 'ru') {
    const names = verifiedColors.map(c => colorFor('ru', c, 'm')).filter(Boolean);
    if (!names.length) return 'Спокойная выверенная палитра комплекта.';
    const mood = toneMode === 'contrast' ? 'выразительный чистый контраст'
      : toneMode === 'tonal' ? 'спокойная тональная гармония'
        : toneMode === 'warm' ? 'тёплая природная палитра'
          : toneMode === 'cool' ? 'собранная прохладная палитра' : 'сбалансированная палитра';
    return `${cap(names.join(', '))} — ${mood}.`;
  }
  if (lang === 'uz') {
    const names = verifiedColors.map(c => colorFor('uz', c)).filter(Boolean);
    if (!names.length) return "Sokin va uyg'un ranglar palitrasi.";
    const mood = toneMode === 'contrast' ? 'aniq va toza kontrast'
      : toneMode === 'tonal' ? "sokin tonli uyg'unlik"
        : toneMode === 'warm' ? 'iliq tabiiy palitra'
          : toneMode === 'cool' ? 'saranjom sovuq palitra' : "muvozanatli palitra";
    return `${cap(names.join(', '))} — ${mood}.`;
  }
  const names = verifiedColors.map(c => titleCase(c));
  const listStr = names.length === 1 ? names[0]
    : names.length === 2 ? `${names[0]} and ${names[1]}`
      : `${names.slice(0, -1).join(', ')}, and ${names.at(-1)}`;
  const mood = toneMode === 'contrast' ? 'crisp light-dark contrast'
    : toneMode === 'tonal' ? 'cohesive tonal depth'
      : toneMode === 'warm' ? 'warm earth harmony'
        : toneMode === 'cool' ? 'cool tailored balance' : 'grounded neutral balance';
  return `${listStr} create ${mood}.`;
}

function buildWhyItFlattersYou(lang, part, items, context, options) {
  const dna = options?.personalStyleDna || context?.personalStyleDna || null;
  const appearance = dna?.appearance || context?.appearanceProfile || null;
  const body = dna?.body || context?.bodyProfile || null;
  const wardrobe = dna?.ownedWardrobeDna || dna?.ownedWardrobeSummary || context?.ownedWardrobeSummary || null;
  const favColors = [
    ...(Array.isArray(dna?.favoriteColors) ? dna.favoriteColors : []),
    ...(Array.isArray(dna?.preferences?.favoriteColors) ? dna.preferences.favoriteColors : []),
    ...(Array.isArray(context?.preferences?.favoriteColors) ? context.preferences.favoriteColors : []),
  ].map(canonicalNoteColor).filter(Boolean);

  const lookColors = [];
  for (const slot of slotOrder) {
    const item = (items || []).find(it => String(it?.macroCategory || '').toLowerCase() === slot);
    const col = verifiedItemColor(item, part?.[slot]);
    if (col && !lookColors.includes(col)) lookColors.push(col);
  }

  const reasons = [];

  // 1. Explicit opt-in appearance / undertone / contrast harmony (NEVER inferred when absent!)
  if (appearance && typeof appearance === 'object' && appearance.consentGranted !== false) {
    const undertone = String(appearance.skinUndertone || '').trim().toLowerCase();
    const contrast = String(appearance.contrastLevel || '').trim().toLowerCase();
    const flattering = new Set([
      ...(Array.isArray(appearance.flatteringColors) ? appearance.flatteringColors : []),
      ...(Array.isArray(appearance.powerNeutrals) ? appearance.powerNeutrals : []),
    ].map(canonicalNoteColor).filter(Boolean));
    const matchedFlattering = lookColors.filter(c => flattering.has(c));
    const hasLight = lookColors.some(c => LIGHT_TONES.has(c));
    const hasDark = lookColors.some(c => DARK_TONES.has(c));

    if (matchedFlattering.length && ['warm', 'cool', 'neutral', 'olive'].includes(undertone)) {
      if (lang === 'ru') {
        const cNames = matchedFlattering.slice(0, 2).map(c => colorFor('ru', c, 'm')).filter(Boolean);
        const uRu = { warm: 'тёплый', cool: 'холодный', neutral: 'нейтральный', olive: 'оливковый' }[undertone];
        if (cNames.length) reasons.push(`Оттенки (${cNames.join(', ')}) поддерживают ваш ${uRu} подтон.`);
      } else if (lang === 'uz') {
        const cNames = matchedFlattering.slice(0, 2).map(c => colorFor('uz', c)).filter(Boolean);
        const uUz = { warm: 'iliq', cool: 'sovuq', neutral: 'neytral', olive: 'zaytun' }[undertone];
        if (cNames.length) reasons.push(`${cap(cNames.join(' va '))} ranglari sizning ${uUz} teri ohangizga mos tushadi.`);
      } else {
        const cNames = matchedFlattering.slice(0, 2).map(c => c.toLowerCase()).join(' and ');
        const contrastTail = contrast === 'high' && hasLight && hasDark
          ? ' with crisp high-contrast framing'
          : contrast === 'low' && !(hasLight && hasDark) ? ' in a softly blended low-contrast range' : '';
        reasons.push(`The ${cNames} tones complement your ${undertone} undertone${contrastTail}.`);
      }
    } else if (['warm', 'cool', 'neutral', 'olive'].includes(undertone)) {
      const undertoneMatch = undertone === 'warm' ? lookColors.some(c => WARM_TONES.has(c))
        : undertone === 'cool' ? lookColors.some(c => COOL_TONES.has(c)) : lookColors.length > 0;
      if (undertoneMatch) {
        if (lang === 'en') {
          reasons.push(`This palette aligns with your ${undertone} undertone${contrast && contrast !== 'unknown' ? ` and ${contrast}-contrast balance` : ''}.`);
        } else if (lang === 'ru') {
          const uRu = { warm: 'тёплым', cool: 'холодным', neutral: 'нейтральным', olive: 'оливковым' }[undertone];
          reasons.push(`Палитра гармонирует с вашим ${uRu} подтоном.`);
        } else {
          const uUz = { warm: 'iliq', cool: 'sovuq', neutral: 'neytral', olive: 'zaytun' }[undertone];
          reasons.push(`Palitra sizning ${uUz} ohangiz bilan uyg'unlashadi.`);
        }
      }
    }
  }

  // 2. Explicit body proportion / fit preference alignment (NEVER fabricates measurements!)
  const fitPref = String(body?.fitPreference || dna?.fitPreference || context?.preferences?.fitPreference || '').trim().toLowerCase();
  const bodyType = String(body?.bodyType || '').trim().toLowerCase().replace(/[\s-]+/g, '_');
  const validBodyTypes = new Set(['inverted_triangle', 'rectangle', 'triangle_pear', 'oval_apple', 'hourglass', 'athletic', 'trapezoid']);
  if (validBodyTypes.has(bodyType) || ['fitted', 'loose', 'balanced'].includes(fitPref)) {
    const hasStructuredLayer = part?.outerwear?.structured || part?.outerwear?.subtype === 'blazer' || part?.outerwear?.subtype === 'coat';
    const hasOpenLayer = part?.outerwear?.closure === 'openable';
    if (lang === 'en') {
      if (hasStructuredLayer && validBodyTypes.has(bodyType)) {
        reasons.push(`The structured ${enRole(part.outerwear) || 'outer layer'} frames your shoulders cleanly while keeping balanced lower proportions.`);
      } else if (hasOpenLayer && validBodyTypes.has(bodyType)) {
        reasons.push(`The open ${enRole(part.outerwear) || 'layer'} creates an elongating vertical line tailored to your proportions.`);
      } else if (['fitted', 'loose', 'balanced'].includes(fitPref)) {
        reasons.push(`The cut respects your ${fitPref} fit preference across the full silhouette.`);
      }
    } else if (lang === 'ru') {
      if (hasStructuredLayer && validBodyTypes.has(bodyType)) {
        reasons.push('Структурированный верхний слой выстраивает чёткую линию плеч и уравновешивает силуэт.');
      } else if (['fitted', 'loose', 'balanced'].includes(fitPref)) {
        const fRu = { fitted: 'приталенный', loose: 'свободный', balanced: 'сбалансированный' }[fitPref];
        reasons.push(`Посадка поддерживает ваш ${fRu} крой по всему силуэту.`);
      }
    } else {
      if (hasStructuredLayer && validBodyTypes.has(bodyType)) {
        reasons.push("Tuzilmali ustki qatlam yelka chizig'ini aniq ko'rsatib, siluetni muvozanatlashtiradi.");
      } else if (['fitted', 'loose', 'balanced'].includes(fitPref)) {
        const fUz = { fitted: 'torroq', loose: 'erkin', balanced: 'muvozanatli' }[fitPref];
        reasons.push(`Bichim siz tanlagan ${fUz} o'tirishga mos keladi.`);
      }
    }
  }

  // 3. Explicit favorite colors or owned wardrobe continuity
  const matchedFav = lookColors.filter(c => favColors.includes(c));
  const wardrobeColors = (Array.isArray(wardrobe?.dominantColors) ? wardrobe.dominantColors : [])
    .map(canonicalNoteColor).filter(Boolean);
  const matchedWardrobe = lookColors.filter(c => wardrobeColors.includes(c));

  if (!reasons.length && matchedFav.length) {
    if (lang === 'en') {
      reasons.push(`Anchors the look in your preferred ${matchedFav.slice(0, 2).join(' and ')} palette.`);
    } else if (lang === 'ru') {
      const names = matchedFav.slice(0, 2).map(c => colorFor('ru', c, 'm')).filter(Boolean);
      if (names.length) reasons.push(`Образ опирается на ваши любимые оттенки: ${names.join(', ')}.`);
    } else {
      const names = matchedFav.slice(0, 2).map(c => colorFor('uz', c)).filter(Boolean);
      if (names.length) reasons.push(`To'plam siz yoqtirgan ${names.join(' va ')} ranglariga tayanadi.`);
    }
  } else if (reasons.length < 2 && matchedWardrobe.length && !matchedFav.length) {
    if (lang === 'en') {
      reasons.push(`Pairs effortlessly with the ${matchedWardrobe[0]} pieces already anchoring your wardrobe.`);
    } else if (lang === 'ru') {
      const cName = colorFor('ru', matchedWardrobe[0], 'm');
      if (cName) reasons.push(`Легко сочетается с вещами вашего гардероба в базовой гамме.`);
    } else {
      reasons.push(`Garderobingizdagi asosiy ranglar bilan oson uyg'unlashadi.`);
    }
  }

  return reasons.length ? reasons.slice(0, 2).join(' ') : null;
}

function buildStylingTips(lang, part, context) {
  const tips = [];
  const temp = Number(context?.weather?.temp ?? context?.weatherCelsius);

  if (part?.outerwear && part?.top) {
    const outerName = roleFor(lang, part.outerwear);
    const topName = roleFor(lang, part.top);
    if (part.outerwear.closure === 'openable' && outerName && topName) {
      if (lang === 'ru') {
        tips.push(`Носите ${outerName.acc} расстёгнутым в движении, чтобы открыть вертикаль ${topName.gen}.`);
      } else if (lang === 'uz') {
        tips.push(`Harakatda ${outerName.nom}ni ochiq qoldiring — bu ${topName.nom} chizig'ini ko'rsatib turadi.`);
      } else {
        tips.push(`Wear the ${outerName.name} open in motion to reveal the clean vertical contrast of the ${topName.name}.`);
      }
    }
  }

  if (part?.top && ['dress-shirt', 'shirt'].includes(part.top.subtype) && part?.bottom?.structured) {
    if (lang === 'ru') {
      tips.push('Заправьте рубашку аккуратно в классические брюки, чтобы подчеркнуть линию талии.');
    } else if (lang === 'uz') {
      tips.push("Bel chizig'ini aniq ko'rsatish uchun ko'ylakni klassik shim ichiga tartibli kiriting.");
    } else {
      tips.push('Tuck the shirt cleanly into the structured trousers to sharpen the waist-to-leg proportion.');
    }
  } else if (part?.bottom && ['chino', 'jeans', 'tailored-trouser', 'trouser'].includes(part.bottom.subtype) && part?.shoes) {
    const shoeRole = roleFor(lang, part.shoes);
    const botRole = roleFor(lang, part.bottom);
    if (shoeRole && botRole) {
      if (lang === 'ru') {
        tips.push(`Держите чистую длину ${botRole.gen} с минимальным заломом, чтобы ${shoeRole.nom} завершали силуэт.`);
      } else if (lang === 'uz') {
        tips.push(`${cap(botRole.nom)} etagi toza tursin — shunda ${shoeRole.nom} siluetni aniq yakunlaydi.`);
      } else {
        tips.push(`Keep a clean, minimal break at the ${botRole.name} hem so the ${shoeRole.name} anchor the silhouette.`);
      }
    }
  }

  if (Number.isFinite(temp)) {
    if (temp >= 26 && part?.outerwear) {
      if (lang === 'ru') tips.push('В тёплую погоду держите верхний слой открытым или слегка подверните манжеты для лёгкости.');
      else if (lang === 'uz') tips.push("Issiq havoda ustki qatlamni ochiq tuting yoki yeng uchlarini yengil qayirib oling.");
      else tips.push('Keep the outer layer unbuttoned and roll the cuffs slightly for warm-weather airflow.');
    } else if (temp <= 10 && part?.outerwear) {
      if (lang === 'ru') tips.push('В прохладную погоду застегните верхний слой, чтобы сохранить чёткий собранный силуэт.');
      else if (lang === 'uz') tips.push("Salqin havoda aniq va issiq siluet uchun ustki qatlamni tugmalab kiying.");
      else tips.push('Fasten the outer layer outdoors to lock in warmth while preserving a sharp silhouette.');
    }
  }

  if (!tips.length && part?.top && part?.bottom) {
    if (lang === 'ru') tips.push('Соблюдайте баланс пропорций между верхом и низом, избегая лишних контрастных деталей.');
    else if (lang === 'uz') tips.push("Yuqori va pastki qism proporsiyasini muvozanatda saqlang, ortiqcha detallardan tiyiling.");
    else tips.push('Keep the proportions clean between top and bottom and let the natural garment lines lead.');
  }

  return tips.slice(0, 3);
}

/**
 * Build personalized, editorial-caliber Stylist's Notes strictly grounded in
 * verified SKU facts, Creative Director `artDirection`, Visual Critic real
 * visual observations, and explicit user `PersonalStyleDna`.
 */
export function buildStylistNotes(look, context, direction, options = {}) {
  const lang = options.language || normalizeStylistLanguage(context?.language);
  const items = Array.isArray(look?.items) ? look.items : [];
  const part = options.part || details(items);
  const verifiedMaterials = verifiedLookMaterials(items, part);

  const rawArtDirection = cleanEditorialText(
    options.artDirection ||
    look?.brief?.artDirection ||
    look?.artDirection ||
    direction?.artDirection ||
    '',
    235,
  );
  const groundedArtDirection = lang === 'en' && isGroundedAgainstLook(rawArtDirection, part, verifiedMaterials)
    ? rawArtDirection
    : '';

  const criticInsights = lang === 'en'
    ? extractVisualCriticInsights(look, options, part, verifiedMaterials)
    : [];

  let stylistNote = '';
  if (lang === 'en') {
    const parts = [];
    if (groundedArtDirection) {
      parts.push(groundedArtDirection.endsWith('.') ? groundedArtDirection : `${groundedArtDirection}.`);
    }
    if (criticInsights.length) {
      const leadCritic = criticInsights[0];
      // Avoid duplicating near-identical phrasing if artDirection already covers it.
      if (!groundedArtDirection || !groundedArtDirection.toLowerCase().includes(leadCritic.slice(0, 32).toLowerCase())) {
        parts.push(leadCritic.endsWith('.') ? leadCritic : `${leadCritic}.`);
      }
    }
    if (!parts.length) {
      const rel = actualRelationship(part);
      const formula = buildSilhouetteFormula('en', part, items);
      stylistNote = rel ? `${formula}. ${rel}` : `${formula}.`;
    } else {
      stylistNote = parts.join(' ');
    }
  } else {
    const formula = buildSilhouetteFormula(lang, part, items);
    const rel = localizedRelationship(lang, part);
    stylistNote = rel ? `${formula}. ${rel}` : `${formula}.`;
  }

  return {
    stylistNote,
    whyItFlattersYou: buildWhyItFlattersYou(lang, part, items, context, options),
    stylingTips: buildStylingTips(lang, part, context),
    colorStory: buildColorStory(lang, part, items),
    silhouetteFormula: buildSilhouetteFormula(lang, part, items),
  };
}

/** Assign each actual selected look a title + explanation + Stylist's Notes,
 * preserving order and keeping a unique headline even when all native concept IDs are equal.
 * The display language comes from context.language ('en' | 'ru' | 'uz'). */
export function editorialDeckCopy(looks, context, directions, options = {}) {
  const lang = normalizeStylistLanguage(context?.language);
  const seen = new Set();
  const basePart = details(looks?.[0]?.items || []);
  return (looks || []).map((look, index) => {
    const native = lookConcept(look.items, context).concept;
    const direction = directions?.[index] || { id: index ? 'alternative' : 'signature' };
    const cardOptions = {
      personalStyleDna: options?.personalStyleDna || context?.personalStyleDna || null,
      artDirection: options?.creativeBriefs?.[index]?.artDirection || look?.brief?.artDirection || look?.artDirection || direction?.artDirection,
      visualVerdict: options?.visualVerdicts?.[index] || look?.visualCritique || look?.visualVerdict || look?.visualReview,
    };
    if (lang !== 'en') {
      const card = localizedCard(lang, seen, index, look, basePart, context, direction, native);
      return {
        ...card,
        ...buildStylistNotes(look, context, direction, { ...cardOptions, language: lang }),
      };
    }
    const part = details(look.items);
    const truthfulNative = groundedNativeHeadline(native.title, part);
    const headlineOpts = index === 0
      ? [truthfulNative]
      : [...headlineCandidates(part, direction), truthfulNative];
    let title = headlineOpts.find(s => s && !seen.has(s.toLowerCase()));
    if (!title) title = `${truthfulNative} · Option ${index + 1}`;
    seen.add(title.toLowerCase());
    return {
      concept: { id: native.id, title, referenceId: look.referenceId ?? null },
      description: shortCopy(basePart, part, context, direction),
      ...buildStylistNotes(look, context, direction, { ...cardOptions, part, language: 'en' }),
    };
  });
}
