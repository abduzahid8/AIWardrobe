import test from 'node:test';
import assert from 'node:assert/strict';
import {
  editorialDeckCopy,
  DECK_COPY_VERSION,
  STYLIST_NOTES_VERSION,
  groundedNativeHeadline,
  buildStylistNotes,
  sanitizeStylistProse,
} from './deck_editorial_copy.mjs';
import { stylingContext } from '../../src/domain/stylingContext.ts';
import { prepareGarment, describeGarment } from '../../src/domain/garmentDescriptor.ts';
import { lookConcept } from '../../src/domain/outfitConcepts.ts';

const context = stylingContext({style:'classic', occasion:'Work', request:'', optionalLayer:true});
const garment = (id, name, slot, color) => ({
  id,name,macroCategory:slot, ...(color?{color}:{}),
});
const first = [
  garment('top1','White cotton dress shirt','top','white'),
  garment('blazer1','Navy wool blazer','outerwear','navy'),
  garment('bottom1','Charcoal tailored trousers','bottom','charcoal'),
  garment('shoes1','Brown leather derby shoes','shoes','brown'),
];
const second = [
  garment('top2','Cream cotton dress shirt','top','cream'),
  garment('blazer2','Grey wool blazer','outerwear','grey'),
  garment('bottom2','Navy tailored trousers','bottom','navy'),
  garment('shoes2','Black leather oxford shoes','shoes','black'),
];
const third = [
  garment('top3','Blue fine knit sweater','top','blue'),
  garment('blazer3','Brown wool blazer','outerwear','brown'),
  garment('bottom3','Beige tailored trousers','bottom','beige'),
  garment('shoes3','Brown leather loafers','shoes','brown'),
];
const looks = [first,second,third].map((items,index) => ({id:'look-'+index,items}));
const directions = [
  {id:'signature',qualified:true,evidence:{structuralSlots:[],paletteSlots:[]}},
  {id:'refresh',qualified:true,evidence:{structuralSlots:[],paletteSlots:['top','bottom']}},
  {id:'discovery',qualified:true,evidence:{structuralSlots:['top'],paletteSlots:['outerwear','shoes']}},
];

test('three cards have distinct, deterministic, evidence-grounded headings',()=>{
  const native=lookConcept(looks[0].items,context).concept;
  const firstPass=editorialDeckCopy(looks,context,directions);
  const secondPass=editorialDeckCopy(looks,context,directions);
  assert.equal(DECK_COPY_VERSION,'s6-grounded-deck-copy-v5');
  assert.deepEqual(firstPass,secondPass);
  assert.equal(firstPass[0].concept.title,native.title);
  assert.equal(new Set(firstPass.map(x=>x.concept.title.toLowerCase())).size,3);
  assert.ok(firstPass[1].concept.title.length<50);
  assert.ok(firstPass[2].concept.title.length<50);
  assert.ok(firstPass[1].description.includes('different interpretation'));
  assert.ok(firstPass[2].description.includes('new styling direction'));
  assert.ok(firstPass.every(x=>!x.description.includes('viewed your photos')));
});

test('unknown construction and missing user information never become invented fashion facts',()=>{
  const sparse = [{id:'a',items:[
    garment('a1','Top','top'),
    garment('a2','Pants','bottom'),
    garment('a3','Shoes','shoes'),
  ]}];
  const noPreferences=stylingContext({style:'casual',occasion:'Casual',optionalLayer:true});
  const copy=editorialDeckCopy(sparse,noPreferences,[{id:'signature',qualified:true}]);
  assert.equal(copy.length,1);
  assert.ok(!/comfort|skin|body shape|photo|saw|waterproof|100%|cashmere/i.test(copy[0].description));
  assert.ok(!/unknown|undefined|null/i.test(copy[0].concept.title));
  assert.ok(!copy[0].description.includes('favorite brand'));
});

test('an unqualified alternative cannot present itself as an adventurous stylist pick',()=>{
  const copy=editorialDeckCopy(looks,context,[
    directions[0],{id:'alternative',qualified:false,evidence:{structuralSlots:['top']}},
    {id:'alternative',qualified:false,evidence:{paletteSlots:['shoes']}},
  ]);
  assert.ok(copy[1].description.includes('not a separate discovery direction'));
  assert.ok(copy[2].description.includes('not a separate discovery direction'));
  assert.equal(new Set(copy.map(x=>x.concept.title)).size,3);
});

test('even identical wardrobe descriptions get honest unique titles, not invented differences',()=>{
  const clones=[0,1,2].map(n=>({id:'clone-'+n,
    items:first.map((item,i)=>({...item,id:`c${n}-${i}`}))}));
  const copy=editorialDeckCopy(clones,context,directions);
  assert.equal(new Set(copy.map(x=>x.concept.title)).size,3);
  assert.ok(copy.every(x=>!x.description.includes('completely different')));
});

test('title and explanation never mutate selected products, native concepts or user preferences',()=>{
  const input=JSON.stringify(looks),ctx=JSON.stringify(context);
  editorialDeckCopy(looks,context,directions);
  assert.equal(JSON.stringify(looks),input);
  assert.equal(JSON.stringify(context),ctx);
});

test('descriptions compare known garment facts, not vague unproved changes',()=>{
  const descriptions=editorialDeckCopy(looks,context,directions).map(x=>x.description);
  assert.match(descriptions[1],/tailored trousers shift from Dark to Cool Dark/i);
  assert.match(descriptions[2],/(knitwear|sweater) replaces the dress shirt/i);
  assert.ok(!descriptions.some(x=>/changes the silhouette or texture|overall mix is appropriate/.test(x)));
});

test('date night is described as date night, not Date',()=>{
  const dateContext=stylingContext({style:'old_money',occasion:'date',optionalLayer:true});
  const output=editorialDeckCopy([looks[0]],dateContext,[directions[0]]);
  assert.match(output[0].description,/date night/);
});

test('unlayered knit polo never gets a fake layering headline',()=>{
  const part={top:{slot:'top',subtype:'knit-polo'},outerwear:null,
    bottom:{slot:'bottom',subtype:'chino'}};
  assert.equal(groundedNativeHeadline('Smart knit layering',part),'Knit Polo & Chinos');
  assert.equal(groundedNativeHeadline('Refined Tailoring',part),'Refined Tailoring');
  assert.equal(groundedNativeHeadline('Smart knit layering',
    {...part,outerwear:{slot:'outerwear',subtype:'cardigan'}}),'Smart knit layering');
});

// ── Localization (v4): ru + uz template tables ──────────────────────────
const CYRILLIC = /[А-Яа-яёЁ]/;
const LATIN = /[A-Za-z]/;
/** A card string never mixes languages: it is written entirely in one script. */
const monolingual = s => !s || !(CYRILLIC.test(s) && LATIN.test(s));
const contextFor = (language, overrides = {}) => stylingContext({
  style:'classic', occasion:'Work', request:'', optionalLayer:true, language, ...overrides,
});

const allNativeTitles = [
  'Refined Ivy','Textured tailoring','Smart knit layering','Precise office tailoring',
  'Classic separates','Tonal evening','Soft evening texture','Modern evening separates',
  'Easy transit','Soft utility','Relaxed knit layers','Active layers','Formal tailoring',
  'Clean lines','Quiet knit texture','Tonal modern separates','Ivy Knit','Soft Tailoring',
  'Relaxed Layers','Tailored','Soft Knit Layers','Refined Tailoring','Relaxed Denim Ease',
  'Clean separates','Clean transit','Soft separates','Textured separates',
];

test('ru and uz decks keep the same uniqueness, determinism and grounding rules',()=>{
  const native=lookConcept(looks[0].items,context).concept;
  for (const language of ['ru','uz']) {
    const localizedContext = contextFor(language);
    const firstPass = editorialDeckCopy(looks,localizedContext,directions);
    const secondPass = editorialDeckCopy(looks,localizedContext,directions);
    assert.deepEqual(firstPass,secondPass);
    assert.equal(firstPass.length,3);
    // Same uniqueness rule per language: three distinct evidence-led headlines.
    assert.equal(new Set(firstPass.map(x=>x.concept.title.toLowerCase())).size,3);
    // The first card keeps the (localized) native concept headline.
    assert.equal(firstPass[0].concept.title,groundedNativeHeadline(native.title,{},language));
    // Grounding survives translation: the dress-shoe fact is stated, never
    // vague praise; the refresh card carries the measured color change.
    assert.ok(firstPass.every(x=>!/comfort|skin|body shape|photo|100%|cashmere/i.test(x.description)));
    assert.match(firstPass[0].description,/Классическая обувь|klassik poyabzal/i);
    assert.match(firstPass[1].description,/Цвет .* меняется|rangi .* o'zgaradi/i);
    // Every title and description is written in exactly one language.
    for (const card of firstPass) {
      assert.ok(monolingual(card.concept.title),`mixed-language title: ${card.concept.title}`);
      assert.ok(monolingual(card.description),`mixed-language description: ${card.description}`);
    }
    // ru is fully Cyrillic, uz fully Latin — a localized deck reads native.
    const script = language === 'ru' ? CYRILLIC : LATIN;
    const wrongScript = language === 'ru' ? LATIN : CYRILLIC;
    for (const card of firstPass) {
      assert.ok(script.test(card.concept.title));
      assert.ok(!wrongScript.test(card.concept.title));
      assert.ok(!wrongScript.test(card.description));
    }
  }
});

test('an en deck never picks up Cyrillic from the localized tables',()=>{
  const copy=editorialDeckCopy(looks,context,directions);
  for (const card of copy) {
    assert.ok(!CYRILLIC.test(card.concept.title));
    assert.ok(!CYRILLIC.test(card.description));
    assert.ok(monolingual(card.description));
  }
});

test('every native concept title has a ru and uz translation, so no card mixes scripts',()=>{
  const outerwearPart={top:{slot:'top',subtype:'shirt'},outerwear:{slot:'outerwear',subtype:'blazer'},
    bottom:{slot:'bottom',subtype:'trouser'},shoes:{slot:'shoes',subtype:'loafer'}};
  for (const title of allNativeTitles) {
    for (const language of ['ru','uz']) {
      const localized=groundedNativeHeadline(title,outerwearPart,language);
      assert.notEqual(localized,title,`missing ${language} concept title: ${title}`);
      assert.ok(monolingual(localized));
    }
  }
});

test('a missing localized template falls back to English per string, never to a mix',()=>{
  // 'Garden party' has no ru occasion template: the description must be the
  // full English sentence, not a ru/en hybrid.
  const gardenContext=contextFor('ru',{occasion:'Garden party'});
  const copy=editorialDeckCopy([looks[0]],gardenContext,[directions[0]]);
  assert.match(copy[0].description,/Closest to your Classic direction/);
  assert.ok(!CYRILLIC.test(copy[0].description));
  // The description stays monolingual even though the title is localized.
  assert.ok(monolingual(copy[0].description));
});

test('unlocalizable facts fall back to English per string, without inventing words',()=>{
  // Descriptor colors are always normalized family tokens, so the practical
  // per-string fallback surfaces are unknown occasions and unknown concept
  // titles. Both must yield a complete English string, never a ru/en hybrid.
  const uzGarden=contextFor('uz',{occasion:'Garden party'});
  const uzCopy=editorialDeckCopy([looks[0]],uzGarden,[directions[0]]);
  assert.match(uzCopy[0].description,/Closest to your Classic direction/);
  assert.ok(!CYRILLIC.test(uzCopy[0].description));
  assert.ok(monolingual(uzCopy[0].description));
  // A native title outside the concept table keeps its (English) title —
  // monolingual — instead of transliterating or mixing.
  const outerwearPart={top:{slot:'top',subtype:'shirt'},outerwear:{slot:'outerwear',subtype:'blazer'},
    bottom:{slot:'bottom',subtype:'trouser'},shoes:{slot:'shoes',subtype:'loafer'}};
  const fallbackTitle=groundedNativeHeadline('Unmapped concept name',outerwearPart,'ru');
  assert.equal(fallbackTitle,'Unmapped concept name');
  assert.ok(monolingual(fallbackTitle));
});

test('uzbek card headlines stay capitalized, prefer paired nouns over bare slot nouns, and avoid unnatural tones',()=>{
  const uniformColorLooks=[0,1,2].map(n=>({id:'u-'+n,items:[
    garment('t-'+n,'Top item','top','mid'),
    garment('b-'+n,'Bottom item','bottom','earth'),
    garment('s-'+n,'Shoe item','shoes','black'),
  ].map((it,idx)=>({...it,subtype:['shirt','trouser','loafer'][idx]}))}));
  const uzCtx=contextFor('uz',{style:'semi_classic',occasion:'Date night'});
  const cards=editorialDeckCopy(uniformColorLooks,uzCtx,[
    {id:'signature',qualified:true,evidence:{structuralSlots:[],paletteSlots:[]}},
    {id:'refresh',qualified:true,evidence:{structuralSlots:['top'],paletteSlots:[]}},
    {id:'discovery',qualified:true,evidence:{structuralSlots:['top'],paletteSlots:[]}},
  ]);
  for(const card of cards){
    assert.match(card.concept.title,/^[A-ZА-ЯЁ]/,`headline must be capitalized: ${card.concept.title}`);
    assert.ok(!/tuproq tusi|o'rtacha|salqin to'q|beshovt/i.test(card.concept.title),`awkward uzbek in title: ${card.concept.title}`);
    assert.ok(!/tuproq tusi|o'rtacha|salqin to'q|beshovt/i.test(card.description),`awkward uzbek in desc: ${card.description}`);
  }
  const rawDateNightCards=editorialDeckCopy(uniformColorLooks,{language:'uz',style:'semi_classic',occasion:'Date night'},[
    {id:'signature',qualified:true,evidence:{structuralSlots:[],paletteSlots:[]}},
    {id:'refresh',qualified:true,evidence:{structuralSlots:['top'],paletteSlots:[]}},
    {id:'discovery',qualified:true,evidence:{structuralSlots:['top'],paletteSlots:[]}},
  ]);
  for(const card of rawDateNightCards){
    assert.match(card.description,/uchrashuv/i,`expected Uzbek Date night copy, got: ${card.description}`);
    assert.doesNotMatch(card.description,/Closest to|interpretation|follows the tailored/i);
  }
  const bareDeck=editorialDeckCopy(
    [0,1,2].map(n=>({id:'b-'+n,items:[
      garment('t2-'+n,'Cotton shirt','top'),
      garment('b2-'+n,'Tailored trousers','bottom'),
      garment('s2-'+n,'Leather loafers','shoes'),
    ]})),
    uzCtx,
    [
      {id:'signature',qualified:true,evidence:{structuralSlots:[],paletteSlots:[]}},
      {id:'refresh',qualified:true,evidence:{structuralSlots:['top'],paletteSlots:[]}},
      {id:'discovery',qualified:true,evidence:{structuralSlots:['bottom'],paletteSlots:[]}},
    ],
  );
  assert.ok(bareDeck.some(c=>/va /i.test(c.concept.title)),`expected paired headline before bare role noun, got: ${bareDeck.map(c=>c.concept.title).join(', ')}`);
  assert.ok(bareDeck.every(c=>/^[A-ZА-ЯЁ]/.test(c.concept.title)));
});


test('real training tee with stale knit taxonomy gets a grounded full-outfit headline, not Knitwear',()=>{
  const top={id:'training-tee',macroCategory:'top',
    name:'TRAINING TECHNICAL LONG SLEEVE T-SHIRT'};
  const bottom={id:'training-joggers',macroCategory:'bottom',name:'Joggers'};
  const shoes={id:'training-shoes',macroCategory:'shoes',name:'TRAINING TRAINERS'};
  // Simulate the real prepared source snapshot that previously produced the
  // misleading native card "Knitwear" for a technical training T-shirt.
  const prepared=[
    prepareGarment(top,{...describeGarment(top),subtype:'fine-knit'}),
    prepareGarment(bottom,{...describeGarment(bottom),subtype:'athletic-trouser'}),
    prepareGarment(shoes),
  ];
  const sportCtx=stylingContext({style:'casual',occasion:'Sport',optionalLayer:true});
  const cards=editorialDeckCopy([looks[0],{id:'training-card',items:prepared}],sportCtx,[
    {id:'signature',qualified:true,evidence:{structuralSlots:[],paletteSlots:[]}},
    {id:'refresh',qualified:true,evidence:{structuralSlots:['top'],paletteSlots:[]}},
  ]);
  assert.match(cards[1].concept.title,/Tee & Joggers/i,
    'an actual technical shirt and athletic trousers form a real outfit');
  assert.doesNotMatch(cards[1].concept.title,/Knitwear|Sport Sneakers/i);
  assert.equal(describeGarment(prepared[0]).subtype,'fine-knit',
    'fashion admission taxonomy stays untouched; only copy is corrected');
  const realPart={top:{slot:'top',subtype:'t-shirt'},
    bottom:{slot:'bottom',subtype:'athletic-trouser'}};
  assert.equal(groundedNativeHeadline('Knitwear',realPart),'Tee & Joggers');
  assert.equal(groundedNativeHeadline('Knitwear',realPart,'ru'),'Футболка и джоггеры');
  assert.equal(groundedNativeHeadline('Knitwear',realPart,'uz'),'Futbolka va sport shim');
});

test('catalog fashion cards name combinations instead of single shoes or one generic top',()=>{
  const sportCtx=stylingContext({style:'casual',occasion:'Sport',optionalLayer:true});
  const cards=editorialDeckCopy([looks[0],{
    id:'gym-set',items:[
      garment('tank','Tank Vigor 7 slim fit technical fabric','top'),
      garment('joggers','Joggers','bottom'),
      garment('trainers','TRAINING TRAINERS','shoes'),
    ],
  }],sportCtx,[
    {id:'signature',qualified:true,evidence:{structuralSlots:[],paletteSlots:[]}},
    {id:'refresh',qualified:true,evidence:{structuralSlots:['shoes'],paletteSlots:[]}},
  ]);
  assert.match(cards[1].concept.title,/Tank Top & (?:Joggers|Trousers)/i);
  assert.doesNotMatch(cards[1].concept.title,/^(?:Sport Sneakers|Tank Top)$/i);
});

test('preserves Creative Director artDirection, Visual Critic observations, and explicit PersonalStyleDna in Stylist Notes', () => {
  assert.equal(STYLIST_NOTES_VERSION, 's6-editorial-stylist-notes-v1');
  const enrichedLook = {
    id: 'look-editorial-1',
    items: first,
    brief: {
      artDirection: 'Anchor the tailored wool silhouette with crisp white cotton contrast under a structured navy blazer and grounded leather derbies.',
    },
    visualCritique: {
      relationships: [
        {
          kind: 'outfit',
          slots: ['outerwear', 'top', 'bottom'],
          observation: 'The sharp navy blazer lapel frames the white shirt cleanly over charcoal trousers.',
        },
      ],
    },
  };
  const dna = {
    appearance: {
      consentGranted: true,
      skinUndertone: 'cool',
      contrastLevel: 'high',
      colorSeason: 'deep_winter',
      flatteringColors: ['navy', 'white', 'charcoal'],
    },
    body: {
      bodyType: 'rectangle',
      fitPreference: 'fitted',
    },
  };

  const cards = editorialDeckCopy([enrichedLook], context, [directions[0]], { personalStyleDna: dna });
  const card = cards[0];
  assert.match(card.stylistNote, /crisp white cotton contrast/);
  assert.match(card.stylistNote, /sharp navy blazer lapel/);
  assert.match(card.whyItFlattersYou, /cool undertone/i);
  assert.match(card.whyItFlattersYou, /high-contrast/i);
  assert.match(card.whyItFlattersYou, /shoulders/i);
  assert.ok(Array.isArray(card.stylingTips) && card.stylingTips.length >= 2);
  assert.match(card.colorStory, /White.*Navy.*Charcoal/i);
  assert.match(card.silhouetteFormula, /structured blazer over dress shirt/i);
});

test('Stylist Notes reject unverified material/slot fabrications and return null whyItFlattersYou without explicit DNA', () => {
  const unlayered = {
    id: 'look-unlayered',
    items: [
      garment('t1', 'White cotton polo', 'top', 'white'),
      garment('b1', 'Beige cotton chinos', 'bottom', 'beige'),
      garment('s1', 'Brown leather loafers', 'shoes', 'brown'),
    ],
    brief: {
      // Fabricates a cashmere cardigan that does NOT exist in the selected SKUs!
      artDirection: 'Drape a 100% cashmere cardigan over the polo with guaranteed free shipping.',
    },
  };
  const plainCtx = stylingContext({ style: 'casual', occasion: 'Casual', optionalLayer: true });
  const notes = buildStylistNotes(unlayered, plainCtx, { id: 'signature' });
  assert.doesNotMatch(notes.stylistNote, /cashmere|cardigan|free shipping|100%/i);
  assert.equal(notes.whyItFlattersYou, null, 'must never invent flattery/skin/body claims when no DNA/preferences provided');
  assert.ok(notes.stylingTips.length >= 1);
  assert.match(notes.silhouetteFormula, /Polo \+ chinos/i);
});

test('localized ru and uz Stylist Notes remain strictly monolingual without leaking English AI prose', () => {
  const englishBriefLook = {
    ...looks[0],
    brief: {
      artDirection: 'Create an elongated clean shoulder line through tailored navy cloth and crisp shirt contrast.',
    },
  };
  const dna = {
    appearance: { consentGranted: true, skinUndertone: 'cool', flatteringColors: ['navy', 'white'] },
    body: { bodyType: 'rectangle', fitPreference: 'balanced' },
  };
  for (const language of ['ru', 'uz']) {
    const cards = editorialDeckCopy([englishBriefLook], contextFor(language), [directions[0]], { personalStyleDna: dna });
    const c = cards[0];
    assert.ok(monolingual(c.stylistNote), `mixed script stylistNote (${language}): ${c.stylistNote}`);
    assert.ok(monolingual(c.whyItFlattersYou), `mixed script whyItFlattersYou (${language}): ${c.whyItFlattersYou}`);
    assert.ok(monolingual(c.colorStory), `mixed script colorStory (${language}): ${c.colorStory}`);
    assert.ok(monolingual(c.silhouetteFormula), `mixed script silhouetteFormula (${language}): ${c.silhouetteFormula}`);
    assert.ok(c.stylingTips.length >= 1);
    for (const tip of c.stylingTips) {
      assert.ok(monolingual(tip), `mixed script stylingTip (${language}): ${tip}`);
    }
    if (language === 'ru') {
      assert.ok(!LATIN.test(c.stylistNote));
      assert.ok(CYRILLIC.test(c.stylistNote));
    } else {
      assert.ok(!CYRILLIC.test(c.stylistNote));
      assert.ok(LATIN.test(c.stylistNote));
    }
  }
});

test('Phase 10 Defects C & D: Date Night never emits "office" headlines and mixed orange/black palettes never claim "warm earth harmony"', () => {
  const dateLook = {
    id: 'look-date-1',
    items: [
      { id: 't1', macroCategory: 'top', subtype: 'dress-shirt', color: 'white', name: 'White Poplin Dress Shirt' },
      { id: 'o1', macroCategory: 'outerwear', subtype: 'blazer', color: 'navy', name: 'Navy Wool Blazer' },
      { id: 'b1', macroCategory: 'bottom', subtype: 'tailored-trouser', color: 'charcoal', name: 'Charcoal Suit Trousers' },
      { id: 's1', macroCategory: 'shoes', subtype: 'loafer', color: 'black', name: 'Black Calf Loafers' },
    ],
  };

  for (const language of ['en', 'ru', 'uz']) {
    const cards = editorialDeckCopy(
      [dateLook],
      { style: 'classic', occasion: 'Date Night', dressCode: 'smart', language },
      [{ id: 'signature', qualified: true }],
    );
    assert.ok(!/\boffice\b|офис|ofis/i.test(cards[0].concept.title),
      `Date Night must never emit an office headline (${language}): ${cards[0].concept.title}`);
  }
  assert.equal(groundedNativeHeadline('Precise office tailoring', {}, 'en', 'Date Night'), 'Evening tailoring');
  assert.equal(groundedNativeHeadline('Precise office tailoring', {}, 'en', 'Casual'), 'Refined tailoring');
  assert.equal(groundedNativeHeadline('Precise office tailoring', {}, 'en', 'Work'), 'Precise office tailoring');

  // Defect D: Grey + Brown + Orange + Black must NEVER claim "warm earth harmony"
  const mixedLook = {
    id: 'look-mixed-1',
    items: [
      { id: 't2', macroCategory: 'top', subtype: 'fine-knit', color: 'grey', name: 'Grey Knit Polo' },
      { id: 'o2', macroCategory: 'outerwear', subtype: 'blazer', color: 'brown', name: 'Brown Linen Blazer' },
      { id: 'b2', macroCategory: 'bottom', subtype: 'chino', color: 'orange', name: 'Caramel Orange Moleskin Trousers' },
      { id: 's2', macroCategory: 'shoes', subtype: 'oxford', color: 'black', name: 'Executive Cap-Toe Oxford Black' },
    ],
  };
  const notes = buildStylistNotes(mixedLook, { style: 'old_money', occasion: 'Work', language: 'en' }, { id: 'signature' });
  assert.ok(!/warm earth harmony/i.test(notes.colorStory),
    `Grey/Brown/Orange/Black must not be labeled warm earth harmony: ${notes.colorStory}`);

  // True earth palette (Cream + Brown + Camel + Brown) SHOULD still earn warm earth harmony
  const trueEarthLook = {
    id: 'look-earth-1',
    items: [
      { id: 't3', macroCategory: 'top', subtype: 'fine-knit', color: 'cream', name: 'Cream Merino Crewneck' },
      { id: 'o3', macroCategory: 'outerwear', subtype: 'blazer', color: 'brown', name: 'Espresso Wool Blazer' },
      { id: 'b3', macroCategory: 'bottom', subtype: 'tailored-trouser', color: 'camel', name: 'Camel Flannel Trousers' },
      { id: 's3', macroCategory: 'shoes', subtype: 'loafer', color: 'brown', name: 'Dark Brown Suede Loafers' },
    ],
  };
  const earthNotes = buildStylistNotes(trueEarthLook, { style: 'old_money', occasion: 'Work', language: 'en' }, { id: 'signature' });
  assert.match(earthNotes.colorStory, /warm earth harmony|crisp light-dark contrast/i);
});

test('Phase 11 editorial voice: preserves bespoke VLM titles & warm notes, strips leaked rule jargon, and corrects misdescribed garment colors', () => {
  const navyLoaferItems = [
    { id: 't1', macroCategory: 'top', subtype: 'dress-shirt', color: 'white', name: 'White Poplin Dress Shirt' },
    { id: 'o1', macroCategory: 'outerwear', subtype: 'blazer', color: 'navy', name: 'Navy Wool Soft Blazer' },
    { id: 'b1', macroCategory: 'bottom', subtype: 'tailored-trouser', color: 'charcoal', name: 'Charcoal Flannel Tailored Trousers' },
    { id: 's1', macroCategory: 'shoes', subtype: 'loafer', color: 'navy', name: 'Navy Suede Penny Loafers' },
  ];

  // 1. Exact real-world regression from ios_sim_v4_work_notes.png:
  const leaked = 'Clean navy-to-white-to-charcoal progression with black loafers matching the cool charcoal trousers per shoe-to-palette temperature rules (S2, SKU tier 1 heuristic rubric).';
  const cleaned = sanitizeStylistProse(leaked, navyLoaferItems);
  assert.doesNotMatch(cleaned, /shoe-to-palette|temperature rules|heuristic|rubric|SKU|tier 1|\bS2\b/i, `jargon must be stripped: ${cleaned}`);
  assert.match(cleaned, /\bnavy loafers\b/i, `misdescribed black loafers must be corrected to navy loafers: ${cleaned}`);
  assert.doesNotMatch(cleaned, /\bblack loafers\b/i);

  // 2. Preserves rich natural color synonyms without breaking human voice:
  const warmItems = [
    { id: 't2', macroCategory: 'top', subtype: 'fine-knit', color: 'cream', name: 'Cream Merino Turtleneck' },
    { id: 'o2', macroCategory: 'outerwear', subtype: 'jacket', color: 'brown', name: 'Brown Suede Blouson Jacket' },
    { id: 'b2', macroCategory: 'bottom', subtype: 'tailored-trouser', color: 'charcoal', name: 'Charcoal Wool Trousers' },
    { id: 's2', macroCategory: 'shoes', subtype: 'loafer', color: 'brown', name: 'Brown Suede Loafers' },
  ];
  const richProse = 'An ivory turtleneck softens the espresso suede jacket while chocolate loafers anchor the charcoal trousers.';
  assert.equal(sanitizeStylistProse(richProse, warmItems), richProse);

  // 3. Preserves bespoke evocative titles & human Stylist Notes instead of "Clean separates"
  const autonomousLook = {
    id: 'auto-1',
    items: navyLoaferItems,
    autonomousMeta: {
      title: 'Midnight Soft Tailoring',
      stylistNote: 'An unstructured navy wool blazer over a crisp white shirt brings quiet confidence to evening plans, anchored by navy suede loafers and charcoal flannel trousers.',
      whyItFlattersYou: 'The open blazer front creates an elongated vertical line while keeping the silhouette relaxed yet sharp.',
      stylingTips: [
        'Leave the top shirt button undone for effortless evening polish.',
        'Keep the blazer unfastened in motion so the crisp white collar leads.',
      ],
    },
  };
  const cards = editorialDeckCopy(
    [autonomousLook],
    { style: 'classic', occasion: 'Date night', language: 'en' },
    [{ id: 'signature', qualified: true }],
  );
  assert.equal(cards[0].concept.title, 'Midnight Soft Tailoring');
  assert.doesNotMatch(cards[0].concept.title, /Clean separates/i);
  assert.match(cards[0].stylistNote, /unstructured navy wool blazer/i);
  assert.match(cards[0].whyItFlattersYou, /elongated vertical line/i);
  assert.equal(cards[0].stylingTips.length, 2);
  assert.match(cards[0].stylingTips[0], /top shirt button undone/i);

  // 4. Rejects unconsented undertone fabrication in bespoke whyItFlattersYou
  const unconsentedLook = {
    id: 'auto-2',
    items: navyLoaferItems,
    autonomousMeta: {
      title: 'Evening Navy Depth',
      stylistNote: 'Refined navy tailoring paired with crisp white shirting and charcoal trousers.',
      whyItFlattersYou: 'These deep tones flatter your warm skin undertone and complexion.',
    },
  };
  const unconsentedNotes = buildStylistNotes(
    unconsentedLook,
    { style: 'classic', occasion: 'Date night', language: 'en' },
    { id: 'signature' },
  );
  assert.equal(unconsentedNotes.whyItFlattersYou, null, 'must never allow unconsented undertone claims');

  // 5. Rejects impossible "top button" advice on buttonless pullovers, while keeping blazer unbuttoning tips
  const crewneckItems = [
    { id: 't3', macroCategory: 'top', subtype: 'fine-knit', color: 'burgundy', name: 'Regular-Fit Rib-Knit Cotton Sweater' },
    { id: 'o3', macroCategory: 'outerwear', subtype: 'blazer', color: 'burgundy', name: 'Tailored Fit Velvet Single-Breasted Blazer' },
    { id: 'b3', macroCategory: 'bottom', subtype: 'tailored-trouser', color: 'charcoal', name: 'Charcoal Flannel Trousers' },
    { id: 's3', macroCategory: 'shoes', subtype: 'oxford', color: 'navy', name: 'Vincent Navy Wholecut Oxford' },
  ];
  const crewneckNotes = buildStylistNotes(
    {
      items: crewneckItems,
      autonomousMeta: {
        title: 'Burgundy Velvet Evening',
        stylistNote: 'The deep burgundy blazer frames crisp tailoring with subtle textural depth.',
        stylingTips: [
          'Keep the top button of the fine knit undone for a relaxed neckline.',
          'Wear the blazer unbuttoned for an easy evening drape.',
          'Polish the navy oxfords before heading out.',
        ],
      },
    },
    { style: 'classic', occasion: 'Date night', language: 'en' },
    { id: 'signature' },
  );
  assert.equal(
    crewneckNotes.stylingTips.some((t) => /top button/i.test(t)),
    false,
    'must reject button advice on buttonless crewneck sweater',
  );
  assert.equal(
    crewneckNotes.stylingTips.some((t) => /blazer unbuttoned/i.test(t)),
    true,
    'must preserve valid blazer unbuttoning tip',
  );

  // 6. Rejects mismatched title color ("Charcoal" on all-navy) and "Evening" on daytime Work
  const allNavyCardiganItems = [
    { id: 't4', macroCategory: 'top', subtype: 'dress-shirt', color: 'navy', name: 'Navy Poplin Shirt' },
    { id: 'o4', macroCategory: 'outerwear', subtype: 'cardigan', color: 'navy', name: 'Macardomon Cardigan Navy' },
    { id: 'b4', macroCategory: 'bottom', subtype: 'tailored-trouser', color: 'navy', name: 'Cold wool suit trousers' },
    { id: 's4', macroCategory: 'shoes', subtype: 'loafer', color: 'brown', name: 'Velasca Brown Loafers' },
  ];
  const beigeCoatItems = [
    { id: 't5', macroCategory: 'top', subtype: 'dress-shirt', color: 'beige', name: 'Slim-Fit Stretch Shirt' },
    { id: 'o5', macroCategory: 'outerwear', subtype: 'coat', color: 'beige', name: 'Light Beige Overcoat' },
    { id: 'b5', macroCategory: 'bottom', subtype: 'tailored-trouser', color: 'beige', name: 'Machino Trousers' },
    { id: 's5', macroCategory: 'shoes', subtype: 'oxford', color: 'black', name: 'TLB Black Oxford' },
  ];
  const workCards = editorialDeckCopy(
    [
      { items: allNavyCardiganItems, autonomousMeta: { title: 'Charcoal Knit Afternoon', stylistNote: 'Cohesive navy layers tailored for quiet daytime confidence.' } },
      { items: beigeCoatItems, autonomousMeta: { title: 'Beige Coat Evening', stylistNote: 'Warm tonal overcoat and beige tailoring anchored by sleek black oxfords.' } },
    ],
    { style: 'old_money', occasion: 'Work', language: 'en' },
    [{ id: 'signature', qualified: true }, { id: 'refresh', qualified: true }],
  );
  assert.doesNotMatch(workCards[0].concept.title, /\bCharcoal\b/i, 'must not keep "Charcoal" title when no item is charcoal');
  assert.doesNotMatch(workCards[1].concept.title, /\bEvening\b/i, 'must not keep "Evening" in a daytime Work title');

  // 7. Phase 12b: dedupes repeated title words, never truncates mid-brand on "J.Crew", and rejects prompt-echo tips
  const jcrewItems = [
    { id: 't6', macroCategory: 'top', subtype: 'dress-shirt', color: 'white', name: 'White Oxford Shirt' },
    { id: 'o6', macroCategory: 'outerwear', subtype: 'blazer', color: 'burgundy', name: 'Tailored Fit Velvet Single-Breasted Blazer' },
    { id: 'b6', macroCategory: 'bottom', subtype: 'tailored-trouser', color: 'navy', name: 'Kenmare Relaxed suit pant in Italian wool' },
    { id: 's6', macroCategory: 'shoes', subtype: 'oxford', color: 'black', name: 'Black Leather Wholecut Oxford' },
  ];
  const jcrewNote = 'A crisp white Oxford shirt layered cleanly beneath a structured velvet evening blazer, worn with rich dark trousers and dark brown footwear anchored comfortably by J.Crew navy Italian wool tailoring and black oxfords for a sharp work silhouette.';
  const sanitizedJcrew = sanitizeStylistProse(jcrewNote, jcrewItems, { maxLen: 215 });
  assert.doesNotMatch(sanitizedJcrew, /\bJ\.$/, `must never cut off mid-brand after "J.": ${sanitizedJcrew}`);

  const phase12bCards = editorialDeckCopy(
    [{
      items: jcrewItems,
      autonomousMeta: {
        title: 'Burgundy Velvet Velvet Hour',
        stylistNote: 'The velvet evening blazer pairs cleanly with crisp white shirting, avoiding any double-knit bulk.',
        stylingTips: [
          'The tailored trouser waistband stays clean for evening, unlike drawstring joggers.',
          'Wear the navy knit as photographed; do not unbutton a crewneck.',
          'Keep the velvet blazer unbuttoned so the crisp collar leads.',
        ],
      },
    }],
    { style: 'classic', occasion: 'Date night', language: 'en' },
    [{ id: 'signature', qualified: true }],
  );
  assert.equal(phase12bCards[0].concept.title, 'Burgundy Velvet Hour');
  assert.doesNotMatch(phase12bCards[0].stylistNote, /double-knit bulk/i);
  assert.equal(phase12bCards[0].stylingTips.some((t) => /do not unbutton|as photographed|unlike drawstring/i.test(t)), false);
  assert.equal(phase12bCards[0].stylingTips.some((t) => /Keep the velvet blazer unbuttoned/i.test(t)), true);

  // 8. Phase 12c: fixes "navy cool wool blazer" + "structured navy silhouette" against black blazer,
  // strips internal "locked <garment>" swap echo, and catches "Buttonless Polo" for top-button tips.
  const buttonlessPoloItems = [
    { id: 't7', macroCategory: 'top', subtype: 'polo', color: 'light-grey', name: 'Extrafine Merino Buttonless Polo Classic' },
    { id: 'o7', macroCategory: 'outerwear', subtype: 'blazer', color: 'black', name: '100% cool wool suit blazer' },
    { id: 'b7', macroCategory: 'bottom', subtype: 'tailored-trouser', color: 'charcoal', name: 'Milan slim-fit suit trousers' },
    { id: 's7', macroCategory: 'shoes', subtype: 'oxford', color: 'brown', name: '514150 - BROWN SUPERBUCK - E' },
  ];
  const buttonlessCards = editorialDeckCopy(
    [{
      items: buttonlessPoloItems,
      autonomousMeta: {
        title: 'Blazer & Tailored Trousers',
        stylistNote: 'This navy cool wool blazer instantly elevates your work wardrobe.',
        whyItFlattersYou: 'The structured navy silhouette sharpens your frame while harmonizing with the locked grey trousers and brown shoes.',
        stylingTips: [
          'Wear it open over the locked light-gray polo for a refined work look.',
          'Pair with the locked brown leather shoes to complete the aesthetic.',
          'Keep the top button undone for a modern, effortless silhouette.',
        ],
      },
    }],
    { style: 'old_money', occasion: 'Work', language: 'en' },
    [{ id: 'signature', qualified: true }],
    { personalStyleDna: { body: { bodyType: 'rectangle', hasExplicitBodyEvidence: true } } },
  );
  assert.match(buttonlessCards[0].stylistNote, /\bblack cool wool blazer\b/i);
  assert.match(buttonlessCards[0].whyItFlattersYou, /\bstructured black silhouette\b/i);
  assert.doesNotMatch(`${buttonlessCards[0].whyItFlattersYou} ${buttonlessCards[0].stylingTips.join(' ')}`, /\blocked\b/i);
  assert.equal(buttonlessCards[0].stylingTips.some((t) => /top button/i.test(t)), false);

  // 9. Phase 13 taxonomy & grounding fixes: overcoat vs stale overshirt, cardigan fine-knit note preservation,
  // wholecut shoe fallback, bare single-item title rejection, and Workwear -> Workday / work tailoring.
  const overcoatStaleItems = [
    { id: 't8', macroCategory: 'top', subtype: 'shirt', color: 'cream', name: 'Ivory Corduroy Shirt' },
    { id: 'o8', macroCategory: 'outerwear', subtype: 'overshirt', color: 'beige', name: 'Light Beige Overcoat' },
    { id: 'b8', macroCategory: 'bottom', subtype: 'chino', color: 'tan', name: 'The Performance Chino | Uniform | Toasted Tan' },
    { id: 's8', macroCategory: 'shoes', subtype: 'unknown', color: 'black', name: 'Rio Wholecut W - Black' },
  ];
  const overcoatCards = editorialDeckCopy(
    [
      {
        items: overcoatStaleItems,
        autonomousMeta: {
          title: 'White Cotton Dress Shirt',
          stylistNote: 'The light beige overcoat drapes cleanly over the ivory corduroy shirt and sleek black wholecuts for refined workwear.',
        },
      },
      {
        items: [
          { id: 't9', macroCategory: 'top', subtype: 'dress-shirt', color: 'white', name: 'Relaxed Pima Cotton-Poplin Shirt - White' },
          { id: 'o9', macroCategory: 'outerwear', subtype: 'cardigan', color: 'navy', name: 'Nuvola navy blue seamless merino wool cardigan' },
          { id: 'b9', macroCategory: 'bottom', subtype: 'tailored-trouser', color: 'navy', name: 'Regular-Fit Suit Pants' },
          { id: 's9', macroCategory: 'shoes', subtype: 'oxford', color: 'brown', name: 'Caffettee' },
        ],
        autonomousMeta: {
          title: 'White Cotton Poplin Workwear',
          stylistNote: 'A fine-knit merino wool cardigan layered over crisp white shirting creates quiet Ivy poise.',
        },
      },
    ],
    { style: 'old_money', occasion: 'Work', language: 'en' },
    [{ id: 'signature', qualified: true }, { id: 'refresh', qualified: true }],
  );
  assert.doesNotMatch(overcoatCards[0].concept.title, /^White Cotton Dress Shirt$/i);
  assert.match(overcoatCards[0].silhouetteFormula, /\bCoat over\b/i);
  assert.doesNotMatch(overcoatCards[0].silhouetteFormula, /\bOvershirt\b/i);
  assert.match(overcoatCards[0].silhouetteFormula, /\bsleek oxfords\b/i);
  assert.match(overcoatCards[0].stylistNote, /\blight beige overcoat\b/i);
  assert.match(overcoatCards[0].stylistNote, /\bwork tailoring\b/i);
  assert.equal(overcoatCards[1].concept.title, 'White Cotton Poplin Workday');
  assert.match(overcoatCards[1].stylistNote, /\bfine-knit merino wool cardigan\b/i);

  // Rejects "sweater unbuttoned at the collar" on buttonless mockneck & 8-word inventory-list titles
  const mockneckCards = editorialDeckCopy(
    [{
      items: [
        { id: 't10', macroCategory: 'top', subtype: 'fine-knit', color: 'burgundy', name: 'Extrafine Merino Mockneck Burgundy' },
        { id: 'o10', macroCategory: 'outerwear', subtype: 'blazer', color: 'burgundy', name: 'Tailored Fit Velvet Single-Breasted Blazer' },
        { id: 'b10', macroCategory: 'bottom', subtype: 'tailored-trouser', color: 'charcoal', name: 'Italian wool flannel slim-fit suit trousers' },
        { id: 's10', macroCategory: 'shoes', subtype: 'loafer', color: 'black', name: 'Black Shell Cordovan Penny Loafer' },
      ],
      autonomousMeta: {
        title: 'Burgundy Blazer Burgundy Knit Charcoal Trousers Loafers',
        stylistNote: 'Deep tonal velvet and merino layering refined for evening.',
        stylingTips: [
          'Keep the burgundy sweater unbuttoned at the collar for a relaxed feel.',
          'Leave the velvet blazer unfastened for easy drape.',
        ],
      },
    }],
    { style: 'classic', occasion: 'Date night', language: 'en' },
    [{ id: 'signature', qualified: true }],
  );
  assert.doesNotMatch(mockneckCards[0].concept.title, /Charcoal Trousers Loafers/i);
  assert.equal(mockneckCards[0].stylingTips.some((t) => /unbuttoned/i.test(t)), false);
  assert.equal(mockneckCards[0].stylingTips.some((t) => /velvet blazer unfastened/i.test(t)), true);
});
