import test from 'node:test';
import assert from 'node:assert/strict';
import { editorialDeckCopy, DECK_COPY_VERSION, groundedNativeHeadline } from './deck_editorial_copy.mjs';
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
