import test from 'node:test';
import assert from 'node:assert/strict';
import { editorialDeckCopy, DECK_COPY_VERSION, groundedNativeHeadline } from './deck_editorial_copy.mjs';
import { stylingContext } from '../../src/domain/stylingContext.ts';
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
  assert.equal(DECK_COPY_VERSION,'s6-grounded-deck-copy-v3');
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
  assert.match(descriptions[1],/shifts from White to Cream/i);
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
