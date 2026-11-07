import { stylingCandidateRepresentatives } from '../../src/domain/catalogCandidates';
import { describeGarment } from '../../src/domain/garmentDescriptor';
import { outfitStyleIntent, stylingFormulas } from '../../src/domain/stylingProfiles';
import { stylingContext } from '../../src/domain/stylingContext';
import { outfitStylingDistance } from '../../src/domain/outfitVisualSignature';
import { assembleDiverseDeck } from '../../src/domain/outfitDiversity';

const slots = ['top','outerwear','bottom','shoes'];
const look = (prefix: string, names: string[], quality = 0.9, rank = 0) => ({quality, rank,
  items: names.map((name,i)=>({id:`${prefix}-${i}`,name,macroCategory:slots[i],isShopItem:true,imageUrl:`https://test.invalid/${prefix}-${i}.jpg`}))});
const tailored = ['White regular poplin dress shirt','Navy wool blazer','Grey straight tailored trousers','Black leather Oxford shoes'];
const ivy = ['Cream regular polo','Brown fine knit cardigan','Beige straight chinos','Brown suede loafers'];
const casual = ['Plain white cotton T-shirt','Olive cotton overshirt','Blue straight denim jeans','White minimal leather sneakers'];

test('different style identities prefer their complete relationships',()=>{
  const descriptors = (names: string[]) => look('x',names).items.map(describeGarment);
  expect(stylingFormulas(stylingContext({style:'classic'}))[0].id).toBe('tailored');
  expect(stylingFormulas(stylingContext({style:'old_money'}))[0].id).toBe('ivy-knit');
  expect(stylingFormulas(stylingContext({style:'semi_classic'}))[0].id).toBe('modern-tailoring');
  expect(outfitStyleIntent(descriptors(ivy),stylingContext({style:'old_money'})).score)
    .toBeGreaterThan(outfitStyleIntent(descriptors(ivy),stylingContext({style:'classic'})).score);
  expect(outfitStyleIntent(descriptors(tailored),stylingContext({style:'classic'})).score)
    .toBeGreaterThan(outfitStyleIntent(descriptors(tailored),stylingContext({style:'old_money'})).score);
  expect(outfitStyleIntent(descriptors(ivy),stylingContext({style:'old_money'})).score)
    .toBeGreaterThan(outfitStyleIntent(descriptors(casual),stylingContext({style:'old_money'})).score);
  expect(outfitStyleIntent(descriptors(casual),stylingContext({style:'casual'})).score)
    .toBeGreaterThan(outfitStyleIntent(descriptors(tailored),stylingContext({style:'casual'})).score);
});
test('retrieved representatives spread styling evidence without buying weak candidates',()=>{
  const rows = Array.from({length:12},(_,i)=>({id:`${i}`,macroCategory:'outerwear',name:'Navy regular wool blazer',quality:0.9}));
  rows.push({id:'knit',macroCategory:'outerwear',name:'Beige fine knit cardigan',quality:0.88},
    {id:'weak',macroCategory:'outerwear',name:'Green casual bomber',quality:0.3});
  const picks = stylingCandidateRepresentatives(rows,i=>i,i=>i.quality,stylingContext({style:'old_money'}),4);
  expect(picks.some(p=>p.id==='knit')).toBe(true);
  expect(picks.some(p=>p.id==='weak')).toBe(false);
});
test('unknown and studio background metadata earn no fabricated visual evidence',()=>{
  const d = describeGarment({name:'Shirt',description:'Product photo on a white background'});
  expect(d.color).toBe('unknown');
  expect(d.volume).toBe('unknown');
  expect(describeGarment({name:'Shirt',fit:'regular',material:['cotton']}).volume).toBe('regular');
  expect(outfitStylingDistance(look('a',['Shirt','Layer','Trousers','Shoes']).items,
    look('b',['Shirt','Layer','Trousers','Shoes']).items)).toBe(0);
});
test('final selection values styling changes over IDs and provider rank within quality band',()=>{
  const candidates=[look('a',tailored,0.9,0),look('b',tailored,0.9,1),look('c',tailored,0.9,2),
    look('d',ivy,0.9,3),look('e',casual,0.9,4)];
  const select = (rows: typeof candidates) => assembleDiverseDeck({candidates:rows,itemsOf:l=>l.items,
    qualityOf:l=>l.quality,rankOf:l=>l.rank,stylingDistance:(a,b)=>outfitStylingDistance(a.items,b.items),
    usedKeys:new Set<string>(),poolItems:rows.flatMap(l=>l.items),target:3,maxSearchSteps:512}).deck;
  const deck=select(candidates);
  expect(deck).toHaveLength(3);
  expect(deck.some(l=>l===candidates[3])).toBe(true);
  expect(deck.some(l=>l===candidates[4])).toBe(true);
  const weak=look('weak',casual,0.3,0);
  expect(select([...candidates.slice(0,3),weak])).not.toContain(weak);
});
