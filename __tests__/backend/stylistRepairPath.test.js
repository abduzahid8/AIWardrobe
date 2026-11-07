/** @jest-environment node */
const fs = require('fs'), path = require('path');
const { prepareStylistResponse } = require('../../features/outfit-generator/utils/stylistResponse');
const { assembleDiverseDeck } = require('../../src/domain/outfitDiversity');
const { passesStyleJudge } = require('../../src/domain/occasionRules');
const { fourSlotOutfitReason } = require('../../src/domain/outfitStructure');
const harness = fs.readFileSync(path.join(__dirname, 'outfitBudgetWire.test.js'), 'utf8').split('const owned=')[0];
function loadHandler(entry) {
  const source = harness.replace(/const entry=.*?;/, `const entry=${JSON.stringify(entry)};`);
  return new Function('require', '__dirname', 'jest', source + '\nreturn load;')(require, __dirname, jest);
}
const current = path.resolve(__dirname, '../../backend/supabase/functions/generate-outfits/index.ts');
const makeLook = (names, suffix = '') => ({items: names.map((name, i) => ({id: `shop_${i}${suffix}`, name,
  macroCategory: ['top','outerwear','bottom','shoes'][i], type: ['shirt','blazer','trousers','shoes'][i],
  imageUrl: `https://retailer.invalid/${i}${suffix}.jpg`, isShopItem: true, isWardrobe: false,
  price: 100, currency: 'USD', shopUrl: `https://retailer.invalid/products/${i}${suffix}`}))});
const strict = [
  ['White poplin dress shirt','Navy unlined wool blazer','Grey tailored wool trousers','Black leather Oxford shoes'],
  ['Blue cotton dress shirt','Beige linen blazer','Navy pleated trousers','Brown leather derby shoes'],
  ['White striped cotton shirt','Grey wool blazer','Black suit trousers','Burgundy leather monk shoes'],
].map((names,i) => makeLook(names, `-${i}`));
const legacy = makeLook(['White cotton shirt','Navy wool blazer','Beige cotton chinos','Black leather Oxford shoes'], '-legacy');
const options = style => ({style, occasion:'Formal', source:'shop', accepts: items => items.reduce((n,i)=>n+i.price,0)<=500});
function display(prepared, inventory = strict.flatMap(o=>o.items)) {
  const candidates = [...prepared.accepted, ...prepared.recovered];
  return assembleDiverseDeck({candidates, itemsOf:o=>o.items, qualityOf:o=>o.quality||0.9,
    target:3, usedKeys:new Set(), poolItems:inventory, activeSlots:['top','outerwear','bottom','shoes']}).deck;
}

test.each(['classic','casual'])('%s + Formal recovers an old successful but client-invalid deck using authentic catalog garments', style => {
  const prepared = prepareStylistResponse([legacy], strict.flatMap(o=>o.items), options(style));
  expect(prepared.accepted).toHaveLength(0);
  expect(Object.keys(prepared.rejections).join(' ')).toContain('formal_requires_tailored_bottom');
  const deck = display(prepared);
  expect(deck).toHaveLength(3);
  for (const look of deck) {
    expect(fourSlotOutfitReason(look.items)).toBeNull();
    expect(passesStyleJudge(look.items,style,'Formal')).toBe(true);
    for (const item of look.items) {
      expect(item).toMatchObject([...strict.flatMap(o=>o.items),...legacy.items].find(row=>row.id===item.id));
      expect(item.shopUrl).toMatch(/^https:\/\/retailer.invalid\/products\//);
    }
  }
});

test('a genuinely insufficient formal pool stays empty and reports the rejected stage', () => {
  const prepared = prepareStylistResponse([legacy], [], options('classic'));
  expect(prepared.accepted).toEqual([]);expect(prepared.recovered).toEqual([]);
  expect(Object.keys(prepared.rejections).length).toBeGreaterThan(0);
});

test('recovery retains source, locked anchor, preferences and verified budget', () => {
  const inventory = strict.flatMap(o=>o.items);
  const prepared = prepareStylistResponse([legacy], inventory, {...options('classic'), anchorId:strict[0].items[0].id,
    preferences:{avoidColors:['black']}});
  const looks = [...prepared.accepted,...prepared.recovered];expect(looks.length).toBeGreaterThan(0);
  for(const look of looks){expect(look.items.some(i=>i.id===strict[0].items[0].id)).toBe(true);expect(look.items.every(i=>!i.name.includes('Black'))).toBe(true);}
  expect(prepareStylistResponse([legacy], inventory, {...options('classic'),accepts:()=>false}).recovered).toEqual([]);
  expect(prepareStylistResponse([legacy], inventory, {...options('classic'),source:'wardrobe'}).recovered).toEqual([]);
});

const entries = [['current', current], ...(process.env.ACTIVE_STYLIST_SOURCE ? [['active-v179', process.env.ACTIVE_STYLIST_SOURCE]] : []), ...(process.env.LIVE_STYLIST_CATALOG ? [['live-catalog', current]] : [])];
test.each(entries)('%s Edge handler → repair → current client validation → budget → diversity produces three Formal looks', async (version, entry) => {
  const rows = version === 'live-catalog' ? Object.entries(JSON.parse(fs.readFileSync(process.env.LIVE_STYLIST_CATALOG,'utf8'))).flatMap(([slot, rows])=>rows.map(row=>({...row,id:'shop_'+row.id,macroCategory:slot,type:row.garment_type,shopUrl:row.source_url,imageUrl:row.image_url,isShopItem:true,isWardrobe:false}))) : version === 'current' ? strict.flatMap(o=>o.items) : Array.from({length:3},(_,i)=>makeLook([
    ['White cotton shirt','Blue cotton shirt','Cream cotton shirt'][i],
    ['Navy wool blazer','Grey wool blazer','Beige linen blazer'][i],
    ['Beige cotton chinos','Navy cotton chinos','Grey cotton chinos'][i],
    ['Black leather Oxford shoes','Brown leather derby shoes','Burgundy leather monk shoes'][i],
  ],`-legacy${i}`)).flatMap(o=>o.items);
  const load = loadHandler(entry), proposals = Array.from({length:3},(_,i)=>rows.slice(i*4,i*4+4).map(row=>row.id));
  const catalogRows = rows.map(item=>({...item,id:item.id.replace(/^shop_/,''),
    category:{top:'tops',bottom:'bottoms',shoes:'shoes',outerwear:'outerwear'}[item.macroCategory],
    garment_type:{top:'upper_body',bottom:'lower_body',shoes:'shoes',outerwear:'outerwear'}[item.macroCategory],
    source_url:item.shopUrl,image_url:item.imageUrl}));
  let slot;
  const q = {};
  for(const method of ['eq','order','limit','range','gt','lte','abortSignal'])q[method]=()=>q;
  q.select=()=>{slot=undefined;return q;};
  q.or=filter=>{slot=filter.includes('category.eq.shoes')?'shoes':filter.includes('category.eq.outerwear')?'outerwear':filter.includes('category.eq.tops')?'top':'bottom';return q;};
  q.then=resolve=>Promise.resolve({data:catalogRows.filter(row=>!slot||row.macroCategory===slot),error:null}).then(resolve);
  const {handler} = load(proposals,null,null,null,q);
  const response = await handler(new Request('https://fixture.invalid',{method:'POST',headers:{Authorization:'Bearer fixture-user'},
    body:JSON.stringify({stylePreferences:'classic',occasion:'Formal',prompt:'Formal',wardrobeItems:rows,
      useProvidedWardrobeOnly:true,catalogOnly:true,limit:5,deckTarget:3,preferences:version==='live-catalog'?{}:{maxShopBudget:{amount:500,currency:'USD'}}})}));
  const body = await response.json();if(response.status!==200)throw new Error(JSON.stringify(body));expect(response.status).toBe(200);expect(body.outfits).toHaveLength(3);
  const inventory = version==='live-catalog' ? rows : strict.flatMap(o=>o.items);
  const prepared = prepareStylistResponse(body.outfits, inventory, {...options('classic'),...(version==='live-catalog'?{accepts:undefined}:{})});
  if(version==='active-v179')expect(prepared.accepted).toHaveLength(0);
  const deck = display(prepared,inventory);expect(deck).toHaveLength(3);
  if(version==='live-catalog'){fs.writeFileSync('/tmp/aiwardrobe-live-formal-smoke.json',JSON.stringify({status:response.status,outfits:deck.map(look=>({items:look.items.map(({id,name,price,currency,shopUrl})=>({id,name,price,currency,shopUrl}))})),rejections:prepared.rejections},null,2));}
  expect(deck.every(look=>passesStyleJudge(look.items,'classic','Formal')&&!fourSlotOutfitReason(look.items))).toBe(true);
});
