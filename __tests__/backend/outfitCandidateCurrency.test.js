/** @jest-environment node */
const {referenceFx}=require('../fixtures/budgetFx');
const fs = require('fs');
const path = require('path');
const ts = require('typescript');
const filename = process.env.OUTFIT_CANDIDATE_SOURCE || path.resolve(__dirname, '../../backend/supabase/functions/generate-outfits/index.ts');
let source = fs.readFileSync(filename, 'utf8');
if (process.env.OUTFIT_SIMULATE_MISSING_CURRENCY === '1') source = source.replace(/\s*currency: row.currency,/g, '');
const compiled = ts.transpileModule(source, {compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
const api = new Function('require','exports','Deno',compiled+'\nreturn {fetchShopItemsForAllSlotsEdge,budgetInCurrencyEdge,inferShopColor};')(
 name=>{
  if(name.includes('styleInference')) return require('../../src/domain/styleInference');
  if(name.includes('outfitDiversity')) return require('../../src/domain/outfitDiversity');
  if(name.includes('outfitRotation')) return require('../../src/domain/outfitRotation');
  if(name.includes('catalogIdentity')) return require('../../src/domain/catalogIdentity');
  if(name.includes('budgetFx.ts')) return require('../../src/domain/budgetFx');
  if(name.includes('server.ts')) return {serve:()=>{}};
  for(const module of ['occasionRules','outfitStructure','garmentContext','stylingContext','editorialIntelligence','outfitConcepts','visualFashion','stylist5VisualContract','garmentDescriptor','catalogCandidates','lookExplanation','validatedQuartets','legacyClientCompatibility','outfitDeckSelection'])if(name.includes(module))return require(path.resolve(__dirname,'../../src/domain/',module+'.ts'));
  return {};
 }, {}, {env:{get:()=>undefined}});
// RPC-first retrieval contract: the compact candidate RPC returns already
// normalized rows. Hydration is a separate RPC and stays empty here so the
// compact picks are asserted directly.
const rpcClient=(rows)=>({rpc:(name)=>{const q={abortSignal:()=>q,then:resolve=>Promise.resolve(
 name==='catalog_stylist_candidate_hydration'?{data:[],error:null}:{data:rows,error:null}).then(resolve)};return q;}});

test.each(['UZS','USD','EUR'])('shop candidates retain authoritative %s currency for budget conversion',async currency=>{
 const row={id:'fixture-shirt',name:'White cotton shirt',fashion_slot:'top',category:'tops',garment_type:'upper_body',price:400000,currency,image_url:'https://example.invalid/shirt.jpg'};
 const rows=await api.fetchShopItemsForAllSlotsEdge(rpcClient([row]),['top'],'casual',1,new Set());
 expect(rows).toHaveLength(1);expect(rows[0]).toMatchObject({id:'shop_fixture-shirt',price:400000,currency});
 if(currency==='UZS') expect(api.budgetInCurrencyEdge({amount:250,currency:'USD',fx:referenceFx},rows[0].currency)).toEqual({amount:3011180,currency:'UZS'});
});

test('candidate fetch preserves the exact retailer URL from the compact RPC row',async()=>{
 const row={id:'fixture-shirt',name:'White cotton shirt',fashion_slot:'top',category:'tops',garment_type:'upper_body',price:50,currency:'USD',image_url:'https://example.invalid/shirt.jpg',source_url:'https://retailer.invalid/products/exact-shirt'};
 const rows=await api.fetchShopItemsForAllSlotsEdge(rpcClient([row]),['top'],'casual',1,new Set());
 expect(rows).toHaveLength(1);
 expect(rows[0].shopUrl).toBe(row.source_url);
});


test('product title color takes precedence over suggested companion colors',()=>{
 expect(api.inferShopColor({name:'Navy blue cotton shirt',description:'Pair with white trousers and black shoes'})).toBe('navy');
 expect(api.inferShopColor({name:'Dark green cardigan',description:'White shirt underneath'})).toBe('dark green');
 expect(api.inferShopColor({color:'burgundy',name:'White shirt'})).toBe('burgundy');
});
