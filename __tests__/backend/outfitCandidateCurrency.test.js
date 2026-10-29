/** @jest-environment node */
const {referenceFx}=require('../fixtures/budgetFx');
const fs = require('fs');
const path = require('path');
const ts = require('typescript');
const filename = process.env.OUTFIT_CANDIDATE_SOURCE || path.resolve(__dirname, '../../backend/supabase/functions/generate-outfits/index.ts');
let source = fs.readFileSync(filename, 'utf8');
if (process.env.OUTFIT_SIMULATE_MISSING_CURRENCY === '1') source = source.replace(/\s*currency: row.currency,/g, '');
const compiled = ts.transpileModule(source, {compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText;
const api = new Function('require','exports','Deno',compiled+'\nreturn {fetchShopItemsForAllSlotsEdge,budgetInCurrencyEdge};')(
 name=>{
  if(name.includes('outfitDiversity')) return require('../../backend/supabase/functions/_shared/outfitDiversity');
  if(name.includes('outfitRotation')) return require('../../backend/supabase/functions/_shared/outfitRotation');
  if(name.includes('budgetFx.ts')) return require('../../backend/supabase/functions/_shared/budgetFx');
  if(name.includes('server.ts')) return {serve:()=>{}};
  if(name.includes('occasionRules') || name.includes('outfitStructure')) {
    const helper={};const file=path.resolve(__dirname,'../../backend/supabase/functions/_shared/',path.basename(name));
    new Function('exports',ts.transpileModule(fs.readFileSync(file,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2022}}).outputText)(helper);return helper;
  }
  return {};
 }, {}, {env:{get:()=>undefined}});
test.each(['UZS','USD','EUR'])('shop candidates retain authoritative %s currency for budget conversion',async currency=>{
 const row={id:'fixture-shirt',name:'White cotton shirt',category:'tops',garment_type:'upper_body',price:400000,currency,image_url:'https://example.invalid/shirt.jpg'};
 const chain={select:()=>chain,eq:()=>chain,limit:()=>chain,or:()=>chain,order:()=>chain,gt:()=>chain,abortSignal:()=>chain,then:resolve=>Promise.resolve({data:[row],error:null}).then(resolve)};
 const rows=await api.fetchShopItemsForAllSlotsEdge({from:()=>chain},['top'],'casual',1,new Set());
 expect(rows).toHaveLength(1);expect(rows[0]).toMatchObject({id:'shop_fixture-shirt',price:400000,currency});
 if(currency==='UZS') expect(api.budgetInCurrencyEdge({amount:250,currency:'USD',fx:referenceFx},rows[0].currency)).toEqual({amount:3011180,currency:'UZS'});
});

test('candidate fetch selects and preserves the exact retailer URL',async()=>{
 const row={id:'fixture-shirt',name:'White cotton shirt',category:'tops',garment_type:'upper_body',price:50,currency:'USD',image_url:'https://example.invalid/shirt.jpg',source_url:'https://retailer.invalid/products/exact-shirt'};
 const select=jest.fn(()=>chain);
 const chain={select,eq:()=>chain,limit:()=>chain,or:()=>chain,order:()=>chain,gt:()=>chain,abortSignal:()=>chain,then:resolve=>Promise.resolve({data:[row],error:null}).then(resolve)};
 const rows=await api.fetchShopItemsForAllSlotsEdge({from:()=>chain},['top'],'casual',1,new Set());
 expect(select.mock.calls[0][0].split(',').map(value=>value.trim())).toContain('source_url');
 expect(rows[0].shopUrl).toBe(row.source_url);
});
