import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {referenceFirstBriefs} from './lookbook_briefs.mjs';
import {proposeVisualFirstDeck} from './prototype.mjs';

const index=JSON.parse(readFileSync(
  fileURLToPath(new URL('../../benchmarks/stylist6/references/image_teaste_46.json',import.meta.url)),'utf8'));
test('46 physically reviewed blueprint records are present, unique, fully layered or explicitly unlayered',()=>{
  assert.equal(index.images.length,46);
  assert.equal(new Set(index.images.map(x=>x.id)).size,46);
  assert.equal(index.images.filter(x=>x.layer_policy==='required').length,39);
  assert.equal(index.images.filter(x=>x.layer_policy==='absent').length,7);
  assert.ok(index.images.every(x=>x.complete_outfit_collage===true));
  assert.ok(index.images.every(x=>x.fit_on_body_verified===false));
  assert.ok(index.images.every(x=>x.media_sha256===null));
});
test('lookbook adapter generates traceable, structurally different briefs for real requests',()=>{
  for(const request of [
    {style:'old_money',occasion:'work',season:'autumn'},
    {style:'classic',occasion:'work',season:'spring'},
    {style:'semi_classic',occasion:'date_night',season:'summer'},
  ]){
    const briefs=referenceFirstBriefs(index,request,8);
    assert.equal(briefs.length,8);
    assert.equal(new Set(briefs.map(b=>b.referenceId)).size,8);
    assert.ok(briefs.every(b=>/^rt_\d{3}$/.test(b.referenceId)));
    assert.ok(briefs.every(b=>['required','absent'].includes(b.layerPolicy)));
    const constructions=new Set(briefs.map(b=>JSON.stringify([b.roles,b.layerPolicy,b.palette,b.silhouette])));
    assert.ok(constructions.size>=4);
  }
});
test('reference-first briefs remain grounded in real catalog SKUs under mocked quality verdict',async()=>{
  const request={style:'old_money',occasion:'work',season:'autumn',sourceMode:'catalog'};
  const briefs=referenceFirstBriefs(index,request,8);
  const seen=new Set(),catalog=[];
  for(const b of briefs)for(const slot of ['top','outerwear','bottom','shoes']){
    for(const subtype of b.roles[slot]){
      const key=slot+'_'+subtype;
      if(seen.has(key))continue;
      seen.add(key);
      for(let i=0;i<2;i++){
        const id=key+'_'+i;
        catalog.push({id,name:subtype+' test SKU '+i,macroCategory:slot,subtype,
          imageUrl:'https://example.com/images/'+id+'.jpg',
          shopUrl:'https://example.com/products/'+id,
          isShopItem:true,price:140000+i*5000,currency:'UZS',
          color:'cream',visualEvidenceVerified:true});
      }
    }
  }
  const reviewed=[];
  const result=await proposeVisualFirstDeck({
    request,catalog,creativeDirector:async()=>briefs,
    visualCritic:async (payload)=>{
      reviewed.push(...payload);
      return payload.map(p=>({
        candidateId:p.id,visualCohesion:0.92,styleFit:0.89,
        occasionFit:0.91,formalityFit:0.94,confidence:0.95,
        evidence:['mocked visual judgement; NOT a real provider'],
      }));
    },
    hardGate:async()=>({ok:true}),
    priceToBudgetCurrency:(i,currency)=>i.currency===currency?i.price:NaN,
  });
  assert.equal(result.outfits.length,3);
  assert.equal(result.status,'complete');
  assert.ok(reviewed.length>0);
  assert.ok(reviewed.every(p=>/^rt_\d{3}$/.test(p.referenceId)));
  for(const look of result.outfits){
    assert.ok(/^rt_\d{3}$/.test(look.concept.referenceId));
    assert.ok(look.items.every(item=>catalog.some(c=>c.id===item.id)));
  }
});
test('corrupt reference index never silently bypasses provenance checks',()=>{
  assert.throws(()=>referenceFirstBriefs({...index,images:index.images.slice(1)},
    {style:'old_money',occasion:'work'}),/incomplete_reference_collection/);
  assert.throws(()=>referenceFirstBriefs({...index,source:'instagram_creator_export'},
    {style:'old_money',occasion:'work'}),/invalid_reference_index/);
});
