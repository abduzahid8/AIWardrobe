import test from 'node:test';
import assert from 'node:assert/strict';
import {proposeVisualFirstDeck,shortlistRetrievalDeck,STYLIST6_VERSION} from './prototype.mjs';

const catalog=[];
function add(slot,subtype,colors,price=170000){
  for(const [n,color] of colors.entries()){
    const id=slot+'-'+subtype+'-'+n;
    catalog.push({id,name:color+' '+subtype,macroCategory:slot,subtype,color,texture:n?'knit':'smooth',
      imageUrl:'https://example.com/p/'+id+'.jpg',
      shopUrl:'https://example.com/buy/'+id,
      price:price+n*7000,currency:'UZS',isShopItem:true,visualEvidenceVerified:true});
  }
}
add('top','shirt',['white','blue']);
add('top','polo',['navy','cream']);
add('top','fine_knit',['grey','brown']);
add('outerwear','blazer',['navy','grey']);
add('outerwear','cardigan',['brown','navy']);
add('outerwear','overshirt',['beige','olive']);
add('bottom','tailored_trouser',['navy','grey']);
add('bottom','chino',['beige','brown']);
add('bottom','jeans',['blue','black']);
add('shoes','loafer',['brown','black']);
add('shoes','derby',['black','brown']);
add('shoes','minimal_sneaker',['white','grey']);

const concepts=[
  ['ivy','Modern Ivy tailoring','shirt','blazer','tailored_trouser','loafer','navy'],
  ['soft','Quiet knit separates','polo','cardigan','chino','loafer','brown'],
  ['relax','Soft sporty layers','polo','overshirt','jeans','minimal_sneaker','beige'],
  ['office','Polished business tailoring','shirt','blazer','chino','derby','grey'],
  ['casual','Understated weekend knits','fine_knit','cardigan','jeans','minimal_sneaker','cream'],
  ['creative','Textured seasonal separates','fine_knit','overshirt','tailored_trouser','derby','olive'],
].map(([id,name,top,outer,bottom,shoes,color],index)=>({
  referenceId:'rt_00'+(index+1),
  id,name,roles:{top:[top],outerwear:[outer],bottom:[bottom],shoes:[shoes]},
  layerPolicy:'required',palette:[color],silhouette:id,
}));
const base={
  request:{style:'old_money',occasion:'Work',sourceMode:'catalog'},catalog,
  creativeDirector:async()=>concepts,
  visualCritic:async(xs)=>xs.map(x=>({candidateId:x.id,visualCohesion:0.90,referenceFidelity:0.91,reject:false,styleFit:0.91,
    occasionFit:0.89,formalityFit:0.92,confidence:0.96,evidence:['mock cohesion','mock reference','mock formality']})),
  hardGate:async()=>({ok:true}),
  priceToBudgetCurrency:(x,currency)=>x.currency===currency?x.price:null,
};

test('dynamic creative briefs produce three catalog-grounded distinct outfits',async()=>{
  const result=await proposeVisualFirstDeck(base);
  assert.equal(result.generationVersion,STYLIST6_VERSION);
  assert.equal(result.status,'complete');
  assert.equal(result.outfits.length,3);
  assert.equal(new Set(result.outfits.map(x=>x.concept.id)).size,3);
  for(const outfit of result.outfits){
    assert.equal(outfit.items.length,4);
    for(const item of outfit.items)assert.ok(catalog.some(source=>source.id===item.id));
  }
});
test('hard budget applies to every complete outfit before visual review',async()=>{
  const result=await proposeVisualFirstDeck({...base,request:{...base.request,budget:{amount:690000,currency:'UZS'}}});
  for(const look of result.outfits)assert.ok(look.items.reduce((sum,item)=>sum+item.price,0)<=690000);
  assert.ok(result.diagnostics.rejections.budget>0);
});
test('without genuine critic evidence the engine abstains instead of faking quality',async()=>{
  const result=await proposeVisualFirstDeck({...base,visualCritic:async()=>[]});
  assert.equal(result.status,'no_verified_looks');
  assert.equal(result.outfits.length,0);
});
test('untrusted creative director cannot inject catalog product IDs',async()=>{
  const altered=concepts.map((c,i)=>i===0?{...c,productIds:['fabricated-product']}:c);
  await assert.rejects(()=>proposeVisualFirstDeck({...base,creativeDirector:async()=>altered}),/brief_cannot_choose_skus/);
});
test('invalid source links and image URLs are denied before ranking',async()=>{
  const dirty=catalog.map((p,i)=>i===0?{...p,shopUrl:'http://unsafe.invalid'}:i===1?{...p,imageUrl:''}:p);
  const result=await proposeVisualFirstDeck({...base,catalog:dirty});
  assert.equal(result.diagnostics.admittedCatalogItems,catalog.length-2);
  for(const outfit of result.outfits)for(const item of outfit.items){
    assert.ok(item.shopUrl.startsWith('https://'));
    assert.ok(item.imageUrl.startsWith('https://'));
  }
});
test('the backend hard gate is mandatory and failed evaluation never bypassed',async()=>{
  await assert.rejects(()=>proposeVisualFirstDeck({...base,hardGate:null}),/trusted_gates_required/);
  const result=await proposeVisualFirstDeck({...base,hardGate:async()=>({ok:false})});
  assert.equal(result.status,'no_verified_looks');
  assert.ok(result.diagnostics.rejections.hardGate>0);
});
test('missing FX is an explicit rejection, never an invented exchange rate',async()=>{
  const mixed=catalog.map((p,i)=>i%3===0?{...p,currency:'USD',price:20}:p);
  const result=await proposeVisualFirstDeck({...base,catalog:mixed,
    request:{...base.request,budget:{amount:1000000,currency:'UZS'}}});
  for(const look of result.outfits)for(const item of look.items)assert.equal(item.currency,'UZS');
  assert.ok(result.diagnostics.rejections.unpriced>0);
});
test('identical garment directions with renamed headers do not count as creativity',async()=>{
  const cloned=concepts.map((c,i)=>({...c,id:'new'+i,name:'Another outfit '+i,
    roles:concepts[0].roles,layerPolicy:'required',palette:['grey'],silhouette:'same'}));
  await assert.rejects(()=>proposeVisualFirstDeck({...base,creativeDirector:async()=>cloned}),/creative_briefs_not_diverse/);
});
test('retrieval shortlist is explicitly marked visual-verdict NOT RUN',async()=>{
  const deck=await shortlistRetrievalDeck(base);
  assert.equal(deck.generationVersion,STYLIST6_VERSION);
  assert.equal(deck.visualVerdict,'NOT_RUN');
  assert.ok(deck.shortlist.length>0);
  assert.ok(!('visualQuality' in (deck.shortlist[0]||{}))||deck.shortlist[0].visualQuality===undefined);
  assert.ok(deck.diagnostics.candidateCount>0);
});

test('duplicate verdicts and unexpected critic failures never create approved outfits',async()=>{
  const duplicate=await proposeVisualFirstDeck({...base,visualCritic:async(xs)=>{
    const [row]=await base.visualCritic(xs);
    return [row,{...row,referenceFidelity:1}];
  }});
  assert.equal(duplicate.status,'no_verified_looks');
  assert.equal(duplicate.diagnostics.visualValid,0);
  assert.equal(duplicate.diagnostics.visualAbstained,duplicate.diagnostics.visualAttempted);
  const failure=await proposeVisualFirstDeck({...base,visualCritic:async()=>{throw new Error('sensitive details')}});
  assert.equal(failure.status,'no_verified_looks');
  assert.equal(failure.diagnostics.criticFailed,true);
  assert.equal(failure.diagnostics.visualVerdict,'ABSTAINED');
  assert.doesNotMatch(JSON.stringify(failure),/sensitive details/);
});

test('a narrow visual lead keeps the feasible seed while a clear lead displaces it',async()=>{
  for(const [fidelityIncrease,winner] of [[0.01,0],[0.08,1]]){
    const seen=[];
    const result=await proposeVisualFirstDeck({...base,options:{target:1,maxHostedCandidates:2},
      visualCritic:async(payload)=>{
        seen.push(...payload.map(({id})=>id));
        const [first,second]=await base.visualCritic(payload);
        return [first,{...second,referenceFidelity:second.referenceFidelity+fidelityIncrease}];
      }});
    assert.equal(result.outfits[0].id,seen[winner]);
  }
});

test('review slots preserve the seed and cover distinct products from the same brief',async()=>{
  const brief=concepts[0];
  const threePerRole=Object.entries(brief.roles).flatMap(([slot,[subtype]])=>
    Array.from({length:3},(_,index)=>{
      const id=slot+'-diversity-'+index;
      return {id,macroCategory:slot,subtype,color:['top','shoes'].includes(slot)||index===0?'navy':'grey',texture:'smooth',
        isShopItem:true,price:50,currency:'UZS',
        imageUrl:'https://example.com/images/'+id,shopUrl:'https://example.com/shop/'+id};
    }));
  const {shortlist}=await shortlistRetrievalDeck({...base,catalog:threePerRole});
  const alternatives=shortlist.filter(({briefId})=>briefId===brief.id).slice(0,3);
  assert.equal(alternatives.length,3);
  assert.ok(alternatives[0].items.every(({color})=>color==='navy'));
  // The second and third strongest seeds can differ by at most one palette
  // match, so limited inventory may require some reuse of the seed pieces.
  assert.ok(new Set(alternatives.flatMap(({items})=>items.map(({id})=>id))).size===8);
});
