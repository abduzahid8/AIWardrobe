import test from 'node:test';
import assert from 'node:assert/strict';
import {proposeVisualFirstDeck,shortlistRetrievalDeck,selectVisualDeck,diversifyBriefHead,outerwearArchetype,STYLIST6_VERSION} from './prototype.mjs';

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
test('AI-directed catalog retrieval keeps its exact verified role subtypes instead of substituting unrelated items', async () => {
  const ai = {id:'ai-director-1',name:'Cream Knit Casual',
    roles:{top:['fine_knit'],outerwear:['cardigan'],bottom:['chino'],shoes:['loafer']},
    palette:['cream','brown'],silhouette:'balanced',layerPolicy:'required'};
  const fashion={prepare:item=>item, family:item=>item.subtype,
    rankItem:()=>0.8, rankLook:()=>({score:0.82})};
  const deck = await shortlistRetrievalDeck({
    ...base, request:{style:'old_money',occasion:'Weekend',sourceMode:'catalog'},
    creativeDirector:async()=>[ai], options:{fashion,perRole:8,perBrief:30,maxVisual:24},
  });
  assert.ok(deck.shortlist.length>=1);
  for(const look of deck.shortlist){
    assert.equal(look.briefId,'ai_director_1');
    for(const item of look.items){
      assert.ok(ai.roles[item.macroCategory].includes(item.subtype),
        'AI plan role must not be replaced by an unrelated SKU subtype: '+item.subtype);
    }
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
  // Diversity: the allocator must cover distinct products across the three
  // alternatives instead of converging on one reused hero set.
  const distinct=new Set(alternatives.flatMap(({items})=>items.map(({id})=>id)));
  assert.ok(distinct.size>=8, `expected broad product coverage, saw ${distinct.size}`);
  // Quality floor: every alternative is a full look scored within a bounded
  // gap of the strongest seed, never a decisively worse filler.
  const best=Math.max(...alternatives.map(({seedScore})=>seedScore));
  for(const alternative of alternatives){
    assert.equal(alternative.items.length,4);
    assert.ok(Number.isFinite(alternative.seedScore));
    assert.ok(alternative.seedScore>=best-0.25, `seed ${alternative.seedScore} too far below best ${best}`);
  }
});


test('visual deck only treats material architecture as a distinct direction when enabled',()=>{
  const names=['top','bottom','shoes'];
  const looks=['a','b','c'].map((id,i)=>({
    id,brief:{id:'b'+i,name:'Test'},briefId:'b'+i,
    items:names.map(slot=>({id:slot+'-'+id,macroCategory:slot,
      subtype:slot==='top'?'shirt':slot==='bottom'?'trouser':'loafer',
      color:slot==='top'?['sky','white','ivory'][i]:'navy',texture:'cotton'})),
  }));
  const critiques=looks.map(({id})=>({candidateId:id,visualCohesion:.95,referenceFidelity:.95,
    styleFit:.95,occasionFit:.95,formalityFit:.95,confidence:.95,reject:false,
    evidence:['top matches bottom','bottom matches shoes','shoes provide support']}));
  const input={generationVersion:STYLIST6_VERSION,shortlist:looks,diagnostics:{}};
  const ordinary=selectVisualDeck(input,critiques,{target:3,minNovelty:0.01});
  const guarded=selectVisualDeck(input,critiques,{target:3,minNovelty:0.01,requireDistinctArchitectures:true});
  assert.equal(ordinary.outfits.length,3);
  assert.equal(guarded.outfits.length,1,'renamed SKUs and new shirt colors alone do not certify independent directions');
  assert.equal(guarded.status,'distinct_inventory_limited');
});

test('visual deck globally finds three independent directions instead of anchoring to a blocker lead',()=>{
  // All four have independently approved visual photographs and exact SKUs.
  // The very highest-scored outfit is structurally too similar to B and C;
  // a REAL three-way independent set exists only in B,C,D. The old greedy
  // leader-only walk either returned 2 or counted a duplicate second look.
  const slots=['top','outerwear','bottom','shoes'];
  const colorSets=[
    ['navy','navy','navy','navy'],
    ['grey','grey','navy','navy'],
    ['navy','navy','brown','brown'],
    ['navy','navy','navy','navy'],
  ];
  const looks=colorSets.map((colors,index)=>({
    id:['dominant','grey-drape','warm-classic','soft-knit'][index],
    briefId:'direction-'+index,brief:{id:'direction-'+index,name:'Different legitimate look'},
    items:slots.map((slot,i)=>({id:`${slot}-${index}`,name:`${colors[i]} ${slot} ${index}`,
      macroCategory:slot,subtype:slot==='top'&&index===3?'fine-knit':
        slot==='top'?'shirt':slot==='outerwear'?'blazer':
          slot==='bottom'?'tailored-trouser':'loafer',
      color:colors[i],texture:slot==='top'&&index===3?'knit':'cotton',
      isShopItem:true,price:100,currency:'USD',
      imageUrl:`https://example.com/${slot}${index}.jpg`,shopUrl:`https://example.com/${slot}${index}`})),
  }));
  const scores=[0.98,0.90,0.88,0.86];
  const verdicts=looks.map((x,i)=>({candidateId:x.id,visualCohesion:scores[i],
    referenceFidelity:scores[i],styleFit:scores[i],occasionFit:scores[i],
    formalityFit:scores[i],confidence:0.98,reject:false,
    evidence:['top versus bottom harmony','outerwear silhouette and top','shoe to hem proportion']}));
  const result=selectVisualDeck({generationVersion:STYLIST6_VERSION,shortlist:looks,diagnostics:{}},
    verdicts,{target:3,minNovelty:0.01,requireDistinctArchitectures:true});
  assert.equal(result.status,'complete');
  assert.deepEqual(new Set(result.outfits.map(x=>x.id)),
    new Set(['grey-drape','warm-classic','soft-knit']));
  assert.equal(result.diagnostics.visualApproved,4);
  assert.ok(result.outfits.every(x=>Array.isArray(x.evidence)));
});

test('visual selector prefers a genuine third direction only among approved near-quality looks',()=>{
  const roles=[
    ['shirt','blazer','tailored-trouser','derby'],
    ['polo','cardigan','chino','loafer'],
    ['fine-knit','overshirt','jeans','sneaker'],
    ['dress-shirt','coat','flannel-trouser','oxford'],
  ];
  const models=[.97,.96,.95,.90];
  const looks=roles.map((types,i)=>({
    id:'candidate-'+i,briefId:'concept-'+i,brief:{id:'concept-'+i,name:'Verified look'},
    fashion:{editorial:{score:i===3?.88:.92}},
    items:['top','outerwear','bottom','shoes'].map((slot,j)=>({
      id:`sku-${i}-${slot}`,name:`${types[j]} photo ${i}`,macroCategory:slot,
      subtype:types[j],color:'navy',texture:'wool',
    })),
  }));
  const critiques=looks.map((l,i)=>({
    candidateId:l.id,visualCohesion:models[i],referenceFidelity:models[i],
    styleFit:models[i],occasionFit:models[i],formalityFit:models[i],
    confidence:.98,reject:false,evidence:['garment construction','color cohesion','dress register'],
  }));
  const input={generationVersion:STYLIST6_VERSION,shortlist:looks,diagnostics:{}};
  const base={target:3,minNovelty:.01,requireDistinctArchitectures:true};
  const plain=selectVisualDeck(input,critiques,base);
  assert.equal(plain.status,'complete');
  const priority=selectVisualDeck(input,critiques,{...base,
    directionDeckBonus:deck=>deck.some(l=>l.id==='candidate-3')?.12:0});
  assert.equal(priority.status,'complete');
  assert.ok(priority.outfits.some(l=>l.id==='candidate-3'),
    'bounded direction gain should displace a close but less distinctive approved outfit');
  assert.ok(priority.outfits.every(l=>l.visualQuality>=.68));
  const vetoed=critiques.map(v=>v.candidateId==='candidate-3'?{...v,reject:true}:v);
  const guarded=selectVisualDeck(input,vetoed,{...base,
    directionDeckBonus:deck=>deck.some(l=>l.id==='candidate-3')?100:0});
  assert.ok(!guarded.outfits.some(l=>l.id==='candidate-3'),
    'even an unbounded caller bonus must never restore an NVIDIA-vetoed candidate');
});

test('model-approved looks favor comparable affordable outfits unless user requests luxury',()=>{
  const kinds=[['shirt','blazer','chino','loafer'],['polo','cardigan','tailored-trouser','derby'],
    ['fine-knit','coat','jeans','chelsea'],['t-shirt','overshirt','trouser','oxford']];
  const looks=kinds.map((k,i)=>({id:`cost-${i}`,briefId:`cost-${i}`,brief:{id:`cost-${i}`,name:`direction ${i}`},
    fashion:{unbudgetedPricePenalty:i===0?0.18:0},
    items:['top','outerwear','bottom','shoes'].map((slot,j)=>({
      id:`p-${i}-${slot}`,name:`${k[j]} ${i}`,macroCategory:slot,subtype:k[j],
      color:['navy','cream','brown','grey'][i],texture:['cotton','wool','leather','suede'][j]})),
  }));
  const verdicts=looks.map((look,i)=>({candidateId:look.id,visualCohesion:i===0?0.95:0.91,
    referenceFidelity:i===0?0.95:0.91,styleFit:i===0?0.95:0.91,occasionFit:i===0?0.95:0.91,
    formalityFit:i===0?0.95:0.91,confidence:0.96,reject:false,
    evidence:['outfit proportions','styling direction','shoe to trouser proportions']}));
  const input={generationVersion:STYLIST6_VERSION,shortlist:looks,diagnostics:{}};
  const base={target:3,minNovelty:.01,requireDistinctArchitectures:true};
  const value=selectVisualDeck(input,verdicts,{...base,preferValue:true});
  assert.equal(value.status,'complete');
  assert.ok(!value.outfits.some(x=>x.id==='cost-0'),
    'an extreme premium price should lose a close visual-quality contest when three approved alternatives exist');
  const luxury=selectVisualDeck(input,verdicts,{...base,preferValue:false});
  assert.ok(luxury.outfits.some(x=>x.id==='cost-0'),
    'explicit luxury request disables the value adjustment');
});

test('approved three-look deck favors distinct main tops within a small visual-quality tradeoff', () => {
  const garment=(id,slot,subtype,color) => ({ id,macroCategory:slot,
    subtype,color,texture:'cotton',name:id });
  const make=(id,score,top,outer,bottom,shoes)=>({
    id,briefId:id,brief:{id,name:id},
    items:[garment(top,'top','shirt','cream'),
      garment('outer-'+id,'outerwear',outer,'navy'),
      garment('bottom-'+id,'bottom',bottom,'brown'),
      garment('shoe-'+id,'shoes',shoes,'black')],
    visualScore:score,
  });
  const looks=[
    make('hero',.96,'hero-shirt','blazer','chino','loafer'),
    make('repeat',.94,'hero-shirt','cardigan','trouser','derby'),
    make('knit',.93,'knit-top','coat','chino','oxford'),
    make('fresh',.89,'new-shirt','overshirt','trouser','chelsea'),
  ];
  const approved=looks.map(look=>({candidateId:look.id,visualCohesion:look.visualScore,
    referenceFidelity:look.visualScore,styleFit:look.visualScore,occasionFit:look.visualScore,
    formalityFit:look.visualScore,confidence:.96,reject:false,
    evidence:['top and bottom complement','layer and silhouette balanced','footwear appropriate']}));
  const r=selectVisualDeck({generationVersion:STYLIST6_VERSION,shortlist:looks,diagnostics:{}},
    approved,{target:3,minNovelty:.1,requireDistinctArchitectures:true});
  assert.equal(r.status,'complete');
  assert.equal(r.outfits.length,3);
  const tops=r.outfits.map(l=>l.items.find(i=>i.macroCategory==='top')?.id);
  assert.equal(new Set(tops).size,3,
    'top reuse should lose a small near-quality tie when three independent approved tops exist');
  assert.ok(r.outfits.every(l=>l.visualQuality>=.68));
});

test('visual selection prefers non-repeated coats and footwear among similarly approved full decks', () => {
  const article=(id,slot,type,color)=>({id,macroCategory:slot,subtype:type,
    name:id,color,texture:slot==='outerwear'?'wool':'cotton'});
  const look=(id,value,top,outer,bottom,shoe,topType,outerType,bottomType,shoeType,color)=>({
    id,briefId:id,brief:{id,name:id},visualScore:value,
    items:[article(top,'top',topType,color),article(outer,'outerwear',outerType,color),
      article(bottom,'bottom',bottomType,color),article(shoe,'shoes',shoeType,color)],
  });
  const approved=[
    look('hero',.96,'white-shirt','navy-hero-coat','grey-trousers','black-chelsea',
      'shirt','coat','tailored-trouser','chelsea','grey'),
    look('near-clone',.955,'navy-polo','navy-hero-coat','stone-chinos','brown-loafer',
      'polo','coat','chino','loafer','navy'),
    look('knit-direction',.93,'cream-knit','camel-blazer','tan-wool','tan-derby',
      'fine-knit','blazer','tailored-trouser','derby','cream'),
    look('dress-shirt-direction',.91,'blue-dress','olive-overcoat','brown-cords','oxblood-oxford',
      'dress-shirt','coat','corduroy-trouser','oxford','brown'),
  ];
  const verdicts=approved.map(l=>({candidateId:l.id,visualCohesion:l.visualScore,
    referenceFidelity:l.visualScore,styleFit:l.visualScore,occasionFit:l.visualScore,
    formalityFit:l.visualScore,confidence:.98,reject:false,
    evidence:['source-backed coherent top and bottom','proportions fit jacket','footwear matched hem']}));
  const selected=selectVisualDeck({generationVersion:STYLIST6_VERSION,shortlist:approved,diagnostics:{}},
    verdicts,{target:3,minNovelty:.01,requireDistinctArchitectures:true});
  assert.equal(selected.status,'complete');
  const layers=selected.outfits.map(l=>l.items.find(it=>it.macroCategory==='outerwear').id);
  const shoes=selected.outfits.map(l=>l.items.find(it=>it.macroCategory==='shoes').id);
  assert.equal(new Set(layers).size,3,'reusing the exact overcoat should lose a near-quality tie');
  assert.equal(new Set(shoes).size,3);
  assert.ok(selected.outfits.every(l=>l.visualQuality>=.68));
  // A user-chosen coat can legitimately repeat across three styled looks.
  const pinned=selectVisualDeck({generationVersion:STYLIST6_VERSION,
    shortlist:approved.slice(0,2),diagnostics:{}},verdicts.slice(0,2),
    {target:2,minNovelty:.01,requiredIds:['navy-hero-coat']});
  assert.equal(pinned.status,'complete');
});

test('joint selection keeps the strongest first look and avoids a second look that blocks a full deck',()=>{
  const slots=['top','outerwear','bottom','shoes'];
  const looks=[[0,0,0,0],[1,2,0,0],[1,1,0,0],[1,2,0,1]].map((keys,index)=>({
    id:'look-'+index,briefId:'rt_001',brief:{id:'rt_001',name:'Test'},
    items:keys.map((key,i)=>({id:slots[i]+'-'+key,macroCategory:slots[i],subtype:slots[i],color:String(key)})),
  }));
  const rows=looks.map(({id},index)=>{
    const score=[.95,.90,.80,.80][index];
    return {candidateId:id,visualCohesion:score,referenceFidelity:score,styleFit:score,occasionFit:score,
      formalityFit:score,confidence:.95,reject:false,evidence:['mock colors','mock formality','mock balance']};
  });
  const result=selectVisualDeck({generationVersion:STYLIST6_VERSION,shortlist:looks,diagnostics:{}},rows);
  assert.ok(Array.isArray(result.diagnostics.visualQualityDecisions), 'every actual AI verdict has a quality receipt');
  assert.equal(result.status,'complete');
  assert.equal(result.outfits[0].id,'look-0');
  assert.deepEqual(new Set(result.outfits.map(({id})=>id)),new Set(['look-0','look-2','look-3']));
});


test('retrieval preserves different garment architectures ahead of new-SKU clones',()=>{
  const item=(id,macroCategory,subtype)=>({id,macroCategory,subtype});
  const make=(id,seedScore,top,layer,bottom,shoe)=>({id,seedScore,items:[
    item(id+'-top','top',top),item(id+'-layer','outerwear',layer),
    item(id+'-bottom','bottom',bottom),item(id+'-shoe','shoes',shoe),
  ]});
  const pool=[
    make('lead',.98,'shirt','blazer','tailored-trouser','loafer'),
    make('sku-clone',.975,'shirt','blazer','tailored-trouser','derby'),
    make('knit-architecture',.97,'fine-knit','cardigan','chino','loafer'),
    make('other-architecture',.965,'polo','overshirt','jeans','loafer'),
  ];
  diversifyBriefHead(pool);
  assert.deepEqual(pool.slice(0,3).map(x=>x.id),[
    'lead','knit-architecture','other-architecture',
  ]);
  assert.equal(pool[3].id,'sku-clone');
});

test('architecture evidence absent falls back to original physical-SKU diversity',()=>{
  const make=(id,seedScore,ids)=>({id,seedScore,items:ids.map(x=>({id:x}))});
  const candidates=[
    make('lead',.98,['t1','b1','s1']),
    make('clone',.975,['t1','b1','s2']),
    make('new-products',.97,['t2','b2','s3']),
  ];
  diversifyBriefHead(candidates);
  assert.equal(candidates[0].id,'lead');
  assert.equal(candidates[1].id,'new-products');
});

test('retrieval diversification prioritizes distinct garment architecture over office-shirt-trouser clones', () => {
  const item = (id, macroCategory, subtype) => ({ id, macroCategory, subtype });
  const make = (id, seedScore, top, layer, bottom, shoe) => ({
    id, seedScore, items: [
      item(id + '-top', 'top', top),
      ...(layer ? [item(id + '-layer', 'outerwear', layer)] : []),
      item(id + '-bottom', 'bottom', bottom),
      item(id + '-shoe', 'shoes', shoe),
    ],
  });

  const pool = [
    make('office-1', .98, 'dress-shirt', null, 'tailored-trouser', 'derby'),
    make('office-2', .975, 'shirt', null, 'tailored-trouser', 'loafer'),
    make('office-3', .972, 'shirt', null, 'chino', 'derby'),
    make('tailored-blazer', .968, 'dress-shirt', 'blazer', 'tailored-trouser', 'oxford'),
    make('knit-layer', .965, 'fine-knit', 'cardigan', 'tailored-trouser', 'loafer'),
  ];

  diversifyBriefHead(pool);
  const headIds = pool.slice(0, 3).map(x => x.id);
  // Must preserve lead and elevate different garment architectures (tailored-blazer / knit-layer)
  assert.equal(headIds[0], 'office-1');
  assert.ok(headIds.includes('tailored-blazer') || headIds.includes('knit-layer'),
    'Architecture-diverse candidates must be promoted over repeated unlayered office clones');
});


test('three model-approved outfit directions favor a coat architecture over a third repeated blazer',()=>{
  const make=(id,visualScore,topType,outerType,bottomType,shoeType,color)=>({
    id,briefId:id,brief:{id,name:id},items:[
      {id:id+'-top',macroCategory:'top',subtype:topType,name:color+' top',color},
      {id:id+'-outer',macroCategory:'outerwear',subtype:outerType,name:color+' '+outerType,color},
      {id:id+'-bottom',macroCategory:'bottom',subtype:bottomType,name:color+' bottoms',color},
      {id:id+'-shoe',macroCategory:'shoes',subtype:shoeType,name:color+' shoes',color},
    ],visualScore,
  });
  const looks=[
    make('navy-tailoring',.98,'shirt','blazer','tailored-trouser','oxford','navy'),
    make('cream-tailoring',.965,'fine-knit','blazer','chino','derby','cream'),
    make('brown-tailoring',.945,'polo','blazer','trouser','loafer','brown'),
    make('gray-overcoat',.925,'dress-shirt','overcoat','pleated-trouser','chelsea','grey'),
  ];
  const verdicts=looks.map(l=>({candidateId:l.id,visualCohesion:l.visualScore,
    referenceFidelity:l.visualScore,styleFit:l.visualScore,occasionFit:l.visualScore,
    formalityFit:l.visualScore,confidence:0.98,reject:false,
    evidence:['model inspected top and bottom','model inspected layer','model inspected shoes']}));
  assert.equal(outerwearArchetype(looks[0]),'tailored_jacket');
  assert.equal(outerwearArchetype(looks[3]),'full_coat');
  assert.equal(outerwearArchetype(looks[0],['navy-tailoring-outer']),null);
  const selected=selectVisualDeck({generationVersion:STYLIST6_VERSION,shortlist:looks,diagnostics:{}},
    verdicts,{target:3,minNovelty:.02,requireDistinctArchitectures:true});
  assert.equal(selected.status,'complete');
  assert.ok(selected.outfits.some(l=>l.id==='gray-overcoat'),
    'one source-confirmed coat must outrank a third near-quality suit blazer');
  assert.ok(new Set(selected.outfits.map(look=>outerwearArchetype(look))).size>=2);
});


test('whole-outfit occasion/fit floors veto aesthetic color wins that fail the actual use case',()=>{
  const look={id:'good-colors-wrong-occasion',briefId:'trial',brief:{id:'trial',name:'Refined date'},
    items:[
      {id:'t',macroCategory:'top',subtype:'shirt',name:'Draped light shirt'},
      {id:'b',macroCategory:'bottom',subtype:'chino',name:'Cream tapered chinos'},
      {id:'s',macroCategory:'shoes',subtype:'loafer',name:'Brown penny loafers'},
    ]};
  const forgedBeauty={candidateId:look.id,visualCohesion:.96,styleFit:.95,
    occasionFit:.58,formalityFit:.92,referenceFidelity:.96,confidence:.99,reject:false,
    evidence:['colors harmonize','shapes balance','but occasion mismatch']};
  const review={generationVersion:STYLIST6_VERSION,shortlist:[look],diagnostics:{}};
  const legacy=selectVisualDeck(review,[forgedBeauty],{target:1});
  assert.equal(legacy.status,'complete','weighted aesthetic score was enough before axis gate');
  const strict=selectVisualDeck(review,[forgedBeauty],{target:1,
    minStyleFit:.72,minOccasionFit:.75,minFormalityFit:.70});
  assert.equal(strict.status,'no_verified_looks');
  assert.equal(strict.diagnostics.visualQualityDecisions[0].reason,'occasion_mismatch');
  const badPersonal={...forgedBeauty,occasionFit:.94,styleFit:.60};
  const rated=selectVisualDeck(review,[badPersonal],{target:1,
    minStyleFit:.72,minOccasionFit:.75,minFormalityFit:.70});
  assert.equal(rated.diagnostics.visualQualityDecisions[0].reason,'style_mismatch');
  const badFormality={...forgedBeauty,occasionFit:.94,formalityFit:.51};
  const formal=selectVisualDeck(review,[badFormality],{target:1,
    minStyleFit:.72,minOccasionFit:.75,minFormalityFit:.70});
  assert.equal(formal.diagnostics.visualQualityDecisions[0].reason,'formality_mismatch');
  const approved={...forgedBeauty,occasionFit:.81};
  assert.equal(selectVisualDeck(review,[approved],{target:1,
    minStyleFit:.72,minOccasionFit:.75,minFormalityFit:.70}).status,'complete');
});


test('a photographed navy or grey garment cannot be styled for a user who explicitly excludes it', () => {
  const look={id:'visible-deck',briefId:'color-brief',brief:{id:'color-brief',name:'Warm date'},
    items:[
      {id:'cream-top',macroCategory:'top',subtype:'fine-knit',name:'Merino knit'},
      {id:'navy-layer',macroCategory:'outerwear',subtype:'blazer',name:'Sartorial jacket'},
      {id:'beige-trousers',macroCategory:'bottom',subtype:'tailored-trouser',name:'Metropolitan trousers'},
      {id:'black-shoes',macroCategory:'shoes',subtype:'derby',name:'Leather Derby'},
    ]};
  const opinion={candidateId:look.id,referenceless:true,
    visualCohesion:.98,styleFit:.96,occasionFit:.93,formalityFit:.96,
    confidence:.99,reject:false,slotColors:{top:'cream',outerwear:'navy',bottom:'beige',shoes:'black'},
    evidence:['model saw actual garments','model saw jacket color','model saw full silhouette']};
  const input={generationVersion:STYLIST6_VERSION,shortlist:[look],diagnostics:{}};
  const aesthetic=selectVisualDeck(input,[opinion],{target:1});
  assert.equal(aesthetic.status,'complete',
    'a beautiful palette on its own would have passed for someone who enjoys navy');
  const excluded=selectVisualDeck(input,[opinion],{target:1,avoidColors:['navy','grey']});
  assert.equal(excluded.status,'no_verified_looks');
  assert.equal(excluded.diagnostics.visualQualityDecisions[0].reason,'excluded_visual_color');
  const grey={...opinion,slotColors:{...opinion.slotColors,outerwear:'charcoal'}};
  assert.equal(selectVisualDeck(input,[grey],{target:1,avoidColors:['grey']})
    .diagnostics.visualQualityDecisions[0].reason,'excluded_visual_color');
  const uncertain={...opinion,slotColors:{...opinion.slotColors,outerwear:'unknown'}};
  assert.equal(selectVisualDeck(input,[uncertain],{target:1,avoidColors:['navy']})
    .diagnostics.visualQualityDecisions[0].reason,'visual_color_unverified');
  const pinned=selectVisualDeck(input,[opinion],{target:1,avoidColors:['navy'],
    requiredIds:['navy-layer']});
  assert.equal(pinned.status,'complete','a specifically pinned owned garment overrides generic preference');
  const clean={...opinion,slotColors:{...opinion.slotColors,outerwear:'camel'}};
  assert.equal(selectVisualDeck(input,[clean],{target:1,avoidColors:['navy','grey']}).status,'complete');
});


test('catalog-quality visual selector refuses to repeat the exact same cardigan across two outfits',()=>{
  const make=(id,layer)=>({id,briefId:id,brief:{id,name:id},items:[
    {id:id+'-top',macroCategory:'top',subtype:({one:'dress-shirt',two:'fine-knit',three:'polo'})[id],name:'Different top '+id},
    {id:layer,macroCategory:'outerwear',subtype:layer==='same-cardigan'?'cardigan':'blazer',name:layer==='same-cardigan'?'Same Wool Cardigan':layer},
    {id:id+'-bottom',macroCategory:'bottom',subtype:({one:'tailored-trouser',two:'chino',three:'trouser'})[id],name:'Different smart trousers '+id},
    {id:id+'-shoes',macroCategory:'shoes',subtype:({one:'derby',two:'loafer',three:'oxford'})[id],name:'Different shoes '+id},
  ]});
  const looks=[make('one','same-cardigan'),make('two','same-cardigan'),make('three','distinct-layer')];
  const verdicts=looks.map(l=>({candidateId:l.id,referenceless:true,
    visualCohesion:.91,styleFit:.90,occasionFit:.88,formalityFit:.91,
    confidence:.94,reject:false,evidence:['photo top and trousers','photo layer and shoes','outfit coherent']}));
  const input={generationVersion:STYLIST6_VERSION,shortlist:looks,diagnostics:{}};
  const before=selectVisualDeck(input,verdicts,{target:3});
  assert.equal(before.outfits.length,3,'old model verdicts allowed repeated exact product');
  const after=selectVisualDeck(input,verdicts,{target:3,requireNoProductReuse:true});
  assert.equal(after.outfits.length,2,'not allowed to call one repeated cardigan three unique outfits');
  assert.equal(after.status,'distinct_inventory_limited',
    'only two independent actual catalog SKU combinations were model-approved');
  assert.ok(!after.outfits.every(l=>l.items.some(it=>it.id==='same-cardigan')));
  const pinned=selectVisualDeck(input,verdicts,{target:3,requireNoProductReuse:true,
    requiredIds:['same-cardigan']});
  assert.equal(pinned.status,'complete','an explicitly pinned shared wardrobe anchor may legitimately repeat');
});
