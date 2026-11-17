/** One fixed six-scenario replay. Source inventory/output decks stay private.
 * The public report contains only measurements and hashes, never products. */
import fs from 'node:fs';import path from 'node:path';import esbuild from 'esbuild';import {createRequire} from 'node:module';import crypto from 'node:crypto';import cp from 'node:child_process';
const require=createRequire(import.meta.url),root=path.resolve(import.meta.dirname,'../..'),dir=process.argv[2];
if(!dir)throw Error('Provide private frozen directory');
const exportsList=['createOutfitPlan','outfitConcepts','stylingContext','prepareGarment','describeGarment','assessOutfitQuality','styleJudgeReason','fourSlotOutfitReason','validateOutfitAgainstStyle','validateOutfitCompatibility','garmentContextReason','visualAdmissionReason','applyVisualFashion','inferShopColor','isFourSlotCandidate','isNonOutfitItem','rejectForStyleOccasion','rejectForClimate','rejectForRequestIntent','rejectForActivity','retainBudgetFloorCandidates','scoreGarmentForContext','outfitStylingDistance'];
async function kernel(baseline){
 const source=fs.readFileSync(path.join(root,baseline?'.v199-balanced-build/v200.ts':'backend/supabase/functions/generate-outfits/index.ts'),'utf8')
  .replace(/^import \{ serve \} from .*$/m,'const serve=()=>{};')
  .replace(/^import \{ createClient \} from .*$/m,'const createClient=()=>{throw Error("network forbidden")};')
  +(baseline?'':"\nimport {outfitStylingDistance} from '../../../../src/domain/outfitVisualSignature.ts';")+'\nexport {'+exportsList.join(',')+'};';
 const b=await esbuild.build({stdin:{loader:'ts',contents:source,resolveDir:root+'/backend/supabase/functions/generate-outfits',sourcefile:'benchmark.ts'},bundle:true,format:'cjs',platform:'node',target:'node22',write:false});
 const module={exports:{}};new Function('module','exports','require','Deno',b.outputFiles[0].text)(module,module.exports,name=>name.startsWith('https:')?{createClient:()=>{throw Error('network forbidden')}}:require(name),{env:{get:()=>undefined}});return module.exports;
}
const [prior,next]=await Promise.all([kernel(true),kernel(false)]);
const enriched=new Map();for(const name of ['shoes','garments','roles','human-photo-review']) {
 const pixels=path.join(dir,'..',name+'-pixels.jsonl');const file=fs.existsSync(pixels)?pixels:path.join(dir,'..',name+'.jsonl');if(fs.existsSync(file))for(const line of fs.readFileSync(file,'utf8').trim().split('\n').filter(Boolean)){const row=JSON.parse(line);enriched.set(row.product_id,row);}
}
const scenarios=[['casual','Travel'],['semi_classic','Work'],['old_money','Work'],['classic','Work'],['semi_classic','Date night'],['minimalist','Everyday']];
const privateReport=[],summary=[];
for(let n=0;n<scenarios.length;n++) {
 const [style,occasion]=scenarios[n];
 const runs=[];
 for(const [label,api] of [['v200',prior],['stylist5',next]]){
  const context=api.stylingContext({style,occasion,sourceMode:'catalog'});
  const retrieval={engine:label==='stylist5'?'stylist5':'v200',dressCode:context.dressCode,concepts:api.outfitConcepts(context).slice(0,3).map(c=>({id:c.id,roles:Object.fromEntries(['top','outerwear','bottom','shoes'].map((slot,i)=>[slot,c.slots[i]])),preferredSubtypes:[...new Set(c.slots.flat())]}))};
  if(label==='stylist5'){const fx=JSON.parse(fs.readFileSync(path.join(dir,'fx.json')));retrieval.rates=fx.uzsPerUnit;}
  const functionName=label==='v200'?'catalog_stylist_candidates_v200':'catalog_stylist_candidates_compact';
  const contextSql=JSON.stringify(retrieval).replaceAll("'","''");
  const sql="set role service_role;select coalesce(jsonb_agg(row),'[]'::jsonb) from public."+functionName+"(array['top','outerwear','bottom','shoes'],'stylist5-fixed-benchmark','"+contextSql+"'::jsonb,100,0,null,null) row";
  const output=cp.execFileSync('/opt/homebrew/opt/postgresql@16/bin/psql',['-h','127.0.0.1','-p','55440','-d','postgres','-X','-A','-t','-q','-v','ON_ERROR_STOP=1','-c',sql],{encoding:'utf8',maxBuffer:32*1024*1024});
  const explain=cp.execFileSync('/opt/homebrew/opt/postgresql@16/bin/psql',['-h','127.0.0.1','-p','55440','-d','postgres','-X','-A','-t','-q','-v','ON_ERROR_STOP=1','-c',`set role service_role;explain (analyze,buffers,format json) select * from public.${functionName}(array['top','outerwear','bottom','shoes'],'stylist5-fixed-benchmark','${contextSql}'::jsonb,100,0,null,null)`],{encoding:'utf8',maxBuffer:32*1024*1024});
  const explainJson=JSON.parse(explain.trim())[0]?.Plan || {};
  const baselinePool=path.join(dir,`baseline-role-pool-${n}.json`);
  const frozen=label==='v200'&&fs.existsSync(baselinePool)?JSON.parse(fs.readFileSync(baselinePool)):JSON.parse(output);
  fs.writeFileSync(path.join(dir,'..',`retrieved-${n}-${label}.json`),JSON.stringify(frozen),{mode:0o600});
  const items=frozen.map(r=>{
   const added=label==='stylist5'?enriched.get(r.id):undefined;
   const item={...r,id:'shop_'+r.id,isShopItem:true,isWardrobe:false,macroCategory:r.fashion_slot,type:r.garment_type,imageUrl:r.image_url,shopUrl:r.source_url,
    color:api.inferShopColor(r),visualFashion:added?.profile||r.visual_fashion||undefined,visualSourceHash:r.source_hash||r.visual_source_hash||added?.profile.sourceHash,visualEmbedding:added?.embedding||r.visual_embedding,pricePeers:r.price_peers};
   const cached=r.fashion_intelligence?.version===1&&r.fashion_intelligence.descriptor?.slot===r.fashion_slot?r.fashion_intelligence.descriptor:undefined;
   return api.prepareGarment(item,cached?api.applyVisualFashion(cached,item):undefined);
  });
  const gates={visual:i=>api.visualAdmissionReason(i),context:i=>api.garmentContextReason(i,context),structure:i=>!api.isFourSlotCandidate(i),accessory:i=>api.isNonOutfitItem(i),style:i=>api.rejectForStyleOccasion(i,style,occasion),climate:i=>api.rejectForClimate(i,null),intent:i=>api.rejectForRequestIntent(i,occasion,null),activity:i=>api.rejectForActivity(i,occasion)};
  const rejections=Object.fromEntries(Object.entries(gates).map(([name,check])=>[name,items.filter(check).length]));
  const eligible=items.filter(i=>!api.visualAdmissionReason(i)&&!api.garmentContextReason(i,context)&&api.isFourSlotCandidate(i)&&!api.isNonOutfitItem(i)&&!api.rejectForStyleOccasion(i,style,occasion)&&!api.rejectForClimate(i,null)&&!api.rejectForRequestIntent(i,occasion,null)&&!api.rejectForActivity(i,occasion));
  const pool=api.retainBudgetFloorCandidates(eligible,eligible,()=>0,()=>true,undefined,24,i=>api.scoreGarmentForContext(i,context),context);
  const map=new Map(pool.map(i=>[i.id,i]));
  const evaluate=parts=>{
   if(api.fourSlotOutfitReason(parts,true))return {valid:false,score:0};
   if(!api.validateOutfitAgainstStyle({items:parts},style,map,true,{}).ok||!api.validateOutfitCompatibility({items:parts},map,style).ok)return {valid:false,score:0};
   const a=api.assessOutfitQuality(parts,style,occasion,null,null,{},context);
   return {valid:a.acceptance.passed&&!api.styleJudgeReason(parts,style,occasion,null,null,{},a,context),score:a.score};
  };
  const start=performance.now(),cpu=process.cpuUsage();
  const plan=api.createOutfitPlan(pool,context,evaluate,{requiredIds:[],target:3,priority:()=>[],cost:i=>i.price,maxEvaluations:256,maxSyncMs:600});
  const wall=performance.now()-start,usage=process.cpuUsage(cpu),deck=plan.seed;
  const descriptors=deck.map(look=>look.items.map(api.describeGarment));
  const identities=descriptors.map(d=>d.map(x=>[x.subtype,x.volume,x.texture].join('/')).join('|'));
  const distances=[];for(let a=0;a<deck.length;a++)for(let b=a+1;b<deck.length;b++)distances.push(next.outfitStylingDistance(deck[a].items,deck[b].items));
  const metrics={label,rejections,eligible:eligible.length,looks:deck.length,concepts:new Set(deck.map(l=>l.concept.id)).size,
   layerTypes:[...new Set(descriptors.map(d=>d.find(x=>x.slot==='outerwear')?.subtype))],
   topTypes:[...new Set(descriptors.map(d=>d.find(x=>x.slot==='top')?.subtype))],
   silhouettes:new Set(identities).size,minimumSemanticDistance:Math.min(1,...distances),
   meanCoherence:deck.reduce((sum,look)=>sum+evaluate(look.items).score,0)/Math.max(1,deck.length),
   photoProducts:pool.filter(i=>api.describeGarment(i).evidence.some(e=>e.startsWith('image:'))).length,poolSize:pool.length,
   maximumUzsItem:Math.max(0,...deck.flatMap(l=>l.items).filter(i=>i.currency==='UZS').map(i=>i.price)),
   evaluations:plan.diagnostics.evaluations,wallMs:Math.round(wall),cpuMs:Math.round((usage.user+usage.system)/1000),
   retrievalRows:frozen.length,retrievalPayloadBytes:Buffer.byteLength(JSON.stringify(frozen)),
   sqlExecutionMs:Math.round(explainJson['Actual Total Time']||0),sqlSharedHitBlocks:explainJson['Shared Hit Blocks']||0,
   sqlSharedReadBlocks:explainJson['Shared Read Blocks']||0};
  runs.push({metrics,deck:deck.map(l=>({concept:l.concept.id,quality:l.quality,value:l.valueScore,items:l.items.map(i=>({id:i.id,name:i.name,price:i.price,currency:i.currency,imageUrl:i.imageUrl,shopUrl:i.shopUrl,macroCategory:i.macroCategory,descriptor:api.describeGarment(i),imageSha256:i.visualFashion?.imageSha256}))})),diagnostics:{...plan.diagnostics,conceptQuality:plan.alternatives.reduce((groups,l)=>{(groups[l.concept.id]||=[]).push({q:l.quality,fit:l.conceptScore,layer:api.describeGarment(l.items.find(i=>i.macroCategory==='outerwear')||{}).subtype});return groups;},{})}});
 }
 privateReport.push({style,occasion,runs});summary.push({style,occasion,before:runs[0].metrics,after:runs[1].metrics});
}
fs.writeFileSync(path.join(dir,'..','benchmark-private.json'),JSON.stringify(privateReport,null,2),{mode:0o600});
const compiled=await esbuild.build({entryPoints:[path.join(root,'backend/supabase/functions/generate-outfits/index.ts')],bundle:true,format:'esm',platform:'neutral',target:'es2022',external:['https://*'],write:false});
const report={bundleSha256:crypto.createHash('sha256').update(compiled.outputFiles[0].contents).digest('hex'),measuredAt:new Date().toISOString(),catalog:JSON.parse(fs.readFileSync(path.join(dir,'freeze.json'))),providerCalls:0,method:'Exact v200 deterministic kernel vs candidate on the same frozen full catalog and exact old/new retrieval procedures and six inputs. Candidate adds offline evidence of those same images. No human-preference or production-latency claim.',scenarios:summary};
fs.writeFileSync(path.join(root,'docs/reports/stylist-5/benchmark.json'),JSON.stringify(report,null,2)+'\n');console.log(JSON.stringify(report,null,2));
